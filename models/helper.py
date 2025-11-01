
# import new Network name here and add in model_class args
import time

# from .Network import MYNET

from .ViT_Network import ViT_MYNET

from utils import *
from tqdm import tqdm
import torch.nn.functional as F

from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from transformers import BertTokenizer, BertModel
import numpy as np
import torch

def build_label_embedding(train_set,session,Bert_model,tokenizer,word_info, args):
    if args.dataset == "cifar100":
        classes = np.unique(train_set.classes)
        print("Number of classes:", len(classes))
        classes_int = np.unique(train_set.targets)
        print("classes_int:",classes_int)
        print('new classes for session {} : {} \n'.format(session, classes[classes_int]))
    elif args.dataset == "mini_imagenet":
        classes = np.unique(train_set.wnids)
        print("Number of classes:", len(classes))
        classes_int = np.unique(train_set.targets)
        print("classes_int:",classes_int)
        print('new classes for session {} : {} \n'.format(session, classes[classes_int]))
    elif args.dataset == "cub200" or args.dataset == "air":
        classes = np.unique(np.array(train_set.labels)[train_set.targets])
        print("Number of classes:", len(classes))
        classes_int = np.unique(train_set.targets)
        print("classes_int:",classes_int)
        print('new classes for session {} : {} \n'.format(session, classes))
        
    else:
        raise KeyError
    
    words_embed = []
    with torch.no_grad():
        Bert_model.eval()
        if args.dataset in ['cifar100', 'mini_imagenet']:
            for cls in classes[classes_int]:
                encoded_input = tokenizer(f'a photo of {cls}', return_tensors='pt')
                output = Bert_model(**encoded_input)
                # words_embed.append(bert_map(output.pooler_output))
                words_embed.append(output.pooler_output)
                word_info["label_text"] = np.append(word_info["label_text"], cls)
        elif args.dataset in ['cub200', 'air']:
            for cls in classes:
                encoded_input = tokenizer(f'a photo of {cls}', return_tensors='pt')
                output = Bert_model(**encoded_input)
                # words_embed.append(bert_map(output.pooler_output))
                words_embed.append(output.pooler_output)
                word_info["label_text"] = np.append(word_info["label_text"], f'a photo of {cls}')
        else:
            raise KeyError
        
    words_embed = torch.cat(words_embed,dim=0)
    
    if word_info["embed"] == None:
        word_info["embed"] = words_embed.cpu()
    else:
        word_info["embed"] = torch.cat([word_info["embed"].cpu(),words_embed.cpu()],dim=0)
        
    word_info["cur_embed"] = words_embed.cpu()
    word_info["cur_label"] = torch.tensor(classes_int).cpu()


def replace_base_fc(trainset, transform, model, args):
    print("[Replace Base FC - Original]")
    model = model.eval()

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=128,
                                              num_workers=4, pin_memory=True, shuffle=False)
    trainloader.dataset.transform = transform
    embedding_list = []
    label_list = []
    with torch.no_grad():
        for i, batch in enumerate(trainloader):
            data, label = [_.cuda() for _ in batch]
            model.module.mode = 'encoder'
            embedding = model(data, query=True)
            embedding_list.append(embedding.cpu())
            label_list.append(label.cpu())
        
    embedding_list = torch.cat(embedding_list, dim=0)
    label_list = torch.cat(label_list, dim=0)

    proto_list = []

    for class_index in range(args.base_class):
        data_index = (label_list == class_index).nonzero()
        embedding_this = embedding_list[data_index.squeeze(-1)]
        embedding_this = embedding_this.mean(0)
        proto_list.append(embedding_this)

    proto_list = torch.stack(proto_list, dim=0)

    model.module.fc.weight.data[:args.base_class] = proto_list

    return model

def cross_entropy(preds, targets, reduction='none'):
    labels = torch.arange(targets.shape[0]).cuda()
    loss = F.cross_entropy(preds,labels, reduction='none')
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()

def base_train(model, trainloader, optimizer, scheduler, epoch, word_info, query_info, class_list, args, loss_curve):
    print("[Base Train]")
    base_mode = model.module.mode
    
    tl = Averager_Loss()
    ta = Averager()
    model = model.train()
    tqdm_gen = tqdm(trainloader, mininterval=1.0)
    model.module.mode = "encoder"
    
    word_cur_embed = word_info['cur_embed'].clone().detach().cuda()
    word_embed = word_info['embed'].clone().detach().cuda()
    for i, batch in enumerate(tqdm_gen, 1):
        data, train_label = [_.cuda() for _ in batch]

        logits, cls_embed, prompt_embed = model(data, base=True)
        logits_ = logits[:, :args.base_class]
        if train_label.dtype != torch.long:
            train_label = train_label.long()  # <--- 增加此行进行类型转换！
        loss_ce = F.cross_entropy(logits_, train_label)
        
        if args.ED:
            loss_tri = triplet(cls_embed, prompt_embed['Vision'], query_info, train_label,loss_curve)
        else:
            loss_tri = torch.zeros(1,device='cuda')
        
        if args.SKD:
            loss_kb = knowledge_boosting(prompt_embed['Language'], word_embed, query_info, train_label,loss_curve, args)
            # loss_kb = knowledge_boosting(prompt_embed['Language'], word_embed, word_cur_embed, train_label)
        else:
            loss_kb = torch.zeros(1,device='cuda')
        
        acc = count_acc(logits_, train_label)
        total_loss = loss_ce + args.ED_hp*loss_tri + loss_kb
        
        lrc = scheduler.get_last_lr()[0]
        tl.add(total_loss.item(), len(train_label))
        ta.add(acc, len(train_label))
        
        tqdm_gen.set_description(
            'Session 0, epo {}, lrc={:.4f},total loss={:.4f}, loss_CE={:.4f}, loss_ED={:.4f}, loss_SKD={:.4f}, acc={:.4f}'.\
                format(epoch, lrc, total_loss.item(), loss_ce.item(), loss_tri.item(), loss_kb.item(), ta.item()))
        
        optimizer.zero_grad()
        total_loss.backward()
        
        grad={}
        grad['expert_prompt']=model.module.expert_prompt.grad.clone().detach().cpu()
        grad['prompt']=model.module.prompt.grad.clone().detach().cpu()
        for n, p in model.module.encoder.blocks[:2].named_parameters():
            if 'attn.qkv.weight' in n:
                grad[n] = torch.norm(p.clone().detach().cpu(),p=2,dim=1).mean()
        loss_curve['grad_list'].append(grad)
        
        optimizer.step()
        
    tl = tl.item()
    ta = ta.item()
    
    model.module.mode = base_mode
    return tl, ta


def triplet(cls_embed, vision_embed, query_info, train_label,loss_curve):
    P_head = query_info['proto'].clone().cuda()
    
    cls_logit = F.linear(cls_embed, P_head)
    cls_gt = F.cross_entropy(cls_logit, train_label, reduction='none')   #* B
    vis_logit = F.linear(vision_embed, P_head)
    vis_gt = F.cross_entropy(vis_logit, train_label, reduction='none')   #* B
    
    idx = torch.arange(vis_logit.shape[0])
    
    cls_logit[idx, train_label]=0.
    vis_logit[idx, train_label]=0.
    
    l_kl = F.kl_div(F.log_softmax(vis_logit,dim=1), F.softmax(cls_logit,dim=1), reduction='batchmean')
    l_ent = vis_gt.mean() + cls_gt.mean()
    
    loss_tri = ((l_ent/l_kl)+1).log()
    return loss_tri

def knowledge_boosting(lang_embed, word_embed, query_info, train_label, loss_curve, args):
    T = args.temperature
    idx= torch.arange(len(train_label))
    #* Original
    P_head = query_info['proto'].clone().cuda()
    
    #* =======================================================================
    lang_logit = F.linear(lang_embed, P_head)    #* Soft pred
    loss_seman = F.cross_entropy(lang_logit, train_label)
    #* KL Feature
    loss_kd = F.kl_div(F.log_softmax(lang_embed/T,dim=1), F.softmax(word_embed[train_label]/T,dim=1), reduction='batchmean')
    
    loss = loss_kd + args.gamma * loss_seman
    return args.base_skd_weight * loss


def test(model, testloader, epoch, args, session, word_info):
    #todo Test시 Prompt Selection is needed..
    test_class = args.base_class + session * args.way
    model = model.eval()
    vl = Averager_Loss()
    va = Averager()
    va_base = Averager()
    va_new = Averager()
    va_base_given_new = Averager()
    va_new_given_base = Averager()
    print("\t\t\t[Test Phase] Session: {}".format(session))
    with torch.no_grad():
        tqdm_gen = tqdm(testloader)
        for i, batch in enumerate(tqdm_gen, 1):
            data, test_label = [_.cuda() for _ in batch]
            logits = model(data, B_tuning=True)
            logits = logits[:, :test_class]
            if test_label.dtype != torch.long:
                test_label = test_label.long()  # <--- 增加此行进行类型转换！
            loss = F.cross_entropy(logits, test_label)
            
            acc = count_acc(logits, test_label)

            base_idxs = test_label < args.base_class
            if torch.any(base_idxs):
                acc_base = count_acc(logits[base_idxs, :args.base_class], test_label[base_idxs])
                acc_base_given_new = count_acc(logits[base_idxs, :], test_label[base_idxs])
                va_base.add(acc_base, len(test_label[base_idxs]))
                va_base_given_new.add(acc_base_given_new, len(test_label[base_idxs]))


            new_idxs = test_label >= args.base_class
            if torch.any(new_idxs):
                acc_new = count_acc(logits[new_idxs, args.base_class:], test_label[new_idxs] - args.base_class)
                acc_new_given_base = count_acc(logits[new_idxs, :], test_label[new_idxs])
                va_new.add(acc_new, len(test_label[new_idxs]))
                va_new_given_base.add(acc_new_given_base, len(test_label[new_idxs]))

            vl.add(loss.item(), len(test_label))
            va.add(acc, len(test_label))

        vl = vl.item()
        va = va.item()

        va_base = va_base.item()
        va_new = va_new.item()
        va_base_given_new = va_base_given_new.item()
        va_new_given_base = va_new_given_base.item()
    print('epo {}, test, loss={:.4f} acc={:.4f}'.format(epoch, vl, va))
    print('base only accuracy: {:.4f}, new only accuracy: {:.4f}'.format(va_base, va_new))
    print('base acc given new : {:.4f}'.format(va_base_given_new))
    print('new acc given base : {:.4f}'.format(va_new_given_base))

    logs = dict(num_session=session + 1, acc=va, base_acc=va_base, new_acc=va_new, base_acc_given_new=va_base_given_new,
                new_acc_given_base=va_new_given_base)

    return vl, va, logs

def build_base_proto(train_loader, model, query_info, args):
    model = model.eval()
    
    embedding_list = []
    label_list = []
    with torch.no_grad():
        for i, batch in enumerate(train_loader):
            data, label = [_.cuda() for _ in batch]
            
            model.module.mode = 'encoder'
            embedding = model(data, query=True)
            embedding_list.append(embedding.cpu())
            label_list.append(label.cpu())
            
    embedding_list = torch.cat(embedding_list, dim=0)
    label_list = torch.cat(label_list, dim=0)

    proto_list = []

    for class_index in range(args.base_class):
        data_index = (label_list == class_index).nonzero()
        embedding_this = embedding_list[data_index.squeeze(-1)]
        embedding_this = embedding_this.mean(0)
        proto_list.append(embedding_this)

    proto_list = torch.stack(proto_list, dim=0) #* num_base, feat_dim
    query_info["proto"] = proto_list
    model.module.mode = args.base_mode
    model = model.train()

"""
Per-Domain Accuracy

acc_base_domain：仅统计来自 base 数据集样本的 Top-1 准确率（使用当前已见全部类别的 logits）。

acc_inc_domain：仅统计来自 inc 数据集样本的 Top-1 准确率。

Cross-Domain Confusion（域间混淆率）

cd_base_to_inc：base 域样本被预测到“任何新类（全局 id ≥ base_class）”的占比。

cd_inc_to_base：inc 域样本被预测到“任何旧类（全局 id < base_class）”的占比。

这两项直接反映是否“偏向某个域的类别”。

Harmonic Mean（平衡性）

类别视角：hm_base_new = H(acc_base_only, acc_new_only)（你已有 base/new-only 的准确率，直接求调和平均）。

域视角：hm_dom = H(acc_base_domain, acc_inc_domain)。

Top-5 Accuracy（overall 或分域）

跨域移植时，top-5 对模型可用性的衡量更稳定。

Balanced Accuracy（宏平均召回）

每个已见类都计算召回，然后求平均，缓解类频不均或域不均的影响。

ECE（Expected Calibration Error）（可选但强烈建议）

用 softmax 置信度做 15 桶标定误差，监控跨域下的校准性。
"""
def count_topk_acc(logits, labels, k=5):
    if logits.numel() == 0:
        return 0.0
    topk = logits.topk(k, dim=1).indices
    correct = topk.eq(labels.unsqueeze(1)).any(dim=1).float().mean().item()
    return float(correct)

def expected_calibration_error(probs, labels, n_bins=15):
    # probs: softmax(logits)
    if probs.numel() == 0:
        return 0.0
    conf, pred = probs.max(dim=1)
    correct = pred.eq(labels).float()
    ece = probs.new_tensor(0.0)
    n = labels.numel()
    bin_edges = torch.linspace(0, 1, steps=n_bins+1, device=probs.device)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i+1]
        mask = (conf > lo) & (conf <= hi if i < n_bins-1 else conf <= hi)
        if mask.any():
            acc_bin = correct[mask].mean()
            conf_bin = conf[mask].mean()
            ece = ece + (mask.float().sum() / n) * (acc_bin - conf_bin).abs()
    return float(ece.item())

def _domain_ids_for_batch(start, end, part_lengths, device):
    """
    返回长度为 (end-start) 的 LongTensor，值为所属子数据集的 index（0,1,...）。
    part_lengths: [len(ds0), len(ds1), ...] 对应 ConcatDataset 子集长度。
    """
    import numpy as np
    cum = np.cumsum([0] + list(part_lengths))
    bsz = end - start
    dom = torch.zeros(bsz, dtype=torch.long, device=device)
    for d in range(len(part_lengths)):
        s = max(start, int(cum[d]))
        e = min(end,   int(cum[d+1]))
        if e > s:
            dom[s-start:e-start] = d
    return dom

@torch.no_grad()
def test_cross_domain(model, testloader, epoch, args, session, word_info):
    """
    跨数据集测试：在保留原有指标的同时，增加域级准确率、跨域混淆率、Top-5、Balanced Acc 与 ECE。
    """
    test_class = args.base_class + session * args.way
    model.eval()

    vl = Averager_Loss()
    va_overall = Averager()

    va_base_only = Averager()
    va_new_only  = Averager()
    va_base_given_new = Averager()
    va_new_given_base = Averager()

    # 域级统计
    va_dom0 = Averager()  # base 域
    va_dom1 = Averager()  # inc  域（若存在）

    top5_hits = 0
    total_samples = 0

    # per-class 统计用于 Balanced Acc
    n_cls = test_class
    correct_per_c = torch.zeros(n_cls, dtype=torch.long)
    count_per_c   = torch.zeros(n_cls, dtype=torch.long)

    # ECE
    all_probs = []
    all_labels = []

    # 识别每个子数据集长度（用于域掩码）
    datasets = getattr(testloader.dataset, 'datasets', None)
    if datasets is not None:
        part_lengths = [len(d) for d in datasets]
    else:
        part_lengths = [len(testloader.dataset)]

    print("\t\t\t[Cross-Domain Test] Session: {}".format(session))
    seen = 0
    tqdm_gen = tqdm(testloader)
    for i, batch in enumerate(tqdm_gen, 1):
        data, test_label = [_.cuda() for _ in batch]
        logits = model(data, B_tuning=True)[:, :test_class]
        if test_label.dtype != torch.long:
            test_label = test_label.long()
        loss = F.cross_entropy(logits, test_label)

        # overall top-1
        acc_overall = count_acc(logits, test_label)

        # 原有 base/new 指标（类别视角）
        base_mask = test_label < args.base_class
        new_mask  = ~base_mask
        if base_mask.any():
            acc_base_only = count_acc(logits[base_mask, :args.base_class], test_label[base_mask])
            acc_base_gn   = count_acc(logits[base_mask, :], test_label[base_mask])
            va_base_only.add(acc_base_only, int(base_mask.sum().item()))
            va_base_given_new.add(acc_base_gn, int(base_mask.sum().item()))
        if new_mask.any():
            acc_new_only = count_acc(logits[new_mask, args.base_class:], test_label[new_mask] - args.base_class)
            acc_new_gb   = count_acc(logits[new_mask, :], test_label[new_mask])
            va_new_only.add(acc_new_only, int(new_mask.sum().item()))
            va_new_given_base.add(acc_new_gb, int(new_mask.sum().item()))

        # ===== 域视角：根据样本全局位置恢复域 id =====
        bsz = data.size(0)
        dom_ids = _domain_ids_for_batch(seen, seen+bsz, part_lengths, device=test_label.device)
        seen += bsz

        dom0 = dom_ids == 0
        dom1 = dom_ids > 0  # 兼容>2子集的场景：把除第0个以外当作“inc 域”

        if dom0.any():
            va_dom0.add(count_acc(logits[dom0], test_label[dom0]), int(dom0.sum().item()))
        if dom1.any():
            va_dom1.add(count_acc(logits[dom1], test_label[dom1]), int(dom1.sum().item()))

        # 域间混淆率
        pred = logits.argmax(dim=1)
        cd_b2i = float( ((dom0) & (pred >= args.base_class)).float().sum().item() / max(1, int(dom0.sum().item())) )
        cd_i2b = float( ((dom1) & (pred <  args.base_class)).float().sum().item() / max(1, int(dom1.sum().item())) )

        # top-5（overall）
        top5_hits += (logits.topk(5, dim=1).indices.eq(test_label.unsqueeze(1))).any(dim=1).float().sum().item()
        total_samples += bsz

        # Balanced Acc 统计
        with torch.no_grad():
            for c in test_label.unique():
                c = int(c.item())
                sel = (test_label == c)
                correct_per_c[c] += (pred[sel] == test_label[sel]).sum().cpu()
                count_per_c[c]   += int(sel.sum().item())

        # ECE
        all_probs.append(F.softmax(logits, dim=1).detach())
        all_labels.append(test_label.detach())

        vl.add(loss.item(), bsz)
        va_overall.add(acc_overall, bsz)

        tqdm_gen.set_description('CD-Test s{} | loss {:.4f} acc {:.4f} | b2i {:.3f} i2b {:.3f}'.format(
            session, loss.item(), acc_overall, cd_b2i, cd_i2b
        ))

    # 汇总
    logs = dict(
        num_session=session+1,
        acc=float(va_overall.item()),
        base_acc=float(va_base_only.item()),
        new_acc=float(va_new_only.item()),
        base_acc_given_new=float(va_base_given_new.item()),
        new_acc_given_base=float(va_new_given_base.item()),
    )

    # 域级
    logs.update(dict(
        acc_base_domain=float(va_dom0.item()),
        acc_inc_domain=float(va_dom1.item()) if total_samples - part_lengths[0] > 0 else 0.0
    ))

    # 调和平均
    def _hmean(a, b):
        if a <= 0 or b <= 0:
            return 0.0
        return 2 * a * b / (a + b)
    logs['hm_base_new'] = _hmean(logs['base_acc'], logs['new_acc'])
    logs['hm_domain']   = _hmean(logs['acc_base_domain'], logs['acc_inc_domain'])

    # Top-5
    logs['top5'] = float(top5_hits / max(1, total_samples))

    # Balanced Accuracy
    valid = count_per_c > 0
    bal = (correct_per_c[valid].float() / count_per_c[valid].float()).mean().item() if valid.any() else 0.0
    logs['balanced_acc'] = float(bal)

    # ECE
    all_probs = torch.cat(all_probs, dim=0) if len(all_probs) else torch.empty(0)
    all_labels = torch.cat(all_labels, dim=0) if len(all_labels) else torch.empty(0, dtype=torch.long)
    logs['ece'] = expected_calibration_error(all_probs, all_labels, n_bins=15) if all_probs.numel() else 0.0

    print('epo {}, CD-test, loss={:.4f} acc={:.4f}'.format(epoch, vl.item(), va_overall.item()))
    print('base/new only: {:.4f}/{:.4f} | base_given_new {:.4f} | new_given_base {:.4f}'.format(
        logs['base_acc'], logs['new_acc'], logs['base_acc_given_new'], logs['new_acc_given_base']))
    print('domain acc (base/inc): {:.4f}/{:.4f} | HM(class/domain): {:.4f}/{:.4f}'.format(
        logs['acc_base_domain'], logs['acc_inc_domain'], logs['hm_base_new'], logs['hm_domain']))
    print('Top-5: {:.4f} | BalancedAcc: {:.4f} | ECE: {:.4f}'.format(
        logs['top5'], logs['balanced_acc'], logs['ece']))

    return vl.item(), va_overall.item(), logs