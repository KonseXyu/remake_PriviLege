
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


def replace_base_fc(trainset, transform, model, args, query_info=None):
    """
    Replace FC with class prototypes computed from base-trainset.
    - encoder  : mean of encoder features (original behavior)
    - proto: mean of 0.5*(CLS + Vision) extracted via prompt_encode with PKT (proto style)
                 and optionally sync query_info['proto'].
    """
    mode = getattr(args, 'replace_base_mode', 'encoder')
    print(f"[Replace Base FC - Mode: {mode}]")
    model = model.eval()

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=128,
                                              num_workers=4, pin_memory=True, shuffle=False)
    # 使用测试的 transform 来对齐评测分布
    trainloader.dataset.transform = transform

    embedding_list, label_list = [], []
    with torch.no_grad():
        for _, batch in enumerate(trainloader):
            data, label = [_.cuda() for _ in batch]

            if mode == 'proto':
                # === proto 风格：与测试/训练同一套特征路径 ===
                # ViT_Network 中已有 prompt_encode 接口；取 CLS 与 Vision 两路后做 0.5*(CLS+Vision)
                # （接口参考）:contentReference[oaicite:3]{index=3}
                cls_embed, prompt_embed = model.module.prompt_encode(
                    data, prompt_feat=True, B_tuning=True, eval=False
                )
                embedding = 0.5 * (cls_embed + prompt_embed['Vision'])
            else:
                # === 原实现：encoder 特征均值 ===
                model.module.mode = 'encoder'
                embedding = model(data, query=True)

            embedding_list.append(embedding.detach().cpu())
            label_list.append(label.cpu())

    embedding_list = torch.cat(embedding_list, dim=0)
    label_list = torch.cat(label_list, dim=0)

    # 按 base_class 逐类取均值作为原型
    proto_list = []
    for class_index in range(args.base_class):
        idx = (label_list == class_index).nonzero().squeeze(-1)
        emb_this = embedding_list.index_select(0, idx)
        proto_list.append(emb_this.mean(0))
    proto_list = torch.stack(proto_list, dim=0)  # [base_class, D]

    # 1) 写回 FC
    model.module.fc.weight.data[:args.base_class] = proto_list.cuda()

    # 2) 可选：把这套原型同步给 query_info（与 proto 一致）
    if (mode == 'proto') and (query_info is not None):
        query_info["proto"] = proto_list.cpu()

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

        if args.proto_classifier:
            # 组合特征：与 proto 一致，用 0.5*(CLS + Vision)
            combined = 0.5 * (cls_embed + prompt_embed['Vision'])
            # 归一化
            combined = F.normalize(combined, dim=1)
            # 取 base-session 的原型（来自 build_base_proto）
            P = query_info['proto'][:args.base_class].clone().cuda()
            P = F.normalize(P, dim=1)
            logits_ = args.proto_temp * F.linear(combined, P)  # 余弦 * 温度
        else:
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


def test(model, testloader, epoch, args, session, word_info, query_info=None):
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
            if args.proto_classifier:
                # 直接用 prompt_encode 避开 forward 的 mode 分支
                cls_embed, prompt_embed = model.module.prompt_encode(
                    data, prompt_feat=True, B_tuning=True, eval=False
                )
                combined = 0.5 * (cls_embed + prompt_embed['Vision'])
                combined = F.normalize(combined, dim=1)

                if query_info is not None and query_info.get('proto', None) is not None:
                    P = query_info['proto'][:test_class].clone().cuda()
                else:
                    # 兜底：用 FC 权重当原型（如果之前已 replace_fc / update_fc_avg，会等价）
                    P = model.module.fc.weight[:test_class].detach()
                P = F.normalize(P, dim=1)

                logits = args.proto_temp * F.linear(combined, P)
            else:
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

    # ==== 选择 base 原型的特征口径 ====
    mode = getattr(args, 'base_proto_mode', None)
    if mode is None:
        # 自动：若启用“原型余弦+温度”分类器，则用 proto；否则 encoder
        mode = 'proto' if getattr(args, 'proto_classifier', False) else 'encoder'
    print(f"[Build Base Proto - Mode: {mode}]")

    embedding_list = []
    label_list = []
    with torch.no_grad():
        for i, batch in enumerate(train_loader):
            data, label = [_.cuda() for _ in batch]

            if mode == 'proto':
                # proto 同款：0.5*(CLS + Vision)
                cls_embed, prompt_embed = model.module.prompt_encode(
                    data, prompt_feat=True, B_tuning=True, eval=False
                )
                embedding = 0.5 * (cls_embed + prompt_embed['Vision'])
            else:
                # 原始实现：encoder 特征
                model.module.mode = 'encoder'
                embedding = model(data, query=True)

            embedding_list.append(embedding.detach().cpu())
            label_list.append(label.detach().cpu())

    embedding_list = torch.cat(embedding_list, dim=0)
    label_list = torch.cat(label_list, dim=0)

    # === 按 base_class 逐类求均值原型 ===
    proto_list = []
    for class_index in range(args.base_class):
        data_index = (label_list == class_index).nonzero().squeeze(-1)
        embedding_this = embedding_list.index_select(0, data_index)
        proto_list.append(embedding_this.mean(0))

    proto_list = torch.stack(proto_list, dim=0)  # [base_class, D]
    query_info["proto"] = proto_list             # 与原实现相同
    model.module.mode = args.base_mode           # 复原运行模式（与原实现一致）
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

@torch.no_grad()
def test_cross_domain(model, testloader, epoch, args, session, word_info, query_info=None):
    """
    Cross-dataset evaluation with domain-aware metrics.

    Assumptions:
    - testloader.dataset is a Concat-like object with attribute `datasets`,
      ordered as [base_test, inc_test_remap, ...], and DataLoader uses shuffle=False.
    - Labels are global class ids.
    - In session s, only first K_s = base_class + s*way logits are evaluated.
    Returns:
      loss_avg, overall_acc, logs(dict with rich metrics)
    """
    import torch
    import torch.nn.functional as F
    from tqdm import tqdm

    device = next(model.parameters()).device
    model.eval()

    test_class = args.base_class + session * args.way  # enabled classes this session

    vl = Averager_Loss()
    va_overall = Averager()
    va_base_only = Averager()
    va_new_only  = Averager()
    va_base_given_new = Averager()
    va_new_given_base = Averager()

    # --- domain partition lengths (for ConcatDataset)
    datasets = getattr(testloader.dataset, 'datasets', None)
    if datasets is not None and len(datasets) >= 1:
        part_lengths = [len(d) for d in datasets]
    else:
        part_lengths = [len(testloader.dataset)]  # single domain fallback

    # --- accumulators
    total = 0
    top5_hits = 0

    n_cls = test_class
    correct_per_c = torch.zeros(n_cls, dtype=torch.long)   # CPU accumulators for balanced_acc
    count_per_c   = torch.zeros(n_cls, dtype=torch.long)

    dom0_total = 0  # base domain (first part)
    dom1_total = 0  # inc domain (all remaining parts)
    dom0_correct = 0
    dom1_correct = 0
    base_to_inc = 0
    inc_to_base = 0

    # --- ECE accumulators (on CPU). We push GPU sums via .item()
    n_bins = 15
    bin_count = torch.zeros(n_bins, dtype=torch.long)      # CPU
    bin_conf_sum = torch.zeros(n_bins, dtype=torch.double) # CPU
    bin_correct_sum = torch.zeros(n_bins, dtype=torch.double) # CPU

    def _dom_masks(seen, bsz, part_lengths, device):
        """
        Return boolean masks for domain-0 (base) and domain-1+ (all inc)
        based on global position `seen` in a non-shuffled ConcatDataset.
        """
        if len(part_lengths) == 0:
            m0 = torch.zeros(bsz, dtype=torch.bool, device=device)
            return m0, ~m0
        dom0_end = part_lengths[0]
        m0_count = max(0, min(bsz, dom0_end - seen))
        m0 = torch.zeros(bsz, dtype=torch.bool, device=device)
        if m0_count > 0:
            m0[:m0_count] = True
        has_inc = sum(part_lengths) > part_lengths[0]
        m1 = (~m0) if has_inc else torch.zeros(bsz, dtype=torch.bool, device=device)
        return m0, m1

    def _hmean(a, b):
        return 0.0 if a <= 0 or b <= 0 else 2 * a * b / (a + b)

    seen = 0
    for batch in tqdm(testloader):
        data, test_label = batch
        data = data.to(device)
        test_label = test_label.to(device).long()

        # forward & restrict to seen classes this session
        if getattr(args, 'proto_classifier', False):
            # 取 CLS 与 Vision 两路特征，并做与训练一致的组合
            cls_embed, prompt_embed = model.module.prompt_encode(
                data, prompt_feat=True, B_tuning=True, eval=False
            )
            combined = 0.5 * (cls_embed + prompt_embed['Vision'])
            combined = F.normalize(combined, dim=1)

            # 取“已见类”的原型作为分类器（优先用 query_info['proto']，兜底用 FC 权重）
            if (query_info is not None) and (query_info.get('proto', None) is not None):
                P = query_info['proto'][:test_class].clone().cuda()
            else:
                P = model.module.fc.weight[:test_class].detach()
            P = F.normalize(P, dim=1)

            logits = getattr(args, 'proto_temp', 10.0) * F.linear(combined, P)
        else:
            logits = model(data, B_tuning=True)[:, :test_class]

        loss = F.cross_entropy(logits, test_label)

        # overall
        acc = count_acc(logits, test_label)

        # base/new (class-view)
        base_mask = test_label < args.base_class
        new_mask  = ~base_mask
        if base_mask.any():
            va_base_only.add(
                count_acc(logits[base_mask, :args.base_class], test_label[base_mask]),
                int(base_mask.sum().item())
            )
            va_base_given_new.add(
                count_acc(logits[base_mask, :], test_label[base_mask]),
                int(base_mask.sum().item())
            )
        if new_mask.any():
            va_new_only.add(
                count_acc(logits[new_mask, args.base_class:], test_label[new_mask] - args.base_class),
                int(new_mask.sum().item())
            )
            va_new_given_base.add(
                count_acc(logits[new_mask, :], test_label[new_mask]),
                int(new_mask.sum().item())
            )

        # predictions
        pred = logits.argmax(dim=1)

        # domain-view
        bsz = data.size(0)
        dom0_mask, dom1_mask = _dom_masks(seen, bsz, part_lengths, device=test_label.device)
        seen += bsz

        if dom0_mask.any():
            dom0_total   += int(dom0_mask.sum().item())
            dom0_correct += int((pred[dom0_mask] == test_label[dom0_mask]).sum().item())
            base_to_inc  += int((pred[dom0_mask] >= args.base_class).sum().item())  # base→inc confusion
        if dom1_mask.any():
            dom1_total   += int(dom1_mask.sum().item())
            dom1_correct += int((pred[dom1_mask] == test_label[dom1_mask]).sum().item())
            inc_to_base  += int((pred[dom1_mask] <  args.base_class).sum().item())  # inc→base confusion

        # top-5 (overall)
        k = 5 if test_class >= 5 else test_class
        if k > 0:
            top5_hits += (logits.topk(k, dim=1).indices.eq(test_label.unsqueeze(1))).any(dim=1).float().sum().item()

        # balanced accuracy (per-class recall) on CPU
        cnt = torch.bincount(test_label, minlength=n_cls).cpu()
        correct_mask = (pred == test_label)
        corr = torch.bincount(test_label[correct_mask], minlength=n_cls).cpu()
        count_per_c[:len(cnt)]   += cnt
        correct_per_c[:len(corr)] += corr

        # ECE (streaming on CPU; move batch stats via .item())
        probs = torch.softmax(logits, dim=1)
        conf, _ = probs.max(dim=1)
        idx = torch.clamp((conf * n_bins).floor().long(), max=n_bins-1)  # [0, n_bins-1]
        correct01 = correct_mask.to(torch.double)
        for b in range(n_bins):
            m = (idx == b)
            if m.any():
                bc = int(m.sum().item())
                bin_count[b]       += bc
                bin_conf_sum[b]    += float(conf[m].double().sum().item())
                bin_correct_sum[b] += float(correct01[m].sum().item())

        # loss/overall
        vl.add(loss.item(), bsz)
        va_overall.add(acc, bsz)
        total += bsz

    # ===== aggregate =====
    overall_acc = float(va_overall.item())
    base_acc    = float(va_base_only.item())
    new_acc     = float(va_new_only.item())
    base_acc_gn = float(va_base_given_new.item())
    new_acc_gb  = float(va_new_given_base.item())

    acc_base_domain = float(dom0_correct / dom0_total) if dom0_total > 0 else 0.0
    acc_inc_domain  = float(dom1_correct / dom1_total) if dom1_total > 0 else 0.0

    hm_base_new = _hmean(base_acc, new_acc)
    hm_domain   = _hmean(acc_base_domain, acc_inc_domain)
    top5 = float(top5_hits / max(1, total))

    valid = (count_per_c > 0)
    balanced_acc = float((correct_per_c[valid].float() / count_per_c[valid].float()).mean().item()) if valid.any() else 0.0

    # ECE (all on CPU)
    N = int(bin_count.sum().item())
    if N > 0:
        nz = bin_count > 0
        bin_acc  = torch.zeros(n_bins, dtype=torch.double)
        bin_conf = torch.zeros(n_bins, dtype=torch.double)
        bin_acc[nz]  = bin_correct_sum[nz] / bin_count[nz].double()
        bin_conf[nz] = bin_conf_sum[nz]   / bin_count[nz].double()
        weights = bin_count.double() / N
        ece = float((weights[nz] * (bin_acc[nz] - bin_conf[nz]).abs()).sum().item())
    else:
        ece = 0.0

    logs = dict(
        num_session = session + 1,
        acc = overall_acc,
        base_acc = base_acc,
        new_acc = new_acc,
        base_acc_given_new = base_acc_gn,
        new_acc_given_base = new_acc_gb,
        acc_base_domain = acc_base_domain,
        acc_inc_domain  = acc_inc_domain,
        hm_base_new = hm_base_new,
        hm_domain   = hm_domain,
        top5 = top5,
        balanced_acc = balanced_acc,
        ece = ece,
        cd_base_to_inc = float(base_to_inc / max(1, dom0_total)),
        cd_inc_to_base = float(inc_to_base / max(1, dom1_total)),
    )

    print(('[CD-Test] epoch {} | overall {:.4f} | base/new {:.4f}/{:.4f} | '
           'dom(base/inc) {:.4f}/{:.4f} | HM(class/domain) {:.4f}/{:.4f} | '
           'Top-5 {:.4f} | BalAcc {:.4f} | ECE {:.4f} | B→I {:.3f} | I→B {:.3f}')
          .format(epoch, overall_acc, base_acc, new_acc,
                  acc_base_domain, acc_inc_domain, hm_base_new, hm_domain,
                  top5, balanced_acc, ece,
                  logs["cd_base_to_inc"], logs["cd_inc_to_base"]))

    return vl.item(), overall_acc, logs

