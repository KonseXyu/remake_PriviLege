
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
            loss_kb = knowledge_boosting(prompt_embed['Language'], word_embed, query_info, train_label,loss_curve)
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

def knowledge_boosting(lang_embed, word_embed, query_info, train_label, loss_curve):
    T = 2.
    idx= torch.arange(len(train_label))
    #* Original
    P_head = query_info['proto'].clone().cuda()
    
    #* =======================================================================
    lang_logit = F.linear(lang_embed, P_head)    #* Soft pred
    loss_seman = F.cross_entropy(lang_logit, train_label)
    #* KL Feature
    loss_kd = F.kl_div(F.log_softmax(lang_embed/T,dim=1), F.softmax(word_embed[train_label]/T,dim=1), reduction='batchmean')
    
    loss = loss_kd + 0.1*loss_seman
    return 0.5*loss


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

