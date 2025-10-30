import os
import os.path as osp

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class MiniImageNet(Dataset):

    def __init__(self, root='./data', train=True,
                 transform=None,
                 index_path=None, index=None, base_sess=None, is_clip=False, global_offset=0): # 接收一个用于全局偏移的新参数
        if train:
            setname = 'train'
        else:
            setname = 'test'
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.train = train  # training set or test set
        self.IMAGE_PATH = os.path.join(root, 'miniimagenet/images')
        self.SPLIT_PATH = os.path.join(root, 'miniimagenet/split')

        csv_path = osp.join(self.SPLIT_PATH, setname + '.csv')
        lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]

        self.data = []
        self.targets = []
        self.data2label = {}
        lb = -1

        self.wnids = []

        for l in lines:
            name, wnid = l.split(',')
            path = osp.join(self.IMAGE_PATH, name)
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1
            self.data.append(path)
            self.targets.append(lb)
            self.data2label[path] = lb
        
        self.classes = self.mapping_clsidx_to_txt()
        for i in range(len(self.wnids)):
            self.wnids[i] = self.classes[list(self.classes.keys())[i]]
        

        if train and index is not None:
            image_size = 224
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size),
                # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])])
            if base_sess:
                self.data, self.targets = self.SelectfromClasses(self.data, self.targets, index)
            else:
                self.data, self.targets = self.SelectfromTxt(self.data2label, index_path)
                # --- !!! 关键修改：将全局标签重新映射为本地标签 (0, 1, 2, ...) !!! ---
                # 获取当前会话中的所有唯一全局标签
                unique_targets = sorted(list(set(self.targets)))
                # 创建映射字典：全局标签 -> 本地标签 (0, 1, 2, ...)
                target_mapping = {original_label: new_label for new_label, original_label in enumerate(unique_targets)}
                # 应用映射
                self.targets = [target_mapping[t] for t in self.targets]
                # -----------------------------------------------------------------------
        else:
            image_size = 224
            self.transform = transforms.Compose([
                transforms.Resize([image_size, image_size]),
                # transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])])
            if index is not None:
                self.data, self.targets = self.SelectfromClasses(self.data, self.targets, index)
                # --- !!! 关键修改：将标签平移到正确的全局偏移位置 !!! ---
                # 对于测试集，这通常是正确的全局标签。对于训练集，如果采用 local remapping，也需要平移。
                # 由于您的训练逻辑 (使用 full 110 head) 需要全局标签，我们在这里进行偏移。
                self.targets = np.array(self.targets) + global_offset
                self.targets = self.targets.tolist()
                # --------------------------------------------------------


    def SelectfromTxt(self, data2label, index_path):
        index=[]
        lines = [x.strip() for x in open(index_path, 'r').readlines()]
        for line in lines:
            index.append(line.split('/')[3])
        data_tmp = []
        targets_tmp = []
        for i in index:
            img_path = os.path.join(self.IMAGE_PATH, i)
            data_tmp.append(img_path)
            targets_tmp.append(data2label[img_path])

        return data_tmp, targets_tmp

    def SelectfromClasses(self, data, targets, index):
        data_tmp = []
        targets_tmp = []
        for i in index:
            ind_cl = np.where(i == targets)[0]
            for j in ind_cl:
                data_tmp.append(data[j])
                targets_tmp.append(targets[j])

        return data_tmp, targets_tmp

    def mapping_clsidx_to_txt(self):
        classes = {}
        lines = [x.strip() for x in open('/root/autodl-tmp/CD-FSCIL-My/dataloader/miniimagenet/map_clsloc.txt', 'r').readlines()][1:]
        for l in lines:
            name, class_num, class_txt = l.split(' ')
            if name not in classes.keys():
                classes[name]=class_txt
        return classes
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):

        path, targets = self.data[i], self.targets[i]
        path = os.path.normpath(path)
        image = self.transform(Image.open(path).convert('RGB'))
        return image, targets


if __name__ == '__main__':
    txt_path = "../../data/index_list/mini_imagenet/session_2.txt"
    base_class = 100
    class_index = np.arange(base_class)
    dataroot = 'D:\Code\pythonCode\data'

    trainset = MiniImageNet(root=dataroot, train=True, transform=None, index_path=txt_path)
    # print(trainset.targets)
    cls = np.unique(trainset.targets)
    batch_size_base = len(trainset)
    print(batch_size_base)
    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size_base, shuffle=True, num_workers=8,
                                              pin_memory=True)
