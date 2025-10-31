import importlib
import os
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import ConcatDataset

from dataloader.sampler import CategoriesSampler


DATASET_REGISTRY = {
    'cifar100': {
        'module': 'dataloader.cifar100.cifar',
        'class_name': 'CIFAR100',
        'base_class': 60,
        'num_classes': 100,
        'way': 5,
        'shot': 5,
        'sessions': 9,
    },
    'cub200': {
        'module': 'dataloader.cub200.cub200',
        'class_name': 'CUB200',
        'base_class': 100,
        'num_classes': 200,
        'way': 10,
        'shot': 5,
        'sessions': 11,
    },
    'mini_imagenet': {
        'module': 'dataloader.miniimagenet.miniimagenet',
        'class_name': 'MiniImageNet',
        'base_class': 60,
        'num_classes': 100,
        'way': 5,
        'shot': 5,
        'sessions': 9,
    },
}


class _ConcatWithTransform(ConcatDataset):
    def __init__(self, datasets, transform=None):
        super().__init__(datasets)
        self.transform = transform


def _load_dataset_info(name):
    config = DATASET_REGISTRY[name]
    module = importlib.import_module(config['module'])
    dataset_class = getattr(module, config['class_name'])
    return SimpleNamespace(
        name=name,
        module=module,
        dataset_class=dataset_class,
        base_class=config['base_class'],
        num_classes=config['num_classes'],
        way=config['way'],
        shot=config['shot'],
        sessions=config['sessions'],
    )

def set_up_datasets(args):
    base_dataset_name = args.base_dataset or args.dataset
    incremental_dataset_name = args.incremental_dataset or base_dataset_name

    base_info = _load_dataset_info(base_dataset_name)
    incremental_info = _load_dataset_info(incremental_dataset_name)

    args.base_dataset = base_dataset_name
    args.incremental_dataset = incremental_dataset_name
    args.base_dataset_info = base_info
    args.incremental_dataset_info = incremental_info

    args.dataset = base_dataset_name
    args.base_class = base_info.base_class

    incremental_sessions = max(incremental_info.sessions, 1)
    args.sessions = incremental_sessions
    args.way = incremental_info.way
    args.shot = incremental_info.shot
    args.num_classes = args.base_class + args.way * (args.sessions - 1)

    args.dataset_modules = {
        base_info.name: base_info,
        incremental_info.name: incremental_info,
    }

    args.incremental_label_map = {}
    args.seen_incremental_source_labels = []

    return args

def get_dataloader(args,session):
    if session == 0:
        trainset, trainloader, testloader = get_base_dataloader(args)
    else:
        trainset, trainloader, testloader = get_new_dataloader(args, session)
    return trainset, trainloader, testloader

def _get_dataset_class(args, dataset_name):
    dataset_info = args.dataset_modules[dataset_name]
    return dataset_info.dataset_class


def get_base_dataloader(args, clip_trsf=None):
    dataset_name = args.base_dataset
    dataset_cls = _get_dataset_class(args, dataset_name)
    class_index = np.arange(args.base_class)

    if dataset_name == 'cifar100':
        trainset = dataset_cls(root=args.dataroot, train=True, download=True, transform=clip_trsf,
                               index=class_index, base_sess=True, is_vit=args.vit, is_clip=args.clip)
        testset = dataset_cls(root=args.dataroot, train=False, download=False,
                              index=class_index, base_sess=True, is_vit=args.vit, is_clip=args.clip)
    elif dataset_name == 'cub200':
        trainset = dataset_cls(root=args.dataroot, train=True,
                               index=class_index, base_sess=True, is_clip=args.clip)
        testset = dataset_cls(root=args.dataroot, train=False, index=class_index, is_clip=args.clip)
    elif dataset_name == 'mini_imagenet':
        trainset = dataset_cls(root=args.dataroot, train=True,
                               index=class_index, base_sess=True, is_clip=args.clip)
        testset = dataset_cls(root=args.dataroot, train=False, index=class_index, is_clip=args.clip)
    else:
        raise ValueError(f"Unsupported base dataset: {dataset_name}")

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=args.batch_size_base, shuffle=True,
                                              num_workers=4)
    testloader = torch.utils.data.DataLoader(
        dataset=testset, batch_size=args.test_batch_size, shuffle=False, num_workers=4)

    return trainset, trainloader, testloader

def get_standard_data_loader(args):
    #todo for the standard CIL Data
    pass

def get_base_dataloader_meta(args):
    dataset_name = args.base_dataset
    dataset_cls = _get_dataset_class(args, dataset_name)
    txt_path = _resolve_session_file(dataset_name, 0)
    class_index = np.arange(args.base_class)

    if dataset_name == 'cifar100':
        trainset = dataset_cls(root=args.dataroot, train=True, download=True,
                               index=class_index, base_sess=True, is_vit=args.vit, is_clip=args.clip)
        testset = dataset_cls(root=args.dataroot, train=False, download=False,
                              index=class_index, base_sess=True, is_vit=args.vit, is_clip=args.clip)
    elif dataset_name == 'cub200':
        trainset = dataset_cls(root=args.dataroot, train=True,
                               index_path=txt_path, is_clip=args.clip)
        testset = dataset_cls(root=args.dataroot, train=False,
                              index=class_index, is_clip=args.clip)
    elif dataset_name == 'mini_imagenet':
        trainset = dataset_cls(root=args.dataroot, train=True,
                               index_path=txt_path, is_clip=args.clip)
        testset = dataset_cls(root=args.dataroot, train=False,
                              index=class_index, is_clip=args.clip)
    else:
        raise ValueError(f"Unsupported base dataset: {dataset_name}")

    sampler = CategoriesSampler(trainset.targets, args.train_episode, args.episode_way,
                                args.episode_shot + args.episode_query)

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_sampler=sampler, num_workers=args.num_workers)

    testloader = torch.utils.data.DataLoader(
        dataset=testset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers)

    return trainset, trainloader, testloader

def _resolve_session_file(dataset_name, session):
    session_filename = f'session_{session + 1}.txt'
    return os.path.join('data', 'index_list', dataset_name, session_filename)


def _extract_session_labels(dataset):
    if hasattr(dataset, 'data2label'):
        labels = [dataset.data2label[path] for path in dataset.data]
    else:
        labels = dataset.targets
    return sorted(set(labels))


def _map_targets(dataset, mapping):
    if hasattr(dataset, 'data2label'):
        dataset.targets = [mapping[int(dataset.data2label[path])] for path in dataset.data]
    else:
        dataset.targets = [mapping[int(target)] for target in dataset.targets]


def get_new_dataloader(args, session, clip_trsf=None):
    dataset_name = args.incremental_dataset
    dataset_cls = _get_dataset_class(args, dataset_name)
    txt_path = _resolve_session_file(dataset_name, session)

    if dataset_name == 'cifar100':
        class_index = open(txt_path).read().splitlines()
        trainset = dataset_cls(root=args.dataroot, train=True, download=False, transform=clip_trsf,
                               index=class_index, base_sess=False, is_vit=args.vit, is_clip=args.clip)
    elif dataset_name == 'cub200':
        trainset = dataset_cls(root=args.dataroot, train=True,
                               index_path=txt_path, is_clip=args.clip)
    elif dataset_name == 'mini_imagenet':
        trainset = dataset_cls(root=args.dataroot, train=True,
                               index_path=txt_path, is_clip=args.clip)
    else:
        raise ValueError(f"Unsupported incremental dataset: {dataset_name}")

    session_labels = [int(label) for label in _extract_session_labels(trainset)]
    global_start = args.base_class + (session - 1) * args.way
    label_mapping = {int(src_label): int(global_start + idx) for idx, src_label in enumerate(session_labels)}

    args.incremental_label_map.update(label_mapping)
    args.seen_incremental_source_labels = sorted(set(args.seen_incremental_source_labels + session_labels))

    _map_targets(trainset, label_mapping)

    if args.batch_size_new == 0:
        batch_size_new = len(trainset)
        trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size_new, shuffle=False,
                                                  num_workers=args.num_workers)
    else:
        trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=args.batch_size_new, shuffle=True,
                                                  num_workers=args.num_workers)

    base_dataset_cls = _get_dataset_class(args, args.base_dataset)
    base_class_index = np.arange(args.base_class)

    if args.base_dataset == 'cifar100':
        base_testset = base_dataset_cls(root=args.dataroot, train=False, download=False,
                                        index=base_class_index, base_sess=True, is_vit=args.vit, is_clip=args.clip)
    elif args.base_dataset == 'cub200':
        base_testset = base_dataset_cls(root=args.dataroot, train=False, index=base_class_index, is_clip=args.clip)
    elif args.base_dataset == 'mini_imagenet':
        base_testset = base_dataset_cls(root=args.dataroot, train=False, index=base_class_index, is_clip=args.clip)
    else:
        raise ValueError(f"Unsupported base dataset: {args.base_dataset}")

    incremental_indices = args.seen_incremental_source_labels
    if dataset_name == 'cifar100':
        testset_inc = dataset_cls(root=args.dataroot, train=False, download=False, transform=clip_trsf,
                                  index=incremental_indices, base_sess=False, is_vit=args.vit, is_clip=args.clip)
    elif dataset_name == 'cub200':
        testset_inc = dataset_cls(root=args.dataroot, train=False,
                                  index=incremental_indices, is_clip=args.clip)
    elif dataset_name == 'mini_imagenet':
        testset_inc = dataset_cls(root=args.dataroot, train=False,
                                  index=incremental_indices, is_clip=args.clip)

    testset_inc.targets = [args.incremental_label_map[int(src_label)] for src_label in testset_inc.targets]

    if args.base_dataset == args.incremental_dataset:
        testset = testset_inc
    else:
        combined_transform = getattr(testset_inc, 'transform', None)
        testset = _ConcatWithTransform([base_testset, testset_inc], transform=combined_transform)

    testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=args.test_batch_size, shuffle=False,
                                             num_workers=args.num_workers)

    return trainset, trainloader, testloader

def get_session_classes(args, session):
    base_classes = list(range(args.base_class))
    incremental_classes = sorted(args.incremental_label_map.values())
    return np.array(base_classes + incremental_classes)
