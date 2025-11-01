
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from dataloader.sampler import CategoriesSampler

# =====================
# Cross-dataset helpers
# =====================
from collections import defaultdict
import random

class RemapTargets(Dataset):
    """
    Wrap a dataset to (1) filter by selected class ids, (2) remap original class ids to GLOBAL ids,
    and (3) (optionally) expose class_names in the selected order (for label embedding).
    """
    def __init__(self, base_ds: Dataset, selected_orig_ids, orig_to_global: dict, class_names=None):
        self.base = base_ds
        self.selected = set(int(x) for x in selected_orig_ids)
        self.orig_to_global = {int(k): int(v) for k, v in orig_to_global.items()}
        # indices belonging to selected classes
        base_targets = getattr(self.base, 'targets')
        self.idxs = [i for i, t in enumerate(base_targets) if int(t) in self.selected]
        # GLOBAL ids
        self.targets = [self.orig_to_global[int(base_targets[i])] for i in self.idxs]
        # (optional) human-readable names in the SAME order as the GLOBAL ids for *this* session
        self.class_names = class_names
        # keep a pointer to transform for downstream code compatibility
        self.transform = getattr(self.base, 'transform', None)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, i):
        x, _t = self.base[self.idxs[i]]
        return x, self.targets[i]


class FewShotSubset(Dataset):
    """
    Pick K samples per class (by GLOBAL ids) from a RemapTargets dataset.
    """
    def __init__(self, remapped_ds: RemapTargets, k_shot: int, rng=None):
        self.base = remapped_ds
        self.k = int(k_shot)
        rng = rng or random.Random(0)
        buckets = defaultdict(list)
        for i, gid in enumerate(self.base.targets):
            buckets[int(gid)].append(i)
        self.idxs = []
        for gid, lst in buckets.items():
            rng.shuffle(lst)
            take = lst[:self.k] if self.k > 0 else lst
            self.idxs.extend(take)
        # keep targets aligned
        self.targets = [self.base.targets[i] for i in self.idxs]
        self.class_names = getattr(self.base, 'class_names', None)
        self.transform = getattr(self.base, 'transform', None)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, i):
        x, _t = self.base[self.idxs[i]]
        return x, self.targets[i]


class ConcatWithTransform(ConcatDataset):
    """
    torch.utils.data.ConcatDataset without losing a 'transform' attribute.
    We simply forward the transform of the first child dataset if available.
    """
    def __init__(self, datasets):
        super().__init__(datasets)
        self.transform = getattr(datasets[0], 'transform', None)


def _defaults_for(dataset_name):
    """Return common FSCIL defaults for each dataset."""
    if dataset_name == 'mini_imagenet':
        return dict(base_class=60, way=5, sessions=9, num_classes=100)
    if dataset_name == 'cub200':
        return dict(base_class=100, way=10, sessions=11, num_classes=200)
    if dataset_name == 'cifar100':
        return dict(base_class=60, way=5, sessions=9, num_classes=100)
    raise KeyError(dataset_name)


def _build_raw(dataset_name, root, train, args):
    """Instantiate the raw dataset for reading all classes; selection happens later via wrappers.
       IMPORTANT: Always pass `index` (full class range). For train=True on some datasets (e.g., CUB/Mini),
       set base_sess=True so they DO NOT try to read a txt `index_path`.
    """
    if dataset_name == 'cub200':
        from dataloader.cub200.cub200 import CUB200
        n = _defaults_for('cub200')['num_classes']
        full_index = np.arange(n)
        if train:
            return CUB200(root=root, train=True, index=full_index, base_sess=True)
        else:
            return CUB200(root=root, train=False, index=full_index)

    elif dataset_name == 'mini_imagenet':
        from dataloader.miniimagenet.miniimagenet import MiniImageNet
        n = _defaults_for('mini_imagenet')['num_classes']
        full_index = np.arange(n)
        if train:
            return MiniImageNet(root=root, train=True, index=full_index, base_sess=True)
        else:
            return MiniImageNet(root=root, train=False, index=full_index)

    elif dataset_name == 'cifar100':
        from dataloader.cifar100.cifar import CIFAR100
        n = _defaults_for('cifar100')['num_classes']
        full_index = np.arange(n)
        if train:
            return CIFAR100(root=root, train=True, download=False, index=full_index, base_sess=True,
                            is_vit=getattr(args, 'vit', False))
        else:
            return CIFAR100(root=root, train=False, download=False, index=full_index, base_sess=True,
                            is_vit=getattr(args, 'vit', False))
    else:
        raise KeyError(dataset_name)


def _names_for(dataset_name, raw_ds, selected_orig_ids):
    """Return readable class names aligned with 'selected_orig_ids'."""
    selected_orig_ids = [int(i) for i in selected_orig_ids]
    if dataset_name in ['cub200', 'air']:
        # CUB-200: 'labels' is a list of class names
        labels = getattr(raw_ds, 'labels', None)
        if labels is not None:
            return [labels[i] for i in selected_orig_ids]
    elif dataset_name == 'mini_imagenet':
        # many MiniImageNet impls expose wnids (or classes). Fall back to str(i) if not found.
        wnids = getattr(raw_ds, 'wnids', None)
        if wnids is not None and len(wnids) > 0:
            return [wnids[i] for i in selected_orig_ids]
        classes = getattr(raw_ds, 'classes', None)
        if classes is not None and len(classes) > 0:
            return [classes[i] for i in selected_orig_ids]
    elif dataset_name == 'cifar100':
        classes = getattr(raw_ds, 'classes', None)
        if classes is not None and len(classes) > 0:
            return [classes[i] for i in selected_orig_ids]
    # fallback
    return [str(i) for i in selected_orig_ids]


def set_up_datasets(args):
    """
    Original single-dataset setup (cifar100 / cub200 / mini_imagenet) + Cross-dataset FSCIL.
    Cross-dataset mode is activated when args.cross_dataset is True, in which case:
      - Base session (0) is trained on args.base_dataset.
      - Incremental sessions (1..N) are trained/evaluated on args.inc_dataset.
    """
    # ===== Cross-dataset branch =====
    if getattr(args, 'cross_dataset', False):
        # sensible defaults
        args.base_dataset = getattr(args, 'base_dataset', None) or 'mini_imagenet'
        args.inc_dataset = getattr(args, 'inc_dataset', None) or 'cub200'
        args.base_dataroot = getattr(args, 'base_dataroot', None) or args.dataroot
        args.inc_dataroot = getattr(args, 'inc_dataroot', None) or args.dataroot

        base_def = _defaults_for(args.base_dataset)
        inc_def = _defaults_for(args.inc_dataset)

        # base_class from base dataset, way/sessions from incremental dataset
        args.base_class = base_def['base_class']
        args.way = getattr(args, 'inc_way', None) or inc_def['way']
        inc_sessions = getattr(args, 'inc_sessions', None) or (inc_def['sessions'] - 1)
        args.sessions = 1 + int(inc_sessions)

        # for model heads or other code that may read this
        args.num_classes = int(args.base_class + (args.sessions - 1) * args.way)

        # raw datasets (we'll wrap them)
        base_train_raw = _build_raw(args.base_dataset, args.base_dataroot, True, args)
        base_test_raw  = _build_raw(args.base_dataset, args.base_dataroot, False, args)
        inc_train_raw  = _build_raw(args.inc_dataset,  args.inc_dataroot,  True, args)
        inc_test_raw   = _build_raw(args.inc_dataset,  args.inc_dataroot,  False, args)

        # global label mapping
        unique_base_ids = sorted(set(int(t) for t in base_train_raw.targets))[:args.base_class]
        base_map = {orig: gid for gid, orig in enumerate(unique_base_ids)}  # -> 0..base_class-1

        # incremental classes pool (we assume datasets do not overlap; keep all inc classes)
        inc_all_ids = sorted(set(int(t) for t in inc_train_raw.targets))
        # randomize inc class order (seeded for reproducibility)
        seed = int(getattr(args, 'seed', 0))
        rng = np.random.RandomState(seed)
        inc_all_ids = np.array(inc_all_ids)
        rng.shuffle(inc_all_ids)
        per = int(args.way)
        inc_slices = [inc_all_ids[i*per:(i+1)*per].tolist() for i in range(inc_sessions)]
        # drop last if incomplete
        inc_slices = [sl for sl in inc_slices if len(sl) == per]

        gid = int(args.base_class)
        inc_map = {}
        for sl in inc_slices:
            for orig in sl:
                inc_map[int(orig)] = gid
                gid += 1

        # base loaders
        base_names = _names_for(args.base_dataset, base_train_raw, unique_base_ids)
        base_train = RemapTargets(base_train_raw, unique_base_ids, base_map, class_names=base_names)
        base_test  = RemapTargets(base_test_raw,  unique_base_ids, base_map, class_names=base_names)

        train_loader_0 = DataLoader(base_train, batch_size=int(getattr(args, 'batch_size_base', 128)), shuffle=True,
                                    num_workers=int(getattr(args, 'num_workers', 4)))
        test_loader_0  = DataLoader(base_test,  batch_size=int(getattr(args, 'test_batch_size', 128)), shuffle=False,
                                    num_workers=int(getattr(args, 'num_workers', 4)))

        session_loaders = []
        session_loaders.append(dict(
            train_set=base_train, train_loader=train_loader_0,
            test_set=base_test,   test_loader=test_loader_0,
            dataset=args.base_dataset, class_names=base_names
        ))

        # incremental loaders per session
        for s_idx, sl in enumerate(inc_slices, start=1):
            names = _names_for(args.inc_dataset, inc_train_raw, sl)
            inc_train_remap = RemapTargets(inc_train_raw, sl, inc_map, class_names=names)
            # test needs all classes seen so far from INC dataset + BASE
            inc_seen_flat = [x for group in inc_slices[:s_idx] for x in group]
            inc_test_remap = RemapTargets(inc_test_raw, inc_seen_flat, inc_map)

            # K-shot training on current inc classes
            k = int(getattr(args, 'low_shot', getattr(args, 'shot', 1)))
            if k > 0:
                rng_py = random.Random(seed)
                inc_train_k = FewShotSubset(inc_train_remap, k_shot=k, rng=rng_py)
            else:
                inc_train_k = inc_train_remap

            # test: base + all seen inc
            test_concat = ConcatWithTransform([base_test, inc_test_remap])

            batch_new = int(getattr(args, 'batch_size_new', 0))
            if batch_new == 0:
                batch_new = len(inc_train_k)
                shuffle_flag = False
            else:
                shuffle_flag = True

            train_loader = DataLoader(inc_train_k, batch_size=batch_new, shuffle=shuffle_flag,
                                      num_workers=int(getattr(args, 'num_workers', 4)))
            test_loader  = DataLoader(test_concat, batch_size=int(getattr(args, 'test_batch_size', 128)), shuffle=False,
                                      num_workers=int(getattr(args, 'num_workers', 4)))

            session_loaders.append(dict(
                train_set=inc_train_k, train_loader=train_loader,
                test_set=test_concat,  test_loader=test_loader,
                dataset=args.inc_dataset, class_names=names
            ))

        # expose for the trainer
        args.session_loaders = session_loaders
        # also keep the Dataset module consistent with base_dataset for legacy code
        if args.base_dataset == 'cifar100':
            import dataloader.cifar100.cifar as Dataset
        elif args.base_dataset == 'cub200':
            import dataloader.cub200.cub200 as Dataset
        elif args.base_dataset == 'mini_imagenet':
            import dataloader.miniimagenet.miniimagenet as Dataset
        else:
            Dataset = None
        args.Dataset = Dataset
        return args

    # ===== Original single-dataset path (kept intact) =====
    if args.dataset == 'cifar100':
        import dataloader.cifar100.cifar as Dataset
        args.base_class = 60
        args.num_classes=100
        args.way = 5
        args.shot = 5
        args.sessions = 9
    if args.dataset == 'cub200':
        import dataloader.cub200.cub200 as Dataset
        args.base_class = 100
        args.num_classes = 200
        args.way = 10
        args.shot = 5
        args.sessions = 11
    if args.dataset == 'mini_imagenet':
        import dataloader.miniimagenet.miniimagenet as Dataset
        args.base_class = 60
        args.num_classes=100
        args.way = 5
        args.shot = 5
        args.sessions = 9
    args.Dataset=Dataset
    return args


def get_dataloader(args,session):
    if session == 0:
        trainset, trainloader, testloader = get_base_dataloader(args)
    else:
        trainset, trainloader, testloader = get_new_dataloader(args, session)
    return trainset, trainloader, testloader


def get_base_dataloader(args, clip_trsf=None):
    class_index = np.arange(args.base_class)
    if args.dataset == 'cifar100':
        trainset = args.Dataset.CIFAR100(root=args.dataroot, train=True, download=True, transform=clip_trsf,
                                         index=class_index, base_sess=True,is_vit=args.vit)
        testset = args.Dataset.CIFAR100(root=args.dataroot, train=False, download=False,
                                            index=class_index, base_sess=True,is_vit=args.vit)

    if args.dataset == 'cub200':
        trainset = args.Dataset.CUB200(root=args.dataroot, train=True, index=class_index, base_sess=True)
        testset = args.Dataset.CUB200(root=args.dataroot, train=False, index=class_index)

    if args.dataset == 'mini_imagenet':
        trainset = args.Dataset.MiniImageNet(root=args.dataroot, train=True, index=class_index, base_sess=True)
        testset = args.Dataset.MiniImageNet(root=args.dataroot, train=False, index=class_index)

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=args.batch_size_base, shuffle=True,
                                              num_workers=4)
    testloader = torch.utils.data.DataLoader(
        dataset=testset, batch_size=args.test_batch_size, shuffle=False, num_workers=4)

    return trainset, trainloader, testloader


def get_standard_data_loader(args):
    #todo for the standard CIL Data
    pass


def get_base_dataloader_meta(args):
    txt_path = "/root/autodl-tmp/remake_PriviLege/data/index_list/" + args.dataset + "/session_" + str(0 + 1) + '.txt'
    class_index = np.arange(args.base_class)
    if args.dataset == 'cifar100':
        trainset = args.Dataset.CIFAR100(root=args.dataroot, train=True, download=True,
                                         index=class_index, base_sess=True,is_vit=args.vit)
        testset = args.Dataset.CIFAR100(root=args.dataroot, train=False, download=False,
                                        index=class_index, base_sess=True,is_vit=args.vit)

    if args.dataset == 'cub200':
        trainset = args.Dataset.CUB200(root=args.dataroot, train=True, index_path=txt_path)
        testset = args.Dataset.CUB200(root=args.dataroot, train=False, index=class_index)
    if args.dataset == 'mini_imagenet':
        trainset = args.Dataset.MiniImageNet(root=args.dataroot, train=True, index_path=txt_path)
        testset = args.Dataset.MiniImageNet(root=args.dataroot, train=False, index=class_index)

    sampler = CategoriesSampler(trainset.targets, args.train_episode, args.episode_way,
                                args.episode_shot + args.episode_query)

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_sampler=sampler, num_workers=args.num_workers)
    testloader = torch.utils.data.DataLoader(
        dataset=testset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers)

    return trainset, trainloader, testloader


def get_new_dataloader(args,session, clip_trsf=None):
    txt_path = "/root/autodl-tmp/remake_PriviLege/data/index_list/" + args.dataset + "/session_" + str(session + 1) + '.txt'
    if args.dataset == 'cifar100':
        if args.clip:
            class_index = open(txt_path).read().splitlines()
            trainset = args.Dataset.CIFAR100(root=args.dataroot, train=True, download=False, transform=clip_trsf,
                                            index=class_index, base_sess=False,is_vit=args.vit)
        else:
            class_index = open(txt_path).read().splitlines()
            trainset = args.Dataset.CIFAR100(root=args.dataroot, train=True, download=False,
                                         index=class_index, base_sess=False,is_vit=args.vit)
    if args.dataset == 'cub200':
        trainset = args.Dataset.CUB200(root=args.dataroot, train=True, index_path=txt_path)
    if args.dataset == 'mini_imagenet':
        trainset = args.Dataset.MiniImageNet(root=args.dataroot, train=True, index_path=txt_path)
    if args.batch_size_new == 0:
        batch_size_new = trainset.__len__()
        trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size_new, shuffle=False,
                                                  num_workers=args.num_workers)
    else:
        trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=args.batch_size_new, shuffle=True,
                                                  num_workers=args.num_workers)

    # test on all encountered classes
    class_new = get_session_classes(args, session)

    if args.dataset == 'cifar100':
        testset = args.Dataset.CIFAR100(root=args.dataroot, train=False, download=False, transform=clip_trsf,
                                            index=class_new, base_sess=False,is_vit=args.vit)
    if args.dataset == 'cub200':
        testset = args.Dataset.CUB200(root=args.dataroot, train=False, index=class_new)
    if args.dataset == 'mini_imagenet':
        testset = args.Dataset.MiniImageNet(root=args.dataroot, train=False, index=class_new)

    testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=args.test_batch_size, shuffle=False,
                                             num_workers=args.num_workers)

    return trainset, trainloader, testloader


def get_session_classes(args,session):
    class_list=np.arange(args.base_class + session * args.way)
    return class_list
