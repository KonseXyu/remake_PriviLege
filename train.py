import argparse
import importlib
from utils import *
import torch

MODEL_DIR=None
DATA_DIR = './data/'
PROJECT='base'

def get_command_line_parser():
    parser = argparse.ArgumentParser()

    # about dataset and network
    parser.add_argument('-project', type=str, default=PROJECT)
    parser.add_argument('-dataset', type=str, default='cub200',
                        choices=['mini_imagenet', 'cub200', 'cifar100'])
    parser.add_argument('-dataroot', type=str, default=DATA_DIR)
    parser.add_argument('-out', type=str, default=None)

    # about pre-training
    # parser.add_argument('-epochs_base', type=int, default=200)
    parser.add_argument('-epochs_base', type=int, default=50)
    # parser.add_argument('-epochs_new', type=int, default=100)
    parser.add_argument('-epochs_new', type=int, default=20)
    parser.add_argument('-lr_base', type=float, default=0.1)
    parser.add_argument('-lr_new', type=float, default=2e-4)
    parser.add_argument('-schedule', type=str, default='Step',
                        choices=['Step', 'Milestone', 'Cosine'])
    parser.add_argument('-milestones', nargs='+', type=int, default=[60, 70])
    parser.add_argument('-step', type=int, default=80)
    parser.add_argument('-decay', type=float, default=0.0005)
    parser.add_argument('-momentum', type=float, default=0.9)

    parser.add_argument('-not_data_init', action='store_true', help='using average data embedding to init or not')

    parser.add_argument('-batch_size_base', type=int, default=128)
    parser.add_argument('-batch_size_new', type=int, default=0, help='set 0 will use all the availiable training image for new')
    parser.add_argument('-test_batch_size', type=int, default=128)
    parser.add_argument('-base_mode', type=str, default='ft_cos',
                        choices=['ft_dot', 'ft_cos']) # ft_dot means using linear classifier, ft_cos means using cosine classifier
    parser.add_argument('-new_mode', type=str, default='avg_cos',
                        choices=['ft_dot', 'ft_cos', 'avg_cos', 'ft_comb', 'ft_euc']) # ft_dot means using linear classifier, ft_cos means using cosine classifier, avg_cos means using average data embedding and cosine classifier

    # for episode learning
    parser.add_argument('-train_episode', type=int, default=50)
    parser.add_argument('-episode_shot', type=int, default=1)
    parser.add_argument('-episode_way', type=int, default=15)
    parser.add_argument('-episode_query', type=int, default=15)

    # for cec
    parser.add_argument('-lrg', type=float, default=0.1) #lr for graph attention network
    parser.add_argument('-low_shot', type=int, default=1)
    parser.add_argument('-low_way', type=int, default=15)

    parser.add_argument('-start_session', type=int, default=1)
    parser.add_argument('-model_dir', type=str, default=MODEL_DIR, help='loading model parameter from a specific dir')
    parser.add_argument('-set_no_val', action='store_true', help='set validation using test set or no validation')

    # about training
    parser.add_argument('-gpu', default='0')
    parser.add_argument('-num_workers', type=int, default=4)
    parser.add_argument('-seed', type=int, default=-1)
    # parser.add_argument('-seeds', nargs='+', help='<Required> Set flag', required=True, default=1)
    parser.add_argument('-debug', action='store_true')
    parser.add_argument('-rotation', action='store_true')
    parser.add_argument('-fraction_to_keep', type=float, default=0.1)
    
    parser.add_argument('-vit', action='store_true')
    parser.add_argument('-baseline', action='store_true')
    parser.add_argument('-clip', action='store_true')
    
    parser.add_argument('-ED', action='store_true')
    parser.add_argument('-ED_hp', type=float, default=0.5)
    
    parser.add_argument('-LT', action='store_true')
    parser.add_argument('-WC', action='store_true')
    parser.add_argument('-MP', action='store_true')

    parser.add_argument('-proto_classifier', action='store_true',
                        help='Use prototype-cosine + temperature as classifier for train/test')
    parser.add_argument('-proto_temp', type=float, default=10.0,
                        help='Temperature for prototype-cosine classifier (default=10.0)')

    parser.add_argument(
        '-replace_base_mode',
        type=str,
        default='encoder',  # 原来就是按 encoder 特征均值来替换 FC
        choices=['encoder', 'proto'],
        help="How to compute prototypes for replace_base_fc: "
             "'encoder' (orig mean enc), 'proto' (mean of 0.5*(CLS+Vision) with PKT)"
    )

    parser.add_argument(
        '-inc_proto_mode',
        type=str,
        default='encoder',
        choices=['encoder', 'proto'],
        help="How to compute prototypes for new classes in incremental sessions: "
             "'encoder' (use self.encode), 'proto' (use 0.5*(CLS+Vision) via prompt_encode)."
    )
    parser.add_argument(
        '-append_new_proto',
        type=int,
        default=1,
        choices=[0, 1],
        help="Whether to append new-class prototypes into query_info['proto'] (1=on, 0=off). Default: 1"
    )

    parser.add_argument(
        '-base_proto_mode',
        type=str,
        default=None,                       # None 表示自动：如果开了 proto_classifier 就用 privilege；否则 encoder
        choices=[None, 'encoder', 'proto'],
        help="How to compute prototypes in BASE build_base_proto: "
             "'encoder' (original), 'proto' (0.5*(CLS+Vision) via prompt_encode). "
             "Default: auto (proto if -proto_classifier else encoder)."
    )

    parser.add_argument('-SKD', action='store_true')
    parser.add_argument('-gamma', type=float, default=0.1)
    parser.add_argument('-inc_gamma', type=float, default=0.1)
    parser.add_argument('-temperature', type=float, default=2.)
    parser.add_argument('-inc_temperature', type=float, default=2.)
    parser.add_argument('-base_skd_weight', type=float, default=0.5)
    parser.add_argument('-inc_skd_weight', type=float, default=0.5)

    parser.add_argument('-prefix', action='store_true')
    parser.add_argument('-pret_clip', action='store_true')
    parser.add_argument('-comp_out', type=int, default=1.)
    
    # parser.add_argument('-scratch', action='store_true')
    parser.add_argument('-scratch', action='store_true')
    parser.add_argument('-taskblock', type=int, default=2)
    parser.add_argument('-ft', action='store_true')
    parser.add_argument('-lp', action='store_true')
    parser.add_argument('-PKT_tune_way', type=int, default=1)

    # === Cross-dataset FSCIL (base & incremental on different datasets) ===
    parser.add_argument('-cross_dataset', action='store_true',
                        help='Use different datasets for base (session 0) and incremental sessions.')
    parser.add_argument('-base_dataset', type=str, default=None,
                        choices=['mini_imagenet', 'cub200', 'cifar100'])
    parser.add_argument('-inc_dataset', type=str, default=None,
                        choices=['mini_imagenet', 'cub200', 'cifar100'])
    parser.add_argument('-base_dataroot', type=str, default=None,
                        help='Data root for the base dataset; default: dataroot')
    parser.add_argument('-inc_dataroot', type=str, default=None,
                        help='Data root for the incremental dataset; default: dataroot')
    parser.add_argument('-inc_way', type=int, default=None,
                        help='#classes added per incremental session; default depends on inc_dataset')
    parser.add_argument('-inc_sessions', type=int, default=None,
                        help='#incremental sessions; default depends on inc_dataset')

    return parser


if __name__ == '__main__':
    torch.set_num_threads(2)
    devices = [d for d in range(torch.cuda.device_count())]
    device_names  = [torch.cuda.get_device_name(d) for d in devices]
    print('GPU COUNT',torch.cuda.device_count())
    print("Devices:",devices)
    print("Device Name:",device_names)
    
    parser = get_command_line_parser()
    args = parser.parse_args()

    # For incremental learning, get same random seed that has been used during pretraining step
    # if args.model_dir is not None:
    #     args.seed = int(args.model_dir.split('/')[-2][-9])
    
    # for seed in args.seed:
    # args.seed = seed
    set_seed(args.seed)
    pprint(vars(args))
    args.num_gpu = set_gpu(args)
    if args.vit:
        trainer = importlib.import_module('models.ViT_fscil_trainer').ViT_FSCILTrainer(args)
    else:
        trainer = importlib.import_module('models.fscil_trainer').FSCILTrainer(args)
    trainer.train()