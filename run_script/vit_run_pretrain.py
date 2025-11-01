import os
import sys
# seeds = [1,2,3,4,5]
seeds = [1]

project = 'base'
# dataset = 'mini_imagenet'
# dataset = 'cifar100'
dataset = 'cub200'

lr_base = 2e-4
lr_new = 2e-4

epochs_bases = [5] #5
epochs_new = 3 #3
milestones_list = ['20 30 45']

#* data_dir = '/local_datasets/'
data_dir = sys.argv[1]
gpu_num = sys.argv[2]

for seed in seeds:
    print("Pretraining -- Seed{}".format(seed))
    for i, epochs_base in enumerate(epochs_bases):
        os.system(''
                'python train.py '
                '-project {} '
                '-dataset {} '
                '-base_mode ft_dot '
                '-new_mode avg_cos '
                '-lr_base {} '
                '-lr_new {} '
                '-decay 0.0005 '
                '-epochs_base {} '
                '-epochs_new {} '
                '-schedule Cosine '
                '-milestones {} '
                '-gpu {} '
                '-gamma 0.1 '
                '-inc_gamma 0.1 '
                '-temperature 2. '
                '-inc_temperature 2. '
                '-base_skd_weight 0.5 '
                '-inc_skd_weight 0.5 '
                '-ED_hp 0.5 '
                '-start_session 0 '
                '-batch_size_base 128 '
                '-seed {} '
                '-vit '
                '-comp_out 1 '
                # '-prefix '
                '-ED '
                '-SKD '
                '-LT '
                # '-cross_dataset '
                '-base_dataset mini_imagenet '
                '-inc_dataset cifar100 '
                # '-base_dataroot dataroot '
                # '-inc_dataroot dataroot '
                  
                '-proto_classifier '
                '-proto_temp 10.0 '

                '-base_proto_mode proto '       #encoder proto 两种实现方式
                '-inc_proto_mode proto '        #encoder proto 两种实现方式
                '-replace_base_mode proto '     #encoder proto 两种实现方式
                '-append_new_proto 1 '
                  
                '-inc_way 5 '
                '-inc_sessions 2 '
                  
                '-out {} '
                '-dataroot {}'.format(project, dataset, lr_base, lr_new, epochs_base, epochs_new, milestones_list[i], gpu_num, seed, 'PriViLege', data_dir)
                )
