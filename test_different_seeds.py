import os
import argparse

import numpy as np
import torch
from coach import Coach
import warnings

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
## model params
parser.add_argument('--num_interval', default=12, type=int, help='number of 6h interval considered in dataset (totoal time length)')
# parser.add_argument('--num_pred_interval', default=8, type=int, help='number of 6h interval considered (prediction time length)')
parser.add_argument('--interval_length', default='6 hours', type=str, help='number of 6h interval considered')
parser.add_argument('--latent_dim', default=128, type=int, help='number of 6h interval considered')
# todo
parser.add_argument('--use_cxr', action='store_true', default=True)
parser.add_argument('--use_lab', action='store_true', default=True)
parser.add_argument('--use_vs', action='store_true', default=True)

parser.add_argument('--use_bsample', action='store_true', default=True)
parser.add_argument('--use_ptt', action='store_true', default=True, help='use pred time transformer') # todo
parser.add_argument('--use_mft', action='store_true', default=True, help='use mm fusion transformer') # todo
parser.add_argument('--use_fagg', action='store_true', default=True, help='use feature aggregation in lab and vs transformer')
parser.add_argument('--use_single_cxr', action='store_true', default=False, help='use feature aggregation in lab and vs transformer')


# test params
parser.add_argument('--num_test_repeat', default=1, type=int,  help='use gml loss')
parser.add_argument('--set_compare_mod', action='store_true', default=False, help='will not test if any mod missing')
parser.add_argument('--test_only', action='store_true', default=True, help='will not test if any mod missing')
parser.add_argument('--cal_flops', action='store_true', default=False, help='will not test if any mod missing')


parser.add_argument('--test_data_name', default='test', type=str,  help='use gml loss')
parser.add_argument('--train_data_name', default='train', type=str,  help='use gml loss')
parser.add_argument('--val_data_name', default='val', type=str,  help='use gml loss')


parser.add_argument('--manual_seed', default=88, type=int, help='Mannual seed') # todo: 0/1/2
parser.add_argument('--cnn_name', default='resnet34', type=str, help='cnn model names')
parser.add_argument('--model_depth', default=34, type=str, help='model depth (18|34|50|101|152|200)')
parser.add_argument('--n_classes', default=3, type=str, help='model output classes')
parser.add_argument('--in_channels', default=1, type=str, help='model input channels (1|3)')
parser.add_argument('--sample_size', default=90, type=str, help='image size')

# loss weights
parser.add_argument('--w_cls', default=1.0, type=float, help='bce loss')
parser.add_argument('--w_int', default=1.0, type=float, help='regression for interval')
parser.add_argument('--w_gml', default=0.0, type=float, help='..')
## training params
parser.add_argument('--lr', default=3e-4, type=float, help='..')
parser.add_argument('--total_epoch', default=400, type=int, help='..')  # 400*50=20,000 steps
parser.add_argument('--batch_size', default=16, type=int, help='..')

parser.add_argument('--ckpt_source', default='', type=str, help='..')
parser.add_argument('--save_dir', default="rebuttal", type=str, help='..') # proj, sep
parser.add_argument('--save_interval', default=10, type=int, help='..') #50

args = parser.parse_args()
ckpt_source_name = args.ckpt_source.split('/')[-1]
# args.save_dir += f'_{ckpt_source_name}'



args.save_dir  = os.path.join('exps', args.save_dir)
os.makedirs(args.save_dir, exist_ok=True)

# save script

torch.manual_seed(args.manual_seed)

mmcoach = Coach(args)
metric_names = ['auroc', 'mae', 'mse', 'accuracy', 'ppv', 'npv', 'sensitivity', 'specificity', 'mae_d', 'f1']
ms_results = {k:[] for k in metric_names}
for i in range(1): # todo
    args.ckpt = args.ckpt_source + str(i) + '/best.pth'
    # args.ckpt = args.ckpt_source + str(i) + '/best.pth'
    # mmcoach.test_draw_roc()
    mmcoach.test()
#     for k in metric_names:
#         ms_results[k].append(results_all[k])
# with open(os.path.join(args.save_dir, 'results.txt'), 'a') as f:
#     for k in metric_names:
#         mean = np.mean(ms_results[k])
#         std =np.std(ms_results[k])
#         print(f'{k}: {mean}+-{std}')
#         f.write(f'{k}: {mean}+-{std}\n')

