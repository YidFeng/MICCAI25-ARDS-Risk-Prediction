import os
import argparse
import torch
from coach import Coach

parser = argparse.ArgumentParser()

# commonly used params

parser.add_argument('--save_dir', default='thesis', type=str, help='..') # proj, sep


## data settings
parser.add_argument('--data_src_dir', default='YOUR DATA PATH', type=str, help='source data directory')

parser.add_argument('--test_data_name', default='test', type=str,  help='')
parser.add_argument('--train_data_name', default='train', type=str,  help='')
parser.add_argument('--val_data_name', default='val', type=str,  help='')

parser.add_argument('--use_cxr', action='store_true', default=True)
parser.add_argument('--use_lab', action='store_true', default=True)
parser.add_argument('--use_vs', action='store_true', default=True)
parser.add_argument('--use_single_cxr', action='store_true', default=False, help='')

parser.add_argument('--num_interval', default=12, type=int, help='number of 6h interval considered in dataset (totoal time length)')
parser.add_argument('--interval_length', default='6 hours', type=str, help='number of 6h interval considered')

# arch settings

parser.add_argument('--use_ptt', action='store_true', default=True, help='use pred time transformer') # todo
parser.add_argument('--use_mft', action='store_true', default=True, help='use mm fusion transformer') # todo
parser.add_argument('--use_fagg', action='store_true', default=True, help='use feature aggregation in lab and vs transformer')
parser.add_argument('--pos_type', type=str, default='learn_rtime_mod', help='sin_naive/learn_naive/sin_rtime/sin_rtime_mod/learn_rtime_mod')
parser.add_argument('--latent_dim', default=128, type=int, help='number of 6h interval considered')


# test params
parser.add_argument('--num_test_repeat', default=2, type=int,  help='')
parser.add_argument('--set_compare_mod', action='store_true', default=False, help='what is the meaning of set compare mod')
parser.add_argument('--test_only', action='store_true', default=False, help='will not test if any mod missing')
parser.add_argument('--cal_flops', action='store_true', default=False, help='will not test if any mod missing')


# train settings
parser.add_argument('--use_bsample', action='store_true', default=True)
parser.add_argument('--manual_seed', default=0, type=int, help='Mannual seed') # todo: 0/1/2
parser.add_argument('--cnn_name', default='resnet34', type=str, help='cnn model names')
parser.add_argument('--model_depth', default=34, type=str, help='model depth (18|34|50|101|152|200)')
parser.add_argument('--n_classes', default=3, type=str, help='model output classes')
parser.add_argument('--in_channels', default=1, type=str, help='model input channels (1|3)')

# loss weights
parser.add_argument('--w_cls', default=1.0, type=float, help='bce loss')
parser.add_argument('--w_int', default=1.0, type=float, help='regression for interval')
## training params
parser.add_argument('--lr', default=3e-4, type=float, help='..')
parser.add_argument('--total_epoch', default=400, type=int, help='..')  # 400*50=20,000 steps
parser.add_argument('--batch_size', default=16, type=int, help='..')
parser.add_argument('--ckpt', default=None, type=str, help='..')
parser.add_argument('--save_interval', default=10, type=int, help='..') #50

args = parser.parse_args()

if args.use_bsample:
    args.save_dir += 'bsample'
else:
    args.save_dir += 'nobsample'
if args.use_ptt:
    args.save_dir += '_ptt'
if args.use_mft:
    args.save_dir += '_mft'
interval_length_txt = args.interval_length.replace(' ', '_')
args.save_dir += f'_total_interval{args.num_interval}_length{interval_length_txt}'
if args.use_cxr:
    args.save_dir += '_cxr'
if args.use_lab:
    args.save_dir += '_lab'
if args.use_vs:
    args.save_dir += '_vs'

if args.w_cls > 0:
    args.save_dir += f'_cls{args.w_cls}'
if args.w_int > 0:
    args.save_dir += f'_predint{args.w_int}'
args.save_dir += f'_pos{args.pos_type}'
args.save_dir += f'_seed{args.manual_seed}'
args.save_dir  = os.path.join('exps', args.save_dir)
os.makedirs(args.save_dir, exist_ok=True)

# save script
def save_script(script_path, save_dir):
    # Read the content of the script

    with open(script_path, 'r') as script_file:
        script_content = script_file.read()

    # Specify the name for the copied script
    script_name = os.path.basename(script_path)
    target_script_path = os.path.join(save_dir, script_name)

    # Write the script content to the new file in the target folder
    with open(target_script_path, 'w') as target_script_file:
        target_script_file.write(script_content)

    print(f"Script copy saved to: {target_script_path}")
    #
script_path = os.path.abspath(__file__)
coach_path = script_path.replace('train.py', 'coach.py')
save_script(coach_path, args.save_dir)
save_script(script_path, args.save_dir)
torch.manual_seed(args.manual_seed)

mmcoach = Coach(args)
if not args.test_only:
    mmcoach.train()
mmcoach.test(is_compare_mod=args.set_compare_mod)
