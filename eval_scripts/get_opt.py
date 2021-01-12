import os
from argparse import Namespace
import utils.paramUtil as paramUtil


opt_conversion = {
    # Base
    'name': str,
    'gpu_id': str,
    'time_counter': lambda s: s == 'True',
    'motion_length': int,
    'dataset_type': str,
    'clip_set': str,
    'checkpoints_dir': str,
    'dim_z': int,
    'hidden_size': int,
    'prior_hidden_layers': int,
    'posterior_hidden_layers': int,
    'decoder_hidden_layers': int,
    'veloc_hidden_layers': int,
    'd_hidden_layers': int,
    'isTrain': lambda s: s == 'True',
    
    # Train
    'batch_size': int,
    'arbitrary_len': lambda s: s == 'True',
    'do_adversary': lambda s: s == 'True',
    'do_recognition': lambda s: s == 'True',
    'do_align': lambda s: s == 'True',
    'skip_prob': float,
    'tf_ratio': float,
    'lambda_kld': float,
    'lambda_align': float,
    'lambda_adversary': float,
    'lambda_recognition': float,
    'is_continue': lambda s: s == 'True',
    'iters': int,
    'plot_every': int,
    "save_every": int,
    "eval_every": int,
    "save_latest": int,
    'print_every': int,

    # Evaluation
    'which_epoch': str,
    'result_path': str,
    'replic_times': int,
    'do_random': lambda s: s == 'True',
    'num_samples': int,

    # Misc. options
    'dim_noise_pose': int,
    'dim_noise_motion': int,
    'hidden_size_pose': int,
    'hidden_size_motion': int,

    # Makeshift options
    'model_file_path_override': str,
}


dataset_opt = {
    'humanact12': {
        'dataset_path': "./dataset/humanact12",
        'input_size_raw': 72,
        'joints_num': 24,
        'label_dec': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        'raw_offsets': paramUtil.shihao_raw_offsets,
        'kinematic_chain': paramUtil.shihao_kinematic_chain,
        'enumerator': paramUtil.shihao_coarse_action_enumerator,
    },
    
    'ntu_rgbd_vibe': {
        'file_prefix': './dataset',
        'motion_desc_file': 'ntu_vibe_list.txt',
        'joints_num': 18,
        'input_size_raw': 54,
        'label_dec': paramUtil.ntu_action_labels,
        'enumerator': paramUtil.ntu_action_enumerator
    },
    'mocap': {
        "dataset_path": "./dataset/mocap/mocap_3djoints/",
        "clip_path": './dataset/mocap/pose_clip.csv',
        "input_size_raw": 60,
        "joints_num": 20,
        'label_dec': [0, 1, 2, 3, 4, 5, 6, 7]
    }
}


def get_opt(opt_path, num_motions, device):
    opt = Namespace()
    opt_dict = vars(opt)
    opt.opt_path = opt_path
    skip = ('------------ Options -------------\n',
            '-------------- End ----------------\n')
    print('Reading', opt_path)
    with open(opt_path) as f:
        for line in f:
            if line not in skip:
                key, value = line.strip().split(': ')
                conversion = opt_conversion.get(key, lambda s: True if s == 'True' else False if s == 'False' else s)
                opt_dict[key] = conversion(value)

    opt_dict['isTrain'] = False

    opt_dict['which_epoch'] = 'latest'
    opt_dict['result_path'] = './eval_results/vae/'     # TODO: remove this
    opt_dict['do_random'] = True
    opt_dict['num_samples'] = num_motions   # but why?

    opt.device = device
    opt.save_root = os.path.join(opt.checkpoints_dir, opt.dataset_type, opt.name)
    opt.model_path = os.path.join(opt.save_root, 'model')
    opt.joints_path = os.path.join(opt.save_root, 'joints')
    opt.model_file_path = os.path.join(opt.model_path, opt.which_epoch + '.tar')
    opt.result_path = os.path.join(opt.result_path, opt.dataset_type, opt.name)
    # print(opt.coarse_grained)

    if not opt.coarse_grained:
        dataset_opt['humanact12'] = dataset_opt['humanact12_fineG']
        # print(dataset_opt['humanact12'])

    if opt.dataset_type == 'humanact13':
        opt.dataset_type = 'humanact12'

    opt_dict.update(dataset_opt[opt.dataset_type])
    # print(opt_dict['label_dec'])

    if 'use_lie' not in opt_dict:
        opt.use_lie = False
    opt.lie_enforce = False

    opt.dim_category = len(opt.label_dec)

    opt.pose_dim = opt.input_size_raw
    if 'time_counter' in opt_dict and opt.time_counter:
        opt.input_size = opt.input_size_raw + opt.dim_category + 1
    else:
        opt.input_size = opt.input_size_raw + opt.dim_category

    opt.veloc_input_size = opt.input_size_raw * 2 + 20

    opt.output_size = opt.input_size_raw
    
    return opt

