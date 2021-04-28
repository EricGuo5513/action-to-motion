from datetime import datetime
import numpy as np
import os
import torch

import utils.paramUtil as paramUtil
from utils.get_opt import get_opt
from utils.load_classifier import load_classifier, load_classifier_for_fid
from motion_loaders.dataset_motion_loader import get_dataset_motion_loader
from motion_loaders.motion_loader import get_motion_loader
from utils.fid import calculate_frechet_distance
from utils.matrix_transformer import MatrixTransformer as mt
from utils.plot_script import *

torch.multiprocessing.set_sharing_strategy('file_system')
def evaluate_accuracy(num_motions, gru_classifier, motion_loaders, dataset_opt, device, file):
    print('========== Evaluating Accuracy ==========')
    for motion_loader_name, motion_loader in motion_loaders.items():
        accuracy = calculate_accuracy(motion_loader, len(dataset_opt.label_dec),
                                      gru_classifier, device)
        print(f'---> [{motion_loader_name}] Accuracy: {np.trace(accuracy)/num_motions:.4f}')
        print(f'---> [{motion_loader_name}] Accuracy: {np.trace(accuracy)/num_motions:.4f}', file=file, flush=True)


def calculate_accuracy(motion_loader, num_labels, classifier, device):
    print('Calculating Accuracies...')
    confusion = torch.zeros(num_labels, num_labels, dtype=torch.long)

    with torch.no_grad():
        for idx, batch in enumerate(motion_loader):
            batch_motion, batch_label = batch
            batch_motion = torch.clone(batch_motion).float().detach_().to(device)
            batch_label = torch.clone(batch_label).long().detach_().to(device)
            batch_prob, _ = classifier(batch_motion, None)
            batch_pred = batch_prob.max(dim=1).indices
            for label, pred in zip(batch_label, batch_pred):
                # print(label.data, pred.data)
                confusion[label][pred] += 1

    return confusion


def evaluate_fid(ground_truth_motion_loader, gru_classifier_for_fid, motion_loaders, device, file):
    print('========== Evaluating FID ==========')
    ground_truth_activations, ground_truth_labels = \
        calculate_activations_labels(ground_truth_motion_loader, gru_classifier_for_fid, device)
    ground_truth_statistics = calculate_activation_statistics(ground_truth_activations)

    for motion_loader_name, motion_loader in motion_loaders.items():
        activations, labels = calculate_activations_labels(motion_loader, gru_classifier_for_fid, device)
        statistics = calculate_activation_statistics(activations)
        fid = calculate_fid(ground_truth_statistics, statistics)
        diversity, multimodality = \
            calculate_diversity_multimodality(activations, labels, len(dataset_opt.label_dec))

        print(f'---> [{motion_loader_name}] FID: {fid:.4f}')
        print(f'---> [{motion_loader_name}] FID: {fid:.4f}', file=file, flush=True)
        print(f'---> [{motion_loader_name}] Diversity: {diversity:.4f}')
        print(f'---> [{motion_loader_name}] Diversity: {diversity:.4f}', file=file, flush=True)
        print(f'---> [{motion_loader_name}] Multimodality: {multimodality:.4f}')
        print(f'---> [{motion_loader_name}] Multimodality: {multimodality:.4f}', file=file, flush=True)


def calculate_fid(statistics_1, statistics_2):
    return calculate_frechet_distance(statistics_1[0], statistics_1[1],
                                      statistics_2[0], statistics_2[1])


def calculate_activations_labels(motion_loader, classifier, device):
    print('Calculating Activations...')
    activations = []
    labels = []

    with torch.no_grad():
        for idx, batch in enumerate(motion_loader):
            batch_motion, batch_label = batch
            batch_motion = torch.clone(batch_motion).float().detach_().to(device)

            activations.append(classifier(batch_motion, None))
            labels.append(batch_label)
        activations = torch.cat(activations, dim=0)
        labels = torch.cat(labels, dim=0)

    return activations, labels


def calculate_activation_statistics(activations):
    activations = activations.cpu().numpy()
    mu = np.mean(activations, axis=0)
    sigma = np.cov(activations, rowvar=False)

    return mu, sigma


def calculate_diversity_multimodality(activations, labels, num_labels):
    print('=== Evaluating Diversity ===')
    diversity_times = 200
    multimodality_times = 20
    labels = labels.long()
    num_motions = len(labels)

    diversity = 0
    first_indices = np.random.randint(0, num_motions, diversity_times)
    second_indices = np.random.randint(0, num_motions, diversity_times)
    for first_idx, second_idx in zip(first_indices, second_indices):
        diversity += torch.dist(activations[first_idx, :],
                                activations[second_idx, :])
    diversity /= diversity_times

    print('=== Evaluating Multimodality ===')
    multimodality = 0
    labal_quotas = np.repeat(multimodality_times, num_labels)
    while np.any(labal_quotas > 0):
        # print(labal_quotas)
        first_idx = np.random.randint(0, num_motions)
        first_label = labels[first_idx]
        if not labal_quotas[first_label]:
            continue

        second_idx = np.random.randint(0, num_motions)
        second_label = labels[second_idx]
        while first_label != second_label:
            second_idx = np.random.randint(0, num_motions)
            second_label = labels[second_idx]

        labal_quotas[first_label] -= 1

        first_activation = activations[first_idx, :]
        second_activation = activations[second_idx, :]
        multimodality += torch.dist(first_activation,
                                    second_activation)

    multimodality /= (multimodality_times * num_labels)

    return diversity, multimodality

def evaluation(log_file):
    with open(log_file, 'w') as f:
        for replication in range(20):
            motion_loaders = {}
            motion_loaders['ground truth'] = ground_truth_motion_loader
            for motion_loader_name, motion_loader_getter in eval_motion_loaders.items():
                motion_loader = motion_loader_getter(num_motions, device)
                motion_loaders[motion_loader_name] = motion_loader
            print(f'==================== Replication {replication} ====================')
            print(f'==================== Replication {replication} ====================', file=f, flush=True)
            print(f'Time: {datetime.now()}')
            print(f'Time: {datetime.now()}', file=f, flush=True)
            evaluate_accuracy(num_motions, gru_classifier, motion_loaders, dataset_opt, device, f)
            print(f'Time: {datetime.now()}')
            print(f'Time: {datetime.now()}', file=f, flush=True)
            evaluate_fid(ground_truth_motion_loader, gru_classifier_for_fid, motion_loaders, device, f)

        print(f'Time: {datetime.now()}')
        print(f'Time: {datetime.now()}', file=f, flush=True)
        print(f'!!! DONE !!!')
        print(f'!!! DONE !!!', file=f, flush=True)

def animation_4_user_study(save_dir, motion_loaders):
    kp_save_dir = os.path.join(save_dir, 'keypoints')
    am_save_dir = os.path.join(save_dir, 'animation')
    enumerator = paramUtil.shihao_coarse_action_enumerator
    label_dec = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    pose_tree = paramUtil.smpl_tree
    for motion_loader_name, motion_loader in motion_loaders.items():
        counter_arr = np.ones(len(label_dec)) * 10
        print(motion_loader_name)
        for batch_data in motion_loader:
            print("%d/%d" % (counter_arr.sum(), 10 * len(label_dec)))
            joints_data, labels = batch_data
            joints_data = joints_data.cpu().numpy()
            for i in range(joints_data.shape[0]):
                motion_orig = joints_data[i]
                labels = labels.long()
                a_id = labels[i]
                if counter_arr[a_id] >= 0:
                    class_type = enumerator[label_dec[a_id]]
                    offset = np.matlib.repmat(np.array([motion_orig[0, 0], motion_orig[0, 1], motion_orig[0, 2]]),
                                              motion_orig.shape[0], 24)
                    motion_mat = motion_orig - offset

                    motion_mat = motion_mat.reshape(-1, 24, 3)
                    kd_cls_path = os.path.join(kp_save_dir, class_type)
                    am_cls_path = os.path.join(am_save_dir, class_type)
                    if not os.path.exists(kd_cls_path):
                        os.makedirs(kd_cls_path)
                    if not os.path.exists(am_cls_path):
                        os.makedirs(am_cls_path)
                    joints_path = os.path.join(kd_cls_path, motion_loader_name + str(counter_arr[a_id]) + '_3d.npy')
                    animate_path = os.path.join(am_cls_path, motion_loader_name + str(counter_arr[a_id]) + '.gif')
                    np.save(joints_path, motion_mat)
                    motion_mat = mt.swap_yz(motion_mat)
                    motion_mat[:, :, 2] *= -1
                    plot_3d_motion(motion_mat, pose_tree, class_type, animate_path, interval=100)
                    counter_arr[a_id] -= 1
                if counter_arr.sum() == 0:
                    break
            if counter_arr.sum() == 0:
                break


if __name__ == '__main__':

    # dataset_opt_path = './checkpoints/vae/ntu_rgbd_vibe/vae_velocS_f0001_t01_trj10_rela/opt.txt'
    # dataset_opt_path = './checkpoints/vae/humanact12/vae_velocR_f0001_t001_trj10_rela_fineG/opt.txt'
    dataset_opt_path = './checkpoints/vae/mocap/vanilla_vae_lie_mse_kld01/opt.txt'
    label_spe = 3
    eval_motion_loaders = {
        'vae_velocR_f0001_t01_trj10_rela': lambda num_motions, device: get_motion_loader(
            './checkpoints/vae/mocap/vae_velocR_f0001_t01_trj10_rela/opt.txt',
            num_motions, 128, device, ground_truth_motion_loader, label_spe),
        'vae_velocS_f0001_t01_trj10_rela': lambda num_motions, device: get_motion_loader(
            './checkpoints/vae/mocap/vae_velocS_f0001_t01_trj10_rela/opt.txt',
            num_motions, 128, device, ground_truth_motion_loader, label_spe),

        #  'vae_velocS_f0001_t01_trj10_rela': lambda num_motions, device: get_motion_loader(
        #      './checkpoints/vae/ntu_rgbd_vibe/vae_velocS_f0001_t01_trj10_rela_fineG/opt.txt',
        #      num_motions, 128, device, ground_truth_motion_loader),
        # 'vae_velocR_f0001_t005_trj10_rela': lambda num_motions, device: get_motion_loader(
        #     './checkpoints/vae/humanact13/vae_velocR_f0001_t005_trj10_rela/opt.txt',
        #     num_motions, 128, device, ground_truth_motion_loader),
        # 'vae_veloc_f0001_t01_optim_seperate_relative': lambda num_motions, device: get_motion_loader(
        #     './checkpoints/vae/humanact13/vae_veloc_f0001_t01_optim_seperate_relative/opt.txt',
        #     num_motions, 128, device, ground_truth_motion_loader),
        # 'vae_velocS_f0001_t001_trj10_rela_fineG': lambda num_motions, device: get_motion_loader(
        #    './checkpoints/vae/humanact12/vae_velocS_f0001_t001_trj10_rela_fineG/opt.txt',
        #    num_motions, 128, device, ground_truth_motion_loader),
        # 'vae_velocR_f0001_t001_trj10_rela_fineG': lambda num_motions, device: get_motion_loader(
        # './checkpoints/vae/humanact12/vae_velocR_f0001_t001_trj10_rela_fineG/opt.txt',
        # num_motions, 128, device, ground_truth_motion_loader),
        # 'vanilla_vae_lie_mse_kld01': lambda num_motions, device: get_motion_loader(
        #    './checkpoints/vae/mocap/vanilla_vae_lie_mse_kld01/opt.txt',
        #    num_motions, 36, device, ground_truth_motion_loader),
        # 'vanilla_vae_lie_mse_kld001_fineG': lambda num_motions, device: get_motion_loader(
        #   './checkpoints/vae/humanact12/vanilla_vae_lie_mse_kld001_fineG/opt.txt',
        #   num_motions, 128, device, ground_truth_motion_loader),
        'vanilla_vae_lie_mse_kld01': lambda num_motions, device: get_motion_loader(
            './checkpoints/vae/mocap/vanilla_vae_lie_mse_kld01/opt.txt',
            num_motions, 128, device, ground_truth_motion_loader, label_spe),

        # 'vanilla_vae_lie_mse_kld01': lambda num_motions, device: get_motion_loader(
        #     './checkpoints/vae/ntu_rgbd_vibe/vanilla_vae_lie_mse_kld01/opt.txt',
        #     num_motions, 128, device, ground_truth_motion_loader),
        #'ground_truth': lambda num_motions, device: get_dataset_motion_loader(
        #    get_opt(dataset_opt_path, num_motions, device), num_motions, device),
        # 'vanila_vae_tf': lambda num_motions, device: get_motion_loader(
        #     './checkpoints/vae/humanact12/vanilla_vae_tf_fineG/opt.txt',
        #     num_motions, 128, device),
        'vanila_vae_tf': lambda num_motions, device: get_motion_loader(
            './checkpoints/vae/mocap/vanila_vae_tf_2/opt.txt',
            num_motions, 128, device, label=label_spe),

        # 'motion_gan': lambda num_motions, device: get_motion_loader(
        #      './checkpoints/humanact12/motion_gan_fineG/opt.txt',
        #      num_motions, 128, device),
        # 'conditionedRNN': lambda num_motions, device: get_motion_loader(
        #      './model_file/conditionedRNN_act12_opt_fineG.txt',
        #      num_motions, 128, device, ground_truth_motion_loader),
        # 'deep_completion': lambda num_motions, device: get_motion_loader(
        #     './model_file/deep_completion_act12_opt_fineG.txt',
        #      num_motions, 128, device),
    }

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(0)
    num_motions = 3000
    # num_motions = 200

    dataset_opt = get_opt(dataset_opt_path, num_motions, device)
    # print(dataset_opt)
    gru_classifier_for_fid = load_classifier_for_fid(dataset_opt, device)
    gru_classifier = load_classifier(dataset_opt, device)

    ground_truth_motion_loader = get_dataset_motion_loader(dataset_opt, num_motions, device, label=label_spe)
    motion_loaders = {}
    # motion_loaders['ground_truth'] = ground_truth_motion_loader
    '''
    for motion_loader_name, motion_loader_getter in eval_motion_loaders.items():
    motion_loader = motion_loader_getter(num_motions, device)
    motion_loaders[motion_loader_name] = motion_loader
    save_dir = './eval_results/user_study'
    animation_4_user_study(save_dir, motion_loaders)
    
    '''
    log_file = 'final_evaluation_mocap_veloc_label3_bk.log'
    evaluation(log_file)

