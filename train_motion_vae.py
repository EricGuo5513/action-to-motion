from torch.utils.data import DataLoader

import models.motion_vae as vae_models
import utils.paramUtil as paramUtil
from trainer.vae_trainer import *
from dataProcessing import dataset
from utils.plot_script import plot_loss
from options.train_vae_options import TrainOptions
import os


if __name__ == "__main__":
    parser = TrainOptions()
    opt = parser.parse()
    device = torch.device("cuda:" + str(opt.gpu_id) if torch.cuda.is_available() else "cpu")

    opt.save_root = os.path.join(opt.checkpoints_dir, opt.dataset_type, opt.name)
    opt.model_path = os.path.join(opt.save_root, 'model')
    opt.joints_path = os.path.join(opt.save_root, 'joints')
    opt.log_path = os.path.join(opt.save_root, "log.txt")

    if not os.path.exists(opt.model_path):
        os.makedirs(opt.model_path)
    if not os.path.exists(opt.joints_path):
        os.makedirs(opt.joints_path)

    dataset_path = ""
    joints_num = 0
    input_size = 72
    data = None

    if opt.dataset_type == "humanact12":
        dataset_path = "./dataset/humanact12"
        input_size = 72
        joints_num = 24
        raw_offsets = paramUtil.humanact12_raw_offsets
        kinematic_chain = paramUtil.humanact12_kinematic_chain
        data = dataset.MotionFolderDatasetHumanAct12(dataset_path, opt, lie_enforce=opt.lie_enforce)

    elif opt.dataset_type == "mocap":
        dataset_path = "./dataset/mocap/mocap_3djoints/"
        clip_path = './dataset/mocap/pose_clip.csv'
        input_size = 60
        joints_num = 20
        raw_offsets = paramUtil.mocap_raw_offsets
        kinematic_chain = paramUtil.mocap_kinematic_chain
        data = dataset.MotionFolderDatasetMocap(clip_path, dataset_path, opt)

    elif opt.dataset_type == "ntu_rgbd_vibe":
        file_prefix = "./dataset"
        motion_desc_file = "ntu_vibe_list.txt"
        joints_num = 18
        input_size = 54
        labels = paramUtil.ntu_action_labels
        raw_offsets = paramUtil.vibe_raw_offsets
        kinematic_chain = paramUtil.vibe_kinematic_chain
        data = dataset.MotionFolderDatasetNtuVIBE(file_prefix, motion_desc_file, labels, opt, joints_num=joints_num,
                                              offset=True, extract_joints=paramUtil.kinect_vibe_extract_joints)
    else:
        raise NotImplementedError('This dataset is unregonized!!!')

    opt.dim_category = len(data.labels)
    # arbitrary_len won't limit motion length, but the batch size has to be 1
    if opt.arbitrary_len:
        opt.batch_size = 1
        motion_loader = DataLoader(data, batch_size=opt.batch_size, drop_last=True, num_workers=1, shuffle=True)
    else:
        motion_dataset = dataset.MotionDataset(data, opt)
        motion_loader = DataLoader(motion_dataset, batch_size=opt.batch_size, drop_last=True, num_workers=2, shuffle=True)
    opt.pose_dim = input_size

    if opt.time_counter:
        opt.input_size = input_size + opt.dim_category + 1
    else:
        opt.input_size = input_size + opt.dim_category

    opt.output_size = input_size
    prior_net = vae_models.GaussianGRU(opt.input_size, opt.dim_z, opt.hidden_size,
                                       opt.prior_hidden_layers, opt.batch_size, device)
    posterior_net = vae_models.GaussianGRU(opt.input_size, opt.dim_z, opt.hidden_size,
                                           opt.posterior_hidden_layers, opt.batch_size, device)
    if opt.use_lie:
        decoder = vae_models.DecoderGRULie(opt.input_size + opt.dim_z, opt.output_size, opt.hidden_size,
                                           opt.decoder_hidden_layers,
                                           opt.batch_size, device)
    else:
        decoder = vae_models.DecoderGRU(opt.input_size + opt.dim_z, opt.output_size, opt.hidden_size,
                                        opt.decoder_hidden_layers,
                                        opt.batch_size, device)

    pc_prior = sum(param.numel() for param in prior_net.parameters())
    print(prior_net)
    print("Total parameters of prior net: {}".format(pc_prior))
    pc_posterior = sum(param.numel() for param in posterior_net.parameters())
    print(posterior_net)
    print("Total parameters of posterior net: {}".format(pc_posterior))
    pc_decoder = sum(param.numel() for param in decoder.parameters())
    print(decoder)
    print("Total parameters of decoder: {}".format(pc_decoder))

    if opt.use_lie:
        # Use Lie representation
        trainer = TrainerLie(motion_loader, opt, device, raw_offsets, kinematic_chain)
    else:
        # Use 3d coordinates representation
        trainer = Trainer(motion_loader, opt, device)

    logs = trainer.trainIters(prior_net, posterior_net, decoder)

    plot_loss(logs, os.path.join(opt.save_root, "loss_curve.png"), opt.plot_every)
    save_logfile(logs, opt.log_path)
