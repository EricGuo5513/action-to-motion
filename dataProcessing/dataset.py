import torch
from torch.utils import data
import pandas as pd
import csv
import os
import numpy as np
import numpy.matlib
import codecs as cs
import scipy.io as sio

import utils.paramUtil as paramUtil
import codecs
import joblib
from lie.pose_lie import *

class MotionFolderDatasetMocap(data.Dataset):
    def __init__(self, filename, datapath, opt, do_offset=True):
        self.clip = pd.read_csv(filename, index_col=False).dropna(how='all').dropna(axis=1, how='all')
        self.datapath = datapath
        self.lengths = []
        self.data = []
        self.labels = []
        self.opt = opt
        for i in range(self.clip.shape[0]):
            motion_name = self.clip.iloc[i]['motion']
            action_type = self.clip.iloc[i]['action_type']
            npy_path = os.path.join(datapath, motion_name + '.npy')

            # motion_length, joints_num, 3
            pose_raw = np.load(npy_path)
            # rescale the pose
            pose_raw = pose_raw / 20

            # Locate the root joint of initial pose at origin
            if do_offset:
                # get the offset and return the final pose
                offset_mat = np.tile(pose_raw[0, 0], (pose_raw.shape[1], 1))
                pose_mat = pose_raw - offset_mat
            else:
                pose_mat = pose_raw

            pose_mat = pose_mat.reshape((-1, 20 * 3))

            # not used any more
            if self.opt.no_trajectory:
                # for lie params, just exclude the root translation part
                if self.opt.lie_enforce:
                    pose_mat = pose_mat[:, 3:]
                else:
                    offset = np.tile(pose_mat[..., :3], (1, int(pose_mat.shape[1] / 3)))
                    pose_mat = pose_mat - offset

            self.data.append((pose_mat, action_type))
            if action_type not in self.labels:
                self.labels.append(action_type)
            self.lengths.append(pose_mat.shape[0])
        self.cumsum = np.cumsum([0] + self.lengths)
        print("Total number of frames {}, videos {}, action types {}".format(self.cumsum[-1], self.clip.shape[0],
                                                                             len(self.labels)))
        self.label_enc = dict(zip(self.labels, np.arange(len(self.labels))))
        self.label_enc_rev = dict(zip(np.arange(len(self.labels)), self.labels))
        with codecs.open(os.path.join(opt.save_root, "label_enc_rev_mocap.txt"), 'w', 'utf-8') as f:
            for item in self.label_enc_rev.items():
                f.write(str(item) + "\n")

    def __len__(self):
        return len(self.data)

    def get_label_reverse(self, enc_label):
        return self.label_enc_rev.get(enc_label)

    def __getitem__(self, index):
        pose_mat, label = self.data[index]
        label = self.label_enc[label]
        return pose_mat, label


class MotionFolderDatasetHumanAct12(data.Dataset):
    def __init__(self, datapath, opt, lie_enforce, do_offset=True, raw_offsets=None, kinematic_chain=None):
        self.datapath = datapath
        self.do_offset = do_offset
        self.lengths = []
        self.data = []
        self.labels = []
        self.opt = opt
        data_list = os.listdir(datapath)
        data_list.sort()

        if lie_enforce:
            raw_offsets = torch.from_numpy(raw_offsets)
            self.lie_skeleton = LieSkeleton(raw_offsets, kinematic_chain, torch.DoubleTensor)

        for file_name in data_list:
            full_path = os.path.join(self.datapath, file_name)
            pose_raw = np.load(full_path)

            # Locate the root joint of initial pose at origin
            if do_offset:
                # get the offset and return the final pose
                offset_mat = np.tile(pose_raw[0, 0], (pose_raw.shape[1], 1))
                pose_mat = pose_raw - offset_mat
            else:
                pose_mat = pose_raw

            # dataLoader will return Lie parameters
            if lie_enforce and opt.isTrain:
                # the first column of lie_params is zeros
                # dim (motion_len, joints_num, 3)
                pose_mat = torch.from_numpy(pose_mat)
                lie_params = self.lie_skeleton.inverse_kinemetics(pose_mat).numpy()
                # use the first column to store root translation information
                pose_mat = np.concatenate((np.expand_dims(pose_mat[:, 0, :], axis=1)
                                           , lie_params[:, 1:, :])
                                           , axis=1)

            pose_mat = pose_mat.reshape((-1, 24 * 3))

            # not used any more
            if self.opt.no_trajectory:
                # for lie params, just exclude the root translation part
                if self.opt.lie_enforce:
                    pose_mat = pose_mat[:, 3:]
                else:
                    offset = np.tile(pose_mat[..., :3], (1, int(pose_mat.shape[1] / 3)))
                    pose_mat = pose_mat - offset

            label = file_name[file_name.find('A') + 1: file_name.find('.')]
            # print(file_name)
            if opt.coarse_grained:
                label = label[:2]
            if label not in self.labels:
                self.labels.append(label)
            self.data.append((pose_mat, label))
            self.lengths.append(pose_mat.shape[0])
        self.labels.sort()
        self.cumsum = np.cumsum([0] + self.lengths)
        print("Total number of frames {}, videos {}, action types {}".format(self.cumsum[-1], len(data_list), len(self.labels)))
        self.label_enc = dict(zip(self.labels, np.arange(len(self.labels))))
        self.label_enc_rev = dict(zip(np.arange(len(self.labels)), self.labels))
        with codecs.open(os.path.join(opt.save_root, "label_enc_rev_humanact13.txt"), 'w', 'utf-8') as f:
            for item in self.label_enc_rev.items():
                f.write(str(item) + "\n")

    def __len__(self):
        return len(self.data)

    def get_label_reverse(self, enc_label):
        return self.label_enc_rev.get(enc_label)

    def __getitem__(self, index):
        pose_mat, label = self.data[index]
        label = self.label_enc[label]
        return pose_mat, label


class MotionFolderDatasetNtuVIBE(data.Dataset):
    def __init__(self, file_prefix, candi_list_desc, labels, opt, joints_num=18, do_offset=True, extract_joints=None):
        self.data = []
        self.labels = labels
        self.lengths = []
        self.label_enc = dict(zip(labels, np.arange(len(labels))))
        self.label_enc_rev = dict(zip(np.arange(len(labels)), labels))
        candi_list = []

        candi_list_desc_name = os.path.join(file_prefix, candi_list_desc)
        with cs.open(candi_list_desc_name, 'r', 'utf-8') as f:
            for line in f.readlines():
                candi_list.append(line.strip())

        for path in candi_list:
            data_org = joblib.load(os.path.join(file_prefix, path))
            # (motion_length, 49, 3)
            # print(os.path.join(file_prefix, path))
            try:
                data_mat = data_org[1]['joints3d']
            except Exception:
                continue
            action_id = int(path[path.index('A') + 1:-4])
            motion_mat = data_mat

            if extract_joints is not None:
                # (motion_length, len(extract_joints, 3))
                motion_mat = motion_mat[:, extract_joints, :]


            # change the root keypoint of skeleton, exchange the location of 0 and 8
            #if opt.use_lie:
            tmp = np.array(motion_mat[:, 0, :])
            motion_mat[:, 0, :] = motion_mat[:, 8, :]
            motion_mat[:, 8, :] = tmp

            # Locate the root joint of initial pose at origin
            if do_offset:
                offset_mat = motion_mat[0][0]
                motion_mat = motion_mat - offset_mat

            # (motion_length, len(extract_joints) * 3)
            motion_mat = motion_mat.reshape((-1, joints_num * 3))

            self.data.append((motion_mat, action_id))
            self.lengths.append(data_mat.shape[0])
        self.cumsum = np.cumsum([0] + self.lengths)
        print("Total number of frames {}, videos {}, action types {}".format(self.cumsum[-1], len(self.data), len(self.labels)))
        with codecs.open(os.path.join(opt.save_root, "label_enc_rev_ntu_vibe.txt"), 'w', 'utf-8') as f:
            for item in self.label_enc_rev.items():
                f.write(str(item) + "\n")

    def get_label_reverse(self, en_label):
        return self.label_enc_rev[en_label]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        pose_mat, label = self.data[item]
        en_label = self.label_enc[label]
        return pose_mat, en_label


class PoseDataset(data.Dataset):
    def __init__(self, dataset, lie_enforce=False, no_trajectory=False):
        self.dataset = dataset
        self.lie_enforce = lie_enforce
        self.no_trajectory = no_trajectory

    def __getitem__(self, item):
        if item != 0:
            motion_id = np.searchsorted(self.dataset.cumsum, item) - 1
            pose_num = item - self.dataset.cumsum[motion_id] - 1
        else:
            motion_id = 0
            pose_num = 0
        motion, label = self.dataset[motion_id]
        pose = motion[pose_num]
        # offset for each pose
        if self.lie_enforce:
            if not self.no_trajectory:
                pose[:3] = 0.0
                pose_o = pose
            else:
                pose_o = pose
        else:
            offset = np.tile(pose[0:3], int(pose.shape[0] / 3))
            pose_o = pose - offset
        return pose_o, label

    def __len__(self):
        return self.dataset.cumsum[-1]


class MotionDataset(data.Dataset):
    def __init__(self, dataset, opt):
        self.dataset = dataset
        self.motion_length = opt.motion_length
        self.opt = opt

    def __getitem__(self, item):
        motion, label = self.dataset[item]
        motion = np.array(motion)
        motion_len = motion.shape[0]
        # Motion can be of various length, we randomly sample sub-sequence
        # or repeat the last pose for padding

        # random sample
        if motion_len >= self.motion_length:
            gap = motion_len - self.motion_length
            start = 0 if gap == 0 else np.random.randint(0, gap, 1)[0]
            end = start + self.motion_length
            r_motion = motion[start:end]
            # offset deduction
            r_motion = r_motion - np.tile(r_motion[0, :3], (1, int(r_motion.shape[-1]/3)))
        # padding
        else:
            gap = self.motion_length - motion_len
            last_pose = np.expand_dims(motion[-1], axis=0)
            pad_poses = np.repeat(last_pose, gap, axis=0)
            r_motion = np.concatenate([motion, pad_poses], axis=0)
        return r_motion, label

    def __len__(self):
        return len(self.dataset)
