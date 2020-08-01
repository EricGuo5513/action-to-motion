import torch
from lie.lie_util import *
from torch import nn


class LieSkeleton(object):
    def __init__(self, raw_translation, kinematic_tree, tensor):
        super(LieSkeleton, self).__init__()
        self.tensor = tensor
        # print(self.tensor)
        self._raw_translation = self.tensor(raw_translation.shape).copy_(raw_translation).detach()
        self._kinematic_tree = kinematic_tree
        self._translation = None
        self._parents = [0] * len(self._raw_translation)
        self._parents[0] = -1
        for chain in self._kinematic_tree:
            for j in range(1, len(chain)):
                self._parents[chain[j]] = chain[j-1]

    def njoints(self):
        return len(self._raw_translation)

    def raw_translation(self):
        return self._raw_translation

    def kinematic_tree(self):
        return self._kinematic_tree

    def parents(self):
        return self._translation

    def get_translation_joints(self, joints):
        # joints/offsets (batch_size, joints_num, 3)
        # print(self._raw_translation.shape)
        _translation = self._raw_translation.clone().detach()
        _translation = _translation.expand(joints.shape[0], -1, -1).clone()
        #print(_translation.shape)
        #print(self._raw_translation.shape)
        for i in range(1, self._raw_translation.shape[0]):
            _translation[:, i, :] = torch.norm(joints[:, i, :] - joints[:, self._parents[i], :], p=2, dim=1)[:, None] * \
                                     _translation[:, i, :]
        self._translation = _translation
        return _translation

    def get_translation_bone(self, bonelengths):
        # bonelength (batch_size, joints_num - 1)
        # offsets (batch_size, joints_num, 3)
        self._translation = self._raw_translation.clone().detach().expand(bonelengths.size(0), -1, -1).clone().to(bonelengths.device)
        self._translation[:, 1:, :] = bonelengths * self._translation[:, 1:, :]

    def inverse_kinemetics(self, joints):
        # joints (batch_size, joints_num, 3)
        # lie_params (batch_size, joints_num, 3)
        lie_params = self.tensor(joints.shape).fill_(0)
        # root_matR (batch_size, 3, 3)
        root_matR = torch.eye(3, dtype=joints.dtype).expand((joints.shape[0], -1, -1)).clone().detach().to(joints.device)
        for chain in self._kinematic_tree:
            R = root_matR
            for j in range(len(chain) - 1):
                # (batch, 3)
                u = self._raw_translation[chain[j + 1]].expand(joints.shape[0], -1).clone().detach().to(joints.device)
                # (batch, 3)
                v = joints[:, chain[j+1], :] - joints[:, chain[j], :]
                # (batch, 3)
                v = v / torch.norm(v, p=2, dim=1)[:, None]
                # (batch, 3, 3)
                R_local = torch.matmul(R.transpose(1, 2), lie_exp_map(lie_u_v(u, v)))
                # print("R_local shape:" + str(R_local.shape))
                # print(R_local)
                lie_params[:, chain[j + 1], :] = matR_log_map(R_local)
                R = torch.matmul(R, R_local)
        return lie_params

    def forward_kinematics(self, lie_params, joints, root_translation, do_root_R = False, scale_inds=None):
        # lie_params (batch_size, joints_num, 3) lie_params[:, 0, :] is not used
        # joints (batch_size, joints_num, 3)
        # root_translation (batch_size, 3)
        # translation_mat (batch_size, joints_num, 3)
        translation_mat = self.get_translation_joints(joints)
        if scale_inds is not None:
            translation_mat[:, scale_inds, :] *= 1.25
        joints = self.tensor(lie_params.size()).fill_(0)
        joints[:, 0] = root_translation
        for chain in self._kinematic_tree:
            # if do_root_R is true, root coordinate system has rotation angulers
            # Plus, for chain not containing root(e.g arms), we use root rotation as the rotation
            # of joints near neck(i.e. beginning of this chain).
            if do_root_R:
                matR = lie_exp_map(lie_params[:, 0, :])
            # Or, root rotation matrix is identity matrix, which means no rotation at global coordinate system
            else:
                matR = torch.eye(3, dtype=joints.dtype).expand((joints.shape[0], -1, -1)).clone().detach().to(joints.device)
            for i in range(1, len(chain)):
                matR = torch.matmul(matR, lie_exp_map(lie_params[:, chain[i], :]))
                translation_vec = translation_mat[:, chain[i], :].unsqueeze_(-1)
                joints[:, chain[i], :] = torch.matmul(matR, translation_vec).squeeze_()\
                                         + joints[:, chain[i-1], :]
        return joints
