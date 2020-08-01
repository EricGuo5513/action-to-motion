import torch
import torch.nn as nn
from torch import tensor
import numpy as np
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"

class Noise(nn.Module):
    def __init__(self, use_noise, sigma=0.2):
        super(Noise, self).__init__()
        self.use_noise = use_noise
        self.sigma = 0.2

    def forward(self, x):
        if self.use_noise:
            return x + self.sigma * tensor(torch.randn(x.size), dtype=torch.double,\
                                           device=device, requires_grad=False)
        return x


class MotionGenerator(nn.Module):
    def __init__(self, dim_z, dim_category, motion_length, hidden_size, opt, joints_num=24, input_size=72, output_size=72):
        super(MotionGenerator, self).__init__()

        self.dim_z = dim_z
        self.opt = opt
        self.motion_length = motion_length
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.dim_category = dim_category
        self.dim_h = hidden_size + dim_category
        self.joints_num = joints_num

        self.layers = 2

        self.recurrent_z = nn.GRU(dim_z, hidden_size, 1)

        self.recurrent_h = nn.GRU(self.dim_h, hidden_size, self.layers)
        self.linear = nn.Linear(self.hidden_size, output_size)

    # Resample noise for each time step
    def sample_z_r(self, num_samples, motion_len=None):
        motion_len = motion_len if motion_len is not None else self.motion_length
        z_t = []
        for pose_num in range(motion_len):
            z_t.append(self.get_normal_noise(num_samples))
        # dim motion_len x (num_samples, dim_z)
        z_m_t = [z_k.view(1, -1, self.dim_z) for z_k in z_t]
        # dim (motion_len, num_samples, dim_z)
        z_m = torch.cat(z_m_t, dim=0)
        return z_m

    # Same noise for each time step
    def sample_z_s(self, num_samples, motion_len=None):
        motion_len = motion_len if motion_len is not None else self.motion_length

        # dim (1, num_samples, dim_z)
        z_s = self.get_normal_noise(num_samples).unsqueeze_(0)

        # dim (motion_len, num_samples, dim_z)
        z_s = z_s.repeat([motion_len, 1]).requires_grad_(False)
        return z_s

    # Sample a category for generation
    def sample_z_categ(self, num_samples, motion_len=None):
        motion_len = motion_len if motion_len is not None else self.motion_length

        if self.dim_category <= 0:
            return None, np.zeros(num_samples)
        # dim (num_samples, )
        classes_to_generate = np.random.randint(self.dim_category, size=num_samples)
        # dim (num_samples, dim_category)
        one_hot = np.zeros((num_samples, self.dim_category), dtype=np.float32)
        one_hot[np.arange(num_samples), classes_to_generate] = 1

        # dim (motion_len, num_samples, dim_category)
        one_hot_motion = np.expand_dims(one_hot, axis=0).repeat(motion_len, axis=0)
        one_hot_motion = torch.from_numpy(one_hot_motion).to(device).requires_grad_(False)

        return one_hot_motion, classes_to_generate

    # fix the category for generation
    def fix_z_categ(self, categories, motion_len=None):
        motion_len = motion_len if motion_len is not None else self.motion_length

        classes_to_generate = np.array(categories).reshape((-1,))
        # dim (num_samples, dim_category)
        one_hot = np.zeros((categories.shape[0], self.dim_category), dtype=np.float32)
        one_hot[np.arange(categories.shape[0]), classes_to_generate] = 1

        # dim (motion_len, num_samples, dim_category)
        one_hot_motion = np.expand_dims(one_hot, axis=0).repeat(motion_len, axis=0)
        one_hot_motion = torch.from_numpy(one_hot_motion).to(device).requires_grad_(False)

        return one_hot_motion, classes_to_generate

    # Motion noises first pass a one-layer GRU, and then concated with the
    # category one-hot vector, yielding the final input
    def sample_z_motion(self, num_samples, motion_len=None):
        # dim (motion_len, num_samples, dim_z)
        z_m = self.sample_z_r(num_samples, motion_len)

        # dim (1, num_samples, hidden_size)
        h_0 = self.init_hidden(num_samples)

        # h_n is the last hidden unit, z_t dim (motion_len, num_samples, hidden_size)
        z_t, h_n = self.recurrent_z(z_m, h_0)

        # z_c dim (motion_len, num_samples, dim_category)
        z_c, z_category_labels = self.sample_z_categ(num_samples, motion_len)
        # dim (motion_len, num_samples, dim_category + hidden_size)
        z = torch.cat((z_t, z_c), dim=-1)
        return z, torch.from_numpy(z_category_labels).requires_grad_(False).to(device)

    # For evaluation, instead of training
    def create_z_motion(self, z_m, z_c, num_samples):
        if z_m.size(0) != z_c.size(0):
            raise ValueError("Length of category and motion don't match")

        if z_m.size(1) != z_c.size(1):
            raise ValueError("Num_samples of category anf motion don't match")

        h_0 = self.init_hidden(num_samples)
        z_t, h_n = self.recurrent_z(z_m, h_0)
        z = torch.cat((z_t, z_c), dim=-1)
        return z

    # For evaluation, instead of training
    def generate_motion_fixed_noise(self, inputs, num_samples):
        h_0 = self.init_hidden(num_samples, self.layers)
        h_outs, h_n = self.recurrent_h(inputs, h_0)
        p_outs = self.linear(h_outs)
        return p_outs, None

    def generate_motion(self, num_samples):
        # dim (motion_len, num_samples, dim_h)
        z, z_category_labels = self.sample_z_motion(num_samples, self.motion_length)

        # dim (layers, num_samples, dim_h)
        h_0 = self.init_hidden(num_samples, self.layers)

        # dim (motion_len, num_samples, hidden_size)
        h_outs, h_n = self.recurrent_h(z, h_0)

        # dim (motion_len, num_samples, output_size)
        p_outs = self.linear(h_outs)
        return p_outs, z_category_labels

    def sample_motion_clips(self, num_samples, motion_len=None):
        motion_len = motion_len if motion_len is not None else self.motion_length
        # dim (motion_length, num_samples, output_size) (num_samples, dim_category)
        p_outs, z_category_labels = self.generate_motion(num_samples)

        # return the whole sequence
        if motion_len >= self.motion_length:
            # dim (num_samples, motion_length, output_size)
            p_outs = p_outs.permute(1, 0, 2)
            return p_outs, z_category_labels
        # return a subsequences(motion clips)
        else:
            upper_bound = self.motion_length - motion_len
            l_inds = np.random.randint(upper_bound)
            u_inds = l_inds + motion_len
            c_outs = p_outs[l_inds:u_inds, :, :]
            # dim (num_samples, motion_len, output_size)
            c_outs = c_outs.permute(1, 0, 2)
            return c_outs, z_category_labels

    def sample_poses(self, num_samples):
        # dim (motion_length, num_samples, output_size) (num_samples, dim_category)
        p_outs, z_category_labels = self.generate_motion(num_samples)

        # Pick a random pose for each motion sequence
        s_pose_inds = np.random.choice(self.motion_length, num_samples).astype(np.int64)
        # dim (1, num_samples, output_size)
        sample_poses = p_outs[s_pose_inds, np.arange(num_samples), :]
        # dim (num_samples, output_size)
        sample_poses = sample_poses.squeeze_()

        # offset for each pose, by subtracting its first coordinates
        if not self.opt.use_lie:
            sample_poses_offset = sample_poses[:, 0:3].repeat([1, self.joints_num])
            sample_poses = sample_poses - sample_poses_offset
        return sample_poses, None

    def init_hidden(self, num_samples, layers=1):
        # return torch.zeros(num_samples, self.hidden_size, device=device, requires_grad=False)
        return torch.randn(layers, num_samples, self.hidden_size, device=device).float().requires_grad_(False)

    def get_normal_noise(self, num_samples):
        return torch.randn(num_samples, self.dim_z, device=device).float().requires_grad_(False)


class MotionGeneratorLie(MotionGenerator):
    def __init__(self, dim_z, dim_category, motion_length, hidden_size, opt, joints_num=24, input_size=72, output_size=72):
        super(MotionGeneratorLie, self).__init__(dim_z,
                                                 dim_category,
                                                 motion_length,
                                                 hidden_size,
                                                 opt,
                                                 joints_num,
                                                 input_size,
                                                 output_size)
        self.opt = opt
        if self.opt.no_trajectory:
            self.linear_lie = nn.Linear(self.hidden_size, self.output_size)
        else:
            self.linear_lie = nn.Linear(self.hidden_size - 6, self.output_size - 3)
            self.linear_traj = nn.Linear(6, 3)
        self.PI = 3.1415926
        # self.batch_norm = nn.BatchNorm2d()

    def generate_motion(self, num_samples):
        # dim (motion_len, num_samples, dim_h)
        z, z_category_labels = self.sample_z_motion(num_samples, self.motion_length)

        # dim (layers, num_samples, dim_h)
        h_0 = self.init_hidden(num_samples, self.layers)

        # dim (motion_len, num_samples, hidden_size)
        h_outs, _ = self.recurrent_h(z, h_0)
        if self.opt.no_trajectory:
            p_lie = self.linear_lie(h_outs)
            p_lie = torch.tanh(p_lie) * self.PI
            p_outs = p_lie
        else:
            p_lie = self.linear_lie(h_outs[..., :-6])
            # dim (motion_len, num_samples, output_size - 3)
            p_lie = torch.tanh(p_lie) * self.PI
            # dim (motion_len, num_samples, 3)
            p_trajec = self.linear_traj(h_outs[..., -6:])
            # dim (motion_len, num_samples, output_size)
            p_outs = torch.cat((p_lie, p_trajec), dim=-1)
        return p_outs, z_category_labels

    def generate_motion_fixed_noise(self, inputs, num_samples):
        h_0 = self.init_hidden(num_samples, self.layers)
        # dim (motion_len, num_samples, hidden_size)
        h_outs, _ = self.recurrent_h(inputs, h_0)

        if self.opt.no_trajectory:
            p_lie = self.linear_lie(h_outs)
            p_lie = torch.tanh(p_lie) * self.PI
            p_outs = p_lie
        else:
            p_lie = self.linear_lie(h_outs[..., :-6])
            # dim (motion_len, num_samples, output_size - 3)
            p_lie = torch.tanh(p_lie) * self.PI
            # dim (motion_len, num_samples, 3)
            p_trajec = self.linear_traj(h_outs[..., -6:])
            # dim (motion_len, num_samples, output_size)
            p_outs = torch.cat((p_lie, p_trajec), dim=-1)

        return p_outs, None


class MotionDiscriminator(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_layer, output_size=1, use_noise=None):
        super(MotionDiscriminator, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.hidden_layer = hidden_layer
        self.use_noise = use_noise

        self.recurrent = nn.GRU(input_size, hidden_size, hidden_layer)
        self.linear1 = nn.Linear(hidden_size, 30)
        self.linear2 = nn.Linear(30, output_size)

    def forward(self, motion_sequence, hidden_unit=None):
        # dim (motion_length, num_samples, hidden_size)
        if hidden_unit is None:
            motion_sequence = motion_sequence.permute(1, 0, 2)
            hidden_unit = self.initHidden(motion_sequence.size(1), self.hidden_layer)
        gru_o, _ = self.recurrent(motion_sequence.float(), hidden_unit)
        # dim (num_samples, 30)
        lin1 = self.linear1(gru_o[-1, :, :])
        lin1 = torch.tanh(lin1)
        # dim (num_samples, output_size)
        lin2 = self.linear2(lin1)
        return lin2, _

    def initHidden(self, num_samples, layer):
        return torch.randn(layer, num_samples, self.hidden_size, device=device, requires_grad=False)


class CategoricalMotionDiscriminator(MotionDiscriminator):
    def __init__(self, input_size, hidden_size, hidden_layer, dim_categorical, output_size=1, use_noise=None):
        super(CategoricalMotionDiscriminator, self).__init__(input_size=input_size,
                                                             hidden_size=hidden_size,
                                                             hidden_layer=hidden_layer,
                                                             output_size=output_size + dim_categorical,
                                                             use_noise=use_noise)
        self.dim_categorical = dim_categorical

    def split(self, vec):
        return vec[:, 0], vec[:, 1:]

    def forward(self, motion_sequence, hidden_unit=None):
        # dim (motion_length, num_samples, input_size)
        motion_sequence = motion_sequence.permute(1, 0, 2)
        # dim (hidden_layers, num_samples, hidden_size)
        hidden_unit = self.initHidden(motion_sequence.size(1), self.hidden_layer)
        pre_vector, _ = super(CategoricalMotionDiscriminator, self).\
            forward(motion_sequence, hidden_unit)
        # dim (num_samples) (num_samples, dim_category)
        labels, categ = self.split(pre_vector)
        return labels, categ


class PoseDiscriminator(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(PoseDiscriminator, self).__init__()

        self.main = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, inputs):
        h = self.main(inputs.float()).squeeze()
        # print(h.size())
        return h, None