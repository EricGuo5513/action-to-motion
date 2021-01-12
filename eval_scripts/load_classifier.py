import torch
import utils.paramUtil as paramUtil
from models.motion_gan import MotionDiscriminator


classifier_model_files = {
    'ntu_rgbd_vibe': './model_file/action_recognition_model_vibe_v2.tar',
    'humanact12': './model_file/action_recognition_model_humanact12.tar',
    'mocap': './model_file/action_recognition_model_mocap_new.tar'
}


def load_classifier(opt, device):
    model = torch.load(classifier_model_files[opt.dataset_type])
    classifier = MotionDiscriminator(opt.input_size_raw, 128, 2, len(opt.label_dec)).to(device)
    classifier.load_state_dict(model['model'])
    classifier.eval()

    return classifier


class MotionDiscriminatorForFID(MotionDiscriminator):
    def forward(self, motion_sequence, hidden_unit=None):
        # dim (motion_length, num_samples, hidden_size)
        if hidden_unit is None:
            motion_sequence = motion_sequence.permute(1, 0, 2)
            hidden_unit = self.initHidden(motion_sequence.size(1), self.hidden_layer)
        gru_o, _ = self.recurrent(motion_sequence.float(), hidden_unit)
        # dim (num_samples, 30)
        lin1 = self.linear1(gru_o[-1, :, :])
        lin1 = torch.tanh(lin1)
        return lin1


def load_classifier_for_fid(opt, device):
    model = torch.load(classifier_model_files[opt.dataset_type])
    # print(len(opt.label_dec))
    classifier = MotionDiscriminatorForFID(opt.input_size_raw, 128, 2, len(opt.label_dec)).to(device)
    classifier.load_state_dict(model['model'])
    classifier.eval()

    return classifier
