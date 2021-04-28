import math
import time
from PIL import Image
import os
from scipy.ndimage import gaussian_filter


def save_logfile(log_loss, save_path):
    with open(save_path, 'wt') as f:
        for k, v in log_loss.items():
            w_line = k
            for digit in v:
                w_line += ' %.3f' % digit
            f.write(w_line + '\n')


def print_current_loss(start_time, niter_state, total_niters, losses, epoch=None, inner_iter=None, current_kld=None):

    def as_minutes(s):
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)

    def time_since(since, percent):
        now = time.time()
        s = now - since
        es = s / percent
        rs = es - s
        return '%s (- %s)' % (as_minutes(s), as_minutes(rs))

    if epoch is not None:
        print('epoch: %2d inner_iter: %3d' % (epoch, inner_iter), end=" ")
    message = '%s niter: %d completed: %3d%%)' % (time_since(start_time, niter_state / total_niters),
                                                niter_state, niter_state / total_niters * 100)
    for k, v in losses.items():
        message += ' %s: %.3f ' % (k, v)
    if current_kld is not None:
        message += ' current_kld_weight: %f' % (current_kld)
    print(message)


def compose_and_save_img(img_list, save_dir, img_name, col=4, row=1, img_size=(256, 200)):
    # print(col, row)
    compose_img = compose_image(img_list, col, row, img_size)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    img_path = os.path.join(save_dir, img_name)
    # print(img_path)
    compose_img.save(img_path)


def compose_image(img_list, col, row, img_size):
    to_image = Image.new('RGB', (col * img_size[0], row * img_size[1]))
    for y in range(0, row):
        for x in range(0, col):
            from_img = Image.fromarray(img_list[y * col + x])
            # print((x * img_size[0], y*img_size[1],
            #                           (x + 1) * img_size[0], (y + 1) * img_size[1]))
            paste_area = (x * img_size[0], y*img_size[1],
                                      (x + 1) * img_size[0], (y + 1) * img_size[1])
            to_image.paste(from_img, paste_area)
            # to_image[y*img_size[1]:(y + 1) * img_size[1], x * img_size[0] :(x + 1) * img_size[0]] = from_img
    return to_image


def temporal_filter(motion, sigma=1):
    motion = motion.reshape(motion.shape[0], -1)
    # print(motion.shape)
    for i in range(motion.shape[0]):
        motion[:, i] = gaussian_filter(motion[:, i], sigma=sigma, mode="nearest")
    return motion