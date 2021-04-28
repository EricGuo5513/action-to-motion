import numpy as np
import matplotlib.pyplot as plt
import math
import numpy as np
import numpy.matlib
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, writers
import mpl_toolkits.mplot3d.axes3d as p3
import time
import cv2

COLORS = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
          [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
          [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

def plot_loss(losses, save_path, intervals=500):
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    for key in losses.keys():
        plt.plot(list_cut_average(losses[key], intervals), label=key)
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(save_path)
    plt.show()


def plot_2d_pose(pose, pose_tree, class_type, save_path=None, excluded_joints=None):
    def init():
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(class_type)

    fig = plt.figure()
    init()
    data = np.array(pose, dtype=float)

    if excluded_joints is None:
        plt.scatter(data[:, 0], data[:, 1], color='b', marker='h', s=15)
    else:
        plot_joints = [i for i in range(data.shape[1]) if i not in excluded_joints]
        plt.scatter(data[plot_joints, 0], data[plot_joints, 1], color='b', marker='h', s=15)

    for idx1, idx2 in pose_tree:
        plt.plot([data[idx1, 0], data[idx2, 0]],
                [data[idx1, 1], data[idx2, 1]], color='r', linewidth=2.0)

    # update(1)
    # plt.show()
    # Writer = writers['ffmpeg']
    # writer = Writer(fps=15, metadata={})
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()
    plt.close()


def list_cut_average(ll, intervals):
    if intervals == 1:
        return ll

    bins = math.ceil(len(ll) * 1.0 / intervals)
    ll_new = []
    for i in range(bins):
        l_low = intervals * i
        l_high = l_low + intervals
        l_high = l_high if l_high < len(ll) else len(ll)
        ll_new.append(np.mean(ll[l_low:l_high]))
    return ll_new



def plot_3d_pose_v2(savePath, kinematic_tree, joints, title=None):
    figure = plt.figure()
    # ax = plt.axes(xlim=(-1, 1), ylim=(-1, 1), zlim=(-1, 1), projection='3d')
    ax = Axes3D(figure)
    ax.set_ylim(-1, 1)
    ax.set_xlim(-1, 1)
    ax.set_zlim(-1, 1)
    if title is not None:
        ax.set_title(title)
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_zlabel('z')
    ax.view_init(elev=110, azim=90)
    # ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], color='black')
    colors = ['red', 'magenta', 'black', 'magenta', 'black', 'green', 'blue']
    for chain, color in zip(kinematic_tree, colors):
        ax.plot3D(joints[chain, 0], joints[chain, 1], joints[chain, 2], linewidth=5.0, color=color)
    plt.axis('off')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    plt.savefig(savePath)


def plot_3d_motion_v2(motion, kinematic_tree, save_path, interval=50, dataset=None):
    matplotlib.use('Agg')

    def init():
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        if dataset == "mocap":
            ax.set_ylim(-1.5, 1.5)
            ax.set_xlim(0, 3)
            ax.set_zlim(-1.5, 1.5)
        else:
            ax.set_ylim(-1, 1)
            ax.set_xlim(-1, 1)
            ax.set_zlim(-1, 1)

    fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    ax = p3.Axes3D(fig)
    init()

    data = np.array(motion, dtype=float)
    colors = ['red', 'magenta', 'black', 'green', 'blue']
    frame_number = data.shape[0]
    # dim (frame, joints, xyz)
    print(data.shape)

    def update(index):
        ax.lines = []
        ax.collections = []
        if dataset == "mocap":
            ax.view_init(elev=110, azim=-90)
        else:
            ax.view_init(elev=110, azim=90)
        for chain, color in zip(kinematic_tree, colors):
            ax.plot3D(motion[index, chain, 0], motion[index, chain, 1], motion[index, chain, 2], linewidth=4.0, color=color)
        # plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

    ani = FuncAnimation(fig, update, frames=frame_number, interval=interval, repeat=False, repeat_delay=200)
    # update(1)
    # plt.show()
    # Writer = writers['ffmpeg']
    # writer = Writer(fps=15, metadata={})
    ani.save(save_path, writer='pillow')
    plt.close()


def plot_3d_motion_with_trajec(motion, kinematic_tree, save_path, interval=50, trajec1=None, trajec2=None, dataset=None):
    matplotlib.use('Agg')

    def init():
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        if dataset == "mocap":
            ax.set_ylim(-1.5, 1.5)
            ax.set_xlim(0, 3)
            ax.set_zlim(-1.5, 1.5)
        else:
            ax.set_ylim(-1, 1)
            ax.set_xlim(-1, 1)
            ax.set_zlim(-1, 1)
        # ax.set_ylim(-1.0, 0.2)
        # ax.set_xlim(-0.2, 1.0)
        # ax.set_zlim(-1.0, 0.4)

    fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    ax = p3.Axes3D(fig)
    init()

    data = np.array(motion, dtype=float)
    colors = ['red', 'magenta', 'black', 'magenta', 'black', 'green', 'blue']
    frame_number = data.shape[0]
    # dim (frame, joints, xyz)
    print(data.shape)

    def update(index):
        ax.lines = []
        ax.collections = []
        if dataset == "mocap":
            ax.view_init(elev=110, azim=-90)
        else:
            ax.view_init(elev=110, azim=90)
        if trajec1 is not None:
            ax.plot3D(trajec1[:index+1, 0], trajec1[:index+1, 1], trajec1[:index+1, 2], linewidth=2.0, color='green')
        if trajec2 is not None:
            ax.plot3D(trajec2[:index+1, 0], trajec2[:index+1, 1], trajec2[:index+1, 2], linewidth=2.0, color='blue')
        for chain, color in zip(kinematic_tree, colors):
            ax.plot3D(motion[index, chain, 0], motion[index, chain, 1], motion[index, chain, 2], linewidth=4.0, color=color)
        # plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

    ani = FuncAnimation(fig, update, frames=frame_number, interval=interval, repeat=False, repeat_delay=200)
    # update(1)
    # plt.show()
    # Writer = writers['ffmpeg']
    # writer = Writer(fps=15, metadata={})
    ani.save(save_path, writer='pillow')
    plt.close()

def plot_3d_trajectory(data, save_path, ground=None):
    matplotlib.use('Agg')

    def init():
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_ylim(-1, 1)
        ax.set_xlim(-1, 1)
        ax.set_zlim(-1, 1)
        # ax.set_ylim(-1.0, 0.2)
        # ax.set_xlim(-0.2, 1.0)
        # ax.set_zlim(-1.0, 0.4)

    fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    ax = p3.Axes3D(fig)
    init()
    colors = ['red', 'magenta', 'black', 'green', 'blue']
    ax.plot3D(data[:, 0], data[:, 1], data[:, 2], linewidth=2.0, color='red')
    if ground is not None:
        ax.plot3D(ground[:, 0], ground[:, 1], ground[:, 2], linewidth=2.0, color='blue')
    plt.savefig(save_path)



def plot_3d_motion(motion, pose_tree, class_type, save_path, interval=300, excluded_joints=None):
    matplotlib.use('Agg')

    def init():
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_ylim(-0.75, 0.75)
        ax.set_xlim(-0.75, 0.75)
        ax.set_zlim(-0.75, 0.75)
        # ax.set_ylim(-1.0, 0.2)
        # ax.set_xlim(-0.2, 1.0)
        # ax.set_zlim(-1.0, 0.4)
        ax.set_title(class_type)

    fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    ax = p3.Axes3D(fig)
    init()

    data = np.array(motion, dtype=float)
    frame_number = data.shape[0]
    # dim (frame, joints, xyz)
    print(data.shape)

    def update(index):
        ax.lines = []
        ax.collections = []
        if excluded_joints is None:
            ax.scatter(data[index, :, 0], data[index, :, 1], data[index, :, 2], color='b', marker='h', s=15)
        else:
            plot_joints = [i for i in range(data.shape[1]) if i not in excluded_joints]
            ax.scatter(data[index, plot_joints, 0], data[index, plot_joints, 1], data[index, plot_joints, 2], color='b', marker='h', s=15)

        for idx1, idx2 in pose_tree:
            ax.plot([data[index, idx1, 0], data[index, idx2, 0]],
                    [data[index, idx1, 1], data[index, idx2, 1]], [data[index, idx1, 2], data[index, idx2, 2]], color='r', linewidth=2.0)

    ani = FuncAnimation(fig, update, frames=frame_number, interval=interval, repeat=False, repeat_delay=200)
    # update(1)
    # plt.show()
    # Writer = writers['ffmpeg']
    # writer = Writer(fps=15, metadata={})
    ani.save(save_path, writer='pillow')
    plt.close()


def plot_2d_motion(motion, pose_tree, axis_0, axis_1, class_type, save_path, interval=300):
    matplotlib.use('Agg')

    fig = plt.figure()
    plt.title(class_type)
    # ax = fig.add_subplot(111, projection='3d')
    data = np.array(motion, dtype=float)
    frame_number = data.shape[0]
    # dim (frame, joints, xyz)
    print(data.shape)

    def update(index):
        plt.clf()
        plt.xlim(-0.7, 0.7)
        plt.ylim(-0.7, 0.7)
        plt.scatter(data[index, :, axis_0], data[index, :, axis_1], color='b', marker='h', s=15)
        for idx1, idx2 in pose_tree:
            plt.plot([data[index, idx1, axis_0], data[index, idx2, axis_0]],
                    [data[index, idx1, axis_1], data[index, idx2, axis_1]], color='r', linewidth=2.0)

    ani = FuncAnimation(fig, update, frames=frame_number, interval=interval, repeat=False, repeat_delay=200)
    # update(1)
    # plt.show()
    # Writer = writers['ffmpeg']
    # writer = Writer(fps=15, metadata={})
    ani.save(save_path, writer='pillow')
    plt.close()