import numpy as np

class MatrixTransformer(object):

    @staticmethod
    def rotate_along_x(matrix, theta):

        Rx = np.array([[1, 0, 0],
                       [0, np.cos(theta), np.sin(theta)],
                       [0, -np.sin(theta), np.cos(theta)]])
        return np.dot(matrix, Rx)

    @staticmethod
    def rotate_along_y(matrix, theta):

        Ry = np.array([[np.cos(theta), 0, -np.sin(theta)],
                       [0, 1, 0],
                       [np.sin(theta), 0, np.cos(theta)]])
        return np.dot(matrix, Ry)

    @staticmethod
    def rotate_along_z(matrix, theta):

        Rz = np.array([[np.cos(theta), np.sin(theta), 0],
                       [-np.sin(theta), np.cos(theta), 0],
                       [0, 0, 1]])
        return np.dot(matrix, Rz)

    @staticmethod
    def swap_yz(matrix):

        S_yz = np.array([[1, 0, 0],
                         [0, 0, 1],
                         [0, 1, 0]])
        return np.dot(matrix, S_yz)

    @staticmethod
    def swap_xz(matrix):

        S_xz = np.array([[0, 0, 1],
                         [0, 1, 0],
                         [1, 0, 0]])
        return np.dot(matrix, S_xz)

    @staticmethod
    def swap_xy(matrix):

        S_xy = np.array([[0, 1, 0],
                         [1, 0, 0],
                         [0, 0, 1]])

        return np.dot(matrix, S_xy)

    def project_3d_to_2d(cam_f, cam_c, verts):
        '''
        project 3d points to original 2d coordinate space.
        Input:
            cam: (1, 3) camera parameters (f, cx, cy) output by model.
            verts: 3d verts output by model.
            proc_param: preprocessing parameters. this is for converting points from crop (model input) to original image.
        Output:
        '''
        fx = cam_f[0]
        fy = cam_f[1]
        tx = cam_c[0]
        ty = cam_c[1]

        verts = verts.reshape(-1, 3)
        verts2d = np.zeros((verts.shape[0], 2))
        print(verts2d.shape)
        verts2d[:, 0] = fx * verts[:, 0] / verts[:, 2] + tx
        verts2d[:, 1] = fy * verts[:, 1] / verts[:, 2] + ty

        return verts2d
