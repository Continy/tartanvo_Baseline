import pypose as pp
import numpy as np
import pandas
import torch
import yaml
import cv2
import os

from os import listdir
from os.path import isdir, isfile
from torch.utils.data import Dataset

from .transformation import pos_quats2SEs, pose2motion, SEs2ses
from .utils import make_intrinsics_layer


def sync_data(ts_src, ts_tar):
    res = []
    j = 0
    for t in ts_tar:
        while j + 1 < len(ts_src) and abs(ts_src[j + 1] - t) <= abs(ts_src[j] -
                                                                    t):
            j += 1
        res.append(j)
    # for i in range(len(res)-1):
    #     if res[i+1] - res[i] <= 0:
    #         print('sync_data error', i, ts_tar[i:i+2], ts_src[max(0,res[i]-5):min(len(ts_src), res[i]+5)])
    return np.array(res)


def intrinsic2matrix(intrinsic):
    fx, fy, cx, cy = intrinsic
    return np.array([fx, 0, cx, 0, fy, cy, 0, 0, 1],
                    dtype=np.float32).reshape(3, 3)


def matrix2intrinsic(m):
    return np.array([m[0, 0], m[1, 1], m[0, 2], m[1, 2]], dtype=np.float32)


def stereo_rectify(left_intrinsic, left_distortion, right_intrinsic,
                   right_distortion, width, height, right2left_pose):
    left_K = intrinsic2matrix(left_intrinsic).astype(np.float64)
    right_K = intrinsic2matrix(right_intrinsic).astype(np.float64)
    left_distortion = left_distortion.astype(np.float64)
    right_distortion = right_distortion.astype(np.float64)
    R = right2left_pose.Inv().rotation().matrix().numpy().astype(np.float64)
    T = right2left_pose.Inv().translation().numpy().astype(np.float64)

    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(left_K,
                                                      left_distortion,
                                                      right_K,
                                                      right_distortion,
                                                      (width, height),
                                                      R,
                                                      T,
                                                      alpha=0)

    left_map = cv2.initUndistortRectifyMap(left_K, left_distortion, R1, P1,
                                           (width, height), cv2.CV_32FC1)
    right_map = cv2.initUndistortRectifyMap(right_K, right_distortion, R2, P2,
                                            (width, height), cv2.CV_32FC1)

    left_intrinsic_new = matrix2intrinsic(P1)
    right_intrinsic_new = matrix2intrinsic(P2)
    right2left_pose_new = pp.SE3([-P2[0, 3] / P2[0, 0], 0, 0, 0, 0, 0,
                                  1]).to(torch.float32)

    return left_intrinsic_new, right_intrinsic_new, right2left_pose_new, left_map, right_map


class TartanAirTrajFolderLoader:

    def __init__(self, datadir):

        ############################## load images ######################################################################
        imgfolder = datadir + '/image_left'
        files = listdir(imgfolder)
        self.rgbfiles = [(imgfolder + '/' + ff) for ff in files
                         if (ff.endswith('.png') or ff.endswith('.jpg'))]
        self.rgbfiles.sort()
        self.rgb_dts = np.ones(len(self.rgbfiles), dtype=np.float32) * 0.1
        self.rgb_ts = np.array([i for i in range(len(self.rgbfiles))],
                               dtype=np.float64) * 0.1

        ############################## load stereo right images ######################################################################
        if isdir(datadir + '/image_right'):
            imgfolder = datadir + '/image_right'
            files = listdir(imgfolder)
            self.rgbfiles_right = [
                (imgfolder + '/' + ff) for ff in files
                if (ff.endswith('.png') or ff.endswith('.jpg'))
            ]
            self.rgbfiles_right.sort()
        else:
            self.rgbfiles_right = None

        ############################## load flow ######################################################################
        if isdir(datadir + '/flow'):
            imgfolder = datadir + '/flow'
            files = listdir(imgfolder)
            self.flowfiles = [(imgfolder + '/' + ff) for ff in files
                              if ff.endswith('_flow.npy')]
            self.flowfiles.sort()
        else:
            self.flowfiles = None

        ############################## load depth ######################################################################
        if isdir(datadir + '/depth_left'):
            imgfolder = datadir + '/depth_left'
            files = listdir(imgfolder)
            self.depthfiles = [(imgfolder + '/' + ff) for ff in files
                               if ff.endswith('_depth.npy')]
            self.depthfiles.sort()
        else:
            self.depthfiles = None

        ############################## load calibrations ######################################################################
        self.intrinsic = np.array([320.0, 320.0, 320.0, 240.0],
                                  dtype=np.float32)
        self.intrinsic_right = np.array([320.0, 320.0, 320.0, 240.0],
                                        dtype=np.float32)
        self.right2left_pose = pp.SE3([0, 0.25, 0, 0, 0, 0,
                                       1]).to(dtype=torch.float32)
        # self.right2left_pose = np.array([0, 0.25, 0,   0, 0, 0, 1], dtype=np.float32)
        self.require_undistort = False

        ############################## load gt poses ######################################################################
        posefile = datadir + '/pose_left.txt'
        self.poses = np.loadtxt(posefile).astype(np.float32)
        self.vels = None

        self.has_imu = False


class TartanAirV2TrajFolderLoader:

    def __init__(self, datadir):

        ############################## load images ######################################################################
        imgfolder = datadir + '/image_lcam_front'
        self.has_imu = False
        files = listdir(imgfolder)
        self.rgbfiles = [(imgfolder + '/' + ff) for ff in files
                         if (ff.endswith('.png') or ff.endswith('.jpg'))]
        self.rgbfiles.sort()
        self.rgb_dts = np.ones(len(self.rgbfiles), dtype=np.float32) * 0.1
        self.rgb_ts = np.array([i for i in range(len(self.rgbfiles))],
                               dtype=np.float64) * 0.1

        ############################## load stereo right images ######################################################################
        if isdir(datadir + '/image_rcam_front'):
            imgfolder = datadir + '/image_rcam_front'
            files = listdir(imgfolder)
            self.rgbfiles_right = [
                (imgfolder + '/' + ff) for ff in files
                if (ff.endswith('.png') or ff.endswith('.jpg'))
            ]
            self.rgbfiles_right.sort()
        else:
            self.rgbfiles_right = None

        ############################## load flow ######################################################################
        if isdir(datadir + '/flow'):
            imgfolder = datadir + '/flow'
            files = listdir(imgfolder)
            self.flowfiles = [(imgfolder + '/' + ff) for ff in files
                              if ff.endswith('_flow.npy')]
            self.flowfiles.sort()
        else:
            self.flowfiles = None

        ############################## load depth ######################################################################
        if isdir(datadir + '/depth_lcam_front'):
            imgfolder = datadir + '/depth_lcam_front'
            files = listdir(imgfolder)
            self.depthfiles = [(imgfolder + '/' + ff) for ff in files
                               if ff.endswith('_depth.npy')]
            self.depthfiles.sort()
        else:
            self.depthfiles = None

        ############################## load calibrations ######################################################################
        self.intrinsic = np.array([320.0, 320.0, 320.0, 320.0],
                                  dtype=np.float32)
        self.intrinsic_right = np.array([320.0, 320.0, 320.0, 320.0],
                                        dtype=np.float32)
        self.right2left_pose = pp.SE3([0, 0.25, 0, 0, 0, 0,
                                       1]).to(dtype=torch.float32)
        # self.right2left_pose = np.array([0, 0.25, 0,   0, 0, 0, 1], dtype=np.float32)
        self.require_undistort = False

        ############################## load gt poses ######################################################################
        posefile = datadir + '/pose_lcam_front.txt'
        self.poses = np.loadtxt(posefile).astype(np.float32)
        self.vels = None


class EuRoCTrajFolderLoader:

    def __init__(self, datadir):
        all_timestamps = []

        ############################## load images ######################################################################
        df = pandas.read_csv(datadir + '/cam0/data.csv')
        timestamps_left = df.values[:, 0].astype(int) // int(1e6)
        all_timestamps.append(timestamps_left)
        self.rgbfiles = datadir + '/cam0/data/' + df.values[:, 1]

        ############################## load stereo right images ######################################################################
        if isfile(datadir + '/cam1/data.csv'):
            df = pandas.read_csv(datadir + '/cam1/data.csv')
            timestamps_right = df.values[:, 0].astype(int) // int(1e6)
            all_timestamps.append(timestamps_right)
            self.rgbfiles_right = datadir + '/cam1/data/' + df.values[:, 1]
        else:
            self.rgbfiles_right = None

        ############################## load calibrations ######################################################################
        with open(datadir + '/cam0/sensor.yaml') as f:
            res = yaml.load(f.read(), Loader=yaml.FullLoader)
            self.intrinsic = np.array(res['intrinsics'], dtype=np.float32)
            distortion = np.array(res['distortion_coefficients'],
                                  dtype=np.float32)
            T_BL = np.array(res['T_BS']['data'],
                            dtype=np.float32).reshape(4, 4)

        if self.rgbfiles_right is not None:
            with open(datadir + '/cam1/sensor.yaml') as f:
                res = yaml.load(f.read(), Loader=yaml.FullLoader)
                self.intrinsic_right = np.array(res['intrinsics'],
                                                dtype=np.float32)
                distortion_right = np.array(res['distortion_coefficients'],
                                            dtype=np.float32)
                T_BR = np.array(res['T_BS']['data'],
                                dtype=np.float32).reshape(4, 4)

        if self.rgbfiles_right is not None:
            T_LR = np.matmul(np.linalg.inv(T_BL), T_BR)
            self.right2left_pose = pp.from_matrix(
                torch.tensor(T_LR), ltype=pp.SE3_type).to(dtype=torch.float32)

            self.require_undistort = True
            img = cv2.imread(self.rgbfiles_right[0])
            h, w = img.shape[:2]
            self.intrinsic, self.intrinsic_right, self.right2left_pose, self.imgmap, self.imgmap_right = stereo_rectify(
                self.intrinsic, distortion, self.intrinsic_right,
                distortion_right, w, h, self.right2left_pose)
        else:
            self.require_undistort = False

        ############################## load gt poses ######################################################################
        df = pandas.read_csv(datadir + '/state_groundtruth_estimate0/data.csv')
        timestamps_pose = df.values[:, 0].astype(int) // int(1e6)
        all_timestamps.append(timestamps_pose)
        self.poses = df.values[:, (1, 2, 3, 5, 6, 7, 4)].astype(np.float32)
        self.vels = df.values[:, 8:11].astype(np.float32)
        accel_bias = df.values[:, 14:17].astype(np.float32)
        gyro_bias = df.values[:, 11:14].astype(np.float32)

        ############################## align timestamps ######################################################################
        timestamps = set(all_timestamps[0])
        for i in range(1, len(all_timestamps)):
            timestamps = timestamps.intersection(set(all_timestamps[i]))
        self.rgbfiles = self.rgbfiles[[
            i for i, t in enumerate(timestamps_left) if t in timestamps
        ]]
        if self.rgbfiles_right is not None:
            self.rgbfiles_right = self.rgbfiles_right[[
                i for i, t in enumerate(timestamps_right) if t in timestamps
            ]]
        self.poses = self.poses[[
            i for i, t in enumerate(timestamps_pose) if t in timestamps
        ]]
        self.vels = self.vels[[
            i for i, t in enumerate(timestamps_pose) if t in timestamps
        ]]
        timestamps = np.array(list(timestamps))
        timestamps.sort()
        self.rgb_dts = np.diff(timestamps).astype(np.float32) * 1e-3
        self.rgb_ts = np.array(timestamps).astype(np.float64) * 1e-3
        self.has_imu = False


class TrajFolderDatasetBase(Dataset):

    def __init__(self,
                 datadir,
                 datatype,
                 transform=None,
                 start_frame=0,
                 end_frame=-1,
                 loader=None):
        if loader is None:
            if datatype == 'v1':
                loader = TartanAirTrajFolderLoader(datadir)
            elif datatype == 'v2':
                loader = TartanAirV2TrajFolderLoader(datadir)
            elif datatype == 'euroc':
                loader = EuRoCTrajFolderLoader(datadir)

        if end_frame <= 0:
            end_frame += len(loader.rgbfiles)

        self.datadir = datadir
        self.datatype = datatype
        self.transform = transform
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.loader = loader

        self.rgbfiles = loader.rgbfiles[start_frame:end_frame]
        self.rgb_dts = loader.rgb_dts[start_frame:end_frame - 1]
        self.rgb_ts = loader.rgb_ts[start_frame:end_frame]
        self.num_img = len(self.rgbfiles)

        try:
            self.rgbfiles_right = loader.rgbfiles_right[start_frame:end_frame]
        except:
            self.rgbfiles_right = None

        try:
            self.flowfiles = loader.flowfiles[start_frame:end_frame - 1]
        except:
            self.flowfiles = None

        try:
            self.depthfiles = loader.depthfiles[start_frame:end_frame]
        except:
            self.depthfiles = None

        self.intrinsic = loader.intrinsic
        try:
            self.intrinsic_right = loader.intrinsic_right
            self.right2left_pose = loader.right2left_pose
        except:
            pass

        self.poses = loader.poses[start_frame:end_frame]

        try:
            self.vels = loader.vels[start_frame:end_frame]
        except:
            self.vels = None

        if loader.require_undistort:
            self.imgmap = loader.imgmap
            try:
                self.imgmap_right = loader.imgmap_right
            except:
                pass
            self.require_undistort = True
        else:
            self.require_undistort = False

        self.links = None
        self.num_link = 0

        del loader

class KITTITrajFolderLoader:

    def __init__(self, datadir):

        ############################## load images ######################################################################
        imgfolder = datadir + '/image_2'
        self.has_imu = False
        files = listdir(imgfolder)
        self.rgbfiles = [(imgfolder + '/' + ff) for ff in files
                         if (ff.endswith('.png') or ff.endswith('.jpg'))]
        self.rgbfiles.sort()
        self.rgb_dts = np.ones(len(self.rgbfiles), dtype=np.float32) * 0.1
        self.rgb_ts = np.array([i for i in range(len(self.rgbfiles))],
                               dtype=np.float64) * 0.1

        ############################## load stereo right images ######################################################################
        if isdir(datadir + '/image_3'):
            imgfolder = datadir + '/image_3'
            files = listdir(imgfolder)
            self.rgbfiles_right = [
                (imgfolder + '/' + ff) for ff in files
                if (ff.endswith('.png') or ff.endswith('.jpg'))
            ]
            self.rgbfiles_right.sort()
        else:
            self.rgbfiles_right = None

        datadir_split = datadir.split('/')
        scene = datadir_split[-1]
        if scene == '00' or scene == '01' or scene == '02':
            self.intrinsic = np.array([718.856, 718.856, 607.1928, 185.2157],
                                    dtype=np.float32)
        elif scene == '03':
            self.intrinsic = np.array([721.5377, 721.5377, 609.5593, 172.854],
                                    dtype=np.float32)
        else:
            self.intrinsic = np.array([707.0912, 707.0912, 601.8873, 183.1104],
                                    dtype=np.float32)
        self.intrinsic_right = self.intrinsic
        self.right2left_pose = pp.SE3([0, 0.53715, 0, 0, 0, 0,
                                       1]).to(dtype=torch.float32)
        # self.right2left_pose = np.array([0, 0.25, 0,   0, 0, 0, 1], dtype=np.float32)
        self.require_undistort = False

        self.poses = None
        self.vels = None

class TrajFolderDataset(TrajFolderDatasetBase):

    def __init__(self,
                 datadir,
                 datatype,
                 transform=None,
                 start_frame=0,
                 end_frame=-1,
                 loader=None,
                 links=None):
        super(TrajFolderDataset, self).__init__(datadir, datatype, transform,
                                                start_frame, end_frame, loader)

        if links is None:
            self.links = [[i, i + 1] for i in range(self.num_img - 1)]
        else:
            self.links = links
        self.num_link = len(self.links)

        self.motions = self.calc_motions_by_links(self.links)

    def __getitem__(self, idx):
        return self.get_pair(self.links[idx][0], self.links[idx][1])

    def __len__(self):
        return self.num_link

    def calc_motions_by_links(self, links):
        if self.poses is None:
            return None

        SEs = pos_quats2SEs(self.poses)
        matrix = pose2motion(SEs, links=links)
        motions = SEs2ses(matrix).astype(np.float32)
        return motions

    def undistort(self, img, is_right=False):
        if not self.require_undistort:
            return img
        imgmap = self.imgmap_right if is_right else self.imgmap
        dst = cv2.remap(img, imgmap[0], imgmap[1], cv2.INTER_AREA)
        return dst

    def get_pair(self, i, j):
        res = {}

        img0 = cv2.imread(self.rgbfiles[i], cv2.IMREAD_COLOR)
        img1 = cv2.imread(self.rgbfiles[j], cv2.IMREAD_COLOR)
        img0 = self.undistort(img0)
        img1 = self.undistort(img1)
        res['img1'] = img0
        res['img2'] = img1

        h, w, _ = img0.shape
        intrinsicLayer = make_intrinsics_layer(w, h, self.intrinsic[0],
                                               self.intrinsic[1],
                                               self.intrinsic[2],
                                               self.intrinsic[3])
        res['intrinsic'] = intrinsicLayer


        if self.transform:
            res = self.transform(res)

        return res
