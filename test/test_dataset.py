import csv
import os
import random
import copy
import cv2
import numpy as np
from torchvision.transforms import transforms
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
# from process import get_preprocessed_tensor
from losses import *
import json

#目录结构
"""
echo_strain/patient/A4C.avi

"""


# 固定的mean和std， 得到一样的预处理结果

transform = transforms.Compose([
    transforms.Resize((112, 112)),
])

DEFAULT_MEAN = np.array([0.12741163, 0.1279413, 0.12912785]) * 255
DEFAULT_STD = np.array([0.19557191, 0.19562256, 0.1965878]) * 255


class test_dataset(Dataset):
    def __init__(self, file_path='D:\\echo_strain', pad=None, frames=32, frequency=2,
                 codebook_path='FileList.csv'
                 ):
        """读取训练的数据集"""
        # 保存需要训练的数据样本 和标签
        self.datas = []
        self.targets = []
        self.weights = []

        # pad 值     下采样帧数     采样周
        self.pad = pad
        self.frames = frames
        self.frequency = frequency

        # std  mean
        self.mean = DEFAULT_MEAN
        self.std = DEFAULT_STD

        # 训练文件 路径
        self.file_path = file_path

        self.codebook_path = codebook_path
        # 对应训练文件路径下的所有文件

        patients = os.listdir(self.file_path)
        # 将读取到的file文件名单独分割出来 同时保存不包含后缀的文件名
        for patient in patients:
            file_path = os.path.join(self.file_path, patient, f"{patient}_A4C.avi")
            self.datas.append(file_path)
            self.targets.append(self.get_target(patient))

        print(self.datas)
        print(self.targets)


    def get_target(self, f):
        reader = csv.reader(open(self.codebook_path, 'r+'))
        next(reader)

        for line in reader:
            if line[0] == f:
                return float(line[-2])

    def __getitem__(self, item):
        """在此处实现对数据集的加载"""
        file = self.datas[item]
        target = self.targets[item]
        # print(target)
        # path = self.file_path + '/' + file + '.avi'

        video = self.load_video(file).astype(np.float32)
        # video = get_preprocessed_tensor(file)
        # video = float32_to_uint8(video).numpy().astype(np.float32)

        # for i in video:
        #     cv2.imshow("image", i[0])
        #     cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # print(video.shape)
        video = video.transpose(3, 0, 1, 2)
        video = self.sample_video(video)

        if self.pad:
            video = self.pad_video(video)

        video = self.normalize_video(video)

        video_tensor = torch.from_numpy(video)

        target = torch.tensor(target)

        video_tensor = transform(video_tensor)
        # 判断是否需要加权 只对训练集有效
        return {'filename': file,
                'video': video_tensor,
                'label': target,
                }

    def __len__(self):
        """返回训练集的长度"""
        return len(self.datas)

    def get_labels(self):
        # 返回全部的数据标签
        return self.targets

    def get_filenames(self):
        # 返回全部文件名
        return self.datas

    def load_video(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(path)

        capture = cv2.VideoCapture(path)

        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        video = np.zeros((count, height, width, 3), np.uint8)

        for i in range(count):
            out, frame = capture.read()
            if not out:
                raise ValueError("Problem when reading frame #{} of {}.".format(i, path))
            video[i, :, :, :] = frame

        return video

    def pad_video(self, video):
        """ 对video进行padding """
        """ 同时包含对video 的randomcrop"""
        if not self.pad:
            return video

        c, t, h, w = video.shape

        tvideo = np.zeros((c, t, h + 2 * self.pad, w + 2 * self.pad), dtype=video.dtype)
        tvideo[:, :, self.pad:- self.pad, self.pad:- self.pad] = video  # pylint: disable=E1130
        i, j = np.random.randint(0, 2 * self.pad, 2)

        video_temp = tvideo[:, :, i:(i + h), j:(j + w)]
        # print(tvideo.shape)
        # print(video_temp.shape)
        return video_temp

    def normalize_video(self, video):
        if isinstance(self.mean, (float, int)):
            video -= self.mean
        else:
            video -= self.mean.reshape(3, 1, 1, 1)

        if isinstance(self.std, (float, int)):
            video /= self.std
        else:
            video /= self.std.reshape(3, 1, 1, 1)

        return video

    def sample_video(self, video):
        c, f, h, w = video.shape
        frames = self.frames

        # 帧数不足， 进行补全
        if f < frames * self.frequency:
            video = np.concatenate((video, np.zeros((c, frames * self.frequency - f, h, w), video.dtype)), axis=1)
            c, f, h, w = video.shape

        start = np.random.choice(f - (frames - 1) * self.frequency, 1)
        # print(start)
        temp = start + self.frequency * np.arange(frames)
        video = tuple(video[:, s + self.frequency * np.arange(frames), :, :] for s in start)
        video = video[0]

        return video

def float32_to_uint8(tensor):
    min_val = tensor.min()
    max_val = tensor.max()

    if max_val == min_val:
        tensor_uint8 = torch.zeros_like(tensor, dtype=torch.uint8)
    else:
        tensor_norm = (tensor - min_val) / (max_val - min_val)
        tensor_uint8 = (tensor_norm * 255).to(torch.uint8)
    return tensor_uint8

if __name__ == '__main__':
    dataset = test_dataset()
    item = dataset[0]
    video = item['video']
    label = item['label']
    print(video.shape)

