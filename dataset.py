import csv
import os
import random
import copy
import cv2
import numpy as np
from torchvision.transforms import transforms
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from utils import get_lds_kernel_window
from scipy.ndimage import convolve1d

from losses import *
import json


transform = transforms.Compose([
    transforms.Resize((112, 112)),
])

DEFAULT_MEAN = np.array([0.12741163, 0.1279413, 0.12912785]) * 255
DEFAULT_STD = np.array([0.19557191, 0.19562256, 0.1965878]) * 255


class EchoData(Dataset):

    def __init__(self, file_path='D:\\train_video', pad=None, frames=32, frequency=2, split_path='split.json', codebook_path='codebook.csv', split='train'
                 ):

        self.datas = []
        self.targets = []
        self.weights = []

        self.pad = pad
        self.frames = frames
        self.frequency = frequency

        self.mean = DEFAULT_MEAN
        self.std = DEFAULT_STD

        self.split = split
        self.file_path = file_path

        # self.codebook = 'codebook.csv'
        self.split_path = split_path
        self.codebook_path = codebook_path

        self.files = []
        files = os.listdir(self.file_path)

        for f in files:
            file_name = f.split('.')[0]
            self.files.append(file_name)

        with open(self.split_path, 'r+') as f:
            split_json = json.load(f)

        for k, v in split_json.items():
            if v == self.split:
                self.datas.append(k)
                self.targets.append(self.get_target(k))

    def get_target(self, f):
        reader = csv.reader(open(self.codebook_path, 'r+'))
        next(reader)

        for line in reader:
            if line[0] == f:
                return float(line[-2])


    def __getitem__(self, item):
        file = self.datas[item]
        target = self.targets[item]
        # print(target)
        path = self.file_path + '/' + file + '.avi'

        video = self.load_video(path).astype(np.float32)
        # print(video.shape)
        video = video.transpose(3, 0, 1, 2)
        video = self.sample_video(video)

        if self.pad:
            video = self.pad_video(video)

        video = self.normalize_video(video)

        video_tensor = torch.from_numpy(video)

        target = torch.tensor(target)

        video_tensor = transform(video_tensor)
        return {'filename': file,
                'video': video_tensor,
                'label': target,
        }

    def __len__(self):
        return len(self.datas)
    
    def  get_labels(self):
        return self.targets
    
    def get_filenames(self):
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
        if not self.pad:
            return video

        c, t, h, w = video.shape

        tvideo = np.zeros((c, t, h + 2 * self.pad, w + 2 * self.pad), dtype=video.dtype)
        tvideo[:, :, self.pad:- self.pad, self.pad:- self.pad] = video  # pylint: disable=E1130
        i, j = np.random.randint(0, 2 * self.pad, 2)

        video_temp = tvideo[:, :, i:(i + h), j:(j + w)]
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

        if f < frames * self.frequency:
            video = np.concatenate((video, np.zeros((c, frames * self.frequency - f, h, w), video.dtype)), axis=1)
            c, f, h, w = video.shape

        start = np.random.choice(f - (frames - 1) * self.frequency, 1)
        #print(start)
        temp = start + self.frequency * np.arange(frames)
        video = tuple(video[:, s + self.frequency * np.arange(frames), :, :] for s in start)
        video = video[0]

        return video

def rotate(tensor):

    a = random.randint(-30, 30)
    images = []
    for image in tensor:
        rotate_image = TF.rotate(image, a)
        images.append(rotate_image)
    tensor1 = torch.stack(images)

    return tensor1


def flip(tensor):
    flag = random.choice([0, 1])
    horizontal_flip = transforms.RandomHorizontalFlip(p=1)
    if flag:
        return tensor
    else:
        images = []
        for image in tensor:
            flip_image = horizontal_flip(image)
            images.append(flip_image)
        tensor1 = torch.stack(images)

    return tensor1
    
def enhance(video):
    video = rotate(video)
    return flip(video)

class dataset_Con(EchoData):
    def __getitem__(self, index):
        view1 = super().__getitem__(index)
        view2 = super().__getitem__(index)
        
        video = view2['video']
        enhance_video = enhance(video)
        
        view2['video'] = enhance_video
        
        return view1, view2
    
    def get_labels(self):
        return super().get_labels()
    
    def get_filenames(self):
        return super().get_filenames()
    
