import csv
import os
import random
import copy
import cv2
import torch
from torch.utils.data import DataLoader
from  utils import *
import numpy as np
from torchvision.transforms import transforms
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from utils import get_lds_kernel_window
from scipy.ndimage import convolve1d
import pydicom
from tqdm import tqdm
from EchoNet.regressor import get_shallow_mlp_head
from EchoNet import uniformer
from process import get_preprocessed_tensor
from losses import *
import json

transform = transforms.Compose([
    transforms.Resize((112, 112)),
])

DEFAULT_MEAN = np.array([0.12741163, 0.1279413, 0.12912785]) * 255
DEFAULT_STD = np.array([0.19557191, 0.19562256, 0.1965878]) * 255


class EchoData(Dataset):
    def __init__(self, file_path='D:\\train_video', pad=None, frames=32, frequency=2, split_path='split.json',
                 codebook_path='codebook.csv', split='train'
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
        path = self.file_path + '/' + file + '.dcm'

        video = dicom2video_array(path).astype(np.float32)

        print(video.shape)
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

    def get_labels(self):
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
        # print(start)
        temp = start + self.frequency * np.arange(frames)
        video = tuple(video[:, s + self.frequency * np.arange(frames), :, :] for s in start)
        video = video[0]

        return video


def dicom2video_array(dicom_file):
    # 检查文件是否存在
    if not os.path.exists(dicom_file):
        raise FileNotFoundError(f"DICOM file not found: {dicom_file}")

    # 读取DICOM文件
    dicom_dataset = pydicom.dcmread(dicom_file, force=True)

    tensor = get_preprocessed_tensor(dicom_file)
    frames, channels, height, width = tensor.shape

    video_array = np.zeros((frames, height, width, 3), dtype=np.uint8)

    for i in range(frames):
        frame = (tensor[i, 1, :, :] * 255).clamp(0, 255).numpy().astype(np.uint8)

        frame_bgr = cv2.merge([frame, frame, frame])

        video_array[i, :, :, :] = frame_bgr

    return video_array

def make_json():
    from datetime import datetime
    import csv
    time = datetime.now()
    time = time.strftime('%Y_%m_%d_%H_%M_%S')
    json_file = 'result.json'
    result = {}
    try:
        with open(json_file, 'r+') as f:
            dic = json.load(f)
    except:
        dic = {}
    reader = csv.reader(open('codebook.csv', 'r+'))
    next(reader)
    result['predict'] = {}
    result['result'] = {}
    for line in reader:
        if line[-1] == 'validation':
            predict = {}
            predict['predict'] = None
            predict['truth'] = float(line[-2])
            result['predict'][line[0]] = predict
        else:
            pass
    result['result']['mAE'] = None
    result['result']['mSE'] = None
    result['result']['R2'] = None
    dic[time] = result
    # with open('result.json', 'w+') as f:
    #     json.dump(dic, f)
    return time, dic


if __name__ == '__main__':
    assert 1 == 0, "please change the source file path in EchoData()"
    seed_everything(42)
    test = EchoData("E:\\data\\",split='train')
    print(f"test length : {len(test)}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_loader = DataLoader(dataset=test, batch_size=1, shuffle=False, num_workers=15)
    criterion = nn.L1Loss()
    predict_total_loss = 0.0
    checkpoint_pth = 'checkpoint4.028.pth'
    model_dict = torch.load(checkpoint_pth, map_location=torch.device('cpu'))['model']
    from EchoNet.uniformer import uniformer_small
    from EchoNet.regressor import get_shallow_mlp_head

    model = uniformer_small()
    model.load_state_dict(model_dict, strict=False)
    model = model.to(device)

    regressor = get_shallow_mlp_head()
    regressor = regressor.to(device)

    regressor_dict = torch.load(checkpoint_pth,map_location=torch.device('cpu'))['regressor']
    regressor.load_state_dict(regressor_dict)
    model.eval()
    regressor.eval()
    print('testing : --------')
    time, test_result = make_json()
    result = test_result[time]['predict']
    evaluation = test_result[time]['result']

    with torch.no_grad():
        for batch_idx, item in tqdm(enumerate(test_loader)):
            data = item['video'].to(device)
            target = item['label'].to(device)
            output, features = model(data)
            outputs = regressor(features)
            # output = output.squeeze(1)
            outputs = outputs.squeeze(1)
            loss = criterion(outputs, target)
            predict_total_loss += loss.item()

            filename = item['filename'][0]
            result[filename]['predict'] = float(outputs[0])
        avg_loss_test = predict_total_loss / len(test_loader)
        print(f'test loss:{avg_loss_test}')

    predicts = []
    truths = []
    for k, v in result.items():
        predicts.append(v['predict'])
        truths.append(v['truth'])
    y_pred = np.array(predicts)
    y_true = np.array(truths)
    mAE = np.mean(np.abs(y_true - y_pred))
    mSE = np.mean((y_true - y_pred) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    evaluation['mAE'] = mAE
    evaluation['mSE'] = mSE
    evaluation['R2'] = r2
    print(evaluation)

    test_result[time]['predict'] = result
    test_result[time]['result'] = evaluation
    with open('result.json', 'w+') as f:
        json.dump(test_result, f)