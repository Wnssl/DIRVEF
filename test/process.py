import torch
from skimage import transform
import numpy as np
import cv2


def read_video_to_array(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        return None  # 没有读取到帧

    # 转成numpy数组，shape形如 (帧数, 高, 宽, 通道数)
    video_array = np.array(frames)
    return video_array

def get_preprocessed_tensor(avi_file_path, orientation="Stanford"):

    gray_frames = read_video_to_array(avi_file_path)[:, :, :, 0]
    # print(gray_frames.shape)

    if orientation == "Stanford":
        for i, frame in enumerate(gray_frames):
            gray_frames[i] = cv2.flip(frame, 1)

    shape_of_frames = gray_frames.shape
    changes = np.zeros((shape_of_frames[1], shape_of_frames[2]))
    changes_frequency = np.zeros((shape_of_frames[1], shape_of_frames[2]))
    binary_mask = np.zeros((shape_of_frames[1], shape_of_frames[2]))
    cropped_frames = []

    for i in range(len(gray_frames) - 1):
        diff = abs(gray_frames[i] - gray_frames[i + 1])
        changes += diff
        nonzero = np.nonzero(diff)
        changes_frequency[nonzero[0], nonzero[1]] += 1
    max_of_changes = np.amax(changes)
    min_of_changes = np.amin(changes)

    for r in range(len(changes)):
        for p in range(len(changes[r])):
            if int(changes_frequency[r][p]) < 10:
                changes[r][p] = 0
            else:
                changes[r][p] = int(255 * ((changes[r][p] - min_of_changes) / (max_of_changes - min_of_changes)))

    nonzero_values_for_binary_mask = np.nonzero(changes)

    binary_mask[nonzero_values_for_binary_mask[0], nonzero_values_for_binary_mask[1]] += 1
    kernel = np.ones((5, 5), np.int32)
    erosion_on_binary_mask = cv2.erode(binary_mask, kernel, iterations=1)
    # image_show(erosion_on_binary_mask)
    binary_mask_after_erosion = np.where(erosion_on_binary_mask, binary_mask, 0)
    # image_show(binary_mask_after_erosion)
    nonzero_values_after_erosion = np.nonzero(binary_mask_after_erosion)
    binary_mask_coordinates = np.array([nonzero_values_after_erosion[0], nonzero_values_after_erosion[1]]).T

    cropped_mask = binary_mask_after_erosion[
                   np.min(binary_mask_coordinates[:, 0]):np.max(binary_mask_coordinates[:, 0]),
                   np.min(binary_mask_coordinates[:, 1]):np.max(binary_mask_coordinates[:, 1])]

    for row in cropped_mask:
        ids = [i for i, x in enumerate(row) if x == 1]
        if len(ids) < 2:
            continue
        row[ids[0]:ids[-1]] = 1

    for i in range(len(gray_frames)):
        masked_image = np.where(erosion_on_binary_mask, gray_frames[i], 0)

        cropped_image = masked_image[np.min(binary_mask_coordinates[:, 0]):np.max(binary_mask_coordinates[:, 0]),
                        np.min(binary_mask_coordinates[:, 1]):np.max(binary_mask_coordinates[:, 1])]
        cropped_frames.append(cropped_image)

    resized_frames = []
    for frame in cropped_frames:
        resized_frame = transform.resize(frame, (224, 224))
        resized_frames.append(resized_frame)
    resized_frames = np.asarray(resized_frames)
    # resized_binary_mask = transform.resize(cropped_mask, (224, 224))

    frames_3ch = []
    for frame in resized_frames:
        new_frame = np.zeros((np.array(frame).shape[0], np.array(frame).shape[1], 3))
        new_frame[:, :, 0] = frame
        new_frame[:, :, 1] = frame
        new_frame[:, :, 2] = frame
        frames_3ch.append(new_frame)
    frames_tensor = np.array(frames_3ch)
    # frames_tensor = frames_tensor.transpose((0, 3, 1, 2))
    # binary_mask_tensor = np.array(resized_binary_mask)
    # frames_tensor = torch.from_numpy(frames_tensor)
    # binary_mask_tensor = torch.from_numpy(binary_mask_tensor)

    x = frames_tensor

    x_min = x.min()
    x_max = x.max()

    x_norm = (x - x_min) / (x_max - x_min)  # 归一化到0-1
    x_255 = (x_norm * 255).astype(np.uint8)

    return x_255

