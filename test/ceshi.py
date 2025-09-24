# import cv2
# import numpy as np
# from process import get_preprocessed_tensor
#
# video = "D:\\echo_strain\\patient001\\patient001_A4C.avi"
#
# tensor = get_preprocessed_tensor(video)
#
# # v
# #
# # cap = cv2.VideoCapture(video)
# #
# # if not cap.isOpened():
# #     print("无法打开视频文件")
# #     exit()
# #
# #
# # _, frame = cap.read()  # 读取一帧
#
# frame = tensor.numpy()
# print(frame.shape)
# print(frame)
# frame = frame.transpose(1,0,2,3)[0][0]
# print(frame.shape)
#
# cv2.imshow("1", frame)
# cv2.waitKey(0)
#
#
#     # cv2.imshow('Video', frame)  # 显示帧
# # print(frame.shape)
#
# # frame = np.array(frame)
# # print(frame)
# # print(frame.shape)
# frame = frame.reshape(1,-1).squeeze(0)
# print(frame.shape)
# import matplotlib.pyplot as plt
# from collections import Counter
#
# # 示例数组
#
# # 统计每个数字出现的次数
# count = Counter(frame)
#
# # 拆分成两个列表：元素和对应频次
# elements = list(count.keys())
# frequencies = list(count.values())
#
# # 绘制柱状图
# plt.bar(elements, frequencies, color='skyblue')
# plt.xlabel('数字')
# plt.ylabel('出现次数')
# plt.title('数组中数字出现次数统计')
# plt.show()



# from dataset import EchoData
# import cv2
#
#
# test = EchoData("D:\\train_video", split='test')
# item = test[0]
# video = item["video"]
# print(type(video))
#
# # cv2.imshow("1", video)
#
# print(video)
# print(video.shape)
#

from test_dataset import test_dataset
test = test_dataset("D:\\echo_strain")
item = test[0]