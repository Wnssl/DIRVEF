from tqdm import tqdm
import os
from process import get_preprocessed_tensor
import cv2
path = "D:\\echo_strain"

os.mkdir("D:\\pre_test")
patients = os.listdir(path)
for p in tqdm(patients):
    file_path = os.path.join(path, p, f"{p}_A4C.avi")
    video = get_preprocessed_tensor(file_path)
    os.mkdir(os.path.join("D:\\pre_test", p))
    outpath = os.path.join("D:\\pre_test", p, f"{p}_pre.avi")

    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 使用XVID编码保存AVI视频
    fps = 30

    out = cv2.VideoWriter(outpath, fourcc, fps, (224, 224))

    for i in video:
        #
        # 如果是RGB格式，需要转换为BGR
        # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(i)

    out.release()
    print(f"{outpath}视频保存完成！")