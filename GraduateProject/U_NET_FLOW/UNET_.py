import cv2
import numpy as np
from unet_model import UNet

# 定义光流法函数
def optical_flow(prev_frame, curr_frame):
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return mag

# 初始化 U-Net 模型
unet = UNet()

# 打开视频文件或摄像头
cap = cv2.VideoCapture('sonar_video.mp4')

# 读取第一帧
ret, prev_frame = cap.read()

while ret:
    # 读取当前帧
    ret, curr_frame = cap.read()
    if not ret:
        break
    
    # 使用 U-Net 进行背景抑制
    mask = unet.predict(curr_frame)
    
    # 应用光流法
    flow_mag = optical_flow(prev_frame, curr_frame)
    
    # 结合 U-Net 和光流法进行目标检测
    detection = cv2.bitwise_and(mask, mask, mask=(flow_mag > 1.0).astype(np.uint8))
    
    # 计算查准率、查全率和 FPS
    # ...计算代码...
    
    # 显示结果
    cv2.imshow('Frame', curr_frame)
    cv2.imshow('Detection', detection)
    
    # 更新帧
    prev_frame = curr_frame
    
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
