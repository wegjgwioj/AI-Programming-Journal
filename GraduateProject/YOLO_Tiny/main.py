import cv2
import numpy as np
from yolov4_tiny import YOLOv4Tiny

# 定义三帧差分法函数
def frame_difference(prev_frame, curr_frame, next_frame):
    diff_frames1 = cv2.absdiff(next_frame, curr_frame)
    diff_frames2 = cv2.absdiff(curr_frame, prev_frame)
    return cv2.bitwise_and(diff_frames1, diff_frames2)

# 初始化 YOLOv4-Tiny 模型
yolo = YOLOv4Tiny()

# 打开视频文件或摄像头
cap = cv2.VideoCapture('sonar_video.mp4')

# 读取三帧
ret, frame1 = cap.read()
ret, frame2 = cap.read()
ret, frame3 = cap.read()

while ret:
    # 应用三帧差分法
    diff_frame = frame_difference(frame1, frame2, frame3)
    
    # 使用 YOLOv4-Tiny 进行目标检测
    detections = yolo.detect(diff_frame)
    
    # 计算查准率、查全率和 FPS
    # ...计算代码...
    
    # 显示结果
    cv2.imshow('Frame', frame3)
    cv2.imshow('Difference Frame', diff_frame)
    
    # 更新帧
    frame1 = frame2
    frame2 = frame3
    ret, frame3 = cap.read()
    
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
