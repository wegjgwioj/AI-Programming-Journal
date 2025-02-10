import cv2
import numpy as np

# 定义改进帧差法的函数
def improved_frame_difference(prev_frame, curr_frame, next_frame, threshold=25):
    diff_frames1 = cv2.absdiff(next_frame, curr_frame)
    diff_frames2 = cv2.absdiff(curr_frame, prev_frame)
    diff = cv2.bitwise_and(diff_frames1, diff_frames2)
    _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    return thresh

# 定义轨迹滤波的函数
def trajectory_filtering(detections, max_distance=50):
    filtered_detections = []
    for i, det in enumerate(detections):
        if i == 0:
            filtered_detections.append(det)
        else:
            prev_det = filtered_detections[-1]
            distance = np.linalg.norm(np.array(det) - np.array(prev_det))
            if distance < max_distance:
                filtered_detections.append(det)
    return filtered_detections

# 在视频流中应用改进帧差法和轨迹滤波进行目标检测
def detect_objects(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, prev_frame = cap.read()
    ret, curr_frame = cap.read()
    detections = []

    while cap.isOpened():
        ret, next_frame = cap.read()
        if not ret:
            break

        gray_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        gray_curr = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        gray_next = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

        fg_mask = improved_frame_difference(gray_prev, gray_curr, gray_next)
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        frame_detections = []
        for contour in contours:
            if cv2.contourArea(contour) > 500:
                x, y, w, h = cv2.boundingRect(contour)
                frame_detections.append((x, y, w, h))
                cv2.rectangle(next_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        filtered_detections = trajectory_filtering(frame_detections)
        detections.extend(filtered_detections)

        cv2.imshow('Frame', next_frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

        prev_frame = curr_frame
        curr_frame = next_frame

    cap.release()
    cv2.destroyAllWindows()
    return detections

# 计算查准率、查全率和 FPS
def evaluate_performance(detections, ground_truths):
    tp = len([det for det in detections if det in ground_truths])
    fp = len(detections) - tp
    fn = len(ground_truths) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    return precision, recall

if __name__ == "__main__":
    video_path = 'path_to_your_video.mp4'
    ground_truths = []  # Load or define your ground truth data here
    detections = detect_objects(video_path)
    precision, recall = evaluate_performance(detections, ground_truths)
    print(f'Precision: {precision}, Recall: {recall}')