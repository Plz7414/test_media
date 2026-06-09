import cv2
import mediapipe as mp
import numpy as np

# 1. 使用新版 Tasks API 定義
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# 2. 下載官方模型權重檔 (新版必須手動載入模型檔)
# 請手動點擊此連結下載，並放到跟這隻程式相同的資料夾下：
# https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_boundary/float16/1/hand_boundary.task
# 或者下載手部關鍵點模型（推薦）：
# https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task

model_path = 'hand_landmarker.task'

# 3. 設定偵測參數
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,  # 指定為影片模式
    num_hands=2  # 最多兩隻手
)

# 4. 讀取影片
video_path = 'your_video.mp4'  # 👈 記得改成你實際的腕力影片檔名！
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"錯誤：無法開啟影片 {video_path}")
    exit()

# 建立偵測器
with HandLandmarker.create_from_options(options) as landmarker:
    print("【成功】新版 MediaPipe 初始化成功！開始播放影片...")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # 取得目前影片的時間戳記 (毫秒)，新版 API 規定必須傳入時間戳記
        frame_timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))

        # 轉換成 MediaPipe 專用的 Image 物件
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

        # 執行偵測
        detection_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

        # 5. 繪製骨架 (因為新版沒有 drawing_utils，我們用簡單的 OpenCV 點位自己畫)
        if detection_result.hand_landmarks:
            h, w, _ = frame.shape
            for hand_landmarks in detection_result.hand_landmarks:
                for landmark in hand_landmarks:
                    # 將比例座標 (0~1) 轉換為影像實際像素座標
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)  # 畫綠色圓點

        cv2.imshow('New MediaPipe Tasks API Test', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()