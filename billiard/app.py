import streamlit as st
import os
import sys
import subprocess
import time

# ==========================================
# 🚨 라이브러리 자동 설치 (MoviePy)
# ==========================================
try:
    from moviepy import ImageSequenceClip
except ImportError:
    try:
        from moviepy.editor import ImageSequenceClip
    except ImportError:
        st.warning("⚠️ 필요한 라이브러리(moviepy) 설치 중... 잠시만 기다려주세요.")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "moviepy"])
            st.success("✅ 설치 완료! 자동으로 새로고침됩니다.")
            time.sleep(2)
            st.rerun()
        except Exception as e:
            st.error(f"❌ 설치 실패: {e}")
            st.stop()

import cv2
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from ultralytics import YOLO
import tempfile

# ==========================================
# ⚙️ 설정 (Rich Feature 전용)
# ==========================================
st.set_page_config(page_title="AI 3-Cushion Referee", page_icon="🎱", layout="wide")

MODEL_PATH = "best_model1_lstm.pth" 
YOLO_PATH = "best.pt"           
TABLE_W, TABLE_H = 1280.0, 720.0        
SEQ_LENGTH = 60                         
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==========================================
# 🧠 Rich 모델 클래스 정의 (입력: 6채널)
# ==========================================
class RichLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        # 🚨 입력 사이즈 6 (x, y, dx, dy, speed, angle)
        self.lstm = nn.LSTM(6, 64, 2, batch_first=True)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.relu(out[:, -1, :])
        return self.sigmoid(self.fc(out))

@st.cache_resource
def load_models():
    model = RichLSTM()
    try:
        state_dict = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
    except Exception as e:
        st.error(f"모델 로드 실패: {e}")
        return None, None
        
    yolo = YOLO(YOLO_PATH)
    return model, yolo

# ==========================================
# 🛠️ 데이터 전처리 (수구 자동 감지 + 6특징)
# ==========================================
# def process_data_rich(df):
#     # 1. 공 색깔별로 분리
#     white = df[df['cls'] == 1].copy()
#     yellow = df[df['cls'] == 2].copy()
    
#     # 속도 계산 함수
#     def get_max_speed(ball_df):
#         if len(ball_df) < 5: return 0
#         ball_df = ball_df.sort_values('frame')
#         dx = ball_df['x'].diff().fillna(0)
#         dy = ball_df['y'].diff().fillna(0)
#         return np.sqrt(dx**2 + dy**2).max()

#     w_speed = get_max_speed(white)
#     y_speed = get_max_speed(yellow)
    
#     # 🚨 [수구 자동 감지] 더 빠른 공을 수구로 선택
#     if w_speed >= y_speed:
#         target, ball_color = white, "White"
#     else:
#         target, ball_color = yellow, "Yellow"

#     if len(target) < 10: return None, None, None

#     # 2. 보간 및 스무딩
#     target = target.sort_values('conf', ascending=False).drop_duplicates('frame').sort_values('frame')
#     target['x'] = target['x'].rolling(window=5, min_periods=1).mean()
#     target['y'] = target['y'].rolling(window=5, min_periods=1).mean()
    
#     min_f, max_f = target['frame'].min(), target['frame'].max()
#     full_range = range(int(min_f), int(max_f) + 1)
#     target = target.set_index('frame').reindex(full_range)
#     target[['x', 'y']] = target[['x', 'y']].interpolate(method='linear')
#     target = target.dropna().reset_index()

#     points_dict = {int(r['frame']): (int(r['x']), int(r['y'])) for _, r in target.iterrows()}

#     # 3. 6가지 특징 계산 (Rich Feature)
#     x_norm = target['x'].values / TABLE_W
#     y_norm = target['y'].values / TABLE_H
    
#     dx = np.diff(x_norm, prepend=x_norm[0])
#     dy = np.diff(y_norm, prepend=y_norm[0])
#     speed = np.sqrt(dx**2 + dy**2)
#     angle = np.arctan2(dy, dx) / np.pi 

#     # 타격 시점 기준 자르기
#     peak_idx = np.argmax(speed)
#     if speed[peak_idx] < 0.005: return None, None, None

#     start_idx = max(0, peak_idx - 5)
#     end_idx = start_idx + SEQ_LENGTH
    
#     features = np.stack([x_norm, y_norm, dx, dy, speed, angle], axis=1)
#     features = features[start_idx : end_idx]

#     # 패딩
#     if len(features) < SEQ_LENGTH:
#         padding = np.zeros((SEQ_LENGTH - len(features), 6))
#         features = np.vstack([features, padding])
        
#     # 정규화
#     mean = features.mean(axis=0)
#     std = features.std(axis=0) + 1e-6
#     features = (features - mean) / std
    
#     return features, points_dict, ball_color

# ==========================================
# 🛠️ 데이터 전처리 (수구 자동 감지 개선판)
# ==========================================
def process_data_rich(df):
    # 1. 공 색깔별로 분리 및 정렬
    white = df[df['cls'] == 1].sort_values('frame')
    yellow = df[df['cls'] == 2].sort_values('frame')
    
    # 🚨 [핵심 수정] 수구 판별 로직 변경 (속도 -> 움직임 시작 타이밍)
    def get_movement_start_frame(ball_df, threshold=2.0):
        if len(ball_df) < 5: return 99999 # 데이터 없으면 아주 늦은 값 반환
        
        # 노이즈 제거를 위한 스무딩 (좌표 튐 방지)
        ball_df['x_smooth'] = ball_df['x'].rolling(window=3, min_periods=1).mean()
        ball_df['y_smooth'] = ball_df['y'].rolling(window=3, min_periods=1).mean()
        
        # 속도 계산
        dx = ball_df['x_smooth'].diff().fillna(0)
        dy = ball_df['y_smooth'].diff().fillna(0)
        speed = np.sqrt(dx**2 + dy**2)
        
        # 특정 속도(threshold)를 넘는 첫 번째 프레임 찾기
        moving_frames = ball_df[speed > threshold]['frame']
        if len(moving_frames) > 0:
            return moving_frames.min()
        else:
            return 99999 # 움직임이 거의 없으면 무시

    # 각각 언제 처음 움직였는지 체크
    w_start = get_movement_start_frame(white)
    y_start = get_movement_start_frame(yellow)
    
    # 더 먼저 움직인 공을 수구로 선정
    # (둘 다 안 움직였거나 비슷하면 기존대로 데이터 많은 쪽 or 흰색 우선)
    if w_start < y_start:
        target, ball_color = white, "White"
    elif y_start < w_start:
        target, ball_color = yellow, "Yellow"
    else:
        # 타이밍이 감지 안되면 데이터 길이로 판단
        if len(white) >= len(yellow):
            target, ball_color = white, "White"
        else:
            target, ball_color = yellow, "Yellow"

    if len(target) < 10: return None, None, None

    # 2. 보간 및 스무딩 (선택된 수구에 대해 정밀 작업)
    target = target.sort_values('conf', ascending=False).drop_duplicates('frame').sort_values('frame')
    target['x'] = target['x'].rolling(window=5, min_periods=1).mean()
    target['y'] = target['y'].rolling(window=5, min_periods=1).mean()
    
    min_f, max_f = target['frame'].min(), target['frame'].max()
    full_range = range(int(min_f), int(max_f) + 1)
    target = target.set_index('frame').reindex(full_range)
    target[['x', 'y']] = target[['x', 'y']].interpolate(method='linear')
    target = target.dropna().reset_index()

    points_dict = {int(r['frame']): (int(r['x']), int(r['y'])) for _, r in target.iterrows()}

    # 3. 6가지 특징 계산 (Rich Feature)
    x_norm = target['x'].values / TABLE_W
    y_norm = target['y'].values / TABLE_H
    
    dx = np.diff(x_norm, prepend=x_norm[0])
    dy = np.diff(y_norm, prepend=y_norm[0])
    speed = np.sqrt(dx**2 + dy**2)
    angle = np.arctan2(dy, dx) / np.pi 

    # 타격 시점 기준 자르기
    peak_idx = np.argmax(speed)
    
    # 속도가 너무 느리면(정지 영상 등) 무시
    if speed[peak_idx] < 0.005: return None, None, None

    start_idx = max(0, peak_idx - 5)
    end_idx = start_idx + SEQ_LENGTH
    
    features = np.stack([x_norm, y_norm, dx, dy, speed, angle], axis=1)
    features = features[start_idx : end_idx]

    # 패딩
    if len(features) < SEQ_LENGTH:
        padding = np.zeros((SEQ_LENGTH - len(features), 6))
        features = np.vstack([features, padding])
        
    # 정규화
    mean = features.mean(axis=0)
    std = features.std(axis=0) + 1e-6
    features = (features - mean) / std
    
    return features, points_dict, ball_color


# ==========================================
# 🎬 비디오 처리 메인 로직
# ==========================================
def process_video(uploaded_file, ai_referee, yolo_model):
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') 
    tfile.write(uploaded_file.read())
    video_path = tfile.name
    tfile.close()

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    frame_list = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_list.append(frame)
    cap.release()
    
    # YOLO 인식 (1번 흰공, 2번 노란공 모두 수집)
    video_data = []
    for i, frame in enumerate(frame_list):
        results = yolo_model.predict(frame, conf=0.05, verbose=False)
        for r in results:
            for box in r.boxes:
                # 🚨 중요: 1(White), 2(Yellow) 모두 리스트에 담음
                if int(box.cls[0]) in [1, 2]: 
                    x, y, w, h = box.xywh[0].cpu().numpy()
                    video_data.append([i, int(box.cls[0]), x, y, float(box.conf[0])])

    if not video_data: return None, None, None, "공을 찾지 못했습니다."

    df = pd.DataFrame(video_data, columns=['frame', 'cls', 'x', 'y', 'conf'])
    
    # Rich 전처리 함수 호출 (여기서 수구 결정됨)
    input_seq, points_dict, ball_color = process_data_rich(df)
    
    if input_seq is None: return None, None, None, "유효한 궤적이 없습니다."

    # AI 판독
    input_tensor = torch.FloatTensor(input_seq).unsqueeze(0).to(device)
    with torch.no_grad():
        score = ai_referee(input_tensor).item()
    
    result_text = "SUCCESS" if score >= 0.5 else "FAIL"
    color_res = (0, 255, 0) if score >= 0.5 else (0, 0, 255)
    
    # MoviePy용 프레임 생성
    output_frames = []
    draw_pts = []
    
    for i, frame in enumerate(frame_list):
        if i in points_dict: draw_pts.append(points_dict[i])
        
        # 그림 그리기 (OpenCV BGR)
        temp_frame = frame.copy()
        
        # 텍스트: 결과 + 수구 색깔 표시

        cv2.putText(temp_frame, f"Cue Ball: {ball_color}", 
                   (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        if len(draw_pts) > 1:
            pts = np.array(draw_pts, np.int32).reshape((-1, 1, 2))
            cv2.polylines(temp_frame, [pts], False, color_res, 3)
            
        # RGB 변환 (MoviePy용)
        frame_rgb = cv2.cvtColor(temp_frame, cv2.COLOR_BGR2RGB)
        output_frames.append(frame_rgb)
    
    # 영상 저장
    timestamp = int(time.time())
    output_filename = f"result_{timestamp}.mp4"
    output_path = os.path.join(os.getcwd(), output_filename)
    
    clip = ImageSequenceClip(output_frames, fps=fps)
    clip.write_videofile(output_path, codec='libx264', audio=False, logger=None, preset='ultrafast')
    
    if os.path.exists(video_path):
        try: os.remove(video_path)
        except: pass
        
    return output_path, result_text, score, None

# ==========================================
# 🖥️ UI
# ==========================================
st.title("3쿠션 판정(예측) 시스템")

ai_model, yolo_model = load_models()

uploaded_file = st.file_uploader("분석할 영상을 업로드하세요", type=["mp4", "avi"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("원본 영상")
        st.video(uploaded_file)
        
    with col2:
        st.subheader("판독 결과")
        if st.button("판독 시작"):
            with st.spinner('분석 및 수구 감지 중...'):
                res_path, res_text, score, err = process_video(uploaded_file, ai_model, yolo_model)
                
                if err:
                    st.error(f"오류: {err}")
                elif res_path and os.path.exists(res_path):
                    if res_text == "SUCCESS":
                        st.success(f"판독 결과: {res_text} ({score*100:.1f}%)")
                    else:
                        st.error(f"판독 결과: {res_text} ({score*100:.1f}%)")
                    
                    # 바이너리 읽기 및 다운로드 버튼
                    with open(res_path, 'rb') as f:
                        video_bytes = f.read()
                    
                    if len(video_bytes) > 0:
                        st.video(video_bytes, format="video/mp4")
                        st.download_button(
                            label="결과 영상 다운로드",
                            data=video_bytes,
                            file_name=os.path.basename(res_path),
                            mime="video/mp4"
                        )
                    else:
                        st.error("생성된 파일이 비어있습니다.")
                else:
                    st.error("영상 생성 실패")