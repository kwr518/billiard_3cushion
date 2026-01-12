import streamlit as st
import os
import sys
import subprocess
import time
import math
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
from ultralytics import YOLO
import tempfile
from sklearn.neighbors import NearestNeighbors
import ollama

# ==========================================
# 🚨 라이브러리 자동 설치
# ==========================================
try:
    from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
except ImportError:
    try:
        from moviepy.editor import ImageSequenceClip
    except ImportError:
        st.warning("⚠️ 필요한 라이브러리(moviepy) 설치 중...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "moviepy"])
        st.success("✅ 설치 완료!")
        time.sleep(1)
        st.rerun()

# ==========================================
# ⚙️ 설정
# ==========================================
st.set_page_config(page_title="AI 3-Cushion Referee", page_icon="🎱", layout="wide")

MODEL_PATH = "best_model1_lstm.pth"
YOLO_PATH = "best.pt"
TABLE_W, TABLE_H = 1280.0, 720.0
SEQ_LENGTH = 60
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 🆕 [설정] 사용할 Ollama 모델 이름
OLLAMA_MODEL = "llama3" 

# 성공 데이터 DB 로드
try:
    SUCCESS_DB = np.load("success_db.npy", allow_pickle=True)
    DB_START_POS = SUCCESS_DB[:, :2] # 검색용 시작점 (앞 2개)
    print(f"✅ 정답 DB 로드 완료: {len(SUCCESS_DB)}개 데이터")
except:
    SUCCESS_DB = None
    print("⚠️ 정답 DB(success_db.npy)가 없습니다. RAG 기능이 제한됩니다.")


# ==========================================
# 🧠 모델 클래스
# ==========================================
class RichLSTM(nn.Module):
    def __init__(self):
        super().__init__()
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
# 🆕 [함수 수정] Ollama 기반 RAG 코칭
# ==========================================
def find_best_ghost_shot(current_start_xy):
    """현재 공 위치와 가장 가까운 성공 샷 궤적 찾기 (Retrieval)"""
    if SUCCESS_DB is None: return None
    
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(DB_START_POS)
    distances, indices = nbrs.kneighbors([current_start_xy])
    
    best_idx = indices[0][0]
    best_shot_data = SUCCESS_DB[best_idx]
    
    trajectory_flat = best_shot_data[2:]
    return trajectory_flat.reshape(-1, 2)

def calculate_angle(trajectory):
    if len(trajectory) < 5: return 0
    dx = trajectory[4][0] - trajectory[0][0]
    dy = trajectory[4][1] - trajectory[0][1]
    deg = math.degrees(math.atan2(dy, dx))
    return abs(deg)

def get_rag_coaching(user_traj_norm, ghost_traj_norm):
    """
    🆕 Ollama를 사용하여 코칭 멘트 생성
    """
    user_angle = calculate_angle(user_traj_norm)
    ghost_angle = calculate_angle(ghost_traj_norm)
    angle_diff = ghost_angle - user_angle
    
    # 프롬프트 (한국어로 작성)
    prompt = f"""
    당신은 3쿠션 당구 전문 코치입니다. 다음 데이터를 분석해 짧고 명확한 조언을 한 줄로 해주세요.
    
    [상황 분석]
    - 사용자 샷 각도: {user_angle:.1f}도
    - 정답(AI) 샷 각도: {ghost_angle:.1f}도
    - 차이: 사용자가 정답보다 {abs(angle_diff):.1f}도 만큼 {"작게(얇게)" if angle_diff > 0 else "크게(두껍게)"} 쳤음.
    
    [지시사항]
    - 위 각도 차이를 언급하며, 두께나 회전을 어떻게 조절해야 득점할 수 있는지 조언하세요.
    - 예시: "두께가 너무 얇았습니다. 진입각을 약 5도 더 키우세요."
    - 반말이나 존댓말 상관없이 코치처럼 말하세요.
    """
    
    try:
        # 🆕 Ollama 호출 부분
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{'role': 'user', 'content': prompt}]
        )
        return response['message']['content']
    except Exception as e:
        return f"⚠️ Ollama 호출 오류: {e}. (터미널에서 'ollama serve'가 켜져 있는지, 모델명이 맞는지 확인하세요)"

# ==========================================
# 🛠️ 데이터 전처리
# ==========================================
def process_data_rich(df):
    white = df[df['cls'] == 1].sort_values('frame')
    yellow = df[df['cls'] == 2].sort_values('frame')
    
    def get_movement_start_frame(ball_df, threshold=2.0):
        if len(ball_df) < 5: return 99999
        ball_df['x_smooth'] = ball_df['x'].rolling(window=3, min_periods=1).mean()
        ball_df['y_smooth'] = ball_df['y'].rolling(window=3, min_periods=1).mean()
        dx = ball_df['x_smooth'].diff().fillna(0)
        dy = ball_df['y_smooth'].diff().fillna(0)
        speed = np.sqrt(dx**2 + dy**2)
        moving_frames = ball_df[speed > threshold]['frame']
        if len(moving_frames) > 0: return moving_frames.min()
        else: return 99999

    w_start = get_movement_start_frame(white)
    y_start = get_movement_start_frame(yellow)
    
    if w_start < y_start: target, ball_color = white, "White"
    elif y_start < w_start: target, ball_color = yellow, "Yellow"
    else: target, ball_color = (white, "White") if len(white) >= len(yellow) else (yellow, "Yellow")

    if len(target) < 10: return None, None, None

    target = target.sort_values('conf', ascending=False).drop_duplicates('frame').sort_values('frame')
    target['x'] = target['x'].rolling(window=5, min_periods=1).mean()
    target['y'] = target['y'].rolling(window=5, min_periods=1).mean()
    
    min_f, max_f = target['frame'].min(), target['frame'].max()
    full_range = range(int(min_f), int(max_f) + 1)
    target = target.set_index('frame').reindex(full_range)
    target[['x', 'y']] = target[['x', 'y']].interpolate(method='linear')
    target = target.dropna().reset_index()

    points_dict = {int(r['frame']): (int(r['x']), int(r['y'])) for _, r in target.iterrows()}

    x_norm = target['x'].values / TABLE_W
    y_norm = target['y'].values / TABLE_H
    
    dx = np.diff(x_norm, prepend=x_norm[0])
    dy = np.diff(y_norm, prepend=y_norm[0])
    speed = np.sqrt(dx**2 + dy**2)
    angle = np.arctan2(dy, dx) / np.pi 

    peak_idx = np.argmax(speed)
    if speed[peak_idx] < 0.005: return None, None, None

    start_idx = max(0, peak_idx - 5)
    end_idx = start_idx + SEQ_LENGTH
    
    features = np.stack([x_norm, y_norm, dx, dy, speed, angle], axis=1)
    features = features[start_idx : end_idx]

    if len(features) < SEQ_LENGTH:
        padding = np.zeros((SEQ_LENGTH - len(features), 6))
        features = np.vstack([features, padding])
        
    mean = features.mean(axis=0)
    std = features.std(axis=0) + 1e-6
    features = (features - mean) / std
    
    return features, points_dict, ball_color

# ==========================================
# 🎬 비디오 처리 메인
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
    
    video_data = []
    for i, frame in enumerate(frame_list):
        results = yolo_model.predict(frame, conf=0.05, verbose=False)
        for r in results:
            for box in r.boxes:
                if int(box.cls[0]) in [1, 2]: 
                    x, y, w, h = box.xywh[0].cpu().numpy()
                    video_data.append([i, int(box.cls[0]), x, y, float(box.conf[0])])

    if not video_data: return None, None, None, "공을 찾지 못했습니다.", None

    df = pd.DataFrame(video_data, columns=['frame', 'cls', 'x', 'y', 'conf'])
    input_seq, points_dict, ball_color = process_data_rich(df)
    
    if input_seq is None: return None, None, None, "유효한 궤적이 없습니다.", None

    input_tensor = torch.FloatTensor(input_seq).unsqueeze(0).to(device)
    with torch.no_grad():
        score = ai_referee(input_tensor).item()
    
    result_text = "SUCCESS" if score >= 0.5 else "FAIL"
    color_res = (0, 255, 0) if score >= 0.5 else (0, 0, 255)
    
    ghost_pts = []
    coaching_msg = None
    
    # 🆕 RAG + Ollama 코칭
    if result_text == "FAIL" and SUCCESS_DB is not None:
        if len(points_dict) > 0:
            start_frame_idx = min(points_dict.keys())
            start_xy_pixel = points_dict[start_frame_idx]
            start_xy_norm = [start_xy_pixel[0] / TABLE_W, start_xy_pixel[1] / TABLE_H]
            
            ghost_traj_norm = find_best_ghost_shot(start_xy_norm)
            
            if ghost_traj_norm is not None:
                # Ollama 코칭 생성
                my_traj_norm = input_seq[:, :2]
                coaching_msg = get_rag_coaching(my_traj_norm, ghost_traj_norm)
                
                # 🛠️ [핵심 수정] 정답 궤적 보정 (내 공 위치로 이동 + 0 제거)
                ghost_start_pixel_x = ghost_traj_norm[0][0] * TABLE_W
                ghost_start_pixel_y = ghost_traj_norm[0][1] * TABLE_H
                
                shift_x = start_xy_pixel[0] - ghost_start_pixel_x
                shift_y = start_xy_pixel[1] - ghost_start_pixel_y
                
                for pt in ghost_traj_norm:
                    # 패딩된 0,0 데이터는 건너뛰기 (중요!)
                    if pt[0] == 0 and pt[1] == 0: continue
                    
                    # 원래 위치로 복원 후 이동량(shift) 더하기
                    px = int((pt[0] * TABLE_W) + shift_x)
                    py = int((pt[1] * TABLE_H) + shift_y)
                    
                    # 화면 밖으로 너무 나가는 값 방지
                    px = max(0, min(int(TABLE_W), px))
                    py = max(0, min(int(TABLE_H), py))
                    
                    ghost_pts.append((px, py))

    output_frames = []
    draw_pts = []
    
    for i, frame in enumerate(frame_list):
        temp_frame = frame.copy()
        
        if i in points_dict: draw_pts.append(points_dict[i])
        
        cv2.putText(temp_frame, f"Cue Ball: {ball_color}", 
                   (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        if len(draw_pts) > 1:
            pts = np.array(draw_pts, np.int32).reshape((-1, 1, 2))
            cv2.polylines(temp_frame, [pts], False, color_res, 3)

        if result_text == "FAIL" and len(ghost_pts) > 1:
            g_pts = np.array(ghost_pts, np.int32).reshape((-1, 1, 2))
            cv2.polylines(temp_frame, [g_pts], False, (0, 255, 127), 2, cv2.LINE_AA)
            end_pt = ghost_pts[-1]
            cv2.putText(temp_frame, "AI Guide", (end_pt[0]-50, end_pt[1]-20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 127), 2)
            
        frame_rgb = cv2.cvtColor(temp_frame, cv2.COLOR_BGR2RGB)
        output_frames.append(frame_rgb)
    
    timestamp = int(time.time())
    output_filename = f"result_{timestamp}.mp4"
    output_path = os.path.join(os.getcwd(), output_filename)
    
    clip = ImageSequenceClip(output_frames, fps=fps)
    clip.write_videofile(output_path, codec='libx264', audio=False, logger=None, preset='ultrafast')
    
    if os.path.exists(video_path):
        try: os.remove(video_path)
        except: pass
        
    return output_path, result_text, score, None, coaching_msg

# ==========================================
# 🖥️ UI
# ==========================================
st.title("🎱 AI 3-Cushion Referee & Coach")

ai_model, yolo_model = load_models()

uploaded_file = st.file_uploader("분석할 영상을 업로드하세요", type=["mp4", "avi"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("원본 영상")
        st.video(uploaded_file)
        
    with col2:
        st.subheader("판독 결과")
        if st.button("판독 시작", type="primary"):
            with st.spinner('AI 심판이 영상을 분석 중입니다...'):
                res_path, res_text, score, err, coach_msg = process_video(uploaded_file, ai_model, yolo_model)
                
                if err:
                    st.error(f"오류: {err}")
                elif res_path and os.path.exists(res_path):
                    if res_text == "SUCCESS":
                        st.success(f"✅ 판독 결과: 성공 ({score*100:.1f}%)")
                    else:
                        st.error(f"❌ 판독 결과: 실패 ({score*100:.1f}%)")
                        if coach_msg:
                            st.info(f"🤖 **AI 코치 피드백 (Ollama):**\n\n{coach_msg}")
                    
                    with open(res_path, 'rb') as f:
                        video_bytes = f.read()
                    
                    if len(video_bytes) > 0:
                        st.video(video_bytes)
                        st.download_button("결과 영상 다운로드", video_bytes, 
                                         file_name="ai_referee_result.mp4", mime="video/mp4")