# AI 3-Cushion Billiards Referee & Commentator
### (AI 기반 3쿠션 당구 심판 및 실시간 해설 시스템)

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Object_Detection-00FFFF?style=flat)
![Ollama](https://img.shields.io/badge/Ollama-Local_LLM-white?style=flat&logo=ollama&logoColor=black)
![Streamlit](https://img.shields.io/badge/Streamlit-Web_App-FF4B4B?style=flat&logo=streamlit&logoColor=white)

## Project Overview
이 프로젝트는 3쿠션 당구 영상을 입력받아 **AI가 득점 성공 여부를 자동으로 판독**하고, 샷의 물리적 데이터(속도, 각도 등)를 분석하여 **생성형 AI(LLM)가 마치 스포츠 캐스터처럼 실시간 해설 및 코칭 멘트를 생성**해주는 멀티모달 AI 시스템입니다.

단순한 비전 인식(CV)을 넘어, **LSTM 시계열 분석**과 **LLM의 자연어 생성 능력**을 결합하여 사용자에게 재미와 정보를 동시에 제공합니다.

---

## Key Features
### 1. AI Referee (자동 판독 시스템)
- **Object Detection:** YOLOv8을 사용하여 영상 내의 당구공(수구, 적구)을 실시간으로 추적합니다.
- **Trajectory Analysis:** 공의 이동 경로(Sequence)를 추출하고 노이즈를 제거(Smoothing)합니다.
- **Decision Making:** 추출된 궤적 데이터를 **LSTM 모델**에 입력하여 득점 성공/실패 여부를 판단합니다. (정확도: 약 90%+)

### 2. AI Caster (생성형 해설위원)
- **Physics Data Extraction:** 샷의 **속도(Speed)**, **진입 각도(Angle)**, **공의 움직임** 등 물리적 특성을 수치화합니다.
- **LLM Integration:** 추출된 데이터를 **Ollama(Llama3/Phi3)** 로컬 LLM에 프롬프트로 주입합니다.
- **Dynamic Commentary:** 상황에 따라 다른 페르소나를 수행합니다.
  - **성공 시:** 스포츠 캐스터처럼 역동적이고 흥분된 어조로 득점 상황 중계.
  - **실패 시:** 분석가(Coach) 모드로 전환하여 실패 원인(두께, 힘 조절 등) 분석 및 조언.

### 3. Visual Feedback
- 판독 결과에 따라 성공(초록색), 실패(빨간색) 궤적을 영상 위에 시각화(Overlay)하여 제공합니다.

---

## Tech Stack
| Category | Technology | Usage |
| :--- | :--- | :--- |
| **Language** | Python | 전체 시스템 개발 |
| **Vision AI** | Ultralytics YOLOv8 | 당구공 객체 탐지 및 좌표 추출 |
| **Sequence Model** | PyTorch (LSTM) | 궤적 기반 성공/실패 분류 모델 학습 및 추론 |
| **Generative AI** | Ollama (Llama3) | 샷 분석 텍스트 생성 (Local LLM) |
| **Data Processing** | OpenCV, Pandas | 영상 처리 및 시계열 데이터 전처리 |
| **Frontend** | Streamlit | 웹 기반 사용자 인터페이스(UI) 구축 |

---

## How to Run

### 1. Prerequisites
이 프로젝트는 **Local LLM**을 사용하므로 [Ollama](https://ollama.com/)가 설치되어 있어야 합니다.

```bash
# 1. Ollama 설치 후 터미널에서 모델 다운로드
ollama pull llama3

📂 Directory Structure
📁 AI-Billiards-Referee
├── 📄 app.py                # 메인 애플리케이션 (Streamlit)
├── 📄 best.pt               # YOLOv8 당구공 탐지 모델
├── 📄 best_model1_lstm.pth  # LSTM 판독 모델 (PyTorch)
├── 📄 requirements.txt      # 의존성 라이브러리 목록
└── 📂 data_csv_v2           # (Optional) 학습용 데이터셋
