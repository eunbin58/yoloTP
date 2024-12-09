import os
import sys
import cv2
import torch
import tempfile
import mimetypes
import numpy as np
import openai
import streamlit as st
# from dotenv import load_dotenv
from ultralytics import YOLO
from dtaidistance import dtw
import warnings
from openai import OpenAI

# 환경 변수 및 경고 설정
# os.environ["QT_QPA_PLATFORM"] = "offscreen"
# warnings.filterwarnings("ignore")

# .env 파일에서 환경 변수 로딩
# load_dotenv()
# openai.api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI()

# YOLO 모델 로드 함수
@st.cache_resource
def load_yolo_model():
    return YOLO('yolov8m-pose.pt', verbose=False)

import cv2
import numpy as np
from ultralytics import YOLO
from dtaidistance import dtw

# YOLO 모델 불러오기
model = YOLO('yolov8m-pose.pt')  # YOLOv8 포즈 모델 경로

# keypoints 좌표를 [0, 1]로 정규화하는 함수
def normalize_keypoints(keypoints, frame_width, frame_height):
    normalized_keypoints = np.copy(keypoints)
    for i in range(0, len(keypoints), 2):
        normalized_keypoints[i] = keypoints[i] / frame_width  # x 좌표
        normalized_keypoints[i + 1] = keypoints[i + 1] / frame_height  # y 좌표
    return normalized_keypoints

# Keypoints 간 상대적 거리 계산
def calculate_relative_distances(keypoints):
    num_keypoints = len(keypoints) // 2
    relative_distances = []
    
    # 각 keypoint 사이의 거리를 계산 (유클리드 거리)
    for i in range(num_keypoints):
        for j in range(i + 1, num_keypoints):
            x1, y1 = keypoints[2 * i], keypoints[2 * i + 1]
            x2, y2 = keypoints[2 * j], keypoints[2 * j + 1]
            distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            relative_distances.append(distance)
    
    return np.array(relative_distances)

# Keypoints 시퀀스를 스무딩하는 함수
def smooth_keypoints(sequence, window_size=3):
    smoothed_sequence = []
    for i in range(sequence.shape[1]):  # 각 keypoint에 대해
        smoothed = np.convolve(sequence[:, i], np.ones(window_size)/window_size, mode='valid')
        smoothed_sequence.append(smoothed)
    smoothed_sequence = np.array(smoothed_sequence).T
    return smoothed_sequence

# 비디오에서 keypoints 추출하는 함수 (시각화 제외)
def extract_keypoints(video_path, model):
    cap = cv2.VideoCapture(video_path)
    keypoints_sequence = []
    max_keypoints = 34  # Keypoints 배열의 고정된 크기 (17개의 keypoints, 각 2D 좌표)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # YOLO로 프레임에서 포즈 추출
        results = model(frame)

        for result in results:
            if result.keypoints is not None:
                # Keypoints 추출 (xy 좌표만 사용)
                keypoints = result.keypoints.xy.cpu().numpy()  # NumPy 배열로 변환
                xy_keypoints = keypoints.flatten()  # 1D로 평탄화
                
                # 좌표 정규화
                normalized_keypoints = normalize_keypoints(xy_keypoints, frame_width, frame_height)

                # Keypoints 배열의 크기를 고정 (34로 맞춤, 부족하면 0으로 패딩)
                if len(normalized_keypoints) < max_keypoints:
                    padded_keypoints = np.zeros(max_keypoints)
                    padded_keypoints[:len(normalized_keypoints)] = normalized_keypoints
                    keypoints_sequence.append(padded_keypoints)
                else:
                    keypoints_sequence.append(normalized_keypoints[:max_keypoints])
    
    cap.release()

    # keypoints_sequence를 배열로 변환
    keypoints_sequence = np.array(keypoints_sequence)

    # keypoints 시퀀스에 스무딩 적용
    if len(keypoints_sequence) > 3:  # 스무딩 적용 가능한 최소 길이 확인
        keypoints_sequence = smooth_keypoints(keypoints_sequence)

    return keypoints_sequence

# 두 시퀀스 간의 DTW 거리 계산 (상대적 거리 기반)
def calculate_dtw_distance(seq1, seq2):
    # 각 시퀀스의 상대적 거리 계산
    seq1_relative = np.array([calculate_relative_distances(frame) for frame in seq1])
    seq2_relative = np.array([calculate_relative_distances(frame) for frame in seq2])

    # 시퀀스 길이 맞추기
    min_len = min(len(seq1_relative), len(seq2_relative))
    seq1_flat = seq1_relative[:min_len]  # 두 시퀀스의 길이를 동일하게 맞춤
    seq2_flat = seq2_relative[:min_len]

    # 프레임별로 DTW 거리 계산
    distances = []
    for i in range(min_len):
        if np.any(np.isnan(seq1_flat[i])) or np.any(np.isnan(seq2_flat[i])):
            distances.append(np.inf)  # NaN이 있는 경우, 무한대로 처리
        else:
            distance = dtw.distance(seq1_flat[i], seq2_flat[i])
            distances.append(distance)

    return np.mean(distances)

# 두 영상의 유사도를 계산하는 메인 함수
def compare_videos(video_path1, video_path2, model):
    # 첫 번째 비디오에서 keypoints 시퀀스를 추출
    keypoints_seq1 = extract_keypoints(video_path1, model)
    # 두 번째 비디오에서 keypoints 시퀀스를 추출
    keypoints_seq2 = extract_keypoints(video_path2, model)

    # 두 시퀀스 간의 DTW 거리 계산 (상대적 거리 기반)
    dtw_distance = calculate_dtw_distance(keypoints_seq1, keypoints_seq2)
    
    # 유사도를 계산 (거리가 작을수록 더 유사함)
    print(f"DTW Distance between the two videos: {dtw_distance}")

    return dtw_distance  # DTW 거리 반환 추가

# DTW 유사도 기반 피드백 생성 함수
def get_advice_based_on_similarity(dtw_distance, action_name):
    user_message = (
        f"사용자와 '{action_name}' 동작을 비교한 결과, DTW 거리 값은 {dtw_distance}입니다.\n"
        "이 값에 기반하여 피드백을 제공해주세요:\n"
        "- 유사도가 낮을 경우: 자세를 교정하기 위한 구체적인 피드백 제공.\n"
        "- 유사도가 높을 경우: 칭찬과 간단한 개선점을 제안.\n"
    )
    messages = [
        {"role": "system", "content": "당신은 피트니스 전문가입니다."},
        {"role": "user", "content": user_message},
    ]
    # try:
    result = client.chat.completions.create(
        model="gpt-4o",  # OpenAI API 호출
        messages=messages,
        temperature=0.7
    )
    # advice = result['choices'][0]['message']['content']
    advice = result.choices[0].message.content
    return advice
    # except Exception as e:
    #     print(f"Error: {str(e)}")
    #     return "피드백을 생성하는 동안 문제가 발생했습니다. 다시 시도해주세요."

# 비디오 파일 저장 함수
def save_uploaded_file(uploaded_file):
    mime_type, _ = mimetypes.guess_type(uploaded_file.name)
    if uploaded_file.size > 10 * 1024 * 1024:
        st.error("파일 크기가 너무 큽니다. 10MB 이하의 파일을 업로드해주세요.")
        return None
    if mime_type and mime_type.startswith('video'):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            temp_file.write(uploaded_file.read())
            return temp_file.name
    else:
        st.error("업로드된 파일은 비디오 파일이어야 합니다.")
        return None

# 비디오 비교 및 유사도 측정 함수
def compare_videos(description_video_path, uploaded_video_path, model):
    keypoints_seq1 = extract_keypoints(description_video_path, model)
    keypoints_seq2 = extract_keypoints(uploaded_video_path, model)
    dtw_distance = calculate_dtw_distance(keypoints_seq1, keypoints_seq2)
    return dtw_distance

# 비디오 처리 함수
def process_video(description_video_path, uploaded_video_path, model):
    try:
        dtw_distance = compare_videos(description_video_path, uploaded_video_path, model)
        advice = get_advice_based_on_similarity(dtw_distance, st.session_state.selected_action)
        return dtw_distance, advice
    except Exception as e:
        st.error(f"분석 중 오류 발생: {e}")
        return None, None

# 스트림릿 페이지 구성
def show():
    st.title("동작 비교 페이지")
    st.write("여기는 동작 비교 페이지입니다.")
    
    model = load_yolo_model()

    # 동작 설명 비디오 처리
    description_video_path = None
    if 'selected_action' in st.session_state:
        action_info = {
            "로우 런지(Low Lunge)": '../src/mp4/video1.mp4',
            "파르브리타 자누 시르사아사나(Revolved Head-to-Knee Pose)": '../src/mp4/video6.mp4',
            "선 활 자세(Standing Split)": '../src/mp4/video3.mp4',
            "런지 사이트 스트레칭(Lunging Side Stretch)": '../src/mp4/video4.mp4'
        }
        video_path = os.path.join(os.path.dirname(__file__), action_info[st.session_state.selected_action])
        if os.path.exists(video_path):
            st.video(video_path)
            description_video_path = video_path
        else:
            st.write("비디오 파일을 찾을 수 없습니다.")

    # 사용자 업로드 비디오 처리
    uploaded_video_path = None
    if 'uploaded_video' in st.session_state:
        st.subheader("사용자 업로드 비디오")
        st.video(st.session_state.uploaded_video)
        uploaded_video_path = save_uploaded_file(st.session_state.uploaded_video)

    # 동작 유사도 측정 버튼
    if description_video_path and uploaded_video_path:
        if st.button("동작 유사도 측정"):
            progress_bar = st.progress(0)
            dtw_distance, advice = process_video(description_video_path, uploaded_video_path, model)
            if dtw_distance is not None:
                st.session_state.dtw_distance = dtw_distance
                st.session_state.advice = advice
                st.write(f"DTW 거리: {dtw_distance}")
                st.write(f"GPT-4 조언: {advice}")
            else:
                st.error("비디오 비교에 실패했습니다.")
            progress_bar.empty()

        if 'similarity_measured' in st.session_state and st.session_state.similarity_measured:
            if st.button("다음"):
                st.session_state.selected_page = "recommendation"
    else:
        st.write("비디오를 선택하거나 업로드해 주세요.")

# CSS 로딩 함수
def load_css(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"CSS 파일을 찾을 수 없습니다: {file_path}")

# CSS 로드 및 적용
css_path = os.path.join(os.path.dirname(__file__), '../src/styles.css')
st.markdown(f"<style>{load_css(css_path)}</style>", unsafe_allow_html=True)
