#!/usr/bin/env python3
"""
realtime_vision_v2.py — 实时视觉测试 v2（12种表情识别）
基于52个blendshapes参数，识别更丰富的表情
"""

import cv2
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import (
    PoseLandmarker, FaceLandmarker, FaceLandmarkerOptions,
    PoseLandmarksConnections,
)
from mediapipe.tasks.python.core.base_options import BaseOptions
import mediapipe as mp_lib
import os
import time
import numpy as np

SCRIPT_DIR = os.path.dirname(__file__)
SAVE_DIR = os.path.join(SCRIPT_DIR, "captures")
MODEL_DIR = os.path.join(SCRIPT_DIR, "models")
os.makedirs(SAVE_DIR, exist_ok=True)


def get_position(landmarks, w):
    ls, rs = landmarks[11], landmarks[12]
    cx = (ls.x + rs.x) / 2
    if cx < 0.35: return "LEFT"
    elif cx > 0.65: return "RIGHT"
    else: return "CENTER"


def get_expression_rich(blendshapes):
    """
    基于52个blendshapes识别12种表情：
    HAPPY / SMILE / SURPRISED / SAD / ANGRY / FEAR
    DISGUST / THINKING / CONFUSED / SLEEPY / WINK / NEUTRAL
    """
    if not blendshapes:
        return "NEUTRAL", (200, 200, 200)

    s = {bs.category_name: bs.score for bs in blendshapes}

    # 提取关键特征
    smile = max(s.get("mouthSmileLeft", 0), s.get("mouthSmileRight", 0))
    jaw_open = s.get("jawOpen", 0)
    brow_inner_up = s.get("browInnerUp", 0)
    brow_outer_up = max(s.get("browOuterUpLeft", 0), s.get("browOuterUpRight", 0))
    brow_down = max(s.get("browDownLeft", 0), s.get("browDownRight", 0))
    mouth_frown = max(s.get("mouthFrownLeft", 0), s.get("mouthFrownRight", 0))
    eye_squint = max(s.get("eyeSquintLeft", 0), s.get("eyeSquintRight", 0))
    eye_wide = max(s.get("eyeWideLeft", 0), s.get("eyeWideRight", 0))
    eye_blink_l = s.get("eyeBlinkLeft", 0)
    eye_blink_r = s.get("eyeBlinkRight", 0)
    nose_sneer = max(s.get("noseSneerLeft", 0), s.get("noseSneerRight", 0))
    mouth_press = max(s.get("mouthPressLeft", 0), s.get("mouthPressRight", 0))
    mouth_pucker = s.get("mouthPucker", 0)
    mouth_stretch = max(s.get("mouthStretchLeft", 0), s.get("mouthStretchRight", 0))
    mouth_funnel = s.get("mouthFunnel", 0)
    cheek_puff = s.get("cheekPuff", 0)
    jaw_left = s.get("jawLeft", 0)
    jaw_right = s.get("jawRight", 0)
    eye_look_up = max(s.get("eyeLookUpLeft", 0), s.get("eyeLookUpRight", 0))

    # 打分制：每个表情算一个分数，取最高分
    scores = {}

    # 😊 HAPPY - 大笑：嘴巴大张+微笑+眉毛外侧上扬
    scores["HAPPY"] = (
        smile * 2.0 +
        jaw_open * 0.5 +
        brow_outer_up * 0.3 +
        cheek_puff * 0.5
    )

    # 😌 SMILE - 微笑：嘴角上扬但嘴巴没大张
    scores["SMILE"] = (
        smile * 2.0 -
        jaw_open * 1.0 +
        mouth_stretch * 0.3
    )

    # 😮 SURPRISED - 惊讶：嘴巴大张+眉毛上扬+眼睛睁大
    scores["SURPRISED"] = (
        jaw_open * 2.0 +
        brow_inner_up * 1.0 +
        brow_outer_up * 0.5 +
        eye_wide * 1.5
    )

    # 😔 SAD - 悲伤：嘴角下撇+眉毛内侧上扬+眼睛微闭
    scores["SAD"] = (
        mouth_frown * 2.0 +
        brow_inner_up * 0.8 +
        eye_squint * 0.5 -
        smile * 2.0
    )

    # 😠 ANGRY - 愤怒：眉毛下压+嘴紧抿+鼻子皱起
    scores["ANGRY"] = (
        brow_down * 2.0 +
        mouth_press * 1.5 +
        nose_sneer * 1.5 +
        eye_squint * 0.5 -
        smile * 2.0
    )

    # 😨 FEAR - 恐惧：眉毛内侧上扬+眼睛睁大+嘴巴紧张
    scores["FEAR"] = (
        brow_inner_up * 1.5 +
        eye_wide * 1.0 +
        mouth_pucker * 1.0 +
        jaw_open * 0.3 -
        smile * 2.0
    )

    # 🤢 DISGUST - 厌恶：鼻子皱+上唇上翻
    scores["DISGUST"] = (
        nose_sneer * 2.0 +
        mouth_funnel * 1.0 +
        brow_down * 0.5 -
        smile * 2.0
    )

    # 🤔 THINKING - 思考：歪嘴+眼睛看向一侧
    scores["THINKING"] = (
        abs(jaw_left - jaw_right) * 2.0 +
        mouth_press * 0.5 +
        abs(s.get("mouthLeft", 0) - s.get("mouthRight", 0)) * 2.0
    )

    # 😕 CONFUSED - 困惑：一边眉上扬一边眉下压
    scores["CONFUSED"] = (
        abs(s.get("browOuterUpLeft", 0) - s.get("browOuterUpRight", 0)) * 3.0 +
        abs(s.get("browDownLeft", 0) - s.get("browDownRight", 0)) * 2.0
    )

    # 😴 SLEEPY - 困倦：眼睛微闭+打哈欠趋势
    scores["SLEEPY"] = (
        (eye_blink_l + eye_blink_r) * 0.5 * 1.5 +
        eye_squint * 0.5 +
        mouth_funnel * 0.3 -
        smile * 1.0
    )

    # 😉 WINK - 眨眼/挤眼：一只眼闭一只眼开
    scores["WINK"] = (
        abs(eye_blink_l - eye_blink_r) * 3.0 +
        smile * 0.5
    )

    # 中性（兜底）
    scores["NEUTRAL"] = 0.1

    # 过滤负分
    scores = {k: max(v, 0) for k, v in scores.items()}

    # 特殊规则：眨眼检测需要一只眼确实闭了
    if min(eye_blink_l, eye_blink_r) < 0.5:
        scores["WINK"] = 0

    # SMILE 和 HAPPY 互斥：如果jaw_open很大就是HAPPY不是SMILE
    if jaw_open > 0.2:
        scores["SMILE"] = 0

    # 取最高分
    best = max(scores, key=scores.get)
    best_score = scores[best]

    # 如果最高分太低，就是中性
    if best_score < 0.3:
        best = "NEUTRAL"

    # 颜色映射
    colors = {
        "HAPPY":     (0, 255, 0),      # 绿
        "SMILE":     (0, 230, 100),    # 浅绿
        "SURPRISED": (0, 200, 255),    # 橙
        "SAD":       (200, 100, 0),    # 蓝
        "ANGRY":     (0, 0, 255),      # 红
        "FEAR":      (180, 0, 180),    # 紫
        "DISGUST":   (0, 160, 160),    # 青
        "THINKING":  (200, 200, 0),    # 黄
        "CONFUSED":  (160, 160, 200),  # 淡紫
        "SLEEPY":    (100, 100, 150),  # 灰蓝
        "WINK":      (0, 255, 255),    # 黄绿
        "NEUTRAL":   (200, 200, 200),  # 灰
    }

    return best, colors.get(best, (200, 200, 200))


def draw_pose(frame, landmarks):
    h, w = frame.shape[:2]
    for conn in PoseLandmarksConnections.POSE_LANDMARKS:
        s, e = landmarks[conn.start], landmarks[conn.end]
        cv2.line(frame, (int(s.x*w), int(s.y*h)), (int(e.x*w), int(e.y*h)), (0, 255, 0), 2)
    for lm in landmarks:
        cv2.circle(frame, (int(lm.x*w), int(lm.y*h)), 4, (0, 255, 0), -1)


def draw_face(frame, landmarks):
    h, w = frame.shape[:2]
    for lm in landmarks:
        cv2.circle(frame, (int(lm.x*w), int(lm.y*h)), 1, (255, 200, 0), -1)


def main():
    print("AI Robot Vision Test v2 - 12 Expressions")

    pose_detector = PoseLandmarker.create_from_model_path(
        os.path.join(MODEL_DIR, "pose_landmarker.task"))
    face_options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=os.path.join(MODEL_DIR, "face_landmarker.task")),
        output_face_blendshapes=True,
        num_faces=3,
        min_face_detection_confidence=0.3,
    )
    face_detector = FaceLandmarker.create_from_options(face_options)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    w_frame = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_frame = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 20

    ts = int(time.time())
    video_path = os.path.join(SAVE_DIR, f"realtime_v2_{ts}.mp4")
    writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w_frame, h_frame))

    fps_timer = time.time()
    fps_count = 0
    display_fps = 0
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_count += 1
        fps_count += 1
        h, w = frame.shape[:2]

        now = time.time()
        if now - fps_timer >= 1.0:
            display_fps = fps_count
            fps_count = 0
            fps_timer = now

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp_lib.Image(image_format=mp_lib.ImageFormat.SRGB, data=rgb)

        # Pose
        pose_result = pose_detector.detect(mp_image)
        if pose_result.pose_landmarks:
            lm = pose_result.pose_landmarks[0]
            position = get_position(lm, w)
            spx = abs(lm[11].x - lm[12].x) * w
            draw_pose(frame, lm)
            cv2.putText(frame, f"Person: {position} ({spx:.0f}px)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Face
        face_result = face_detector.detect(mp_image)
        expr_text = ""
        expr_color = (200, 200, 200)

        if face_result.face_landmarks and len(face_result.face_landmarks) > 0:
            face_lm = face_result.face_landmarks[0]
            draw_face(frame, face_lm)

            if face_result.face_blendshapes and len(face_result.face_blendshapes) > 0:
                expr_text, expr_color = get_expression_rich(face_result.face_blendshapes[0])

                # Top 5 blendshapes 右上角显示
                bs_dict = {bs.category_name: bs.score for bs in face_result.face_blendshapes[0]}
                top5 = sorted(bs_dict.items(), key=lambda x: x[1], reverse=True)[:5]
                for i, (name, val) in enumerate(top5):
                    cv2.putText(frame, f"{name}: {val:.2f}", (w - 260, 25 + i * 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)

        # Big expression text
        if expr_text:
            (tw, th), _ = cv2.getTextSize(expr_text, cv2.FONT_HERSHEY_SIMPLEX, 1.8, 3)
            cx = w // 2 - tw // 2
            cv2.rectangle(frame, (cx - 15, 50), (cx + tw + 15, 50 + th + 30), (30, 30, 30), -1)
            cv2.rectangle(frame, (cx - 15, 50), (cx + tw + 15, 50 + th + 30), expr_color, 3)
            cv2.putText(frame, expr_text, (cx, 50 + th + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.8, expr_color, 3)

        cv2.putText(frame, f"FPS: {display_fps}", (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, "Q to quit", (w - 110, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        writer.write(frame)
        cv2.imshow("AI Robot - Vision v2 (12 Expressions)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    pose_detector.close()
    face_detector.close()
    print(f"Done! Video: {video_path}")


if __name__ == "__main__":
    main()
