#!/usr/bin/env python3
"""
auto_vision_test.py — 后台自动视觉测试（无GUI，新版MediaPipe tasks API）
拍一张照片，运行人体检测+表情识别，保存标注后的图片和JSON状态
"""

import cv2
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import (
    PoseLandmarker, PoseLandmarkerOptions,
    FaceLandmarker, FaceLandmarkerOptions,
    RunningMode,
)
import mediapipe as mp_lib
import math
import numpy as np
import os
import json
import time

SCRIPT_DIR = os.path.dirname(__file__)
SAVE_DIR = os.path.join(SCRIPT_DIR, "captures")
MODEL_DIR = os.path.join(SCRIPT_DIR, "models")
os.makedirs(SAVE_DIR, exist_ok=True)

STATUS_FILE = os.path.join(SCRIPT_DIR, "vision_status.json")


def get_position(landmarks, w):
    """根据肩膀中点判断人体位置"""
    left_shoulder = landmarks[11]   # left_shoulder
    right_shoulder = landmarks[12]  # right_shoulder
    center_x = (left_shoulder.x + right_shoulder.x) / 2
    if center_x < 0.35:
        return "left"
    elif center_x > 0.65:
        return "right"
    else:
        return "center"


def get_expression(blendshapes):
    """根据面部blendshapes判断表情"""
    # blendshapes是一个列表，每个有category_name和score
    scores = {}
    for bs in blendshapes:
        scores[bs.category_name] = bs.score

    # 关键blendshapes
    mouth_smile = max(
        scores.get("mouthSmileLeft", 0),
        scores.get("mouthSmileRight", 0),
    )
    mouth_open = scores.get("jawOpen", 0)
    brow_up = max(
        scores.get("browInnerUp", 0),
        scores.get("browOuterUpLeft", 0),
        scores.get("browOuterUpRight", 0),
    )
    eye_squint = max(
        scores.get("eyeSquintLeft", 0),
        scores.get("eyeSquintRight", 0),
    )

    if mouth_open > 0.3 and brow_up > 0.2:
        return "surprised 😮", 0.9
    elif mouth_smile > 0.4:
        return "happy 😊", 0.85
    elif mouth_smile > 0.2:
        return "slight_happy 😌", 0.7
    elif eye_squint > 0.3:
        return "sad 😔", 0.6
    else:
        return "neutral 😐", 0.8


def draw_pose_landmarks(img, landmarks, connections):
    """在图片上画骨架"""
    h, w = img.shape[:2]
    # 画关键点
    for lm in landmarks:
        x, y = int(lm.x * w), int(lm.y * h)
        cv2.circle(img, (x, y), 3, (0, 255, 0), -1)

    # 画连接线
    for conn in connections:
        start = landmarks[conn.start]
        end = landmarks[conn.end]
        sx, sy = int(start.x * w), int(start.y * h)
        ex, ey = int(end.x * w), int(end.y * h)
        cv2.line(img, (sx, sy), (ex, ey), (0, 255, 0), 2)


def draw_face_landmarks(img, landmarks):
    """在图片上画面部关键点"""
    h, w = img.shape[:2]
    for lm in landmarks:
        x, y = int(lm.x * w), int(lm.y * h)
        cv2.circle(img, (x, y), 1, (255, 200, 0), -1)


def main():
    print("=" * 50)
    print("👁️ 自动视觉测试（MediaPipe Tasks API）")
    print("=" * 50)

    # 加载模型
    pose_path = os.path.join(MODEL_DIR, "pose_landmarker.task")
    face_path = os.path.join(MODEL_DIR, "face_landmarker.task")

    pose_detector = PoseLandmarker.create_from_model_path(pose_path)
    face_detector = FaceLandmarker.create_from_model_path(face_path)
    print("✅ 模型加载成功")

    # 拍照
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 无法打开摄像头")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # 丢掉前几帧
    for _ in range(5):
        cap.read()

    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("❌ 拍照失败")
        return

    h, w = frame.shape[:2]
    print(f"📸 拍照成功: {w}x{h}")

    # 转成MediaPipe Image
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp_lib.Image(image_format=mp_lib.ImageFormat.SRGB, data=rgb)

    # === 状态初始化 ===
    status = {
        "timestamp": time.time(),
        "persons": [],
        "faces": [],
        "summary": {
            "person_count": 0,
            "face_count": 0,
        }
    }

    # === 人体检测 ===
    pose_result = pose_detector.detect(mp_image)
    if pose_result.pose_landmarks:
        landmarks = pose_result.pose_landmarks[0]
        position = get_position(landmarks, w)

        # 肩宽像素
        shoulder_px = abs(landmarks[11].x - landmarks[12].x) * w

        status["persons"].append({
            "id": 1,
            "position": position,
            "shoulder_width_px": round(shoulder_px, 1),
        })
        status["summary"]["person_count"] = 1

        # 画骨架
        from mediapipe.tasks.python.vision import PoseLandmarksConnections
        draw_pose_landmarks(frame, landmarks, PoseLandmarksConnections.POSE_LANDMARKS)

        print(f"✅ 人体: 位置={position}, 肩宽={shoulder_px:.0f}px")
        cv2.putText(frame, f"Person: {position} ({shoulder_px:.0f}px)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        print("⚠️ 未检测到人体")
        cv2.putText(frame, "No person", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # === 表情识别 ===
    face_result = face_detector.detect(mp_image)
    if face_result.face_landmarks:
        for i, face_lm in enumerate(face_result.face_landmarks):
            draw_face_landmarks(frame, face_lm)

            # blendshapes
            expr = "neutral 😐"
            conf = 0.5
            if face_result.face_blendshapes and i < len(face_result.face_blendshapes):
                expr, conf = get_expression(face_result.face_blendshapes[i])

            status["faces"].append({
                "id": i + 1,
                "expression": expr,
                "confidence": conf,
            })

            # 在人脸附近显示表情
            nose = face_lm[1]
            tx = int(nose.x * w) - 30
            ty = int(nose.y * h) - 40
            cv2.rectangle(frame, (tx - 5, ty - 25), (tx + 180, ty + 5), (0, 0, 0), -1)
            cv2.putText(frame, expr, (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            print(f"✅ 表情: Face {i+1} → {expr} (conf={conf:.0%})")

        status["summary"]["face_count"] = len(face_result.face_landmarks)
    else:
        print("⚠️ 未检测到人脸")

    # 右上角汇总
    info = [
        f"Persons: {status['summary']['person_count']}",
        f"Faces: {status['summary']['face_count']}",
    ]
    for f_data in status["faces"]:
        info.append(f"  -> {f_data['expression']}")

    for i, line in enumerate(info):
        cv2.putText(frame, line, (w - 250, 25 + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
                    cv2.LINE_AA)

    # 保存标注图
    img_path = os.path.join(SAVE_DIR, f"vision_test_{int(time.time())}.jpg")
    cv2.imwrite(img_path, frame)
    print(f"\n📸 标注图: {img_path}")

    # 保存JSON状态
    with open(STATUS_FILE, 'w') as f:
        json.dump(status, f, ensure_ascii=False, indent=2)
    print(f"📝 状态文件: {STATUS_FILE}")
    print(json.dumps(status, ensure_ascii=False, indent=2))

    pose_detector.close()
    face_detector.close()
    print("\n✅ 测试完成！")


if __name__ == "__main__":
    main()
