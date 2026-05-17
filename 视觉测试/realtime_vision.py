#!/usr/bin/env python3
"""
realtime_vision.py — 实时视觉测试（带GUI窗口）
功能：
  - 打开摄像头实时显示画面
  - 实时人体检测 + 骨架
  - 实时表情识别 + 显示
  - 同时录制视频
  - 按 Q 退出
"""

import cv2
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import (
    PoseLandmarker, FaceLandmarker,
    PoseLandmarksConnections,
)
import mediapipe as mp_lib
import os
import time
import json

SCRIPT_DIR = os.path.dirname(__file__)
SAVE_DIR = os.path.join(SCRIPT_DIR, "captures")
MODEL_DIR = os.path.join(SCRIPT_DIR, "models")
os.makedirs(SAVE_DIR, exist_ok=True)


def get_position(landmarks, w):
    ls = landmarks[11]
    rs = landmarks[12]
    cx = (ls.x + rs.x) / 2
    if cx < 0.35:
        return "LEFT"
    elif cx > 0.65:
        return "RIGHT"
    else:
        return "CENTER"


def get_expression(blendshapes):
    if not blendshapes:
        return "NEUTRAL -_-", (200, 200, 200)
    scores = {}
    for bs in blendshapes:
        scores[bs.category_name] = bs.score

    mouth_smile = max(scores.get("mouthSmileLeft", 0), scores.get("mouthSmileRight", 0))
    mouth_open = scores.get("jawOpen", 0)
    brow_up = max(scores.get("browInnerUp", 0), scores.get("browOuterUpLeft", 0), scores.get("browOuterUpRight", 0))
    eye_squint = max(scores.get("eyeSquintLeft", 0), scores.get("eyeSquintRight", 0))
    mouth_frown = max(scores.get("mouthFrownLeft", 0), scores.get("mouthFrownRight", 0))

    if mouth_open > 0.3 and brow_up > 0.15:
        return "SURPRISED", (0, 200, 255)
    elif mouth_smile > 0.5:
        return "HAPPY ^_^", (0, 255, 0)
    elif mouth_smile > 0.25:
        return "SMILE :]", (0, 230, 100)
    elif mouth_frown > 0.3 or eye_squint > 0.4:
        return "SAD T_T", (200, 100, 0)
    else:
        return "NEUTRAL -_-", (200, 200, 200)


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
    print("=" * 50)
    print("👁️ 实时视觉测试")
    print("  按 Q 退出")
    print("=" * 50)

    pose_detector = PoseLandmarker.create_from_model_path(
        os.path.join(MODEL_DIR, "pose_landmarker.task"))
    # 关键：必须设置 output_face_blendshapes=True 才能识别表情！
    from mediapipe.tasks.python.core.base_options import BaseOptions
    face_options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=os.path.join(MODEL_DIR, "face_landmarker.task")),
        output_face_blendshapes=True,
        num_faces=3,
        min_face_detection_confidence=0.3,
    )
    face_detector = FaceLandmarker.create_from_options(face_options)
    print("✅ 模型加载成功")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    w_frame = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_frame = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 20
    print(f"✅ 摄像头: {w_frame}x{h_frame}")

    # 录制视频
    ts = int(time.time())
    video_path = os.path.join(SAVE_DIR, f"realtime_{ts}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(video_path, fourcc, fps, (w_frame, h_frame))

    print(f"📹 同时录制到: {video_path}")
    print("\n🎬 开始！对着摄像头做表情吧！\n")

    frame_count = 0
    fps_timer = time.time()
    fps_count = 0
    display_fps = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        fps_count += 1
        h, w = frame.shape[:2]

        # 计算FPS
        now = time.time()
        if now - fps_timer >= 1.0:
            display_fps = fps_count
            fps_count = 0
            fps_timer = now

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp_lib.Image(image_format=mp_lib.ImageFormat.SRGB, data=rgb)

        # 人体检测
        pose_result = pose_detector.detect(mp_image)
        person_text = "No person"
        person_color = (0, 0, 255)

        if pose_result.pose_landmarks:
            lm = pose_result.pose_landmarks[0]
            position = get_position(lm, w)
            shoulder_px = abs(lm[11].x - lm[12].x) * w
            draw_pose(frame, lm)
            person_text = f"Person: {position} ({shoulder_px:.0f}px)"
            person_color = (0, 255, 0)

        # 表情识别
        face_result = face_detector.detect(mp_image)
        expr_text = ""
        expr_color = (200, 200, 200)

        if face_result.face_landmarks and len(face_result.face_landmarks) > 0:
            face_lm = face_result.face_landmarks[0]
            draw_face(frame, face_lm)

            if face_result.face_blendshapes and len(face_result.face_blendshapes) > 0:
                expr_text, expr_color = get_expression(face_result.face_blendshapes[0])

            # 大字显示表情（画面中央上方）
            if expr_text:
                # 背景框
                (tw, th), _ = cv2.getTextSize(expr_text, cv2.FONT_HERSHEY_SIMPLEX, 1.8, 3)
                cx = w // 2 - tw // 2
                # 色块背景（用表情对应的颜色）
                cv2.rectangle(frame, (cx - 15, 40), (cx + tw + 15, 40 + th + 30), (30, 30, 30), -1)
                cv2.rectangle(frame, (cx - 15, 40), (cx + tw + 15, 40 + th + 30), expr_color, 3)
                cv2.putText(frame, expr_text, (cx, 40 + th + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.8, expr_color, 3)

        # 左上角信息
        cv2.putText(frame, person_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, person_color, 2)

        # 左下角FPS
        cv2.putText(frame, f"FPS: {display_fps}", (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # 右下角提示
        cv2.putText(frame, "Press Q to quit", (w - 180, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        # 录制
        writer.write(frame)

        # 显示窗口
        cv2.imshow("AI Robot - Vision Test", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    pose_detector.close()
    face_detector.close()

    print(f"\n✅ 测试结束！")
    print(f"   📹 视频: {video_path}")
    print(f"   📊 总帧数: {frame_count}")


if __name__ == "__main__":
    main()
