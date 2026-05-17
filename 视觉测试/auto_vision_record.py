#!/usr/bin/env python3
"""
auto_vision_record.py — 录制视频 + 实时人体/表情检测 + 保存状态
功能：
  - 录制一段视频（默认10秒）
  - 每帧运行人体检测+表情识别
  - 在视频上画骨架、表情标注
  - 保存标注后的视频和JSON状态
"""

import cv2
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import (
    PoseLandmarker, FaceLandmarker,
    PoseLandmarksConnections,
)
import mediapipe as mp_lib
import os
import json
import time

SCRIPT_DIR = os.path.dirname(__file__)
SAVE_DIR = os.path.join(SCRIPT_DIR, "captures")
MODEL_DIR = os.path.join(SCRIPT_DIR, "models")
os.makedirs(SAVE_DIR, exist_ok=True)

STATUS_FILE = os.path.join(SCRIPT_DIR, "vision_status.json")

RECORD_SECONDS = 10  # 录制时长


def get_position(landmarks, w):
    left_shoulder = landmarks[11]
    right_shoulder = landmarks[12]
    center_x = (left_shoulder.x + right_shoulder.x) / 2
    if center_x < 0.35:
        return "left"
    elif center_x > 0.65:
        return "right"
    else:
        return "center"


def get_expression(blendshapes):
    if not blendshapes:
        return "neutral 😐", 0.5
    scores = {}
    for bs in blendshapes:
        scores[bs.category_name] = bs.score

    mouth_smile = max(scores.get("mouthSmileLeft", 0), scores.get("mouthSmileRight", 0))
    mouth_open = scores.get("jawOpen", 0)
    brow_up = max(scores.get("browInnerUp", 0), scores.get("browOuterUpLeft", 0), scores.get("browOuterUpRight", 0))
    eye_squint = max(scores.get("eyeSquintLeft", 0), scores.get("eyeSquintRight", 0))

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


def draw_pose(frame, landmarks):
    h, w = frame.shape[:2]
    # 画连接线
    for conn in PoseLandmarksConnections.POSE_LANDMARKS:
        s = landmarks[conn.start]
        e = landmarks[conn.end]
        cv2.line(frame, (int(s.x*w), int(s.y*h)), (int(e.x*w), int(e.y*h)), (0, 255, 0), 2)
    # 画关键点
    for lm in landmarks:
        cv2.circle(frame, (int(lm.x*w), int(lm.y*h)), 3, (0, 255, 0), -1)


def draw_face(frame, landmarks):
    h, w = frame.shape[:2]
    for lm in landmarks:
        cv2.circle(frame, (int(lm.x*w), int(lm.y*h)), 1, (255, 200, 0), -1)


def main():
    print("=" * 50)
    print("🎬 视觉测试 + 视频录制")
    print(f"   录制时长: {RECORD_SECONDS}秒")
    print("=" * 50)

    # 加载模型
    pose_detector = PoseLandmarker.create_from_model_path(
        os.path.join(MODEL_DIR, "pose_landmarker.task"))
    face_detector = FaceLandmarker.create_from_model_path(
        os.path.join(MODEL_DIR, "face_landmarker.task"))
    print("✅ 模型加载成功")

    # 打开摄像头
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    w_frame = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_frame = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 20
    print(f"✅ 摄像头: {w_frame}x{h_frame}, {fps:.0f}fps")

    # 视频写入器
    ts = int(time.time())
    video_path = os.path.join(SAVE_DIR, f"vision_record_{ts}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(video_path, fourcc, fps, (w_frame, h_frame))
    print(f"📹 保存到: {video_path}")

    # 状态文件
    status_log = []

    start_time = time.time()
    frame_count = 0

    print(f"\n🎬 开始录制 {RECORD_SECONDS} 秒...")
    print("   刀哥对着摄像头做做表情！😊😮😔")

    while True:
        elapsed = time.time() - start_time
        if elapsed >= RECORD_SECONDS:
            break

        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp_lib.Image(image_format=mp_lib.ImageFormat.SRGB, data=rgb)

        # 当前帧状态
        frame_status = {
            "frame": frame_count,
            "time": round(elapsed, 2),
            "person": None,
            "face": None,
        }

        # 人体检测
        pose_result = pose_detector.detect(mp_image)
        if pose_result.pose_landmarks:
            lm = pose_result.pose_landmarks[0]
            position = get_position(lm, w)
            shoulder_px = abs(lm[11].x - lm[12].x) * w

            draw_pose(frame, lm)

            frame_status["person"] = {"position": position, "shoulder_px": round(shoulder_px, 1)}

            cv2.putText(frame, f"Person: {position} ({shoulder_px:.0f}px)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No person", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 表情识别
        face_result = face_detector.detect(mp_image)
        if face_result.face_landmarks and len(face_result.face_landmarks) > 0:
            face_lm = face_result.face_landmarks[0]
            draw_face(frame, face_lm)

            expr, conf = "neutral 😐", 0.5
            if face_result.face_blendshapes and len(face_result.face_blendshapes) > 0:
                expr, conf = get_expression(face_result.face_blendshapes[0])

            frame_status["face"] = {"expression": expr, "confidence": conf}

            nose = face_lm[1]
            tx, ty = int(nose.x * w) - 30, int(nose.y * h) - 40
            cv2.rectangle(frame, (tx - 5, ty - 25), (tx + 200, ty + 5), (0, 0, 0), -1)
            cv2.putText(frame, expr, (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # 右上角倒计时
        remaining = RECORD_SECONDS - elapsed
        cv2.putText(frame, f"{remaining:.1f}s", (w - 80, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # 右上角帧数
        cv2.putText(frame, f"Frame: {frame_count}", (w - 160, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 写入视频
        writer.write(frame)

        # 记录状态（每5帧记一次）
        if frame_count % 5 == 0:
            status_log.append(frame_status)

        # 每30帧打印一次
        if frame_count % 30 == 0:
            person_info = frame_status.get("person", {})
            face_info = frame_status.get("face", {})
            print(f"  [{elapsed:.1f}s] person={person_info.get('position', 'None')} "
                  f"face={face_info.get('expression', 'None')}")

    # 清理
    cap.release()
    writer.release()
    pose_detector.close()
    face_detector.close()

    # 保存最终状态
    final_status = {
        "timestamp": time.time(),
        "video_file": video_path,
        "frames": frame_count,
        "duration": round(elapsed, 2),
        "status_log": status_log,
        "summary": {
            "person_detected_frames": sum(1 for s in status_log if s.get("person")),
            "face_detected_frames": sum(1 for s in status_log if s.get("face")),
            "expressions_seen": list(set(
                s["face"]["expression"] for s in status_log if s.get("face")
            )),
        }
    }

    with open(STATUS_FILE, 'w') as f:
        json.dump(final_status, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 录制完成！")
    print(f"   📹 视频: {video_path}")
    print(f"   📊 {frame_count}帧 / {elapsed:.1f}秒")
    print(f"   👤 人体检测: {final_status['summary']['person_detected_frames']}/{len(status_log)} 帧")
    print(f"   😊 人脸检测: {final_status['summary']['face_detected_frames']}/{len(status_log)} 帧")
    print(f"   🎭 识别到的表情: {final_status['summary']['expressions_seen']}")
    print(f"   📝 状态: {STATUS_FILE}")


if __name__ == "__main__":
    main()
