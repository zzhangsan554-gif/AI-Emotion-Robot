#!/usr/bin/env python3
"""
test4_vision_status.py — 视觉状态整合 + 写入JSON
功能：
  - 同时运行人体检测 + 表情识别
  - 把结果整合到一个状态字典
  - 实时写入 vision_status.json（模拟给PicoClaw读取）
  - 可视化显示所有信息
  - 按 S 截图，按 Q 退出

这个就是最终视觉模块的原型！
"""

import cv2
import mediapipe as mp
import math
import json
import os
import time

SAVE_DIR = os.path.join(os.path.dirname(__file__), "captures")
os.makedirs(SAVE_DIR, exist_ok=True)

STATUS_FILE = os.path.join(os.path.dirname(__file__), "vision_status.json")


def distance(p1, p2):
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)


def get_position(landmarks, w):
    """人体水平位置"""
    left_shoulder = landmarks[11]
    right_shoulder = landmarks[12]
    center_x = (left_shoulder.x + right_shoulder.x) / 2
    if center_x < 0.35:
        return "left"
    elif center_x > 0.65:
        return "right"
    else:
        return "center"


def get_expression(face_landmarks):
    """简化版表情识别"""
    lm = face_landmarks.landmark

    mouth_width = distance(lm[61], lm[291])
    mouth_open = distance(lm[13], lm[14])
    left_corner_up = lm[61].y < lm[0].y
    right_corner_up = lm[291].y < lm[17].y

    smile_score = 0
    if left_corner_up and right_corner_up:
        smile_score += 1
    if mouth_width > 0.15:
        smile_score += 1

    if mouth_open > 0.04:
        return "surprised"
    elif smile_score >= 2:
        return "happy"
    elif smile_score == 1:
        return "slight_happy"
    else:
        return "neutral"


def main():
    print("👁️ 视觉状态整合测试")
    print("  人体检测 + 表情识别 → vision_status.json")
    print("  按 S → 截图 | 按 Q → 退出")
    print()

    # MediaPipe 初始化
    mp_pose = mp.solutions.pose
    mp_face = mp.solutions.face_mesh
    mp_draw = mp.solutions.drawing_utils

    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    face_mesh = mp_face.FaceMesh(
        max_num_faces=3,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 无法打开摄像头！")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    w_frame = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_frame = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"✅ 摄像头: {w_frame}x{h_frame}")
    print(f"📝 状态文件: {STATUS_FILE}")
    print()

    save_count = 0
    last_write_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]

        # === 状态初始化 ===
        status = {
            "timestamp": time.time(),
            "persons": [],         # 检测到的人体列表
            "faces": [],           # 检测到的人脸+表情
            "summary": {
                "person_count": 0,
                "face_count": 0,
                "alert": None      # 异常告警
            }
        }

        # === 人体检测 ===
        pose_result = pose.process(rgb)
        if pose_result.pose_landmarks:
            mp_draw.draw_landmarks(
                frame, pose_result.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

            landmarks = pose_result.pose_landmarks.landmark
            position = get_position(landmarks, w)

            # 肩宽（用于粗略距离参考）
            shoulder_px = abs(landmarks[11].x - landmarks[12].x) * w

            status["persons"].append({
                "id": 1,
                "position": position,
                "shoulder_width_px": round(shoulder_px, 1),
            })
            status["summary"]["person_count"] = 1

            # 画面上显示
            cv2.putText(frame, f"Person: {position}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No person", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # === 表情识别 ===
        face_result = face_mesh.process(rgb)
        if face_result.multi_face_landmarks:
            for i, face_lm in enumerate(face_result.multi_face_landmarks):
                # 画轮廓（轻量，不画全部网格）
                mp_draw.draw_landmarks(
                    frame, face_lm,
                    mp_face.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_contours_style()
                )

                expr = get_expression(face_lm)
                expr_cn = {"happy": "开心😊", "slight_happy": "微笑😌",
                           "neutral": "中性😐", "surprised": "惊讶😮"}.get(expr, expr)

                status["faces"].append({
                    "id": i + 1,
                    "expression": expr,
                })

                # 在人脸附近显示表情
                nose = face_lm.landmark[1]
                tx = int(nose.x * w) - 30
                ty = int(nose.y * h) - 30
                cv2.putText(frame, expr_cn, (tx, ty),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            status["summary"]["face_count"] = len(face_result.multi_face_landmarks)

        # === 右上角显示状态摘要 ===
        info_lines = [
            f"Persons: {status['summary']['person_count']}",
            f"Faces: {status['summary']['face_count']}",
        ]
        # 如果有人脸，显示表情
        for f in status["faces"]:
            info_lines.append(f"Face {f['id']}: {f['expression']}")

        for i, line in enumerate(info_lines):
            cv2.putText(frame, line, (w - 250, 30 + i * 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
                        cv2.LINE_AA)

        # === 写入JSON（每200ms更新一次，不刷盘太频繁） ===
        now = time.time()
        if now - last_write_time > 0.2:
            try:
                with open(STATUS_FILE, 'w') as f:
                    json.dump(status, f, ensure_ascii=False, indent=2)
                last_write_time = now
            except Exception as e:
                print(f"⚠️ 写入状态文件失败: {e}")

        cv2.imshow("AI小车 - 视觉状态", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break
        elif key == ord('s') or key == ord('S'):
            save_count += 1
            filename = f"vision_{int(time.time())}_{save_count}.jpg"
            filepath = os.path.join(SAVE_DIR, filename)
            cv2.imwrite(filepath, frame)
            print(f"📸 已保存: {filepath}")

    cap.release()
    cv2.destroyAllWindows()
    pose.close()
    face_mesh.close()

    # 清理状态文件
    if os.path.exists(STATUS_FILE):
        os.remove(STATUS_FILE)
        print("🧹 已清理状态文件")

    print("👋 退出")


if __name__ == "__main__":
    main()
