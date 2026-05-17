#!/usr/bin/env python3
"""
test2_pose_detection.py — MediaPipe 人体检测 + 骨架可视化
功能：
  - 实时检测人体姿态（33个关键点）
  - 画骨架和关键点
  - 显示人体位置（左/中/右）
  - 显示人体数量
  - 按 S 截图，按 Q 退出
"""

import cv2
import mediapipe as mp
import os
import time

SAVE_DIR = os.path.join(os.path.dirname(__file__), "captures")
os.makedirs(SAVE_DIR, exist_ok=True)

def get_position_label(landmarks, w):
    """根据人体中心点判断位置（左/中/右）"""
    # 用肩膀中点作为人体中心
    left_shoulder = landmarks[11]
    right_shoulder = landmarks[12]
    center_x = (left_shoulder.x + right_shoulder.x) / 2

    if center_x < 0.35:
        return "左侧", (0, 0, 255)   # 红色
    elif center_x > 0.65:
        return "右侧", (255, 0, 0)   # 蓝色
    else:
        return "中间", (0, 255, 0)   # 绿色


def main():
    print("🤖 人体检测测试（MediaPipe Pose）")
    print("  按 S → 截图保存")
    print("  按 Q → 退出")
    print()

    mp_pose = mp.solutions.pose
    mp_draw = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles

    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=0,       # 0=最轻量, 1=中等, 2=最准
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 无法打开摄像头！")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"✅ 摄像头: {w}x{h}")

    save_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # BGR → RGB（MediaPipe需要RGB）
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        person_count = 0

        if results.pose_landmarks:
            person_count = 1  # MediaPipe Pose 一次只检测一个人

            # 画骨架
            mp_draw.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style()
            )

            landmarks = results.pose_landmarks.landmark

            # 判断位置
            pos_label, pos_color = get_position_label(landmarks, w)

            # 计算肩宽像素（可以用来估算距离）
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]
            shoulder_px = abs(left_shoulder.x - right_shoulder.x) * w

            # 显示信息
            cv2.putText(frame, f"Person: {person_count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Pos: {pos_label}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, pos_color, 2)
            cv2.putText(frame, f"Shoulder: {shoulder_px:.0f}px", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        else:
            cv2.putText(frame, "No person detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("AI小车 - 人体检测", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break
        elif key == ord('s') or key == ord('S'):
            save_count += 1
            filename = f"pose_{int(time.time())}_{save_count}.jpg"
            filepath = os.path.join(SAVE_DIR, filename)
            cv2.imwrite(filepath, frame)
            print(f"📸 已保存: {filepath}")

    cap.release()
    cv2.destroyAllWindows()
    pose.close()
    print("👋 退出")


if __name__ == "__main__":
    main()
