#!/usr/bin/env python3
"""
test3_face_expression.py — 面部关键点 + 表情识别
功能：
  - MediaPipe Face Mesh 检测468个面部关键点
  - 基于关键点简单规则判断表情（开心/中性/惊讶/不开心）
  - 画面部网格
  - 显示表情结果
  - 按 S 截图，按 Q 退出
"""

import cv2
import mediapipe as mp
import math
import os
import time

SAVE_DIR = os.path.join(os.path.dirname(__file__), "captures")
os.makedirs(SAVE_DIR, exist_ok=True)


def distance(p1, p2):
    """两个关键点之间的距离"""
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)


def detect_expression(landmarks):
    """
    基于面部关键点的简单规则判断表情
    返回: (表情名称, 置信度描述)

    关键点索引参考：
    - 61, 291: 嘴角左右
    - 13, 14: 上下唇中心
    - 159, 145: 左眼上下
    - 386, 374: 右眼上下
    - 70: 左眉内侧, 107: 左眉外侧（高度对比）
    - 300: 右眉内侧, 336: 右眉外侧
    - 10: 额头中心
    """

    # 嘴巴宽度（嘴角距离）
    mouth_width = distance(landmarks[61], landmarks[291])

    # 嘴巴张开程度（上下唇距离）
    mouth_open = distance(landmarks[13], landmarks[14])

    # 嘴角上扬程度（嘴角y坐标 vs 上唇y坐标）
    # 嘴角比上唇高（y值小）= 微笑
    left_corner_up = landmarks[61].y < landmarks[0].y   # 左嘴角比唇底高
    right_corner_up = landmarks[291].y < landmarks[17].y  # 右嘴角比唇底高

    # 眼睛张开程度
    left_eye_open = distance(landmarks[159], landmarks[145])
    right_eye_open = distance(landmarks[386], landmarks[374])
    avg_eye_open = (left_eye_open + right_eye_open) / 2

    # 眉毛高度（相对于眼睛）
    left_brow_h = landmarks[70].y    # 左眉内侧
    left_eye_h = landmarks[159].y    # 左眼上
    right_brow_h = landmarks[300].y
    right_eye_h = landmarks[386].y
    avg_brow_eye_dist = ((left_eye_h - left_brow_h) + (right_eye_h - right_brow_h)) / 2

    # 判断规则
    smile_score = 0
    if left_corner_up and right_corner_up:
        smile_score += 1
    if mouth_width > 0.15:  # 嘴巴比较宽 = 微笑
        smile_score += 1

    # 表情判断
    if mouth_open > 0.04 and avg_eye_open > 0.03:
        return "惊讶 😮", "high"
    elif smile_score >= 2:
        return "开心 😊", "high"
    elif smile_score == 1:
        return "微微笑 😌", "medium"
    elif mouth_open < 0.01 and avg_brow_eye_dist < 0.02:
        return "不开心 😔", "medium"
    else:
        return "中性 😐", "high"


def main():
    print("😊 表情识别测试（MediaPipe Face Mesh）")
    print("  按 S → 截图保存")
    print("  按 Q → 退出")
    print()

    mp_face = mp.solutions.face_mesh
    mp_draw = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles

    face_mesh = mp_face.FaceMesh(
        max_num_faces=3,            # 最多检测3张脸
        refine_landmarks=True,      # 精细化（包含虹膜）
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 无法打开摄像头！")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print(f"✅ 摄像头已打开")
    save_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            for i, face_lm in enumerate(results.multi_face_landmarks):
                # 画面部网格
                mp_draw.draw_landmarks(
                    frame,
                    face_lm,
                    mp_face.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_styles.get_default_face_mesh_tesselation_style()
                )
                # 画轮廓
                mp_draw.draw_landmarks(
                    frame,
                    face_lm,
                    mp_face.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_styles.get_default_face_mesh_contours_style()
                )

                # 判断表情
                landmarks = face_lm.landmark
                expr, conf = detect_expression(landmarks)

                # 在人脸附近显示表情
                # 用鼻尖位置( landmark 1 ) 作为文字位置参考
                nose = landmarks[1]
                h, w = frame.shape[:2]
                text_x = int(nose.x * w) - 40
                text_y = int(nose.y * h) - 30

                # 背景
                cv2.rectangle(frame, (text_x - 5, text_y - 25), (text_x + 160, text_y + 5), (0, 0, 0), -1)
                cv2.putText(frame, expr, (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # 显示检测到的人脸数
            cv2.putText(frame, f"Faces: {len(results.multi_face_landmarks)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No face detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("AI小车 - 表情识别", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break
        elif key == ord('s') or key == ord('S'):
            save_count += 1
            filename = f"face_{int(time.time())}_{save_count}.jpg"
            filepath = os.path.join(SAVE_DIR, filename)
            cv2.imwrite(filepath, frame)
            print(f"📸 已保存: {filepath}")

    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()
    print("👋 退出")


if __name__ == "__main__":
    main()
