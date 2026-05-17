#!/usr/bin/env python3
"""
test1_camera_capture.py — 摄像头采集 + 截帧保存
功能：打开摄像头实时预览，按 S 截图保存，按 Q 退出
"""

import cv2
import os
import time

# 截图保存目录
SAVE_DIR = os.path.join(os.path.dirname(__file__), "captures")
os.makedirs(SAVE_DIR, exist_ok=True)

def main():
    print("📷 摄像头测试")
    print("  按 S → 截图保存")
    print("  按 Q → 退出")
    print()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 无法打开摄像头！")
        return

    # 设置分辨率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"✅ 摄像头已打开: {w}x{h}, {fps:.0f}fps")
    print()

    frame_count = 0
    save_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ 读取帧失败")
            break

        frame_count += 1

        # 左上角显示信息
        info = f"{w}x{h} | Frame: {frame_count}"
        cv2.putText(frame, info, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("AI小车 - 摄像头测试", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            print("👋 退出")
            break
        elif key == ord('s') or key == ord('S'):
            save_count += 1
            filename = f"capture_{int(time.time())}_{save_count}.jpg"
            filepath = os.path.join(SAVE_DIR, filename)
            cv2.imwrite(filepath, frame)
            print(f"📸 已保存: {filepath}")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n总共采集 {frame_count} 帧，保存了 {save_count} 张截图")

if __name__ == "__main__":
    main()
