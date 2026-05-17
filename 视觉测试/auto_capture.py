#!/usr/bin/env python3
"""
auto_capture.py — 后台自动拍照测试（不弹窗口，不需要GUI）
拍一张照片保存，用来验证摄像头是否正常工作
"""

import cv2
import os
import time

SAVE_DIR = os.path.join(os.path.dirname(__file__), "captures")
os.makedirs(SAVE_DIR, exist_ok=True)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ 无法打开摄像头")
    exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# 丢掉前几帧（摄像头刚启动可能模糊）
for _ in range(5):
    cap.read()

ret, frame = cap.read()
if ret:
    filepath = os.path.join(SAVE_DIR, "auto_test.jpg")
    cv2.imwrite(filepath, frame)
    h, w = frame.shape[:2]
    print(f"✅ 拍照成功: {filepath} ({w}x{h})")
else:
    print("❌ 拍照失败")

cap.release()
