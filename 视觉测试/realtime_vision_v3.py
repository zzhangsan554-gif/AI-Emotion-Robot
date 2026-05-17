#!/usr/bin/env python3
"""
realtime_vision_v3.py — 完整视觉模块测试（人体+表情+动作识别）
新增：挥手 / 点头 / 摇头 / 举手 动作识别
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
from collections import deque

SCRIPT_DIR = os.path.dirname(__file__)
SAVE_DIR = os.path.join(SCRIPT_DIR, "captures")
MODEL_DIR = os.path.join(SCRIPT_DIR, "models")
os.makedirs(SAVE_DIR, exist_ok=True)


# ==================== 动作识别 ====================

class ActionDetector:
    """基于Pose关键点时序的动作识别"""
    
    def __init__(self, history_len=15):
        self.nose_history = deque(maxlen=history_len)      # 鼻尖位置历史
        self.wrist_history = deque(maxlen=history_len)      # 手腕位置历史
        self.last_action = "NONE"
        self.last_action_time = 0
        self.cooldown = 1.0  # 动作冷却时间（秒）
    
    def detect(self, landmarks, w, h):
        """每帧调用，返回检测到的动作"""
        now = time.time()
        
        # 冷却期内不检测新动作
        if now - self.last_action_time < self.cooldown:
            return self.last_action
        
        # 鼻尖（landmark 0）
        nose = landmarks[0]
        # 左右手腕（landmark 15, 16）
        left_wrist = landmarks[15]
        right_wrist = landmarks[16]
        # 左右肩（landmark 11, 12）
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        
        # 记录历史
        self.nose_history.append((nose.x, nose.y, now))
        self.wrist_history.append({
            "lw": (left_wrist.x, left_wrist.y),
            "rw": (right_wrist.x, right_wrist.y),
            "ls": (left_shoulder.x, left_shoulder.y),
            "rs": (right_shoulder.x, right_shoulder.y),
            "time": now,
        })
        
        if len(self.nose_history) < 8:
            return "NONE"
        
        action = "NONE"
        
        # --- 挥手检测 ---
        action = self._detect_wave()
        if action != "NONE":
            self.last_action = action
            self.last_action_time = now
            return action
        
        # --- 点头检测 ---
        action = self._detect_nod()
        if action != "NONE":
            self.last_action = action
            self.last_action_time = now
            return action
        
        # --- 摇头检测 ---
        action = self._detect_shake()
        if action != "NONE":
            self.last_action = action
            self.last_action_time = now
            return action
        
        # --- 举手检测 ---
        action = self._detect_raise_hand()
        if action != "NONE":
            self.last_action = action
            self.last_action_time = now
            return action
        
        return "NONE"
    
    def _detect_wave(self):
        """挥手：手腕在肩膀上方 + 左右快速移动"""
        if len(self.wrist_history) < 8:
            return "NONE"
        
        recent = list(self.wrist_history)[-8:]
        
        for side, wk, sk in [("LEFT", "lw", "ls"), ("RIGHT", "rw", "rs")]:
            # 手腕必须在肩膀上方
            wrist_above = sum(1 for f in recent if f[wk][1] < f[sk][1] - 0.05) / len(recent)
            if wrist_above < 0.6:
                continue
            
            # 手腕x坐标变化方向翻转次数（挥手=来回摆动）
            xs = [f[wk][0] for f in recent]
            direction_changes = 0
            for i in range(2, len(xs)):
                d1 = xs[i-1] - xs[i-2]
                d2 = xs[i] - xs[i-1]
                if d1 * d2 < 0 and abs(d1) > 0.005 and abs(d2) > 0.005:
                    direction_changes += 1
            
            if direction_changes >= 2:
                return f"WAVE {side}"
        
        return "NONE"
    
    def _detect_nod(self):
        """点头：鼻尖y坐标先下降再上升"""
        if len(self.nose_history) < 8:
            return "NONE"
        
        recent = list(self.nose_history)[-8:]
        ys = [p[1] for p in recent]
        
        # 找到先下降后上升的模式
        min_idx = ys.index(min(ys))
        if 1 < min_idx < len(ys) - 2:
            # 下降幅度和上升幅度都要够大
            drop = ys[min_idx] - ys[0]
            rise = ys[-1] - ys[min_idx]
            if drop < -0.01 and rise > 0.01:
                return "NOD"
        
        return "NONE"
    
    def _detect_shake(self):
        """摇头：鼻尖x坐标左右摆动"""
        if len(self.nose_history) < 8:
            return "NONE"
        
        recent = list(self.nose_history)[-10:]
        xs = [p[0] for p in recent]
        
        # x方向变化翻转次数
        direction_changes = 0
        for i in range(2, len(xs)):
            d1 = xs[i-1] - xs[i-2]
            d2 = xs[i] - xs[i-1]
            if d1 * d2 < 0 and abs(d1) > 0.008 and abs(d2) > 0.008:
                direction_changes += 1
        
        if direction_changes >= 2:
            return "SHAKE HEAD"
        
        return "NONE"
    
    def _detect_raise_hand(self):
        """举手：手腕在头部上方"""
        if len(self.wrist_history) < 3:
            return "NONE"
        
        recent = list(self.wrist_history)[-3:]
        nose_y = self.nose_history[-1][1] if self.nose_history else 1
        
        for side, wk in [("LEFT", "lw"), ("RIGHT", "rw")]:
            hand_above = sum(1 for f in recent if f[wk][1] < nose_y - 0.08) / len(recent)
            if hand_above > 0.7:
                return f"HAND UP {side}"
        
        return "NONE"


# ==================== 表情识别 ====================

def get_expression_rich(blendshapes):
    if not blendshapes:
        return "NEUTRAL", (200, 200, 200)
    s = {bs.category_name: bs.score for bs in blendshapes}

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
    cheek_puff = s.get("cheekPuff", 0)
    mouth_funnel = s.get("mouthFunnel", 0)
    jaw_left = s.get("jawLeft", 0)
    jaw_right = s.get("jawRight", 0)

    scores = {}
    scores["HAPPY"] = smile*2 + jaw_open*0.5 + brow_outer_up*0.3 + cheek_puff*0.5
    scores["SMILE"] = smile*2 - jaw_open*1.0
    scores["SURPRISED"] = jaw_open*2 + brow_inner_up + eye_wide*1.5
    scores["SAD"] = mouth_frown*2 + brow_inner_up*0.8 + eye_squint*0.5 - smile*2
    scores["ANGRY"] = brow_down*2 + mouth_press*1.5 + nose_sneer*1.5 - smile*2
    scores["FEAR"] = brow_inner_up*1.5 + eye_wide + mouth_pucker - smile*2
    scores["DISGUST"] = nose_sneer*2 + mouth_funnel + brow_down*0.5 - smile*2
    scores["THINKING"] = abs(jaw_left-jaw_right)*2 + mouth_press*0.5
    scores["SLEEPY"] = (eye_blink_l+eye_blink_r)*0.75 + eye_squint*0.5 - smile
    scores["WINK"] = abs(eye_blink_l-eye_blink_r)*3 + smile*0.5
    scores["NEUTRAL"] = 0.1

    scores = {k: max(v, 0) for k, v in scores.items()}
    if min(eye_blink_l, eye_blink_r) < 0.5: scores["WINK"] = 0
    if jaw_open > 0.2: scores["SMILE"] = 0

    best = max(scores, key=scores.get)
    if scores[best] < 0.3: best = "NEUTRAL"

    colors = {
        "HAPPY":(0,255,0), "SMILE":(0,230,100), "SURPRISED":(0,200,255),
        "SAD":(200,100,0), "ANGRY":(0,0,255), "FEAR":(180,0,180),
        "DISGUST":(0,160,160), "THINKING":(200,200,0), "SLEEPY":(100,100,150),
        "WINK":(0,255,255), "NEUTRAL":(200,200,200),
    }
    return best, colors.get(best, (200,200,200))


# ==================== 人体位置 ====================

def get_position(landmarks, w):
    ls, rs = landmarks[11], landmarks[12]
    cx = (ls.x + rs.x) / 2
    if cx < 0.35: return "LEFT"
    elif cx > 0.65: return "RIGHT"
    else: return "CENTER"


# ==================== 绘制 ====================

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


# ==================== 主程序 ====================

def main():
    print("AI Robot Vision v3 - Full Test")

    pose_detector = PoseLandmarker.create_from_model_path(
        os.path.join(MODEL_DIR, "pose_landmarker.task"))
    face_options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=os.path.join(MODEL_DIR, "face_landmarker.task")),
        output_face_blendshapes=True, num_faces=3, min_face_detection_confidence=0.3,
    )
    face_detector = FaceLandmarker.create_from_options(face_options)
    action_detector = ActionDetector()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    w_frame = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_frame = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ts = int(time.time())
    video_path = os.path.join(SAVE_DIR, f"v3_full_{ts}.mp4")
    writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 20, (w_frame, h_frame))

    fps_timer, fps_count, display_fps = time.time(), 0, 0

    while True:
        ret, frame = cap.read()
        if not ret: break
        fps_count += 1
        h, w = frame.shape[:2]
        now = time.time()
        if now - fps_timer >= 1.0:
            display_fps = fps_count; fps_count = 0; fps_timer = now

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp_lib.Image(image_format=mp_lib.ImageFormat.SRGB, data=rgb)

        # ---- Pose ----
        pose_result = pose_detector.detect(mp_image)
        action_text = "NONE"
        action_color = (100, 100, 100)

        if pose_result.pose_landmarks:
            lm = pose_result.pose_landmarks[0]
            position = get_position(lm, w)
            spx = abs(lm[11].x - lm[12].x) * w
            draw_pose(frame, lm)

            # 动作识别
            action_text = action_detector.detect(lm, w, h)
            if action_text != "NONE":
                action_color = (0, 255, 255)  # 黄色

            cv2.putText(frame, f"Person: {position} ({spx:.0f}px)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # ---- Face ----
        face_result = face_detector.detect(mp_image)
        expr_text, expr_color = "", (200, 200, 200)

        if face_result.face_landmarks and len(face_result.face_landmarks) > 0:
            face_lm = face_result.face_landmarks[0]
            draw_face(frame, face_lm)
            if face_result.face_blendshapes and len(face_result.face_blendshapes) > 0:
                expr_text, expr_color = get_expression_rich(face_result.face_blendshapes[0])

        # ---- 显示表情 ----
        if expr_text:
            (tw, th), _ = cv2.getTextSize(expr_text, cv2.FONT_HERSHEY_SIMPLEX, 1.6, 3)
            cx = w // 2 - tw // 2
            cv2.rectangle(frame, (cx - 12, 45), (cx + tw + 12, 45 + th + 25), (30, 30, 30), -1)
            cv2.rectangle(frame, (cx - 12, 45), (cx + tw + 12, 45 + th + 25), expr_color, 3)
            cv2.putText(frame, expr_text, (cx, 45 + th + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.6, expr_color, 3)

        # ---- 显示动作 ----
        if action_text != "NONE":
            (aw, ah), _ = cv2.getTextSize(action_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)
            ay = h - 60
            ax = w // 2 - aw // 2
            cv2.rectangle(frame, (ax - 10, ay - 5), (ax + aw + 10, ay + ah + 15), (0, 0, 0), -1)
            cv2.putText(frame, action_text, (ax, ay + ah + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, action_color, 2)

        # ---- 信息 ----
        cv2.putText(frame, f"FPS: {display_fps}", (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, "Q to quit", (w - 110, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        # ---- 动作提示 ----
        cv2.putText(frame, "Try: Wave / Nod / Shake head / Raise hand", (10, h - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), 1)

        writer.write(frame)
        cv2.imshow("AI Robot - Vision v3 (Full)", frame)

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
