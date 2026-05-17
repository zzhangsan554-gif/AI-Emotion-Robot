# AI情绪感知巡逻陪伴机器人 🤖

> 一个能在实验室/办公室巡逻的AI小车，能听、能看、能感知你的情绪，像一个有情感的AI伙伴。

## 核心体验

用户视角：**就是跟一个AI正常聊天，这个AI碰巧有身体，能真的动。**

- 🗣️ 贾维斯级语音交互：自然对话，不需要唤醒词
- 👀 摄像头看表情 + 语音双模态情绪判断
- 🧠 大模型理解语义（不是命令词），各种说法都能理解
- 🔧 自绘PCB硬件，完整软硬件系统

## 硬件

- **主控：** 树莓派5 4GB
- **下位机：** STM32F103C8T6（自绘PCB）
- **语音：** USB麦克风 + 3W扬声器
- **视觉：** USB摄像头
- **电机：** 4x麦克拉姆轮 + TB6612FNG驱动
- **传感器：** 3x超声波 + JY61P IMU

## 项目结构

```
├── 技术方案-v3.1.md          # 完整技术方案（最新）
├── 技术方案-v3.0.md          # 上一版方案
├── YOLO-vs-MediaPipe对比.md  # 视觉方案对比
├── 情绪映射逻辑说明.md
├── USB麦克风和喇叭推荐.md
├── 回音处理方案.md
├── 已确认设备.md
├── 树莓派蓝牙音箱连接指南.md
├── 淘宝采购链接.md
├── 语音交互方案调研-2026-04-24.md
├── GLM-4-Voice-Prompt和上下文管理.md
│
├── 视觉测试/                  # MediaPipe视觉模块
│   ├── realtime_vision.py     # 实时人体检测+表情识别（带GUI）
│   ├── auto_vision_test.py    # 后台自动测试
│   ├── auto_vision_record.py  # 录制+检测
│   ├── auto_capture.py        # 快速拍照
│   ├── test1_camera_capture.py
│   ├── models/                # MediaPipe模型（需单独下载）
│   └── captures/              # 截图/视频（git忽略）
│
├── 音频测试/                   # 豆包端到端语音测试
│   ├── test_01_connect.py     # WebSocket连接
│   ├── test_02_text_chat.py   # 文字聊天
│   ├── test_04_full.py        # 完整语音交互
│   ├── test_06_inject.py      # RAG注入
│   ├── mic_voice_chat.py      # 麦克风语音聊天
│   ├── streaming_player.py    # 流式音频播放
│   └── ...
│
└── 归档/                      # 历史版本
```

## 视觉模块

基于 **MediaPipe Tasks API**：

| 功能 | 说明 | 状态 |
|------|------|------|
| 人体检测 | Pose Landmarker，33个关键点 | ✅ 完成 |
| 位置判断 | 左/中/右 | ✅ 完成 |
| 表情识别 | Face Landmarker + Blendshapes | ✅ 完成 |
| 录制回放 | 视频录制+状态JSON | ✅ 完成 |

**支持的表情：** HAPPY / SMILE / SURPRISED / SAD / NEUTRAL

### 视觉模块使用

```bash
# 安装依赖
pip install opencv-python mediapipe

# 下载模型（首次）
mkdir -p 视觉测试/models
# Pose模型
curl -o 视觉测试/models/pose_landmarker.task \
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
# Face模型
curl -o 视觉测试/models/face_landmarker.task \
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"

# 运行实时测试
cd 视觉测试
python3 realtime_vision.py
```

## 语音架构

- **豆包端到端全双工** — 唯一的嘴和耳朵
- **OpenClaw/GLM-4.7** — 后台安全兜底
- **Python主程序** — 硬件协调+意图解析

## 技术栈

- Python 3.9+
- OpenCV
- MediaPipe 0.10+
- STM32 HAL
- Flask（上位机）
- WebSocket（语音通信）

## License

MIT
