# GLM-4-Voice 的 Prompt 和 上下文管理

> 创建时间: 2026-04-22

## 一、GLM-4-Voice 的"内置提示词"是什么？

GLM-4-Voice 确实有自己的系统人设：

- 一个语音对话AI
- 有默认的说话风格和语气

**但我们不需要了解它默认长什么样！** 我们每次调用时都可以覆盖它。

---

## 二、我们需要做什么？

### 2.1 构建每次的完整 Prompt

每次调用GLM-4-Voice时，我们需要传一个**完整的prompt**，包含：

```python
prompt = f"""{system_prompt}

当前状态：
- 用户情绪：{emotion}（由视觉表情+语音语气判断）
- 当前场景：{scene}（巡逻/跟随/交互/静默）
- 当前时间：{time}
- 用户位置：{user_pos}（来自视觉）

对话历史：
{conversation_history}
最近3-5轮对话...

{dynamic_instruction}
- 音色要求："用温柔的御姐音回复，语速放慢"
- 或："用欢快的少女音回复"
- 或："用低沉的大叔音回复，语调平和"

请根据当前状态和对话历史，给予合适的回应。
"""
```

### 2.2 三个层次的 Prompt

#### 系统提示词（System Prompt）- 每次都固定

```python
system_prompt = """
你是"EmoBot"，一个实验室AI陪伴机器人。

核心职责：
- 理解用户的语义内容（说了什么）
- 感知用户的情绪状态（通过语音语气和面部表情）
- 给予合适的、有温度的回应

人设特点：
- 有个性但不过度拟人化
- 有边界，不回答不当内容
- 简洁明了，回复不超过3句话

音色与语气：
- 默认用中性的御姐音
- 可以根据情绪调整为温柔/欢快/平和
- 可以根据语速调整为慢速/正常/快速

场景感知：
- 巡逻时：主动观察，不打扰
- 跟随时：跟随但不贴太近
- 交互时：停下，专注对话
- 静默时：不说话，只做指示（转向/停止）
"""
```

#### 状态信息（Context）- 每次实时更新

- 当前视觉情绪（来自表情识别）
- 当前用户位置（来自YOLO人体检测）
- 当前运动状态（巡逻/跟随/静止）
- 当前时间
- 电池电量

#### 对话历史（Memory）- 自己管理

GLM-4-Voice 上下文窗口只有 8K tokens：

- 大约能存 3-5 轮对话（每次约 1.5K）
- 我们需要自己维护一个对话历史列表

```python
class ConversationHistory:
    def __init__(self, max_tokens=8000):
        self.history = []  # [(role, content)]
        self.max_tokens = max_tokens

    def add(self, role, content):
        """添加一轮对话"""
        tokens = self._estimate_tokens(content)
        # 如果超过上限，删除最旧的对话
        while self._estimate_total() + tokens > self.max_tokens:
            self.history.pop(0)  # 删最老的一轮

        self.history.append((role, content))

    def get_context(self):
        """获取上下文文本"""
        return "\n".join([
            f"{role}: {content}"
            for role, content in self.history
        ])

    def _estimate_tokens(self, text):
        """估算token数量（中文约 1:1.6）"""
        return len(text) * 1.6

# 使用
conversation = ConversationHistory(max_tokens=8000)

# 添加用户输入
conversation.add("user", "我好累啊")

# 添加模型回复
conversation.add("assistant", "你已经学了很久了，休息一下吧")

# 获取上下文（发给GLM-4-Voice）
context = conversation.get_context()
```

#### 动态指令（Dynamic Instruction）- 根据情绪/场景调整

```python
def get_dynamic_instruction(emotion, visual_emotion, scene):
    """根据情绪生成动态指令"""

    if emotion == "疲惫" and visual_emotion == "疲惫":
        return """
        用温柔关心的御姐音回复，语速放慢一点。
        小车缓慢靠近用户，表示关心。
        """

    elif emotion == "开心":
        return """
        用欢快活泼的少女音回复，语速正常稍快。
        小车可以转个圈或靠近，表示陪伴的快乐。
        """

    elif emotion == "愤怒":
        return """
        用平和低沉的大叔音回复，语速放慢，语调缓和。
        小车后退，不要火上浇油。
        保持安静，听用户发泄。
        """

    elif scene == "巡逻":
        return """
        你在巡逻模式，观察环境。
        如果发现人，切换到跟随模式。
        不要主动说话。
        """

    elif scene == "跟随":
        return """
        你在跟随模式。
        保持适当的距离（1-2米），不要太近也不要太远。
        如果用户说话，停下来回应。
        """

    else:
        return """
        用中性的御姐音回复，语速正常。
        根据用户情绪自然调整。
        """
```

---

## 三、完整的调用流程

```python
from zai import ZhipuAiClient
import base64

# 1. 初始化
conversation = ConversationHistory(max_tokens=8000)
client = ZhipuAiClient(api_key="your_api_key")

def process_user_audio(audio_file, visual_emotion, scene):
    """处理用户语音"""

    # 2. 读取音频
    with open(audio_file, "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode()

    # 3. 估算用户情绪（基于语音语气+视觉表情）
    user_emotion = detect_user_emotion(audio_file, visual_emotion)

    # 4. 获取上下文（对话历史）
    context = conversation.get_context()

    # 5. 生成动态指令
    dynamic_instruction = get_dynamic_instruction(user_emotion, visual_emotion, scene)

    # 6. 构建完整prompt
    prompt = f"""{system_prompt}

当前状态：
- 用户情绪：{user_emotion}
- 视觉情绪：{visual_emotion}
- 当前场景：{scene}
- 对话历史：
{context}

{dynamic_instruction}
请根据以上信息，给予合适的回应。回复要简短（1-3句话）。
"""

    # 7. 调用GLM-4-Voice
    response = client.chat.completions.create(
        model="glm-4-voice",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "input_audio", "input_audio": {"data": audio_b64, "format": "wav"}}
            ]
        }],
        stream=True
    )

    # 8. 提取音频
    for chunk in response:
        if chunk.choices[0].delta.audio:
            # 流式播放
            audio_data = base64.b64decode(chunk.choices[0].delta.audio['data'])
            # 播放 audio_data...

    # 9. 添加到对话历史
    final_text = extract_text_from_response(response)
    conversation.add("assistant", final_text)

    return final_text
```

---

## 四、关键点总结

| 问题                   | 答案                                           |
| -------------------- | -------------------------------------------- |
| GLM-4-Voice需要内置提示词吗？ | 不需要！我们每次调用都传完整的prompt（系统人设+状态+历史+动态指令）       |
| 啥的？                  | 啥的是"系统人设""动态指令"，不是"提示词"，但作用类似                |
| 还是说直接调用模型就好？         | 可以直接调，但需要管理上下文（对话历史）                         |
| 上下文怎么记忆？             | 自己维护对话历史，每次调用时把最近3-5轮对话传进去（8K tokens约够存3-5轮） |
| 需要自己管理吗？             | 是！GLM-4-Voice没有对话记忆，我们必须自己维护                 |

---

## 五、简化版（如果不需要复杂记忆）

如果刀哥觉得上面太复杂，可以简化：

```python
# 最简版本：只传用户情绪和当前对话（最近2-3轮）
def simple_call(audio_b64, emotion, history_text):
    prompt = f"""你是EmoBot。

当前用户情绪：{emotion}
{history_text}

用温柔关心的御姐音回复，语速放慢。回复简短（1-2句话）。"""

    response = client.chat.completions.create(
        model="glm-4-voice",
        messages=[
            {"type": "text", "text": prompt},
            {"type": "input_audio", "input_audio": {"data": audio_b64, "format": "wav"}}
        ]
    )

    return response
```

**建议：先用简化版，后期如果需要"多轮对话记忆"再加完整版。**
