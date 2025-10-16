import base64
import hashlib
import io
import json
import os
from collections import OrderedDict

import dashscope
import folder_paths
import numpy as np
from PIL import Image


# ================================
# 插件配置
# ================================
PLUGIN_NAME = "ComfyUI-PromptPlusUtils"
PLUGIN_VERSION = "1.0.1"

# 在文件顶部或适当位置添加
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# API密钥文件路径
key_path = os.path.join(
    folder_paths.get_folder_paths("custom_nodes")[0],
    "ComfyUI-PromptPlusUtils",
    "api_key.txt"
)

# 确保目录存在
os.makedirs(os.path.dirname(key_path), exist_ok=True)

IMAGE_SYSTEM_PROMPT_ZH = '''
# 角色设定

你是一位资深的影视导演兼分镜师，精通视觉叙事语言与镜头逻辑。  
你擅长根据一张参考图片与一段场景描述，构建完整的叙事与视觉弧线，并将其转化为专业、连贯且富有电影感的分镜序列。


## 输入内容

- **参考图片**：提供画面基调与视觉元素以及人物特征，并且作为首帧。
- **场景描述**：提供叙事核心、人物关系与情绪走向。

## 总体目标

请从参考图中提取视觉、人物特征等风格，再基于场景描述（自动补全的影视化片段内容）规划出清晰的“叙事与视觉弧线”，最后生成一组逻辑连贯、节奏自然、情感递进的镜头序列。  
每个镜头都要体现出“故事推进 + 视觉设计 + 节奏感”的结合。

## 阶段一：视觉分析与叙事规划

### Step 1. 参考图片分析（作为分镜序列首帧）

分析参考图中的关键视觉信息，包括：

- 人物特征（外貌、姿态、情绪、关系）
- 环境元素（空间构成、背景物体、景深层次）
- 光影与色彩（光线方向、色调氛围、时间感）
- 构图语言（主视角、焦点、视觉引导线）
- 整体氛围（宁静 / 紧张 / 温暖 / 神秘 等）
- 提炼视觉风格关键词（如“昏黄街灯”“极简构图”“潮湿夜色”等）；  

根据以上分析，输出一句 `first_frame_prompt`（中文）
后续分镜需在叙事、空间与氛围上自然衔接首帧画面。

### Step 2. 叙事与视觉弧线规划

用户输入的“场景描述”可能简短或抽象。  
你的首要任务是根据该描述自动补全一个完整的影视化片段，包括：

- 角色设定（是谁？动机是什么？）
- 时间与空间（在哪？什么时间？）
- 叙事起点与结尾（这件事如何开始、如何结束？）

并在脑中规划出一条“叙事与视觉弧线”的总体叙事规划，明确：

- **核心事件**：这一段主要讲述了什么？  
  （例如：“两人在操场自拍时情感逐渐升温”）
- **情绪曲线（起 → 承 → 转 → 合）**：  
  （例如：“轻松 → 亲密 → 紧张 → 平静”）
- **视觉节奏规划**：  
  镜头景别和运动的变化如何呼应情绪？  
  （例如：“远景建立 → 中景互动 → 特写情感 → 广角收尾”）  
  或（例如：“静态建立 → 快速运动 → 动作高潮 → 轻松收尾”）

以此生成一段 `visual_narrative_plan`，清晰描述整段片段的视觉走向与节奏逻辑。

## 阶段二：分镜序列生成

请基于首帧分析与叙事规划，以及 `visual_narrative_plan`，生成一个连续且视觉节奏自然的分镜序列。

每个镜头必须以 “Next Scene:” 开头，并且包含以下元素但不需要格式化，并且最终需要把这些元素进行整合，整理成自然语言或者流畅的导演指令语言，且整理成一行内容，即一个分镜一行内容：

**Next Scene:**  
整合以下元素为一行导演指令语言：

- 景别（Shot Size）：远景 / 全景 / 中景 / 近景 / 特写 / 极特写
- 摄影语言：（示例：手持跟拍 / 稳定器运镜 / 长焦压缩 / 广角动态等）
- 摄影机角度（Camera Angle）：平视 / 俯拍 / 仰拍 / 主观视角 / 侧角
- 摄影机运动（Camera Motion）：固定 / 推镜 / 拉镜 / 跟拍 / 移动 / 摇摄 / 升降 / 环绕
- 构图（Composition）：三分法 / 对称构图 / 主体居中 / 景深层次 / 遮挡前景
- 光线类型（Lighting）：自然光 / 逆光 / 顶光 / 侧光 / 柔光 / 冷光 / 人工光
- 色调与氛围（Tone & Mood）：暖调 / 冷调 / 高对比 / 柔和 / 阴影 / 日落 / 夜色
- 环境描述（Environment Details）：空间、场景、时间、天气、氛围，环境背景下的道具（或物体、其他新增的人物）的细节描述
- 人物动作与情绪（Character Action & Emotion）：人物的动作、表情、互动
- 叙事作用（Narrative Purpose）：此镜头的剧情功能（如建立、推进、冲突、情绪转折、结尾）
- 衔接方式（Transition Type）：剪切 / 动作匹配 / 溶解 / 拉镜过渡 / 视线匹配 / 蒙太奇
- 节奏控制（Rhythm）：慢 / 中 / 快（与情绪曲线一致）
- 音效与氛围声（Optional）：环境音 / 呼吸声 / 背景音乐情绪

### 分镜生成要求

- 所有镜头必须自然衔接，保持时空与情绪的连续性。
- 分镜序列内容，一个分镜一行内容，且保证自然语言或者流畅的导演指令语言
- 不得跳时、跳地或变光线基调。
- 整体节奏从首帧的“静态氛围”逐步过渡到“剧情推动”或“情绪爆发”。
- 镜头间过渡需自然，可使用视觉元素（光影、动作、音乐）形成连续性。
- 最后一个镜头应有视觉或情绪收尾感。

'''

IMAGE_SYSTEM_PROMPT_EN = '''
# Character Setting

You are an experienced film director and storyboard artist, proficient in visual narrative language and shot logic.
You excel at constructing complete narrative and visual arcs based on a reference image and a scene description, transforming them into professional, coherent, and cinematic storyboard sequences.

---

## Input Content

- **Reference Image**: Provides the visual tone, visual elements, and character traits, and serves as the first frame.
- **Scene Description**: Provides the narrative core, character relationships, and emotional direction.

---

## Overall Objective

Extract the visual and character style from the reference image, then plan a clear "narrative and visual arc" based on the scene description (automatically completed cinematic segment content), and finally generate a set of logically coherent, naturally paced, and emotionally progressive shot sequences.
Each shot must reflect the combination of "story progression + visual design + rhythm."

---

## Phase 1: Visual Analysis and Narrative Planning

### Step 1. Reference Image Analysis (Serves as the First Frame of the Storyboard Sequence)

Analyze key visual information in the reference image, including:

- Character traits (appearance, posture, emotion, relationships)
- Environmental elements (spatial composition, background objects, depth of field)
- Lighting and color (light direction, color tone, sense of time)
- Composition language (main perspective, focus, visual guiding lines)
- Overall atmosphere (calm / tense / warm / mysterious, etc.)

Based on the above analysis, output a `first_frame_prompt` (in English), extracting visual style keywords (e.g., "dim streetlights," "minimalist composition," "damp night," etc.);
Subsequent shots must naturally connect with the first frame in terms of narrative, space, and atmosphere.

---

### Step 2. Narrative and Visual Arc Planning

The user's "scene description" may be brief or abstract.
Your primary task is to automatically complete a full cinematic segment based on this description, including:

- Character setting (Who are they? What is their motivation?)
- Time and space (Where? When?)
- Narrative start and end (How does the event begin and end?)

And mentally plan an overall narrative structure for a "narrative and visual arc," clarifying:

- **Core Event**: What is the main focus of this segment?
  (e.g., "Two people's emotions gradually升温 while taking selfies on the playground.")
- **Emotional Curve (Beginning → Development → Turn → Resolution)**:
  (e.g., "Lighthearted → Intimate → Tense → Calm")
- **Visual Rhythm Planning**:
  How do changes in shot size and movement correspond to the emotions?
  (e.g., "Establish with wide shot → Interact with medium shot → Emotion with close-up → Conclude with wide angle")
  Or (e.g., "Static establishment → Rapid movement → Action climax → Light conclusion")

Use this to generate a `visual_narrative_plan`, clearly describing the visual direction and rhythm logic of the entire segment.

---

## Phase 2: Storyboard Sequence Generation

Based on the first frame analysis, narrative planning, and the `visual_narrative_plan`, generate a continuous and visually rhythmic storyboard sequence.

Each shot must start with "Next Scene:" and include the following elements without formatting. Ultimately, integrate these elements into natural language or smooth director's instruction language, with each shot condensed into a single line:

**Next Scene:**
Integrate the following elements into one line of director's instruction language:

- Shot Size: Extreme Long Shot / Long Shot / Medium Shot / Close-Up / Extreme Close-Up
- Cinematic Language: (e.g., Handheld follow / Gimbal movement / Telephoto compression / Wide-angle dynamic, etc.)
- Camera Angle: Eye-Level / High Angle / Low Angle / Point-of-View / Dutch Angle
- Camera Motion: Fixed / Dolly In / Dolly Out / Tracking / Trucking / Pan / Tilt / Pedestal / Orbit
- Composition: Rule of Thirds / Symmetrical / Centered / Depth Layering / Foreground Framing
- Lighting Type: Natural Light / Backlight / Top Light / Side Light / Soft Light / Cool Light / Artificial Light
- Tone & Mood: Warm Tone / Cool Tone / High Contrast / Soft / Shadowy / Sunset / Night
- Environment Details: Space, setting, time, weather, atmosphere, details of props (or objects, other additional characters) in the background
- Character Action & Emotion: Character's actions, expressions, interactions
- Narrative Purpose: The plot function of this shot (e.g., establishment, progression, conflict, emotional turn, conclusion)
- Transition Type: Cut / Match Cut / Dissolve / Whip Pan / Eye-Line Match / Montage
- Rhythm Control: Slow / Medium / Fast (consistent with the emotional curve)
- Sound & Ambiance (Optional): Ambient sound / Breathing / Background music emotion

---

### Storyboard Generation Requirements

- All shots must connect naturally, maintaining continuity in time, space, and emotion.
- Do not jump in time, location, or change lighting tone abruptly.
- The overall rhythm should gradually transition from the "static atmosphere" of the first frame to "plot propulsion" or "emotional climax."
- Transitions between shots should be natural, using visual elements (lighting, action, music) to create continuity.
- The final shot should provide a visual or emotional sense of conclusion.
'''

# ================================
# 缓存系统
# ================================
class PromptCache:
    def __init__(self, max_size=100):
        self.cache = OrderedDict()
        self.max_size = max_size
        self.hit_count = 0
        self.miss_count = 0

    def _generate_image_hash(self, image_tensor):
        """生成图像内容的感知哈希"""
        try:
            # 将tensor转换为PIL图像
            pil_images = tensor2pil(image_tensor)
            if not pil_images:
                return "no_image"

            # 使用第一张图像生成哈希
            img = pil_images[0]

            # 缩小图像以生成感知哈希（对微小变化不敏感）
            img_small = img.resize((8, 8), Image.LANCZOS).convert('L')  # 转为灰度
            pixels = list(img_small.getdata())

            # 计算平均值
            avg = sum(pixels) / len(pixels)

            # 生成哈希：大于平均值为1，否则为0
            hash_str = ''.join('1' if pixel > avg else '0' for pixel in pixels)

            # 转为16进制存储
            return hashlib.md5(hash_str.encode()).hexdigest()

        except Exception as e:
            print(f"[{PLUGIN_NAME}] Error generating image hash: {e}")
            return "error_hash"

    def _generate_cache_key(self, prompt, image, model, mode):
        """生成缓存键"""
        prompt_part = hashlib.md5(prompt.strip().encode('utf-8')).hexdigest()
        model_part = model
        mode_part = mode

        if image is not None and mode == "image-to-image":
            image_part = self._generate_image_hash(image)
            return f"{prompt_part}_{image_part}_{model_part}_{mode_part}"
        else:
            return f"{prompt_part}_{model_part}_{mode_part}"

    def get(self, prompt, image, model, mode):
        """从缓存获取结果"""
        key = self._generate_cache_key(prompt, image, model, mode)

        if key in self.cache:
            # 移动到最近使用
            value = self.cache.pop(key)
            self.cache[key] = value
            self.hit_count += 1
            print(f"[{PLUGIN_NAME}] Cache hit! Key: {key[:16]}...")
            return value

        self.miss_count += 1
        return None

    def set(self, prompt, image, model, mode, result):
        """设置缓存结果"""
        key = self._generate_cache_key(prompt, image, model, mode)

        # 如果达到最大大小，移除最旧的条目
        if len(self.cache) >= self.max_size:
            self.cache.popitem(last=False)

        self.cache[key] = result
        print(f"[{PLUGIN_NAME}] Cached result. Cache size: {len(self.cache)}")

    def clear(self):
        """清空缓存"""
        cache_size = len(self.cache)
        self.cache.clear()
        self.hit_count = 0
        self.miss_count = 0
        print(f"[{PLUGIN_NAME}] Cache cleared. Previous size: {cache_size}")

    def get_stats(self):
        """获取缓存统计"""
        total = self.hit_count + self.miss_count
        hit_rate = (self.hit_count / total * 100) if total > 0 else 0
        return {
            "cache_size": len(self.cache),
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": hit_rate
        }


# 全局缓存实例
prompt_cache = PromptCache(max_size=100)




def encode_image(pil_image, save_tokens=True):
    buffered = io.BytesIO()
    if save_tokens:
        image = resize_to_limit(pil_image)
        image.save(buffered, format="JPEG", optimize=True, quality=75)
    else:
        pil_image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def resize_to_limit(img, max_pixels=262144):
    width, height = img.size
    total_pixels = width * height

    if total_pixels <= max_pixels:
        return img

    scale = (max_pixels / total_pixels) ** 0.5
    new_width = int(width * scale)
    new_height = int(height * scale)
    return img.resize((new_width, new_height), Image.LANCZOS)


def tensor2pil(image):
    batch_count = image.size(0) if len(image.shape) > 3 else 1
    if batch_count > 1:
        out = []
        for i in range(batch_count):
            out.extend(tensor2pil(image[i]))
        return out
    return [Image.fromarray(np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))]


def get_caption_language(prompt):
    ranges = [
        ('\u4e00', '\u9fff'),  # CJK Unified Ideographs
    ]
    for char in prompt:
        if any(start <= char <= end for start, end in ranges):
            return 'zh'
    return 'en'


def api_edit(prompt, img_list, model="qwen-vl-max-latest", save_tokens=True, api_key=None, kwargs={}):
    if not api_key:
        raise EnvironmentError("API_KEY is not set!")

    print(f'Using "{model}" for prompt rewriting...')
    assert model in ["qwen-vl-max", "qwen-vl-max-latest", "qwen-vl-max-2025-08-13",
                     "qwen-vl-max-2025-04-08"], f'"{model}" is not available for the "Qwen-Image-Edit" style.'
    sys_promot = "you are a helpful assistant, you should provide useful answers to users."
    messages = [
        {"role": "system", "content": sys_promot},
        {"role": "user", "content": []}]
    for img in img_list:
        messages[1]["content"].append(
            {"image": f"data:image/png;base64,{encode_image(img, save_tokens=save_tokens)}"})
    messages[1]["content"].append({"text": f"{prompt}"})

    response_format = kwargs.get('response_format', None)

    response = dashscope.MultiModalConversation.call(
        api_key=api_key,
        model=model,
        # For example, use qwen-plus here. You can change the model name as needed. Model list: https://help.aliyun.com/zh/model-studio/getting-started/models
        messages=messages,
        result_format='message',
        response_format=response_format,
    )

    if response.status_code == 200:
        return response.output.choices[0].message.content[0]['text']
    else:
        raise Exception(f'Failed to post: {response}')


def polish_prompt_edit(api_key, prompt, img, model="qwen-vl-max-latest", max_retries=10, save_tokens=True):
    retries = 0
    prompt_text = f"{IMAGE_SYSTEM_PROMPT_EN}\n\nUser Input: {prompt}\n\nRewritten Prompt:"

    while retries < max_retries:
        try:
            result = api_edit(prompt_text, img, model=model, save_tokens=save_tokens, api_key=api_key)

            if isinstance(result, str):
                result = result.replace('```json', '').replace('```', '')
                result = json.loads(result)
            else:
                result = json.loads(result)

            polished_prompt = result['Rewritten'].strip().replace("\n", " ")
            return polished_prompt
        except Exception as e:
            error = e
            retries += 1
            print(f"[Warning] Error during API call (attempt {retries}/{max_retries}): {e}")

    raise EnvironmentError(f"Error during API call: {error}")


def api(prompt, model, api_key=None, kwargs={}):
    if not api_key:
        raise EnvironmentError("API_KEY is not set!")

    print(f'Using "{model}" for prompt rewriting...')
    assert model in ["qwen-vl-max", "qwen-vl-max-latest", "qwen-vl-max-2025-08-13", "qwen-plus", "qwen-max",
                     "qwen-plus-latest", "qwen-max-latest"], f'"{model}" is not available for the "Qwen-Image" style.'
    messages = [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': prompt}
    ]

    response_format = kwargs.get('response_format', None)

    response = dashscope.Generation.call(
        api_key=api_key,
        model=model,
        # For example, use qwen-plus here. You can change the model name as needed. Model list: https://help.aliyun.com/zh/model-studio/getting-started/models
        messages=messages,
        result_format='message',
        response_format=response_format,
    )

    if response.status_code == 200:
        return response.output.choices[0].message.content
    else:
        raise Exception(f'Failed to post: {response}')


def polish_prompt(api_key, prompt, model="qwen-plus", max_retries=10):
    retries = 0
    lang = get_caption_language(prompt)
    system_prompt = IMAGE_SYSTEM_PROMPT_ZH if lang == 'zh' else IMAGE_SYSTEM_PROMPT_EN
    magic_prompt = "超清，4K，电影级构图" if lang == 'zh' else "Ultra HD, 4K, cinematic composition"

    prompt_text = f"{system_prompt}\n\nUser Input: {prompt}\n\nRewritten Prompt:"

    while retries < max_retries:
        try:
            result = api(prompt_text, model=model, api_key=api_key)
            polished_prompt = result.strip().replace("\n", " ")
            return polished_prompt + magic_prompt
        except Exception as e:
            error = e
            retries += 1
            print(f"[Warning] Error during API call (attempt {retries}/{max_retries}): {e}")

    raise EnvironmentError(f"Error during API call: {error}")

def get_api_key(api_key_input):
    """获取API密钥，优先使用输入，其次使用文件"""
    _api_key = api_key_input.strip()
    if _api_key:
        print(f"[{PLUGIN_NAME}] Using API key from input")
        return _api_key

    if os.path.exists(key_path):
        try:
            with open(key_path, "r", encoding="utf-8") as f:
                _api_key = f.read().strip()
            if _api_key:
                print(f"[{PLUGIN_NAME}] Using API key from file: {key_path}")
                return _api_key
            else:
                print(f"[{PLUGIN_NAME}] API key file is empty")
        except Exception as e:
            print(f"[{PLUGIN_NAME}] Error reading API key file: {e}")
    else:
        print(f"[{PLUGIN_NAME}] API key file not found: {key_path}")

    return None


class StoryboardPromptFirstFrameHelper:
    """
    StoryboardPromptHelper v1.0.0
    基于用户输入文本和首帧图片生成：
    1. 分镜提示词汇总（一行一个）
    2. 首帧图反推图片提示词
    """

    @classmethod
    def INPUT_TYPES(s):
        vision_models = ["qwen-vl-max", "qwen-vl-max-latest", "qwen-vl-max-2025-08-13", "qwen-vl-max-2025-04-08"]
        text_models = ["qwen-plus", "qwen-max", "qwen-plus-latest", "qwen-max-latest"]

        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "", "tooltip": "用户描述文本"}),
                "model": (text_models + vision_models, {"default": "qwen-vl-max-latest", "tooltip": "选择模型"}),
                "api_key": ("STRING", {"default": "", "tooltip": "阿里云API密钥"}),
                "max_retries": ("INT", {"default": 3, "min":1, "max":10, "tooltip":"最大重试次数"}),
            },
            "optional": {
                "image": ("IMAGE", {"tooltip": "首帧图片"})
            }
        }

    RETURN_TYPES = ("STRING","STRING","INT")
    RETURN_NAMES = ("storyboard_prompts","first_frame_prompt","storyboard_count")
    FUNCTION = "generate_storyboard_prompts"
    CATEGORY = "AI Tools/Prompt"
    DESCRIPTION = "StoryboardPromptFirstFrameHelper v1.0.0 - 分镜提示词生成器-首帧版"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    def generate_storyboard_prompts(self, prompt, model, api_key, max_retries, image=None):
        """
        核心处理逻辑：
        1. 构建完整 LLM 提示词
        2. 调用 API 获取 JSON 返回
        3. 解析 JSON 并返回两个字符串
        """
        _api_key = get_api_key(api_key)
        if not _api_key:
            raise EnvironmentError("API_KEY未设置！")

        lang = get_caption_language(prompt)
        system_prompt = IMAGE_SYSTEM_PROMPT_ZH if lang == 'zh' else IMAGE_SYSTEM_PROMPT_EN

        # 构建 LLM 提示词，要求 JSON 返回两个字段
        llm_prompt = f"""
    {system_prompt}
    用户输入: {prompt}
    请输出JSON，格式如下:
    {{
        "storyboard_prompts": "每行一个分镜提示词，文本中保证没有空白行，整理成以一个分镜一行，并且把语句进行优化，每行以 'Next Scene' 开头",
        "first_frame_prompt": "基于首帧图反推出来的图片提示词"
    }}
    注意：直接输出JSON，不要添加额外文本
        """

        # 图像处理
        img_list = tensor2pil(image) if image is not None else None

        # 调用 Qwen LLM
        retries = 0
        while retries < max_retries:
            try:
                result_str = api_edit(llm_prompt, img_list, model=model, save_tokens=True, api_key=_api_key)

                if isinstance(result_str, str):
                    result_str = result_str.replace('```json','').replace('```','').strip()
                    result = json.loads(result_str)
                else:
                    result = result_str

                raw_storyboard = result.get("storyboard_prompts","").strip()

                print("=== RAW STORYBOARD ===")
                print(repr(raw_storyboard))
                print("=== END RAW ===")

                # 简洁且安全的方法
                if raw_storyboard.startswith("Next Scene"):
                    # 直接替换，但更精确地处理
                    storyboard_prompts = raw_storyboard.replace("Next Scene", "\nNext Scene")
                    # 移除开头的换行符和所有尾随空白
                    storyboard_prompts = storyboard_prompts.lstrip('\n').strip()
                else:
                    storyboard_prompts = raw_storyboard

                # 最后再清理一次可能的连续空白行
                lines = storyboard_prompts.splitlines()
                cleaned_lines = [line.strip() for line in lines if line.strip()]
                storyboard_prompts = "\n".join(cleaned_lines)

                print("=== PROCESSED STORYBOARD ===")
                print(storyboard_prompts)
                print("=== END PROCESSED ===")

                # storyboard_count 分镜数量 根据"Next Scene"个数来统计
                storyboard_count = len(lines)
                first_frame_prompt = result.get("first_frame_prompt","").strip()
                return storyboard_prompts, first_frame_prompt, storyboard_count

            except Exception as e:
                print(f"[StoryboardPromptHelper] API调用失败，第{retries+1}次重试: {e}")
                retries += 1

        raise EnvironmentError("API调用失败，超过最大重试次数")


# 注册节点到 ComfyUI
NODE_CLASS_MAPPINGS["StoryboardPromptFirstFrameHelper"] = StoryboardPromptFirstFrameHelper
NODE_DISPLAY_NAME_MAPPINGS["StoryboardPromptFirstFrameHelper"] = "StoryboardPromptFirstFrameHelper v1.0.0 - 分镜提示词生成器-首帧版"

print("\033[1;34m[StoryboardPromptFirstFrameHelper] 节点注册完成，可在 ComfyUI 中使用!\033[0m")