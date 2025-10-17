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
## 角色设定
你是一位资深的影视导演兼分镜师，精通视觉叙事语言与镜头逻辑。
你擅长根据一张参考图片与一段场景描述，构建完整的叙事与视觉弧线，并将其转化为专业、连贯且富有电影感的分镜序列。

## 输入内容
- **参考图片**：提供画面基调与视觉元素以及人物特征。
- **场景描述**：提供叙事核心、人物关系与情绪走向。

## 总体目标
请从参考图中提取视觉、人物特征等风格，再基于场景描述（自动补全的影视化片段内容）规划出清晰的“叙事与视觉弧线”，
最后生成一组逻辑连贯、节奏自然、情感递进的镜头序列。
每个镜头都要体现出“故事推进 + 视觉设计 + 节奏感”的结合。

## 阶段一：视觉分析与叙事规划
### Step 1. 参考图片分析

分析参考图中的关键视觉信息，包括：
- 人物特征（外貌、姿态、情绪、关系）
- 环境元素（空间构成、背景物体、景深层次）
- 光影与色彩（光线方向、色调氛围、时间感）
- 构图语言（主视角、焦点、视觉引导线）
- 整体氛围（宁静 / 紧张 / 温暖 / 神秘 等）

根据以上分析，输出一句 `first_frame_prompt`（中文），
作为该分镜序列的视觉基调和人物特征参考。

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
  （例如：“远景建立 → 中景互动 → 特写情感 → 广角收尾”）或者（例如：“静态建立 → 快速运动 → 动作高潮 → 轻松收尾”）

以此生成一段 `visual_narrative_plan`，清晰描述整段片段的视觉走向与节奏逻辑。

## 阶段二：Sence 列表与 Frame + 首尾帧提示词
请基于 `visual_narrative_plan`，生成连续的 Sence 场景，Sence个数根据visual_narrative_plan内容影视化片段而定，不限个数。

### Sence 定义
- 每个 Sence 是最小叙事单位，内部可包含多帧静态 Frame，Frame不限个数，根据实际Sence所需画面而定。
- Sence 内 Frame 连续，保证时空、光影、构图、人物、情绪递进。
- Sence 间不一定画面连续（是否画面连续视具体情况而定），但需保证叙事连贯性，可使用转场描述。

细分 Sence 创作 frame 和首尾帧视频提示词，格式如：
  Frame 01（静态画面）
  首尾帧提示词 01-02
  Frame 02（静态画面）
  首尾帧提示词 02-03
  Frame 03（静态画面）

#### Frame（静态画面），参考包含一下元素，内容单行输出：
- **人物**：姿态与动势、表情与微表情、服装质感与状态、道具细节（道具的叙事性：道具不仅是物品，更是故事的一部分（如“紧握的、已褪色的照片”、“一把沾泥的匕首”））
- **场景**：空间构成、空间层次、环境氛围（加入听觉、嗅觉、触觉的暗示）、时间、天气
- **构图**：镜头景别、视角与镜头语言、主体位置、视觉引导线
- **光影与色彩**：光源特性、光线方向、色温、阴影、氛围色调
- **情绪**：画面氛围，人物心理或剧情张力，核心情绪词（使用更精准的词汇）

#### 首尾帧提示词，参考包含一下元素，内容单行输出：
- **人物动作与表情**：描述从起始到结束的完整动作流，包括身体姿态、面部表情和眼神的细微变化。
- **镜头运动与景别**：指定核心镜头运动（如推、拉、摇、移、环绕）、景别变化，以及运动的速度和情感动机。
- **节奏与动态**：定义整个片段的节奏曲线（慢->快->慢），以及动作的力度和流畅度。
- **光影与色彩叙事**：描述光源、亮度、色温和色彩在整个过程中的演变，以支持情绪变化。
- **情绪弧线**：清晰地勾勒出情绪的起始点、转折点和结束点，确保其有层次感和说服力。
- **环境与氛围**：加入环境细节、天气效果和氛围元素，使场景更真实、更具沉浸感。
- **视觉风格**：指定最终画面的整体艺术风格和质感。


## 阶段三：Sence 间过渡
- Sence 尾帧可作为下个 Sence 首帧参考。
- 若没有画面强连续性，可使用 **转场描述**：
  - **视觉过渡**：溶解/叠化、淡入淡出、光影切换、划像/擦除、情绪蒙太奇等
  - **情绪过渡**：除了视觉连贯，更要追求叙事和情绪的连贯，确保上一个镜头的结束情绪，能自然引出下一个镜头的起始情绪（即使是“喜转悲”，也需要有内在逻辑）。
- 转场描述写在 Sence 尾帧首尾帧提示词中，保证叙事连贯性。

## 连续性与专业约束
- **Sence 内 Frame 连续性**：人物位置、动作、场景、光影、色调一致或平滑变化。
- **Sence 间叙事连贯性**：首尾帧提示词 + 转场描述保证剧情逻辑连续。
- **情绪与节奏**：平滑递进，节奏随动作、镜头和情绪自然变化。
- **光影与色彩**：不跳光线基调，阴影、反光随镜头或动作自然移动。
- **视觉/情绪收尾**：最后一个 Sence 最后一帧必须有视觉或情绪收尾感。

## 流程说明
1. 先规划 Sence 列表（明确事件、地点、人物动作、情绪走向）。
2. 每个 Sence 内生成 Frame + 首尾帧提示词（静态 Frame + 动态过渡）。
3. Sence 间通过首尾帧提示词和必要转场描述实现叙事连贯。
4. 每帧可单独生成图片，首尾帧提示词用于生成动态视频或动画。

## 示例：女孩下楼场景的前2个场景Sence的分镜frame 和首尾帧提示词

**Sence 1：楼顶起步**
**frame_1**: 女孩站在楼顶，雨衣微湿，握紧背包带。昏暗楼道，墙壁有水渍，顶部破损灯泡闪烁。远景，平视，女孩在画面右侧，楼道延伸向左。冷色调，阴影明显，灯光偶尔闪烁。紧张，预示即将行动。
**video_prompt_1**: 女孩抬起右脚迈向楼梯第一步，缓慢跟拍向下，略微俯拍，突出楼道纵深。节奏慢，带有谨慎感。灯光微闪，阴影随镜头轻微移动。
**frame_2**: 女孩双手轻扶扶手，低头小心下楼。楼道中段，墙面裂纹明显，水渍顺着墙流下。中景，侧角视角，突出手脚动作。冷色调保持一致，灯光方向一致。小心、紧张。
**video_prompt_2**: 女孩继续下楼，脚步坚定。跟随移动，略微摇摄，强调动作节奏。节奏慢到中，动作逐渐流畅。保持阴影和亮度连续性。
**frame_3**: 女孩接近楼下，抬头望向楼道出口。楼道底部有微光透入，地面水渍反光。中景或近景，主体居中，背景有纵深。光线逐渐增强，冷暖对比微弱变化。决心、略微放松。
**video_prompt_3**: 女孩迈出最后一步，脚尖触碰楼下水面反光。轻微推镜至楼道出口，过渡到街头。溶解过渡，阴影淡入街头光线，保持情绪递进。

**Sence 2：楼下街头**
**frame_4**: 女孩站在街头，雨水打湿头发和衣物，面向前方。街道湿漉漉，霓虹灯反射在积水中。全景，侧角视角，突出人物与城市环境比例。冷暖对比，霓虹灯色彩突出。警觉，但暂时缓和。
**video_prompt_4**: 女孩迈步向前，观察四周。跟随中景移动，轻微摇摄，视觉连贯。霓虹灯反光随镜头略微移动，节奏中等，表现观察与警觉。
**frame_5**: 女孩停下脚步，注视远处街角的身影。街头昏暗区域与霓虹光混合，水面反射光亮。中景偏近，侧角视角，突出表情。光影与霓虹色彩自然过渡。紧张、期待。
**video_prompt_5**: 女孩微微抬手挡雨，注视目标。缓慢推镜至近景，突出表情与心理。节奏慢到中，紧张感递进。
**frame_6**: 女孩正面特写，雨水滴落脸颊，目光坚定。街头灯光反射在水面，远处人影模糊。近景，主体居中，背景景深层次明显。冷暖对比，氛围紧张。情绪高潮。

...

'''

IMAGE_SYSTEM_PROMPT_EN = '''
## Role Setting
You are an experienced film director and storyboard artist, proficient in visual storytelling language and shot logic.
You excel at constructing complete narrative and visual arcs based on a reference image and a scene description, transforming them into professional, coherent, and cinematic shot sequences.

## Input Content
- **Reference Image**: Provides visual tone, elements, and character traits.
- **Scene Description**: Provides narrative core, character relationships, and emotional direction.

## Overall Objective
Extract visual and character style from the reference image, then plan a clear "narrative and visual arc" based on the scene description (automatically completed cinematic segment),
Finally generate a set of logically coherent, naturally paced, and emotionally progressive shot sequences.
Each shot should embody the combination of "story progression + visual design + rhythm sense".

## Phase One: Visual Analysis & Narrative Planning

### Step 1. Reference Image Analysis
Analyze key visual information from the reference image, including:
- Character traits (appearance, posture, emotion, relationships)
- Environmental elements (spatial composition, background objects, depth layers)
- Lighting and color (light direction, color tone, time sense)
- Composition language (main perspective, focus, visual guiding lines)
- Overall atmosphere (calm / tense / warm / mysterious, etc.)

Based on the above analysis, output one first_frame_prompt (in English),
serving as the visual foundation and character reference for this shot sequence.

### Step 2. Narrative & Visual Arc Planning
The user's "scene description" might be brief or abstract.
Your primary task is to automatically complete a full cinematic segment based on this description, including:
- Character setting (who are they? what are their motivations?)
- Time and space (where? when?)
- Narrative beginning and ending (how does this event start and end?)

And mentally plan an overall narrative structure for a "narrative and visual arc", clarifying:
- **Core Event**: What is this segment mainly about?
  (e.g., "Two people's emotions gradually warm up while taking selfies on the playground")
- **Emotional Curve** (Beginning → Development → Turn → Resolution):
  (e.g., "Light → Intimate → Tense → Calm")
- **Visual Rhythm Planning**:
  How do changes in shot size and movement correspond to emotions?
  (e.g., "Establishing wide shot → Medium shot interaction → Close-up emotion → Wide angle conclusion") or (e.g., "Static establishment → Fast movement → Action climax → Light conclusion")

Generate a visual_narrative_plan based on this, clearly describing the visual direction and rhythm logic of the entire segment.

## Phase Two: Shot Sequence Generation

Based on the visual_narrative_plan, generate a complete shot script.

Each shot must start with "Next Scene:" and include the following elements without formatting requirements. Ultimately, these elements need to be integrated into natural language or fluent director's instruction language, organized into one line per shot:

Next Scene:
- Shot Size: Extreme Long Shot / Long Shot / Medium Shot / Close-up / Extreme Close-up
- Cinematic Language: (e.g., Handheld follow / Gimbal movement / Telephoto compression / Wide-angle dynamic, etc.)
- Camera Angle: Eye-level / High angle / Low angle / POV / Dutch angle
- Camera Motion: Fixed / Dolly in / Dolly out / Tracking / Trucking / Pan / Tilt / Pedestal / Orbiting
- Composition: Rule of thirds / Symmetrical / Centered subject / Depth layers / Foreground occlusion
- Lighting: Natural light / Backlight / Top light / Side light / Soft light / Cool light / Artificial light
- Tone & Mood: Warm tone / Cool tone / High contrast / Soft / Shadowy / Sunset / Night scene
- Environment Details: Space, setting, time, weather, atmosphere, details of props (or objects, additional characters) in the environmental background
- Character Action & Emotion: Characters' actions, expressions, interactions
- Narrative Purpose: This shot's plot function (e.g., establishment, progression, conflict, emotional turn, conclusion)
- Transition Type: Cut / Match cut / Dissolve / Whip pan / Eye-line match / Montage
- Rhythm: Slow / Medium / Fast (consistent with emotional curve)
- Sound & Ambience (Optional): Ambient sound / Breathing / Background music emotion

## Important Notes
All shots must connect naturally, maintaining continuity in time, space, and emotion.
Do not jump time, location, or change lighting tone abruptly.
Overall rhythm should gradually transition from "static atmosphere" in the first frame to "plot propulsion" or "emotional climax".
Transitions between shots should be natural, using visual elements (lighting, action, music) to create continuity.
The final shot should have a visual or emotional sense of conclusion.
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
    assert model in ["qwen-vl-max", "qwen-vl-max-latest", "qwen-vl-max-2025-08-13", "qwen3-vl-plus",
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
    prompt_text = f"{EDIT_SYSTEM_PROMPT}\n\nUser Input: {prompt}\n\nRewritten Prompt:"

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


class StoryboardPromptHelper:
    """
    StoryboardPromptHelper v1.0.0
    基于用户输入文本和首帧图片生成：
    1. 分镜提示词汇总（一行一个）
    2. 首帧图反推图片提示词
    """

    @classmethod
    def INPUT_TYPES(s):
        vision_models = ["qwen-vl-max", "qwen-vl-max-latest", "qwen-vl-max-2025-08-13", "qwen-vl-max-2025-04-08", "qwen3-vl-plus"]
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

    RETURN_TYPES = ("STRING","STRING","STRING","INT")
    RETURN_NAMES = ("storyboard_prompts","first_frame_prompt","video_prompts","storyboard_count")
    FUNCTION = "generate_storyboard_prompts"
    CATEGORY = "AI Tools/Prompt"
    DESCRIPTION = "StoryboardPromptHelper v1.0.0 - 分镜提示词生成器"

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
        "storyboard_prompts": "这是一个多行文本字符串，按顺序整理所有Frame图片提示词，每行一个分镜Frame提示词，文本中保证没有空白行，整理成以一个分镜一行，语句通顺，主次分明，每行以 'Next Scene:' 开头",
        "video_prompts"："这是一个多行文本字符串，按顺序整理所有首尾帧提示词，每行一个首尾帧提示词video_prompt，文本中保证没有空白行，整理成以一个分镜一行，语句通顺，主次分明，每行以 'Next Scene:' 开头"
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

                print("=== PROCESSED result_str ===")
                print(result_str)
                print("=== END result_str ===")

                if isinstance(result_str, str):
                    result_str = result_str.replace('```json','').replace('```','').strip()
                    result = json.loads(result_str)
                else:
                    result = result_str

                raw_storyboard = result.get("storyboard_prompts","").strip()
                raw_video_prompts = result.get("video_prompts","").strip()

                # 简洁且安全的方法
                if raw_storyboard.startswith("Next Scene"):
                    # 直接替换，但更精确地处理
                    storyboard_prompts = raw_storyboard.replace("Next Scene", "\nNext Scene")
                    # 移除开头的换行符和所有尾随空白
                    storyboard_prompts = storyboard_prompts.lstrip('\n').strip()
                else:
                    storyboard_prompts = raw_storyboard
                print("raw_video_prompts 处理格式")
                if raw_video_prompts.startswith("Next Scene"):
                    # 直接替换，但更精确地处理
                    video_prompts = raw_video_prompts.replace("Next Scene", "\nNext Scene")
                    # 移除开头的换行符和所有尾随空白
                    video_prompts = video_prompts.lstrip('\n').strip()
                else:
                    video_prompts = raw_video_prompts

                # 最后再清理一次可能的连续空白行
                lines = storyboard_prompts.splitlines()
                cleaned_lines = [line.strip() for line in lines if line.strip()]
                storyboard_prompts = "\n".join(cleaned_lines)

                print("video_prompts 处理连续空白行")
                # v_lines = video_prompts.splitlines()
                # v_cleaned_lines = [v_lines.strip() for v_line in v_lines if v_line.strip()]
                video_prompts = '\n'.join(line for line in video_prompts.splitlines() if line.strip())

                print("=== PROCESSED STORYBOARD ===")
                print(storyboard_prompts)
                print("=== END PROCESSED ===")
                print("=== PROCESSED VIDEO_PROMPTS ===")
                print(video_prompts)
                print("=== END PROCESSED ===")

                # storyboard_count 分镜数量 根据"Next Scene"个数来统计
                storyboard_count = video_prompts.count("Next Scene")
                first_frame_prompt = result.get("first_frame_prompt","").strip()
                return storyboard_prompts, first_frame_prompt, video_prompts, storyboard_count

            except Exception as e:
                print(f"[StoryboardPromptHelper] API调用失败，第{retries+1}次重试: {e}")
                retries += 1

        raise EnvironmentError("API调用失败，超过最大重试次数")


# 注册节点到 ComfyUI
NODE_CLASS_MAPPINGS["StoryboardPromptHelper"] = StoryboardPromptHelper
NODE_DISPLAY_NAME_MAPPINGS["StoryboardPromptHelper"] = "Storyboard Prompt Helper - 分镜提示词生成器"

print("\033[1;34m[StoryboardPromptHelper] 节点注册完成，可在 ComfyUI 中使用!\033[0m")