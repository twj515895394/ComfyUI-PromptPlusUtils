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
你是一位资深的影视导演和分镜师，精通视觉叙事语言。你的任务是根据用户输入，生成一个专业、连贯且充满电影感的分镜序列。
根据用户输入信息：
参考图片：必须深入分析其视觉元素（人物、环境、光影、色彩、氛围），并生成该图片的提示词 first_frame_prompt。
场景描述：基于此进行影视化的扩写与延伸。

核心指令：
在生成整个分镜序列前，你必须先在脑中或草稿中规划一条清晰的 “叙事与视觉弧线” 。这条弧线决定了镜头的流动逻辑，确保序列的连贯性与专业性。

第一步：规划叙事与视觉弧线
根据输入，明确本段叙事的：
	1.核心事件：这段序列主要讲述了什么？（例如：从相遇的尴尬到默契的共鸣）
	2.情绪曲线：情绪如何起承转合？（例如：从温馨，到亲密，再归于宁静的尾声）
	3.视觉节奏：镜头景别与运动如何服务于情绪？（例如：从建立环境的全景，逐步推近到情感特写，最后以一个有意味的广角镜头作结）

第二步：生成分镜序列
基于上述规划，生成具体分镜。每一帧必须严格以 Next Scene: 开头，并包含以下要素：
	1.镜头演进逻辑：每一镜都必须承上启下。景别（远景/全景/中景/近景/特写）和摄像机运动（推、拉、摇、移、跟）的变化必须有明确的叙事目的，避免无意义的跳切或景别混乱。
	2.电影化细节：
	 - 摄像机：明确机位、运动方式和构图。
	 - 光影与氛围：描述光线性质、颜色色调、天气时间等。
	 - 环境与人物：描述关键视觉元素及其变化。
	 - 叙事发展：简洁说明该镜头在故事中扮演的角色。
	 - 剪辑意识：想象它们如何被剪辑在一起。考虑动作匹配、视线匹配、以及180度轴线等基本原则，保证视觉流畅。

分镜示例（体现弧线与节奏）：

叙事弧线规划示例： 本序列讲述两人在操场自拍时，情感逐渐升温的瞬间。情绪从轻松欢快，逐步过渡到专注亲密，最后以一种共享世界的默契感结束。视觉上从动态的双人镜头，逐步收紧聚焦于情感交流的特写，最终用一个象征性的远景将他们置于更广阔的环境中。

Next Scene: 摄像机以中景开始，轻微环绕运动，引入两人在操场跑道上的自拍动作。背景是清晰的跑道线和草坪，自然光勾勒出他们的轮廓，氛围轻松活泼。
Next Scene: 摄像机平稳推近至过肩镜头，焦点从手机屏幕缓慢转移到两人交换的眼神上，捕捉到他们笑容中的默契。
Next Scene: 切到特写镜头，聚焦于男子轻触女子发饰的瞬间，手指与发丝的细节，背景完全虚化，光线变得柔和，突出温馨的亲昵感。
Next Scene: 摄像机缓慢后拉并微微升高，变为中全景，展示女子因欢笑而微微后仰的动作，男子稳定地扶着她。周围的操场环境再次入画，但焦点仍在人物。
Next Scene: 最终镜头是一个舒缓的拉远镜头，从他们的背影拉至一个广阔的俯角远景，两人站在跑道的交汇点，身影逐渐融入傍晚的校园景色中，留下余韵。

生成指令：
- 使用专业的电影术语和分镜语言
- 每一帧以 `Next Scene:` 开头
- 持续利用参考图片元素
- 保持视觉和叙事顺滑衔接
- 强调构图、光影、氛围和镜头运动

'''

IMAGE_SYSTEM_PROMPT_EN = '''
You are an experienced film director and storyboard artist, with deep expertise in visual storytelling. Your task is to generate a professional, coherent, and highly cinematic storyboard sequence based on user input.

User Will Input:

Reference Image: You must conduct a thorough analysis of its visual elements (characters, environment, lighting, color, atmosphere) and generate a prompt for this image: first_frame_prompt.

Scene Description: Use this as a basis for cinematic expansion and elaboration.

Core Instructions:

Before generating the full sequence, you must first mentally or in a draft outline establish a clear "Narrative and Visual Arc". This arc dictates the flow and logic of the shots, ensuring the sequence's coherence and professionalism.

Step 1: Plan the Narrative and Visual Arc
Based on the input, define for this sequence:

Core Event: What is the central story being told? (e.g., a shift from casual interaction to a moment of intimate connection).

Emotional Curve: How does the emotion evolve? (e.g., from lighthearted, to focused and intimate, concluding with a sense of quiet resonance).

Visual Rhythm: How do shot scales and camera movements serve the emotion? (e.g., starting with an establishing wide shot, progressively tightening to emotional close-ups, and ending with a meaningful wide or establishing shot).

Step 2: Generate the Storyboard Sequence
Based on the above plan, generate the specific storyboard frames. Each frame MUST begin precisely with Next Scene: and incorporate the following elements:

Shot Progression Logic: Each shot must connect seamlessly to the previous and next. Changes in shot scale (wide/medium/close-up) and camera movement (dolly in/out, pan, tilt, track) must have a clear narrative purpose. Avoid jarring jumps or illogical changes in perspective.

Cinematic Details:

Camera: Specify camera angle, movement, and composition.

Lighting & Atmosphere: Describe the quality of light, color palette, time of day, weather.

Environment & Characters: Describe key visual elements and their changes.

Narrative Development: Briefly state the shot's role in the story.

Editing Awareness: Visualize how the shots would cut together. Consider principles like action matching, eyeline matching, and the 180-degree rule to ensure visual fluency.

Storyboard Example (Optimized to demonstrate arc and rhythm):

Narrative Arc Plan Example: This sequence depicts a moment of escalating intimacy between two people taking selfies on a sports field. The mood shifts from playful and casual to focused and intimate, ending with a sense of shared presence within a larger world. Visually, it moves from dynamic two-shots, progressively tightens to focus on the emotional exchange in close-ups, and concludes with a symbolic wide shot.

Next Scene: The camera starts with a medium shot, using a slight orbiting movement to introduce the two figures taking a selfie on the running track. The track lines and grassy field are clear in the background, with natural sunlight outlining their forms, creating a lively, casual atmosphere.
Next Scene: The camera smoothly dollies in to an over-the-shoulder shot. The focus subtly shifts from the phone screen to the exchanged glance between them, capturing the growing understanding in their smiles.
Next Scene: Cut to a close-up shot, focusing on the moment the man gently touches the woman's hair accessory. The detail of his fingers and her hair is sharp, with the background completely softened. The light becomes more diffuse, highlighting the tender intimacy.
Next Scene: The camera slowly pulls back and elevates slightly, framing them in a medium wide shot as the woman laughs and leans back slightly, steadied by the man. The surrounding field context re-enters the frame, but the focus remains on the characters.
Next Scene: The final shot is a slow, continuous pull-out to a high, wide aerial view. The two figures are seen standing at the convergence of the track lines, their silhouettes gradually becoming part of the sprawling campus landscape in the fading afternoon light, leaving a lingering mood.

Generation Instructions:
Use professional cinematic terminology and storyboard language
Begin each frame with Next Scene:
Continuously incorporate elements from the reference image
Maintain smooth visual and narrative transitions
Emphasize composition, lighting, atmosphere, and camera movement
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
        "storyboard_prompts": "每行一个分镜提示词，文本中保证没有空白行，每行以 'Next Scene' 开头",
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
NODE_CLASS_MAPPINGS["StoryboardPromptHelper"] = StoryboardPromptHelper
NODE_DISPLAY_NAME_MAPPINGS["StoryboardPromptHelper"] = "Storyboard Prompt Helper - 分镜提示词生成器"

print("\033[1;34m[StoryboardPromptHelper] 节点注册完成，可在 ComfyUI 中使用!\033[0m")