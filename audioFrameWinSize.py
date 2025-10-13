import torch
from comfy import io

class AudioFrameWinSize(io.ComfyNode):
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "audio_encoder_output": ("AUDIO_ENCODER_OUTPUT",)
        }}

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("frame_window_size",)
    FUNCTION = "calculate"
    CATEGORY = "Audio"

    @classmethod
    def NODE_DISPLAY_NAME(cls):
        return "音频滑动窗口值计算"

    def calculate(self, audio_encoder_output):
        if audio_encoder_output is None:
            raise ValueError("[AudioFrameWinSize] 无法从输入中找到有效 embeddings 或 samples")

        # 获取 audio_feat
        all_layers = audio_encoder_output.get("encoded_audio_all_layers", None)
        if all_layers is None:
            raise ValueError("[AudioFrameWinSize] audio_encoder_output 内没有 encoded_audio_all_layers")

        audio_feat = torch.stack(all_layers, dim=0).squeeze(1)  # [num_layers, T, 512]
        # 计算最优 t 值（这里选择最大时间长度）
        t_val = audio_feat.shape[1]

        return (t_val,)
        

# 注册节点
NODE_CLASS_MAPPINGS = {
    "AudioFrameWinSize": AudioFrameWinSize,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AudioFrameWinSize": "音频滑动窗口值计算",
}

print("[Plugin] AudioFrameWinSize node loaded")
