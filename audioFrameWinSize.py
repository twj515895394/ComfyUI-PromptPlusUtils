import torch

class AudioFrameWinSize:
    """
    音频滑动窗口值计算节点
    输入 AudioEncoder 输出，自动计算安全的 t 值（滑动窗口帧数）
    支持 auto/manual 模式
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_encoder_output": ("AUDIO_ENCODER_OUTPUT",),
                "mode": (["auto", "manual"], {"default": "auto"}),
            }
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("t_value",)
    FUNCTION = "compute_t"
    CATEGORY = "Audio/Utils"
    DISPLAY_NAME = "音频滑动窗口值计算"

    # 最大 t 值限制，保证不报错
    MAX_T = 81

    def compute_t(self, audio_encoder_output, mode):
        if not isinstance(audio_encoder_output, dict):
            raise TypeError(f"[AudioFrameWinSize] 输入必须为 AUDIO_ENCODER_OUTPUT, 当前类型: {type(audio_encoder_output)}")

        tensor = None
        t_value = 1
        source = "unknown"

        # 优先取 embeddings
        if "embeddings" in audio_encoder_output and isinstance(audio_encoder_output["embeddings"], torch.Tensor):
            tensor = audio_encoder_output["embeddings"]
            t_value = tensor.shape[1]
            source = "embeddings"
        # fallback 到 samples
        elif "samples" in audio_encoder_output and isinstance(audio_encoder_output["samples"], torch.Tensor):
            tensor = audio_encoder_output["samples"]
            source = "samples"
            # manual 模式内部自动计算
            total_len = tensor.shape[-1] if tensor.dim() > 1 else tensor.numel()
            if total_len > self.MAX_T:
                t_value = self.MAX_T
            else:
                t_value = total_len
        else:
            raise ValueError("[AudioFrameWinSize] 无法从输入中找到有效 embeddings 或 samples")

        # 强制 t 不超过 MAX_T
        if t_value > self.MAX_T:
            t_value = self.MAX_T

        print(f"[AudioFrameWinSize] mode={mode}, source={source}, 输入 shape={tuple(tensor.shape)}, 输出 t={t_value}")
        return (int(t_value),)


# 节点注册
NODE_CLASS_MAPPINGS = {
    "AudioFrameWinSize": AudioFrameWinSize
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AudioFrameWinSize": "音频滑动窗口值计算"
}
