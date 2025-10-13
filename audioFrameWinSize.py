import torch

class AudioFrameWinSize:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # 参考官方节点风格，用一个元组支持多种类型
                "audio_data": ("AUDIO", "TENSOR", "ANY"),
                "window_size": ("INT", {"default": 1024, "min": 1}),
                "step_size": ("INT", {"default": 512, "min": 1}),
            }
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("t_value",)
    FUNCTION = "compute_t"
    CATEGORY = "Audio/Utils"
    DISPLAY_NAME = "音频滑动窗口值计算"

    def compute_t(self, audio_data, window_size, step_size):
        # 解析实际 tensor
        tensor = None

        # audio_data 可能是 dict（官方 AudioEncoder 输出）  
        if isinstance(audio_data, dict):
            # 官方 audio encoder 节点输出里常见 key 是 "samples"
            if "samples" in audio_data:
                tensor = audio_data["samples"]
        elif hasattr(audio_data, "tensor"):
            tensor = audio_data.tensor
        elif hasattr(audio_data, "latents"):
            tensor = audio_data.latents
        elif isinstance(audio_data, torch.Tensor):
            tensor = audio_data

        if tensor is None:
            raise TypeError(f"[AudioFrameWinSize] 无法识别输入类型: {type(audio_data)}")

        # 计算长度（假设最后一个维度是 time 轴）
        if tensor.dim() >= 2:
            total_len = tensor.shape[-1]
        else:
            total_len = tensor.numel()

        if total_len < window_size:
            t_value = 1
        else:
            # 用滑动窗口公式
            t_value = (total_len - window_size) // step_size + 1

        # 打印调试信息
        print(f"[AudioFrameWinSize] total_len={total_len}, window_size={window_size}, step_size={step_size}, t_value={t_value}")
        return (t_value,)


NODE_CLASS_MAPPINGS = {
    "AudioFrameWinSize": AudioFrameWinSize
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AudioFrameWinSize": "音频滑动窗口值计算"
}
