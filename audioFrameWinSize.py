import torch

class AudioFrameWinSize:
    """
    根据输入的音频特征 Tensor 自动计算合适的滑动窗口帧数（t 值）。
    主要用于解决 WanVideo 模型中因 t 不匹配导致的 shape mismatch 报错：
        b (t n) c -> (b t) n c
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_tensor": ("TENSOR",),
                "max_t": ("INT", {"default": 200, "min": 1, "max": 1000}),
                "tolerance": ("INT", {"default": 3, "min": 0, "max": 50}),
            }
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("frame_window_t",)
    FUNCTION = "process"
    CATEGORY = "WanVideo/音频工具"

    def process(self, input_tensor: torch.Tensor, max_t: int, tolerance: int) -> tuple:
        # 检查 tensor 维度合法性
        if input_tensor.ndim < 2:
            raise ValueError(f"输入 Tensor 形状无效: {input_tensor.shape}")

        seq_len = input_tensor.shape[1]

        # 查找能整除 seq_len 的 t 值
        valid_t_values = [t for t in range(1, max_t + 1) if seq_len % t == 0]

        if valid_t_values:
            # 优先取最大的整除值
            t = valid_t_values[-1]
        else:
            # 容忍小范围误差，例如 seq_len=99840, t=73 时的余数
            best_t, min_remainder = 1, seq_len
            for guess in range(1, max_t + 1):
                remainder = seq_len % guess
                if remainder <= tolerance and remainder < min_remainder:
                    best_t = guess
                    min_remainder = remainder
            t = best_t

        print(f"[音频滑动窗口值计算] 输入序列长度={seq_len}, 计算得到 t={t}")
        return (t,)


NODE_CLASS_MAPPINGS = {
    "AudioFrameWinSize": AudioFrameWinSize
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AudioFrameWinSize": "🎧 音频滑动窗口值计算"
}
