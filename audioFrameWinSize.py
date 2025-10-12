import math
from comfy import Node

class AudioFrameWinSize(Node):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # ANY 类型可以接任何输入，包括 AudioEncoder 输出的 dict
                "input_tensor": ("ANY",),
                "max_t": ("INT", {"default": 81, "min": 1, "max": 200}),
                "tolerance": ("INT", {"default": 3, "min": 0, "max": 50}),
            }
        }

    RETURN_TYPES = ("INT",)
    FUNCTION = "compute_t"
    CATEGORY = "音频处理"
    DISPLAY_NAME = "音频滑动窗口值计算"

    def compute_t(self, input_tensor, max_t=81, tolerance=3):
        # 尝试提取实际 tensor
        tensor = None
        if isinstance(input_tensor, dict) and "samples" in input_tensor:
            tensor = input_tensor["samples"]
        elif hasattr(input_tensor, "tensor"):
            tensor = input_tensor.tensor
        elif hasattr(input_tensor, "latents"):
            tensor = input_tensor.latents
        elif hasattr(input_tensor, "shape"):
            tensor = input_tensor
        else:
            print(f"[AudioFrameWinSize] 无法识别输入类型: {type(input_tensor)}")
            return (1,)

        # 总长度
        seq_len = tensor.shape[1]  # 假设 shape [B, T, ...]，取第二维

        # 找能整除的 t 值，t <= max_t
        for t_candidate in range(min(max_t, seq_len), 0, -1):
            n, r = divmod(seq_len, t_candidate)
            if r <= tolerance:
                print(f"[AudioFrameWinSize] 选择滑动窗口 t={t_candidate} (seq_len={seq_len}, n={n}, remainder={r})")
                return (t_candidate,)

        # 如果没有找到合适的 t，就返回 1
        print(f"[AudioFrameWinSize] 没有找到合适的 t，返回 t=1")
        return (1,)
