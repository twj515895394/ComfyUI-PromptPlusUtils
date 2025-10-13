import torch

class AudioFrameWinSize:
    """
    音频滑动窗口值计算节点
    支持输入 ANY 类型（可兼容 AudioEncoder 输出）
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_tensor": ("ANY",),
                "window_size": ("INT", {"default": 1024, "min": 1}),
                "step_size": ("INT", {"default": 512, "min": 1}),
            }
        }

    RETURN_TYPES = ("ANY",)
    FUNCTION = "compute_window"
    CATEGORY = "Audio/Utils"
    DISPLAY_NAME = "音频滑动窗口值计算"

    def compute_window(self, input_tensor, window_size, step_size):
        # 兼容多种输入类型
        if hasattr(input_tensor, "tensor"):
            tensor = input_tensor.tensor
        elif hasattr(input_tensor, "latents"):
            tensor = input_tensor.latents
        elif isinstance(input_tensor, torch.Tensor):
            tensor = input_tensor
        else:
            raise TypeError(f"Unsupported input type: {type(input_tensor)}")

        # 确保为2D张量 [channels, samples]
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)

        total_len = tensor.shape[-1]
        windows = []

        for start in range(0, total_len - window_size + 1, step_size):
            end = start + window_size
            win = tensor[..., start:end]
            windows.append(win)

        if not windows:
            return (tensor,)

        stacked = torch.stack(windows, dim=0)
        return (stacked,)


# 节点注册信息
NODE_CLASS_MAPPINGS = {
    "AudioFrameWinSize": AudioFrameWinSize
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AudioFrameWinSize": "🎧 音频滑动窗口值计算",
}
