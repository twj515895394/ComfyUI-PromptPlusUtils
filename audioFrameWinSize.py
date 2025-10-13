import torch

class AudioFrameWinSize:
    """
    音频滑动窗口值计算节点
    自动根据音频特征长度计算帧窗口数量 (t值)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_tensor": (["AUDIO", "LATENT", "IMAGE", "TENSOR", "ANY"],),
                "window_size": ("INT", {"default": 1024, "min": 1}),
                "step_size": ("INT", {"default": 512, "min": 1}),
            }
        }

    RETURN_TYPES = ("INT",)
    FUNCTION = "compute_t_value"
    CATEGORY = "Audio/Utils"
    DISPLAY_NAME = "音频滑动窗口值计算"

    def compute_t_value(self, input_tensor, window_size, step_size):
        # 自动解析 tensor
        tensor = None

        if isinstance(input_tensor, dict):
            if "samples" in input_tensor:
                tensor = input_tensor["samples"]
        elif hasattr(input_tensor, "tensor"):
            tensor = input_tensor.tensor
        elif hasattr(input_tensor, "latents"):
            tensor = input_tensor.latents
        elif isinstance(input_tensor, torch.Tensor):
            tensor = input_tensor
        else:
            raise TypeError(f"Unsupported input type: {type(input_tensor)}")

        if not isinstance(tensor, torch.Tensor):
            raise ValueError("无法从输入中解析出有效的 torch.Tensor")

        # 计算 t 值
        total_len = tensor.shape[-1] if tensor.dim() > 1 else tensor.numel()
        if total_len < window_size:
            t_value = 1
        else:
            t_value = (total_len - window_size) // step_size + 1

        print(f"[AudioFrameWinSize] 输入 shape={tuple(tensor.shape)}, 输出 t={t_value}")
        return (t_value,)


# 注册节点
NODE_CLASS_MAPPINGS = {
    "AudioFrameWinSize": AudioFrameWinSize
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AudioFrameWinSize": "音频滑动窗口值计算"
}
