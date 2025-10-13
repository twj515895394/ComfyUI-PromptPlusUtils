import torch

class AudioFrameWinSize:
    """
    音频滑动窗口值计算节点
    接收 AudioEncoderEncode 输出，返回最大 frame_window_size (t值)
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_encoder_output": ("AUDIO_ENCODER_OUTPUT",),
            }
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("frame_window_size",)
    FUNCTION = "calculate"
    CATEGORY = "Audio"

    def calculate(self, audio_encoder_output):
        """
        audio_encoder_output: AudioEncoderEncode 输出，dict 格式，包含 encoded_audio_all_layers
        """
        if audio_encoder_output is None or "encoded_audio_all_layers" not in audio_encoder_output:
            raise ValueError("[AudioFrameWinSize] 无法从输入中找到有效 embeddings 或 samples")

        # 合并所有层 embedding
        all_layers = audio_encoder_output["encoded_audio_all_layers"]  # list of torch.Tensor
        if len(all_layers) == 0:
            raise ValueError("[AudioFrameWinSize] encoded_audio_all_layers 为空")

        # 合并为 [num_layers, T, 512] 的 tensor
        audio_feat = torch.stack(all_layers, dim=0).squeeze(1)  # shape: [num_layers, T, 512]

        # 取最大长度 t
        _, T, _ = audio_feat.shape
        frame_window_size = max(1, T)

        return (frame_window_size,)


# 注册节点
NODE_CLASS_MAPPINGS = {
    "AudioFrameWinSize": AudioFrameWinSize,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AudioFrameWinSize": "音频滑动窗口值计算",
}
