import torch
from ComfyUI import io

class AudioFrameWinSize(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="AudioFrameWinSize",
            category="audio",
            inputs={
                "audio_encoder_output": ("AUDIO_ENCODER_OUTPUT", {"optional": False, "tooltip": "来自 AudioEncoderEncode 的输出"}),
                "max_t": ("INT", {"default": 81, "min": 1, "max": 1000, "tooltip": "frame_window_size 最大上限"})
            },
            outputs={
                "frame_window_size": ("INT", {"tooltip": "计算得到的最优滑动窗口帧数"}),
                "audio_total_frames": ("INT", {"tooltip": "音频编码器总帧数"})
            }
        )

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "audio_encoder_output": ("AUDIO_ENCODER_OUTPUT",),
            "max_t": ("INT", {"default": 81, "min": 1, "max": 1000}),
        }}

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("frame_window_size", "audio_total_frames")
    FUNCTION = "compute"
    CATEGORY = "AUDIO/音频滑动窗口值计算"

    def compute(self, audio_encoder_output, max_t=81):
        if audio_encoder_output is None:
            raise ValueError("[AudioFrameWinSize] 无法从输入中找到有效 audio_encoder_output")

        if "encoded_audio_all_layers" not in audio_encoder_output:
            raise ValueError("[AudioFrameWinSize] audio_encoder_output 缺少 encoded_audio_all_layers")

        all_layers = audio_encoder_output["encoded_audio_all_layers"]
        if not isinstance(all_layers, list) or len(all_layers) == 0:
            raise ValueError("[AudioFrameWinSize] encoded_audio_all_layers 为空")

        # 获取总帧数
        audio_feat = torch.stack(all_layers, dim=0).squeeze(1)  # [num_layers, T, dim]
        total_frames = audio_feat.shape[1]

        # 找最大可整除值
        candidates = [t for t in range(1, min(max_t, total_frames)+1) if total_frames % t == 0]
        if candidates:
            frame_window_size = max(candidates)
        else:
            frame_window_size = min(max_t, total_frames)

        return frame_window_size, total_frames


# 注册节点
NODE_CLASS_MAPPINGS = {
    "AudioFrameWinSize": AudioFrameWinSize,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AudioFrameWinSize": "音频滑动窗口值计算",
}
