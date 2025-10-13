import torch
import math

class AudioFrameWinSize:
    """
    计算音频滑动窗口值 frame_window_size
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_encoder_output": ("AUDIO_ENCODER_OUTPUT",),
                "total_frames": ("INT", {"default": 256, "min": 1, "tooltip": "音频设计总帧数"}),
            },
            "optional": {
                "desired_max": ("INT", {"default": 81, "min": 1, "tooltip": "希望的最大滑动窗口值"})
            }
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("frame_window_size",)
    FUNCTION = "compute"
    CATEGORY = "Audio"

    def compute(self, audio_encoder_output, total_frames, desired_max=81):
        # 打印节点加载信息
        print("[AudioFrameWinSize] 节点加载成功!")

        if total_frames <= desired_max:
            frame_window_size = total_frames
            print(f"[AudioFrameWinSize] total_frames ({total_frames}) <= desired_max ({desired_max}), frame_window_size={frame_window_size}")
        else:
            num_blocks = math.ceil(total_frames / desired_max)
            frame_window_size = round(total_frames / num_blocks)
            print(f"[AudioFrameWinSize] total_frames={total_frames}, desired_max={desired_max}, num_blocks={num_blocks}, frame_window_size={frame_window_size}")

        return (frame_window_size,)


NODE_CLASS_MAPPINGS = {
    "AudioFrameWinSize": AudioFrameWinSize,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AudioFrameWinSize": "音频滑动窗口值计算",
}


print("\033[1;34m[AudioFrameWinSize] 节点注册完成，可在 ComfyUI 中使用!\033[0m")
