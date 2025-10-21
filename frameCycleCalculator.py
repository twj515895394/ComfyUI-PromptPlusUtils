
class FrameCycleCalculator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "total_frames": ("INT", {"default": 100, "min": 1, "max": 10000}),
                "frames_per_cycle": ("INT", {"default": 77, "min": 1, "max": 1000}),
            }
        }

    RETURN_TYPES = ("INT", "INT", "INT")
    RETURN_NAMES = ("cycle_count", "frames_per_cycle", "last_cycle_frames")
    FUNCTION = "calculate"
    CATEGORY = "math/frames"

    def calculate(self, total_frames, frames_per_cycle):
        # 如果总帧数不超过每次帧数的2倍，就只循环1次
        if total_frames <= frames_per_cycle * 2:
            cycle_count = 1
            last_cycle_frames = total_frames
            frames_per_cycle = last_cycle_frames
        else:
            # 计算理论循环次数和余数
            cycle_count = (total_frames + frames_per_cycle - 1) // frames_per_cycle
            remainder = total_frames % frames_per_cycle

            if remainder > 0:
                # 有余数：减少一次循环，将余数合并到前一次
                cycle_count = cycle_count - 1
                last_cycle_frames = frames_per_cycle + remainder
            else:
                # 刚好整除：正常循环
                last_cycle_frames = frames_per_cycle

        # 每次帧数始终与输入参数一致
        return (cycle_count, frames_per_cycle, last_cycle_frames)


# 注册节点
NODE_CLASS_MAPPINGS = {
    "FrameCycleCalculator": FrameCycleCalculator
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FrameCycleCalculator": "for 帧循环计算器"
}
print("\033[1;34m[FrameCycleCalculator] 节点注册完成，可在 ComfyUI 中使用!\033[0m")