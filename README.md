# ComfyUI-PromptPlusUtils

一个基于通义千问模型的 ComfyUI 提示词优化插件，支持文生图和图生图两种模式的智能提示词优化。

## 功能特点

- 🤖 **智能提示词优化**：使用 Qwen 模型优化和扩展您的提示词
- 🖼️ **双模式支持**：支持文生图和图生图两种优化模式
- 🔄 **模型自动匹配**：根据模式自动推荐合适的模型
- 🌐 **多语言支持**：自动检测中英文并优化
- 💾 **安全配置**：支持 API 密钥文件存储

## 安装

1. 将插件克隆到 ComfyUI 的 `custom_nodes` 目录：
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/your-username/ComfyUI-PromptPlusUtils.git

```
安装依赖：
```
bash
pip install dashscope>=1.19.0 Pillow>=10.0.0
```

配置 API 密钥：

方法一：在节点中直接输入 API 密钥

方法二：创建 ComfyUI/custom_nodes/ComfyUI-PromptPlusUtils/api_key.txt 文件并填入密钥

使用方法
文生图模式 (text-to-image)
输入您的提示词

选择 text-to-image 模式

选择合适的文本模型（如 qwen-plus, qwen-max）

图生图模式 (image-to-image)
输入图片和编辑指令

选择 image-to-image 模式

必须选择视觉模型（如 qwen-vl-max-latest）

节点说明
PromptImageHelper
主节点，提供以下输入参数：

prompt: 需要优化的提示词

mode: 优化模式（文生图/图生图）

model: Qwen 模型选择

api_key: 阿里云 API 密钥

image: 图生图模式的输入图片（可选）

skip_rewrite: 跳过优化直接返回

支持的模型
文本模型（文生图）
qwen-plus, qwen-max

qwen-plus-latest, qwen-max-latest

视觉模型（图生图）
qwen-vl-max, qwen-vl-max-latest

qwen-vl-max-2025-08-13, qwen-vl-max-2025-04-08
