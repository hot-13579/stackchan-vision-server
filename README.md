# StackChan Vision Server

Stack-chan 视觉伴侣的 PC 端服务器，提供：
- 人脸检测（OpenCV YuNet）
- 多模态大模型视觉问答
- TTS 语音合成

## 安装

```bash
pip install -r requirements.txt
```

## 运行

```bash
# 设置 API Key（二选一）
set DASHSCOPE_API_KEY=你的阿里云百炼 Key
# 或 set OPENAI_API_KEY=你的 OpenAI Key

python vision_server.py
```

启动后会显示局域网 IP，例如：
```
========================================================
 Stack-chan 视觉伴侣服务器
 局域网地址:  http://192.168.2.14:8787
 ...
========================================================
```

## API

- `GET /healthz` - 健康检查
- `POST /face` - 发送 JPEG，返回人脸偏移 `{"found": bool, "x": float, "y": float}`
- `POST /describe?q=问题` - 发送 JPEG + 问题，返回描述文本 `{"text": "..."}`
- `GET /tts?text=文字` - 返回中文语音 WAV

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| PORT | 8787 | 服务器端口 |
| VISION_BASE_URL | dashscope.aliyuncs.com | 大模型 API 地址 |
| VISION_MODEL | qwen-vl-max | 大模型名称 |
| EDGE_VOICE | zh-CN-XiaoxiaoNeural | 语音音色 |
| DASHSCOPE_API_KEY | - | 阿里云百炼 API Key |
| OPENAI_API_KEY | - | OpenAI 兼容 API Key |
