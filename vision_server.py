# -*- coding: utf-8 -*-
"""
Stack-chan 视觉伴侣服务器（运行在 PC 上，与机器人同一局域网）

提供三个接口：
  POST /face            请求体为 JPEG -> OpenCV 人脸检测 -> {"found": bool, "x": ..., "y": ...}
                        x/y 为人脸中心相对画面中心的归一化偏移（右/下为正，范围约 -0.5~0.5）
  POST /describe?q=...  请求体为 JPEG -> 多模态大模型 -> {"text": "..."}
  GET  /<文本>          远程 TTS（edge-tts 中文语音）-> audio/wav, 24kHz 单声道 16bit
                        对应社区固件 tts.type = "remote" 的拉流协议

API Key 只保存在 PC 上，机器人固件里不放任何密钥。
环境变量：
  DASHSCOPE_API_KEY   阿里云百炼 API Key（推荐，国内直连）
  OPENAI_API_KEY      或者任意 OpenAI 兼容服务的 Key（二选一）
  VISION_BASE_URL     默认 https://dashscope.aliyuncs.com/compatible-mode/v1
  VISION_MODEL        默认 qwen-vl-max
  EDGE_VOICE          默认 zh-CN-XiaoxiaoNeural（可选 zh-CN-YunxiNeural 等）
  PORT                默认 8787
"""

import asyncio
import base64
import io
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

PORT = int(os.environ.get("PORT", "8787"))
VISION_BASE_URL = os.environ.get("VISION_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
VISION_MODEL = os.environ.get("VISION_MODEL", "qwen-vl-max")
EDGE_VOICE = os.environ.get("EDGE_VOICE", "zh-CN-XiaoxiaoNeural")
API_KEY = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""

# ---------- 人脸检测器（YuNet，OpenCV 4.5+/5.x） ----------
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL_FILE = os.path.join(MODEL_DIR, "face_detection_yunet_2023mar.onnx")
MODEL_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/"
             "face_detection_yunet/face_detection_yunet_2023mar.onnx")


def _ensure_model() -> bool:
    """模型缺失时自动下载（约 230KB）。"""
    if os.path.isfile(MODEL_FILE):
        return True
    os.makedirs(MODEL_DIR, exist_ok=True)
    print(f"[model] downloading YuNet face model -> {MODEL_FILE}", flush=True)
    try:
        with urllib.request.urlopen(MODEL_URL, timeout=60) as resp, open(MODEL_FILE, "wb") as f:
            f.write(resp.read())
        return os.path.isfile(MODEL_FILE) and os.path.getsize(MODEL_FILE) > 100_000
    except Exception as e:  # noqa: BLE001
        print(f"[model] download failed: {e}", flush=True)
        return False


_face_detector = None
_face_lock = threading.Lock()  # ThreadingHTTPServer 多线程保护

if _ensure_model() and hasattr(cv2, "FaceDetectorYN"):
    try:
        # OpenCV 5.x 用位置参数：(model, config, input_size, score_thr, nms_thr)
        _face_detector = cv2.FaceDetectorYN.create(MODEL_FILE, "", (320, 240), 0.7, 0.3)
    except cv2.error:
        # 旧版 OpenCV 4.x 兼容写法
        _face_detector = cv2.FaceDetectorYN.create(
            MODEL_FILE, "", (320, 240), score_threshold=0.7, nms_threshold=0.3
        )
elif not hasattr(cv2, "FaceDetectorYN"):
    print("[face] 当前 OpenCV 版本不支持 YuNet，请升级: pip install -U opencv-python-headless")
    _face_detector = None


def detect_face(jpeg_bytes: bytes):
    """返回 (found, x_offset, y_offset)。偏移为相对画面中心的归一化值，右/下为正。"""
    if _face_detector is None:
        return False, 0.0, 0.0
    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return False, 0.0, 0.0
    h, w = img.shape[:2]
    with _face_lock:
        _face_detector.setInputSize((w, h))
        _, faces = _face_detector.detect(img)
    if faces is None:
        return False, 0.0, 0.0
    if faces is None:
        return False, 0.0, 0.0
    faces = np.asarray(faces)
    if faces.size == 0:
        return False, 0.0, 0.0
    faces = faces.reshape(-1, faces.shape[-1]) if faces.ndim > 1 else faces.reshape(1, -1)
    rows = [row for row in faces if len(row) > 14 and row[14] > 0.6]
    if not rows:
        return False, 0.0, 0.0
    # 取最大的一张脸（通常就是正在说话的人）
    fx, fy, fw, fh = max(rows, key=lambda r: r[2] * r[3])[:4]
    cx = fx + fw / 2.0
    cy = fy + fh / 2.0
    return True, (cx - w / 2.0) / w, (cy - h / 2.0) / h


def describe_image(jpeg_bytes: bytes, question: str) -> str:
    """调用 OpenAI 兼容协议的多模态大模型描述图片。"""
    if not API_KEY:
        return "我的大脑还没配好。请在电脑上设置 DASHSCOPE_API_KEY 环境变量后重启视觉服务器。"
    b64 = base64.b64encode(jpeg_bytes).decode("ascii")
    payload = {
        "model": VISION_MODEL,
        "messages": [
            {
                "role": "system",
                "content": "你是一个桌面机器人 Stack-chan 的眼睛。回答必须口语化、简短，"
                "一两句话以内，不要使用列表、标题或表情符号。",
            },
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text", "text": question or "用一两句中文简洁描述你看到的画面"},
                ],
            },
        ],
        "max_tokens": 150,
    }
    req = urllib.request.Request(
        VISION_BASE_URL.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + API_KEY,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:  # noqa: BLE001
        print(f"[describe] LLM error: {e}", flush=True)
        return "我的视觉大脑暂时连不上，请检查服务器的网络和密钥配置。"


def _edge_tts_mp3(text: str) -> bytes:
    import edge_tts

    async def _run():
        communicate = edge_tts.Communicate(text, EDGE_VOICE)
        buf = io.BytesIO()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                buf.write(chunk["data"])
        return buf.getvalue()

    return asyncio.run(_run())


def _mp3_to_wav(mp3_bytes: bytes, sample_rate: int = 24000) -> bytes:
    import imageio_ffmpeg

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f_in:
        f_in.write(mp3_bytes)
        in_path = f_in.name
    out_path = in_path + ".wav"
    try:
        subprocess.run(
            [ffmpeg, "-y", "-i", in_path, "-ar", str(sample_rate), "-ac", "1",
             "-f", "wav", "-acodec", "pcm_s16le", out_path],
            check=True, capture_output=True,
        )
        with open(out_path, "rb") as f:
            return f.read()
    finally:
        for p in (in_path, out_path):
            try:
                os.unlink(p)
            except OSError:
                pass


def tts_wav(text: str) -> bytes:
    return _mp3_to_wav(_edge_tts_mp3(text))


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # 保持控制台干净
        pass

    def _send_json(self, obj, status=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_wav(self, wav: bytes):
        self.send_response(200)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", str(len(wav)))
        self.end_headers()
        self.wfile.write(wav)

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(length) if length > 0 else b""

    # ---------- TTS：GET /<url编码的文本> 或 GET /tts?text=... ----------
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        query = urllib.parse.parse_qs(parsed.query)
        if parsed.path == "/healthz":
            self._send_json({"ok": True})
            return
        if parsed.path == "/tts":
            text = query.get("text", [""])[0]
        else:
            text = urllib.parse.unquote(parsed.path.lstrip("/"))
        if not text.strip():
            self._send_json({"error": "empty text"}, status=400)
            return
        print(f"[tts] {text[:40]}", flush=True)
        try:
            self._send_wav(tts_wav(text))
        except Exception as e:  # noqa: BLE001
            print(f"[tts] error: {e}", flush=True)
            self._send_json({"error": str(e)}, status=503)

    # ---------- POST /face 与 /describe ----------
    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        jpeg = self._read_body()
        if not jpeg:
            self._send_json({"error": "empty body"}, status=400)
            return
        if parsed.path == "/face":
            found, x, y = detect_face(jpeg)
            self._send_json({"found": bool(found), "x": round(x, 4), "y": round(y, 4)})
        elif parsed.path == "/describe":
            query = urllib.parse.parse_qs(parsed.query)
            question = query.get("q", [""])[0]
            print(f"[describe] q={question[:40]}", flush=True)
            self._send_json({"text": describe_image(jpeg, question)})
        else:
            self._send_json({"error": "unknown path"}, status=404)


def lan_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return "127.0.0.1"


def main():
    ip = lan_ip()
    print("=" * 56)
    print(" Stack-chan 视觉伴侣服务器")
    print(f" 局域网地址:  http://{ip}:{PORT}")
    print(f" 大模型:      {VISION_MODEL}  ({'已配置密钥' if API_KEY else '未配置密钥！'})")
    print(f" TTS 声音:    {EDGE_VOICE}")
    print(" 请把 mod.js 顶部的 SERVER_HOST 改成上面的 IP，")
    print(" 并在 Stack-chan 的 tts 设置里填同样的 host 和 port。")
    print("=" * 56, flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
