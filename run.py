import os
import socket
import sys

import uvicorn

# Virtual muhitdagi Scripts papkasini PATH ga qo'shish (ffmpeg va demucs uchun)
venv_scripts = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".venv", "Scripts")
if os.path.exists(venv_scripts):
    os.environ["PATH"] = venv_scripts + os.pathsep + os.environ.get("PATH", "")

# Konsol UTF-8 ni qo'llamasa (masalan, chiqish faylga yo'naltirilganda) yiqilib qolmaslik uchun.
for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(errors="replace")
    except (AttributeError, ValueError):
        pass


def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    local_ip = get_local_ip()

    print("=" * 60)
    print("  Demucs Audio Stem Separator ishga tushmoqda...")
    print("  Brauzerda quyidagi manzilni oching:")
    print(f"  -> Local:   http://localhost:{port} yoki http://127.0.0.1:{port}")
    if host == "0.0.0.0" and local_ip != "127.0.0.1":
        print(f"  -> Network: http://{local_ip}:{port}")
    print("=" * 60)
    uvicorn.run("app.main:app", host=host, port=port, reload=False, log_level="info")
