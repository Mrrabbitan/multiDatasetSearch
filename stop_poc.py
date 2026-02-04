import os
import signal
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PID_FILE = ROOT / "poc_app.pid"


def main() -> None:
    if not PID_FILE.exists():
        print(f"未找到 PID 文件: {PID_FILE}，可能服务未启动。")
        return

    pid_text = PID_FILE.read_text(encoding="utf-8").strip()
    if not pid_text:
        print("PID 文件为空，自动删除。")
        PID_FILE.unlink(missing_ok=True)
        return

    try:
        pid = int(pid_text)
    except ValueError:
        print(f"PID 文件内容非法: {pid_text}，自动删除。")
        PID_FILE.unlink(missing_ok=True)
        return

    try:
        if sys.platform.startswith("win"):
            # Windows 下使用 taskkill 结束进程
            subprocess.run(["taskkill", "/PID", str(pid), "/F", "/T"], check=False)
        else:
            os.kill(pid, signal.SIGTERM)
        print(f"已尝试停止进程 PID={pid}")
    except Exception as exc:
        print(f"停止进程失败: {exc}")

    try:
        PID_FILE.unlink()
    except OSError:
        pass


if __name__ == "__main__":
    main()
