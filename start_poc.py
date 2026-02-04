import sys
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PID_FILE = ROOT / "poc_app.pid"
LOG_FILE = ROOT / "poc_app.log"


def main() -> None:
    if PID_FILE.exists():
        print(f"PID 文件已存在: {PID_FILE}，可能已经在运行。如果确认没有运行，可以手动删除该文件。")
        return

    cmd = [sys.executable, "-m", "streamlit", "run", "poc/app/app.py"]

    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with LOG_FILE.open("a", encoding="utf-8") as log:
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=log,
            stderr=log,
        )

    PID_FILE.write_text(str(proc.pid), encoding="utf-8")
    print(f"已启动多模态视联 POC，PID={proc.pid}，日志文件: {LOG_FILE}")


if __name__ == "__main__":
    main()
