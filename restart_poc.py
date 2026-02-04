import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def main() -> None:
    stop_script = ROOT / "stop_poc.py"
    start_script = ROOT / "start_poc.py"

    print("先尝试停止已有服务（如果存在）...")
    if stop_script.exists():
        subprocess.run([sys.executable, str(stop_script)], check=False)

    time.sleep(1.0)

    print("启动服务...")
    if not start_script.exists():
        raise SystemExit(f"未找到 {start_script}")

    subprocess.run([sys.executable, str(start_script)], check=True)


if __name__ == "__main__":
    main()
