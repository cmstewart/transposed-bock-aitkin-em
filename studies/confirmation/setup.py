"""Install the study in an isolated Python 3.11 environment, without changing Jupyter."""
import argparse
from pathlib import Path
import subprocess
import sys
import venv

ROOT = Path(__file__).resolve().parents[2]
TORCH_MEASURE = "a461a8f24eb2960f692afdac0a2202c7d1d507ad"


def interpreter(environment):
    return environment / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


def main():
    if sys.version_info[:2] != (3, 11):
        raise SystemExit("Use a Python 3.11 notebook kernel. The numerical environment is pinned to Python 3.11.")
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=Path, default=ROOT / ".simulation-env")
    args = parser.parse_args()
    python = interpreter(args.environment)
    if not python.exists():
        venv.EnvBuilder(with_pip=True).create(args.environment)
    pip = [str(python), "-m", "pip", "install", "--disable-pip-version-check", "--no-cache-dir"]
    subprocess.run(pip + ["-r", str(Path(__file__).with_name("requirements.txt"))], check=True)
    # torch_measure's optional model dependencies are not used in this study.
    # The imports required by Rasch/TwoPL are installed explicitly above.
    subprocess.run(pip + ["--no-deps", "git+https://github.com/aims-foundations/torch_measure.git@" + TORCH_MEASURE], check=True)
    subprocess.run(pip + ["--no-deps", str(ROOT)], check=True)
    subprocess.run([str(python), "-c", "from transposed_em import LaplaceIRT; from torch_measure.models.twopl import TwoPL; from py_irt.models import IrtModel; print('All three frameworks import successfully')"], check=True)
    print(f"Study interpreter: {python}")


if __name__ == "__main__":
    main()
