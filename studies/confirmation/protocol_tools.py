"""Freeze scientific settings, source files and numerical dependencies per output tree."""
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import platform
import sys

STUDY = Path(__file__).resolve().parent
ROOT = STUDY.parents[1]
OUTPUT = ROOT / "simulation-output"
PACKAGES = ("numpy", "scipy", "pandas", "jax", "jaxlib", "torch", "pyro-ppl",
            "py-irt", "torch_measure", "filelock")


def protocol():
    return json.loads((STUDY / "protocol.json").read_text())


def write_json(path, value):
    import numpy as np
    def convert(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, np.generic):
            return x.item()
        raise TypeError(type(x))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, default=convert) + "\n")
    temp.replace(path)


def manifest():
    import transposed_em
    sources = list(STUDY.glob("*.py")) + [STUDY / "protocol.json", STUDY / "requirements.txt"]
    package = Path(transposed_em.__file__).resolve().parent
    hashes = {"study/" + p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    hashes.update({"transposed_em/" + p.relative_to(package).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in package.rglob("*.py")})
    for name in ("torch_measure", "py_irt"):
        distribution = metadata.distribution(name.replace("_", "-"))
        for file in distribution.files or []:
            if str(file).startswith(name + "/") and str(file).endswith(".py"):
                hashes[str(file)] = hashlib.sha256(distribution.locate_file(file).read_bytes()).hexdigest()
    return {"sources": hashes, "versions": {p: metadata.version(p) for p in PACKAGES},
            "python": platform.python_version(), "system": platform.system(),
            "machine": platform.machine(), "protocol": protocol()}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def verify(output=OUTPUT, require_lock=True):
    actual = manifest()
    lock = Path(output) / "protocol.lock.json"
    if lock.exists():
        if json.loads(lock.read_text()) != actual:
            raise RuntimeError("Sources, protocol or environment changed. Use the original environment; do not overwrite the lock.")
    elif require_lock:
        raise RuntimeError("Run the notebook's smoke test before production.")
    return digest(actual)


def freeze(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "protocol.lock.json"
    if path.exists():
        return verify(output)
    with path.open("x") as stream:
        json.dump(manifest(), stream, indent=2, sort_keys=True)
    return verify(output)
