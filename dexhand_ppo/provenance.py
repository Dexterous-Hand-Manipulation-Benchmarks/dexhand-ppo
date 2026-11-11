"""Per-run record of everything needed to reproduce or report a run."""
import importlib
import json
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PACKAGES = ("torch", "numpy", "hydra", "omegaconf", "gymnasium", "genesis", "dexmachina", "taichi", "wandb",
            "dexhand_bench")
REPOS = {"dexhand_ppo": "dexhand_ppo", "dexmachina": "dexmachina", "genesis": "genesis",
         "dexhand_bench": "dexhand_bench"}


def _git(path: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(path), *args], capture_output=True, text=True, check=True).stdout.strip()


def git_state(module_name: str, run_dir: Path):
    """Commit and dirty flag of the git repo containing ``module_name``; saves the diff if dirty."""
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        return None
    module_file = getattr(module, "__file__", None)
    if module_file is None:
        return None
    path = Path(module_file).resolve().parent
    try:
        root = Path(_git(path, "rev-parse", "--show-toplevel"))
        state = {"root": str(root), "commit": _git(root, "rev-parse", "HEAD"),
                 "branch": _git(root, "rev-parse", "--abbrev-ref", "HEAD")}
        diff = _git(root, "diff", "HEAD")
        untracked = _git(root, "ls-files", "--others", "--exclude-standard")
    except subprocess.CalledProcessError as e:  # not a git repo, or no commits yet
        return {"root": str(path), "commit": None, "error": (e.stderr or "").strip()}
    except FileNotFoundError:
        return {"root": str(path), "commit": None, "error": "git not installed"}
    state["dirty"] = bool(diff or untracked)
    if diff:
        diff_file = run_dir / f"git_diff_{module_name}.patch"
        diff_file.write_text(diff + "\n")
        state["diff_file"] = diff_file.name
    if untracked:
        state["untracked"] = untracked.splitlines()
    return state


def package_versions() -> dict:
    versions = {}
    for name in PACKAGES:
        try:
            module = importlib.import_module(name)
            versions[name] = getattr(module, "__version__", "unknown")
        except Exception:  # noqa: BLE001 - optional packages may fail to import for many reasons
            versions[name] = None
    return versions


def hardware() -> dict:
    info = {"hostname": socket.gethostname(), "platform": platform.platform(), "python": sys.version.split()[0],
            "cpu_count": os.cpu_count()}
    try:
        import torch
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["cuda"] = torch.version.cuda
    except ImportError:
        pass
    return info


def write_provenance(run_dir: str, resolved_cfg: dict, extra: dict | None = None) -> dict:
    """Writes provenance.json and config.json into run_dir and returns the record (also logged to W&B)."""
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    record = {
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "cwd": os.getcwd(),
        "repos": {name: git_state(module, run_path) for name, module in REPOS.items()},
        "packages": package_versions(),
        "hardware": hardware(),
        "env_vars": {k: os.environ.get(k) for k in ("WANDB_MODE", "CUDA_VISIBLE_DEVICES", "SLURM_JOB_ID",
                                                     "UV_PROJECT_ENVIRONMENT")},
    }
    if extra:
        record.update(extra)
    (run_path / "provenance.json").write_text(json.dumps(record, indent=1, default=str) + "\n")
    (run_path / "config.json").write_text(json.dumps(resolved_cfg, indent=1, default=str) + "\n")
    return record
