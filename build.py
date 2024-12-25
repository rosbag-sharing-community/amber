import shutil
from distutils import log as distutils_log
from pathlib import Path
from typing import Any, Dict

import skbuild
import skbuild.constants
import subprocess
import sysconfig
import os

__all__ = ("build",)


def get_poetry_venv_path() -> Path:
    """Get the path to the Poetry virtual environment."""
    try:
        result = subprocess.run(
            ["poetry", "env", "info", "--path"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        venv_path = result.stdout.strip()
        return Path(venv_path)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Failed to get Poetry virtual environment: {e.stderr}"
        ) from e


def build(setup_kwargs: Dict[str, Any]) -> None:
    """Build C-extensions."""
    skbuild.setup(**setup_kwargs, script_args=["build_ext"])

    src_dir = Path(skbuild.constants.CMAKE_INSTALL_DIR())

    # Get the installation directory for the Poetry virtual environment
    venv_path = get_poetry_venv_path()
    dest_dir = (
        venv_path / "lib" / f"python{sysconfig.get_python_version()}" / "site-packages"
    )

    print("Copy from " + str(src_dir) + " to " + str(dest_dir))

    # Copy built C-extensions back to the Poetry virtual environment.
    copy_files(src_dir, dest_dir, "**/*.pyd")
    copy_files(src_dir, dest_dir, "**/*.so")


def remove_files(target_dir: Path, pattern: str) -> None:
    """Delete files matched with a glob pattern in a directory tree."""
    for path in target_dir.glob(pattern):
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        distutils_log.info(f"removed {path}")


def copy_files(src_dir: Path, dest_dir: Path, pattern: str) -> None:
    """Copy files matched with a glob pattern in a directory tree to another."""
    for src in src_dir.glob(pattern):
        dest = dest_dir / src.relative_to(src_dir)
        if src.is_dir():
            # NOTE: inefficient if subdirectories also match to the pattern.
            copy_files(src, dest, "*")
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
            distutils_log.info(f"copied {src} to {dest}")


if __name__ == "__main__":
    build({})
