"""Configuration loader and path resolution utility for JioStar Contract Intelligence Assistant."""

import os
from pathlib import Path
from typing import Any, Dict
import yaml


def get_project_root() -> Path:
    """Return the absolute path to the project root directory."""
    # This file is in src/config.py, so parent is src/, parent of that is project root
    return Path(__file__).resolve().parent.parent


def load_yaml_config(config_path: Path | str | None = None) -> Dict[str, Any]:
    """Load and parse a YAML configuration file relative to project root.

    Args:
        config_path: Path to YAML config file. If None, defaults to configs/data.yaml.

    Returns:
        Parsed configuration dictionary with resolved paths.
    """
    root = get_project_root()
    if config_path is None:
        target_path = root / "configs" / "data.yaml"
    else:
        target_path = Path(config_path)
        if not target_path.is_absolute():
            target_path = root / target_path

    if not target_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {target_path}")

    with open(target_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


def get_resolved_paths(config: Dict[str, Any] | None = None) -> Dict[str, Path]:
    """Resolve all configured paths to absolute Path objects based on project root.

    Args:
        config: Optional parsed configuration dict. If None, loaded from configs/data.yaml.

    Returns:
        Dictionary mapping path keys to absolute Path instances.
    """
    root = get_project_root()
    if config is None:
        config = load_yaml_config()

    raw_paths = config.get("paths", {})
    resolved: Dict[str, Path] = {"project_root": root}

    for key, val in raw_paths.items():
        p = Path(val)
        if not p.is_absolute():
            resolved[key] = (root / p).resolve()
        else:
            resolved[key] = p.resolve()

    return resolved
