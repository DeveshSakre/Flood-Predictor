"""YAML configuration loader supporting configurable paths."""
from pathlib import Path
from typing import Any, Dict
import yaml


def get_repo_root() -> Path:
    """Return the absolute path to the repository root directory."""
    return Path(__file__).resolve().parents[2]


def load_config(config_path: str = "configs/data_config.yaml") -> Dict[str, Any]:
    """
    Load a YAML configuration file.
    
    Args:
        config_path: Relative or absolute path to the configuration YAML.
        
    Returns:
        Nested dictionary of configuration options.
    """
    path = Path(config_path)
    if not path.exists():
        # Check relative to project root
        project_root = get_repo_root()
        alt_path = project_root / config_path
        if alt_path.exists():
            path = alt_path
        else:
            raise FileNotFoundError(f"Configuration file not found at: {config_path} or {alt_path}")
            
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def load_data_config() -> Dict[str, Any]:
    """Shortcut to load data_config.yaml."""
    return load_config("configs/data_config.yaml")
