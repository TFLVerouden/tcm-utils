from __future__ import annotations

from pathlib import Path
import configparser
from tkinter import Tk, filedialog

CONFIG_DIRNAME = ".config"
CONFIG_FILENAME = "file_dialog.ini"
CONFIG_SECTION = "paths"


def find_repo_root(start: Path | None = None) -> Path:
    """Return the repository root by walking upward until a .git folder is found."""
    path = Path(start or Path.cwd()).resolve()
    for candidate in [path] + list(path.parents):
        if (candidate / ".git").exists():
            return candidate
    return path.anchor and Path(path.anchor) or path


def _config_path(repo_root: Path) -> Path:
    config_dir = repo_root / CONFIG_DIRNAME
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir / CONFIG_FILENAME


def _load_config(config_path: Path) -> configparser.ConfigParser:
    config = configparser.ConfigParser()
    if config_path.exists():
        config.read(config_path)
    if CONFIG_SECTION not in config:
        config[CONFIG_SECTION] = {}
    return config


def _save_config(config_path: Path, config: configparser.ConfigParser) -> None:
    with config_path.open("w", encoding="utf-8") as fh:
        config.write(fh)


def _get_last_dir(config: configparser.ConfigParser, key: str, default_dir: Path) -> Path:
    stored = config[CONFIG_SECTION].get(key)
    return Path(stored).expanduser() if stored else default_dir


def _remember_last_dir(config_path: Path, config: configparser.ConfigParser, key: str, directory: Path) -> None:
    config[CONFIG_SECTION][key] = str(directory)
    _save_config(config_path, config)


def ask_open_file(
    key: str,
    title: str,
    filetypes: tuple[tuple[str, str], ...] | list[tuple[str, str]] = (("All files", "*.*"),),
    default_dir: Path | None = None,
    start: Path | None = None,
) -> Path | None:
    """Open a file dialog, remembering the last chosen directory per key."""
    repo_root = find_repo_root(start)
    config_path = _config_path(repo_root)
    config = _load_config(config_path)

    fallback_dir = default_dir or repo_root
    initial_dir = _get_last_dir(config, key, fallback_dir)

    root = Tk()
    root.overrideredirect(True)  # Remove window decorations
    root.geometry('0x0+0+0')  # Make it tiny and offscreen
    root.withdraw()
    
    # Center the root window (helps with macOS dialog positioning)
    try:
        root.eval('tk::PlaceWindow . center')
    except Exception:
        pass  # Fallback if tk::PlaceWindow not available
    
    # TODO: Test this on Windows
    selected = filedialog.askopenfilename(
        title=title,
        initialdir=str(initial_dir),
        filetypes=filetypes,
    )
    root.destroy()

    if not selected:
        return None

    chosen_path = Path(selected).expanduser().resolve()
    _remember_last_dir(config_path, config, key, chosen_path.parent)
    return chosen_path
