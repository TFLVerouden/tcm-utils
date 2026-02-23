from __future__ import annotations

"""Small Tkinter-based file/directory pickers with per-repo persistence.

This module provides thin wrappers around :mod:`tkinter.filedialog` and a few
lightweight helpers for persisting arbitrary per-repo state (e.g., remembered
paths or connection details). It:

- Remembers the last-used directory for each logical dialog (via a user-provided
    ``key``), stored in an INI file.
- Stores that INI file *per git repository* by locating the nearest ``.git``
    directory and writing to ``<repo>/.config/file_dialog.ini``.

The intent is to make CLI or script workflows less annoying: the next time you
open a dialog or need to remember a setting, the helper starts in the state you
left it in.
"""

from pathlib import Path
import configparser
from tkinter import Tk, filedialog

CONFIG_DIRNAME = ".config"
CONFIG_FILENAME = "file_dialog.ini"
CONFIG_SECTION = "paths"


def find_repo_root(start: Path | None = None, prefer_cwd: bool = True) -> Path:
    """Find the git repository root to use for per-repo configuration.

    This walks upward from ``start`` (default: current working directory) until a
    directory containing a ``.git`` folder is found.

    Parameters
    ----------
    start:
        Path to start searching from. If omitted, uses the current working
        directory.
    prefer_cwd:
        When True (default), the repository that contains the current working
        directory is preferred over the repository containing ``start``.

        This matters when calling these helpers from code that lives inside an
        installed package: ``start`` might point into ``site-packages`` while the
        *user* is running a script inside their own git repo. Preferring CWD keeps
        the remembered dialog locations stored alongside the user's project.

    Returns
    -------
    Path
        The repository root if one can be found, otherwise a best-effort
        fallback (drive root / filesystem anchor / ``start``).
    """

    def _git_root(path: Path) -> Path | None:
        for candidate in [path] + list(path.parents):
            if (candidate / ".git").exists():
                return candidate
        return None

    path = Path(start or Path.cwd()).resolve()
    start_root = _git_root(path)
    cwd_root = _git_root(Path.cwd().resolve()) if prefer_cwd else None

    if prefer_cwd and cwd_root is not None:
        if start_root is None or start_root != cwd_root:
            return cwd_root

    # Fall back to filesystem root/anchor when we're not inside a git repository.
    return start_root or cwd_root or (path.anchor and Path(path.anchor)) or path


def _config_path(repo_root: Path) -> Path:
    """Return the default INI config path for a given repository root.

    Side effect: ensures the ``.config`` directory exists.
    """
    config_dir = repo_root / CONFIG_DIRNAME
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir / CONFIG_FILENAME


def get_config_path(
    filename: str,
    start: Path | None = None,
    prefer_cwd: bool = True,
) -> Path:
    """Return a file path inside the repo's ``.config`` directory.

    This is a small convenience for other modules that want to store per-repo
    state in the same place as these dialogs.
    """
    repo_root = find_repo_root(start, prefer_cwd=prefer_cwd)
    config_dir = repo_root / CONFIG_DIRNAME
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir / filename


def _load_config(
    config_path: Path,
    section: str = CONFIG_SECTION,
) -> configparser.ConfigParser:
    """Load the INI config file and ensure ``section`` exists."""
    config = configparser.ConfigParser()
    if config_path.exists():
        config.read(config_path)
    if section not in config:
        config[section] = {}
    return config


def _save_config(config_path: Path, config: configparser.ConfigParser) -> None:
    """Persist the INI config to disk (UTF-8 encoded)."""
    with config_path.open("w", encoding="utf-8") as fh:
        config.write(fh)


def _get_last_dir(config: configparser.ConfigParser, key: str, default_dir: Path) -> Path:
    """Return the last remembered directory for ``key``.

    If the key has not been stored yet, returns ``default_dir``.
    """
    stored = config[CONFIG_SECTION].get(key)
    return Path(stored).expanduser() if stored else default_dir


def _remember_last_dir(config_path: Path, config: configparser.ConfigParser, key: str, directory: Path) -> None:
    """Store the directory for ``key`` in the INI file."""
    config[CONFIG_SECTION][key] = str(directory)
    _save_config(config_path, config)


def ask_open_file(
    key: str,
    title: str,
    filetypes: tuple[tuple[str, str], ...] | list[tuple[str, str]] = (
        ("All files", "*.*"),),
    default_dir: Path | None = None,
    start: Path | None = None,
) -> Path | None:
    """Ask the user to select a file.

    The dialog starts in the last directory used for the same ``key`` and stores
    the directory again after a successful selection.

    Parameters
    ----------
    key:
        Stable identifier used to remember the last-used directory in the INI
        file (e.g. ``"calibration_csv"``).
    title:
        Dialog title.
    filetypes:
        File type filters in the format expected by Tkinter. Note: the current
        implementation always passes an "All files" filter to Tkinter.
    default_dir:
        Fallback directory if no directory has been remembered for ``key``.
        Defaults to the repository root.
    start:
        Optional path used as a hint for locating the repository root.

    Returns
    -------
    Path | None
        The selected file path, or None when the user cancels.
    """
    repo_root = find_repo_root(start, prefer_cwd=True)
    config_path = _config_path(repo_root)
    config = _load_config(config_path)

    fallback_dir = default_dir or repo_root
    initial_dir = _get_last_dir(config, key, fallback_dir)

    # Tkinter requires a root window even when we only show a file dialog.
    # We hide it so we don't get an extra empty window popping up.
    root = Tk()
    root.overrideredirect(True)  # Remove window decorations
    root.geometry("0x0+0+0")  # Make it tiny and offscreen
    root.withdraw()

    # Center the root window (helps with macOS dialog positioning).
    # Some Tk builds don't include tk::PlaceWindow, hence the broad except.
    try:
        root.eval('tk::PlaceWindow . center')
    except Exception:
        pass  # Fallback if tk::PlaceWindow not available

    # Keep a small UX hint in the console for CLI workflows.
    print(title)
    path = filedialog.askopenfilename(title=title, initialdir=str(initial_dir),
                                      filetypes=(("All files", "*.*"),))
    root.destroy()
    if not path:
        return None

    chosen_path = Path(path).expanduser().resolve()
    _remember_last_dir(config_path, config, key, chosen_path.parent)
    return chosen_path


def ask_directory(
    key: str,
    title: str = "Select directory",
    default_dir: Path | None = None,
    start: Path | None = None,
) -> Path | None:
    """Ask the user to select a directory.

    Works like :func:`ask_open_file`, but uses a directory picker and remembers
    the last used directory per ``key``.

    Returns
    -------
    Path | None
        The selected directory path, or None when the user cancels.
    """
    repo_root = find_repo_root(start, prefer_cwd=True)
    config_path = _config_path(repo_root)
    config = _load_config(config_path)

    fallback_dir = default_dir or repo_root
    initial_dir = _get_last_dir(config, key, fallback_dir)

    # Create and hide a root window; required for the dialog to work.
    root = Tk()
    root.overrideredirect(True)
    root.geometry("0x0+0+0")
    root.withdraw()

    try:
        root.eval('tk::PlaceWindow . center')
    except Exception:
        pass

    print(title)
    selected = filedialog.askdirectory(
        title=title, initialdir=str(initial_dir))
    root.destroy()

    if not selected:
        return None

    chosen_path = Path(selected).expanduser().resolve()
    print(f"Selected directory: {chosen_path}")
    _remember_last_dir(config_path, config, key, chosen_path.parent)
    return chosen_path
