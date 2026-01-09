import re
import xml.etree.ElementTree as ET
import numpy as np
import os
import json
from pathlib import Path

from tcm_utils.io_utils import (
    path_relative_to,
    save_metadata_json,
    copy_file_to_raw_subfolder,
    create_timestamped_filename,
)
from tcm_utils.time_utils import timestamp_str, timestamp_from_file
from tcm_utils.file_dialogs import find_repo_root


def recursive_search(d, target):
    """
    Recursively search for a key in a nested dictionary.

    Parameters
    ----------
    d : dict
        Dictionary to search in
    target : str
        Key to search for

    Returns
    -------
    any
        Value of the found key, or None if not found
    """
    if isinstance(d, dict):
        for k, v in d.items():
            if k == target:
                return v
            res = recursive_search(v, target)
            if res is not None:
                return res
    return None


def extract_cihx_metadata(filepath, output_folder=None, output_file="cihx_metadata",
                          save=True, verbose=True, timestamp_source="file",
                          copy_raw=True):
    """
    Extracts the embedded XML metadata from a .cihx file, prints key info, and saves as a JSON file.

    This function extracts ALL metadata keys from the XML and saves them for comprehensive access,
    but only prints the important/commonly used settings to the console for readability.

    Parameters
    ----------
    filepath : str or Path
        Path to the .cihx file
    output_folder : str or Path, optional
        Directory where output files will be saved. If None, uses docs/cihx_analysis in repo root.
    output_file : str
        Base name for the output file (extension will be added automatically)
    save : bool
        Whether to save the metadata to file
    verbose : bool
        Whether to print important extracted metadata to console
    timestamp_source : str
        Use 'file' for file creation/mod time or 'now' for current time
    copy_raw : bool
        Whether to copy the original .cihx file to a raw_data subfolder

    Returns
    -------
    dict
        Dictionary containing the extracted metadata
    """
    # Convert filepath to Path object
    filepath = Path(filepath).expanduser().resolve()

    # Check if input file exists
    if not filepath.exists():
        raise FileNotFoundError(f"Could not find input file: {filepath}")

    # Setup output folder
    if output_folder is None:
        repo_root = find_repo_root(Path(__file__))
        output_folder = repo_root / "docs" / "cihx_analysis"
    else:
        output_folder = Path(output_folder).expanduser().resolve()

    output_folder.mkdir(parents=True, exist_ok=True)

    # Get base filename and save-time timestamp
    base_filename = filepath.stem
    if timestamp_source == "file":
        timestamp_save = timestamp_from_file(filepath, prefer_creation=True)
        timestamp_source_description = "file_creation_time"
    else:
        timestamp_save = timestamp_str()
        timestamp_source_description = "current_time"

    with open(filepath, "rb") as f:
        data = f.read()

    # Extract XML portion
    xml_match = re.search(rb"<cih>.*</cih>", data, flags=re.DOTALL)
    if not xml_match:
        raise ValueError("No XML metadata found in file.")

    xml_data = xml_match.group(0).decode("utf-8", errors="ignore")
    root = ET.fromstring(xml_data)

    # Convert XML into nested dictionary
    def xml_to_dict(element):
        d = {element.tag: {}}
        children = list(element)
        if children:
            dd = {}
            for dc in map(xml_to_dict, children):
                for k, v in dc.items():
                    if k in dd:
                        if not isinstance(dd[k], list):
                            dd[k] = [dd[k]]
                        dd[k].append(v)
                    else:
                        dd[k] = v
            d = {element.tag: dd}
        if element.text and element.text.strip():
            text = element.text.strip()
            if children or element.attrib:
                d[element.tag]["text"] = text
            else:
                d[element.tag] = text
        return d

    metadata_dict = xml_to_dict(root)

    # Print important camera/recording settings if present
    important_keys = [
        # Camera/system info
        ("date", "Recording date"),
        ("time", "Recording time"),
        ("deviceName", "Camera model"),
        ("firmware", "Firmware version"),

        # Recording parameters
        ("recordRate", "Frame rate [fps]"),
        ("shutterSpeedNsec", "Shutter speed [ns]"),
        ("totalFrame", "Nr of frames"),

        # Image properties
        ("resolution", "Resolution"),
        ("effectiveBit", "Effective bit depth"),
        ("fileFormat", "File format"),

        # Image orientation
        ("flipH", "Horizontal flip"),
        ("flipV", "Vertical flip"),
        ("rotate", "Rotation [deg]"),
    ]

    print("\n=== Extracted Metadata ===")
    if verbose:
        for key, label in important_keys:
            value = recursive_search(metadata_dict, key)
            if value is not None:
                print(f"{label}: {value}")

    # Extract ALL possible keys from the metadata
    def extract_all_keys(d, prefix=""):
        """Extract all keys and their values from nested dictionary"""
        all_keys = {}
        if isinstance(d, dict):
            for k, v in d.items():
                current_key = f"{prefix}.{k}" if prefix else k
                if isinstance(v, dict):
                    all_keys.update(extract_all_keys(v, current_key))
                elif isinstance(v, list):
                    all_keys[current_key] = v
                    # Also extract from list items if they're dictionaries
                    for i, item in enumerate(v):
                        if isinstance(item, dict):
                            all_keys.update(extract_all_keys(
                                item, f"{current_key}[{i}]"))
                else:
                    all_keys[current_key] = v
        return all_keys

    # Get all keys and values
    all_metadata_keys = extract_all_keys(metadata_dict)

    # Derive camera timestamp from camera metadata (date + time)
    camera_date = recursive_search(metadata_dict, "date")
    camera_time = recursive_search(metadata_dict, "time")

    def _camera_timestamp(date_val, time_val) -> str | None:
        if not date_val or not time_val:
            return None
        date_parts = re.findall(r"\d+", str(date_val))
        time_parts = re.findall(r"\d+", str(time_val))
        if len(date_parts) < 3 or len(time_parts) < 3:
            return None
        try:
            year = int(date_parts[0]) % 100
            month = int(date_parts[1])
            day = int(date_parts[2])
            hour = int(time_parts[0])
            minute = int(time_parts[1])
            second = int(time_parts[2])
        except ValueError:
            return None
        return f"{year:02d}{month:02d}{day:02d}_{hour:02d}{minute:02d}{second:02d}"

    camera_ts = _camera_timestamp(camera_date, camera_time)
    if camera_ts:
        timestamp = camera_ts
        timestamp_source_description = "camera_metadata"
    else:
        print("Warning: could not parse camera date/time; using save timestamp instead.")
        timestamp = timestamp_save

    # Save comprehensive metadata as JSON file
    if save:
        # Get repo root for relative paths
        try:
            repo_root = find_repo_root(Path(__file__))
        except Exception:
            repo_root = filepath.parent

        # Copy raw file if requested
        moved_raw_path = None
        if copy_raw:
            moved_raw_path = copy_file_to_raw_subfolder(
                filepath, output_folder)
            print(f"\nCopied original file to {moved_raw_path}")

        # Create metadata output filename
        metadata_filename = create_timestamped_filename(
            base_filename, timestamp, "metadata", "json"
        )
        metadata_path = output_folder / metadata_filename

        # Build metadata dictionary
        metadata_output = {
            "timestamp": timestamp,
            "timestamp_save": timestamp_save,
            "timestamp_source": timestamp_source_description,
            "analysis_run_time": timestamp_str(),
            "input_file_original": path_relative_to(filepath, repo_root),
            "camera_metadata": {
                "date": recursive_search(metadata_dict, "date"),
                "time": recursive_search(metadata_dict, "time"),
                "deviceName": recursive_search(metadata_dict, "deviceName"),
                "firmware": recursive_search(metadata_dict, "firmware"),
                "recordRate": recursive_search(metadata_dict, "recordRate"),
                "shutterSpeedNsec": recursive_search(metadata_dict, "shutterSpeedNsec"),
                "totalFrame": recursive_search(metadata_dict, "totalFrame"),
                "resolution": recursive_search(metadata_dict, "resolution"),
                "effectiveBit": recursive_search(metadata_dict, "effectiveBit"),
                "fileFormat": recursive_search(metadata_dict, "fileFormat"),
                "flipH": recursive_search(metadata_dict, "flipH"),
                "flipV": recursive_search(metadata_dict, "flipV"),
                "rotate": recursive_search(metadata_dict, "rotate"),
            },
            "all_metadata_keys": all_metadata_keys,
            "raw_xml_structure": metadata_dict,
        }

        if moved_raw_path:
            metadata_output["raw_data_path"] = path_relative_to(
                moved_raw_path, repo_root)

        # Save metadata
        save_metadata_json(metadata_output, metadata_path)

        print(f"\nMetadata saved to {metadata_path}")
        print(f"Total keys extracted and saved: {len(all_metadata_keys)}")

    return metadata_dict


def list_all_metadata_keys(metadata_dict, prefix=""):
    """
    Recursively prints all metadata keys found in the dictionary with their values.

    Parameters
    ----------
    metadata_dict : dict
        Nested dictionary of metadata (from extract_cihx_metadata)
    prefix : str
        Used internally for recursion (for nested keys)
    """
    if isinstance(metadata_dict, dict):
        for k, v in metadata_dict.items():
            new_prefix = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                list_all_metadata_keys(v, new_prefix)
            elif isinstance(v, list):
                for i, item in enumerate(v):
                    list_all_metadata_keys(item, f"{new_prefix}[{i}]")
            else:
                print(f"{new_prefix}: {v}")
    else:
        print(f"{prefix}: {metadata_dict}")


def load_cihx_metadata(filepath):
    """
    Load previously saved CIHX metadata from a JSON file.

    Parameters
    ----------
    filepath : str or Path
        Path to the JSON file containing saved metadata

    Returns
    -------
    dict
        Dictionary containing the loaded metadata
    """
    filepath = Path(filepath)

    # Ensure file has correct extension
    if not filepath.suffix == '.json':
        filepath = filepath.with_suffix('.json')

    # Check if file exists
    if not filepath.exists():
        raise FileNotFoundError(f"Metadata file not found: {filepath}")

    # Load the data
    with filepath.open('r', encoding='utf-8') as fh:
        loaded_data = json.load(fh)

    print(f"Loaded metadata from {filepath}")
    return loaded_data


if __name__ == "__main__":
    print("CIHX metadata extraction tool")
    print("=" * 40)
    print("Select a .cihx file to extract metadata...")

    try:
        from tcm_utils.file_dialogs import ask_open_file

        # Get repo root and default directory
        repo_root = find_repo_root(Path(__file__))
        default_dir = repo_root / "docs" / "cihx_analysis"

        # Open file dialogue
        selected_file = ask_open_file(
            key="cihx_metadata_extraction",
            title="Select a .cihx file",
            filetypes=[("CIHX files", "*.cihx"), ("All files", "*.*")],
            default_dir=default_dir,
            start=Path(__file__),
        )

        if selected_file:
            print(f"Selected file: {selected_file}")

            # Extract metadata from selected file
            metadata = extract_cihx_metadata(
                selected_file,
                save=True,
                verbose=True,
                timestamp_source="file",
                copy_raw=True,
            )
        else:
            print("No file selected. Exiting.")

    except ImportError as e:
        print(f"Error: {e}")
        print("Please ensure tcm_utils is properly installed.")
        print("Usage from code: extract_cihx_metadata('/path/to/file.cihx')")
