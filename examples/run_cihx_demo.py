#!/usr/bin/env python3
"""
Demo script for testing CIHX metadata extraction functionality.

This script demonstrates the JSON-based metadata export system
and tests the updated read_cihx module.
"""
from __future__ import annotations
from tcm_utils.file_dialogs import ask_open_file, find_repo_root
from tcm_utils.read_cihx import extract_cihx_metadata, load_cihx_metadata

import sys
from pathlib import Path

# Ensure local src/ is on the path when running from the repo
REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

print("=" * 60)
print("CIHX Metadata Extraction Demo")
print("=" * 60)
print("\nThis script demonstrates:")
print("  1. Extracting metadata from .cihx files")
print("  2. Saving metadata as JSON (not .npz)")
print("  3. Copying raw files to organized folders")
print("  4. Using timestamps from file creation time")
print("\n" + "=" * 60)

# Get repository root
repo_root = find_repo_root(Path(__file__))
examples_dir = repo_root / "examples" / "read_cihx_outputs"

print(f"\nLooking for .cihx files in: {examples_dir}")

# Ask user to select a .cihx file
selected_file = ask_open_file(
    key="cihx_demo",
    title="Select a .cihx file for metadata extraction",
    filetypes=[("CIHX files", "*.cihx"), ("All files", "*.*")],
    default_dir=examples_dir,
    start=Path(__file__),
)

if not selected_file:
    print("\nNo file selected. Exiting.")
    sys.exit(1)

print(f"\nSelected file: {selected_file.name}")
print(f"Full path: {selected_file}")

# Setup output directory
output_dir = repo_root / "examples" / "cihx_demo_outputs"
output_dir.mkdir(parents=True, exist_ok=True)

print(f"\nOutput directory: {output_dir}")
print("\nExtracting metadata...")
print("-" * 60)

# Extract metadata with file timestamp
metadata = extract_cihx_metadata(
    selected_file,
    output_folder=output_dir,
    save=True,
    verbose=True,
    timestamp_source="file",
    copy_raw=True,
)

print("\n" + "-" * 60)
print("Metadata extraction complete!")

# Find the saved metadata file
base_filename = selected_file.stem
metadata_files = list(output_dir.glob(f"{base_filename}_*_metadata.json"))

if metadata_files:
    # Load and display the saved metadata
    latest_metadata_file = sorted(metadata_files)[-1]
    print(f"\nLoading saved metadata from: {latest_metadata_file.name}")

    loaded_metadata = load_cihx_metadata(latest_metadata_file)

    print("\n" + "=" * 60)
    print("Saved Metadata Structure:")
    print("=" * 60)

    # Display top-level keys
    print("\nTop-level keys in saved JSON:")
    for key in loaded_metadata.keys():
        if isinstance(loaded_metadata[key], dict):
            print(
                f"  - {key}: (dict with {len(loaded_metadata[key])} keys)")
        elif isinstance(loaded_metadata[key], list):
            print(
                f"  - {key}: (list with {len(loaded_metadata[key])} items)")
        else:
            print(f"  - {key}: {loaded_metadata[key]}")

    # Display camera metadata if present
    if "camera_metadata" in loaded_metadata:
        print("\nCamera Metadata Summary:")
        cam_meta = loaded_metadata["camera_metadata"]
        for key, value in cam_meta.items():
            if value is not None:
                print(f"  - {key}: {value}")

    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)

    print(f"\nOutput files location: {output_dir}")
    print(f"Raw file copy in: {output_dir / 'raw_data'}")

else:
    print("\nWarning: Could not find saved metadata file.")
