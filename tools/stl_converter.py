#!/usr/bin/env python3
"""
STL Binary to ASCII Converter

A command-line tool to convert binary STL files to ASCII STL format.
This tool can process multiple STL files and output them to a specified directory.

Usage:
    python stl_converter.py input1.stl input2.stl -o output_directory
    python stl_converter.py *.stl -o output_directory
    python stl_converter.py --help

Author: Auto-generated tool for TAUV project
"""

import argparse
import os
import sys
from pathlib import Path

try:
    import numpy as np
    import stl
    from stl import mesh
except ImportError as e:
    print(f"Error: Required dependency not found: {e}")
    print("Please install numpy-stl: pip install numpy-stl")
    sys.exit(1)


def is_binary_stl(filepath: str) -> bool:
    """
    Determine if an STL file is in binary format.

    Binary STL files start with an 80-byte header, followed by a 4-byte unsigned integer
    indicating the number of triangular facets, then the facets themselves.
    ASCII STL files start with the word "solid".

    Args:
        filepath: Path to the STL file

    Returns:
        True if the file is binary STL, False if ASCII
    """
    try:
        with open(filepath, 'rb') as f:
            # Read the first 5 bytes to check for "solid" keyword
            header = f.read(5)
            if header.startswith(b'solid'):
                # Could be ASCII, but some binary files also start with "solid"
                # Read more to be sure - if it's truly ASCII, we should see readable text
                f.seek(0)
                chunk = f.read(1024)
                try:
                    # If we can decode it as ASCII and it contains STL keywords, it's ASCII
                    text = chunk.decode('ascii', errors='strict')
                    if 'facet' in text.lower() or 'vertex' in text.lower():
                        return False
                except UnicodeDecodeError:
                    pass

            # If we can't determine from the header, try to parse as binary
            return True
    except Exception as e:
        print(f"Warning: Could not determine STL format for {filepath}: {e}")
        return True  # Default to binary


def convert_stl_to_ascii(input_path: str, output_path: str) -> bool:
    """
    Convert a binary STL file to ASCII format.

    Args:
        input_path: Path to the input STL file
        output_path: Path for the output ASCII STL file

    Returns:
        True if conversion was successful, False otherwise
    """
    try:
        # Check if input file is already ASCII
        if not is_binary_stl(input_path):
            print(f"  File {input_path} is already in ASCII format")
            # Copy the file to output directory
            import shutil

            shutil.copy2(input_path, output_path)
            return True

        # Load the STL mesh
        stl_mesh = mesh.Mesh.from_file(input_path)

        # Save as ASCII
        stl_mesh.save(output_path, mode=stl.Mode.ASCII)

        return True

    except Exception as e:
        print(f"  Error converting {input_path}: {e}")
        return False


def validate_inputs(input_files: list[str], output_dir: str) -> tuple[list[str], str]:
    """
    Validate input files and output directory.

    Args:
        input_files: List of input STL file paths
        output_dir: Output directory path

    Returns:
        Tuple of (valid_input_files, validated_output_dir)
    """
    # Validate input files
    valid_files = []
    for file_path in input_files:
        if not os.path.isfile(file_path):
            print(f"Warning: Input file does not exist: {file_path}")
            continue

        if not file_path.lower().endswith('.stl'):
            print(f"Warning: File does not have .stl extension: {file_path}")
            continue

        valid_files.append(file_path)

    if not valid_files:
        print("Error: No valid STL files found in input")
        sys.exit(1)

    # Validate and create output directory
    output_path = Path(output_dir)
    try:
        output_path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Error: Cannot create output directory {output_dir}: {e}")
        sys.exit(1)

    return valid_files, str(output_path)


def main():
    """Main function for the STL converter CLI."""
    parser = argparse.ArgumentParser(
        description='Convert binary STL files to ASCII format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s file1.stl file2.stl -o output_folder
  %(prog)s *.stl -o ascii_stls
  %(prog)s model.stl -o . --verbose
        """,
    )

    parser.add_argument('input_files', nargs='+', help='One or more STL files to convert')

    parser.add_argument(
        '-o', '--output', required=True, help='Output directory for converted ASCII STL files'
    )

    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output')

    parser.add_argument(
        '--overwrite', action='store_true', help='Overwrite existing files in output directory'
    )

    args = parser.parse_args()

    # Validate inputs
    valid_files, output_dir = validate_inputs(args.input_files, args.output)

    print(f"Converting {len(valid_files)} STL file(s) to ASCII format...")
    print(f"Output directory: {output_dir}")

    success_count = 0
    total_files = len(valid_files)

    for i, input_file in enumerate(valid_files, 1):
        input_path = Path(input_file)
        output_path = Path(output_dir) / input_path.name

        # Check if output file already exists
        if output_path.exists() and not args.overwrite:
            print(
                f"[{i}/{total_files}] Skipping {input_file} (output exists, use --overwrite to force)"
            )
            continue

        if args.verbose:
            print(f"[{i}/{total_files}] Processing: {input_file}")
            print(f"  -> {output_path}")
        else:
            print(f"[{i}/{total_files}] Converting: {input_path.name}")

        if convert_stl_to_ascii(str(input_path), str(output_path)):
            success_count += 1
            if args.verbose:
                # Get file sizes for comparison
                input_size = input_path.stat().st_size
                output_size = output_path.stat().st_size
                print(f"  Success! Size: {input_size} bytes -> {output_size} bytes")
        else:
            print(f"  Failed to convert {input_file}")

    print("\nConversion complete!")
    print(f"Successfully converted: {success_count}/{total_files} files")

    if success_count < total_files:
        print(f"Failed conversions: {total_files - success_count}")
        sys.exit(1)


if __name__ == '__main__':
    main()
