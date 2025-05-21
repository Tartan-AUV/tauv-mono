# STL Binary to ASCII Converter

A command-line tool for converting binary STL files to ASCII STL format. This tool is useful when you need to work with STL files in text format for debugging, editing, or compatibility with software that only supports ASCII STL files.

## Features

- Convert multiple binary STL files to ASCII format in one command
- Automatically detect if files are already in ASCII format (and copy them if needed)
- Batch processing with progress indication
- Verbose mode for detailed output
- Overwrite protection with optional force flag
- Input validation and error handling

## Installation

First, install the required dependencies:

```bash
cd tools
pip install -r requirements.txt
```

This will install `numpy-stl` and other required packages.

## Usage

### Basic Usage

Convert one or more STL files:

```bash
python stl_converter.py input.stl -o output_directory
```

Convert multiple files:

```bash
python stl_converter.py file1.stl file2.stl file3.stl -o output_directory
```

Use wildcards to convert all STL files in a directory:

```bash
python stl_converter.py *.stl -o output_directory
```

### Command Line Options

- `input_files`: One or more STL files to convert (required)
- `-o, --output`: Output directory for converted files (required)
- `-v, --verbose`: Enable verbose output showing detailed progress
- `--overwrite`: Overwrite existing files in output directory

### Examples

1. **Convert a single file with verbose output:**
   ```bash
   python stl_converter.py model.stl -o ascii_models --verbose
   ```

2. **Convert all STL files in current directory:**
   ```bash
   python stl_converter.py *.stl -o converted_stls
   ```

3. **Convert files and overwrite existing output:**
   ```bash
   python stl_converter.py part1.stl part2.stl -o output --overwrite
   ```

4. **Get help:**
   ```bash
   python stl_converter.py --help
   ```

## Output

The tool will:

1. Check if input files exist and have `.stl` extensions
2. Create the output directory if it doesn't exist
3. For each file:
   - Detect if it's already ASCII format (and copy if so)
   - Convert binary STL to ASCII format
   - Show progress and results
4. Provide a summary of successful and failed conversions

### Example Output

```
Converting 3 STL file(s) to ASCII format...
Output directory: /path/to/output

[1/3] Converting: model1.stl
[2/3] Converting: model2.stl
  File model2.stl is already in ASCII format
[3/3] Converting: model3.stl

Conversion complete!
Successfully converted: 3/3 files
```

## STL Format Detection

The tool uses intelligent format detection:

1. **Header Analysis**: Checks if the file starts with "solid" (typical ASCII STL marker)
2. **Content Validation**: Reads a chunk of the file to verify it contains ASCII STL keywords like "facet" and "vertex"
3. **Fallback**: If detection is uncertain, assumes binary format

## File Size Considerations

- **Binary STL**: More compact, faster to read/write
- **ASCII STL**: Larger file size (typically 5-10x larger), human-readable, easier to debug

The tool shows file size changes when using verbose mode.

## Error Handling

The tool handles common errors gracefully:

- Missing input files
- Invalid file extensions
- Permission issues
- Corrupted STL files
- Disk space problems

Failed conversions are reported at the end, and the tool exits with status code 1 if any conversions fail.

## Technical Details

This tool uses the `numpy-stl` library, which provides robust STL file parsing and writing capabilities. The library can handle:

- Both binary and ASCII STL files
- Malformed files (with warnings)
- Large meshes efficiently
- Various STL variations and edge cases

## Integration with TAUV Project

This tool is particularly useful in the TAUV project context for:

- Converting CAD-exported STL files to ASCII format for Stonefish simulation
- Debugging mesh geometry issues
- Preparing models for version control (ASCII files diff better)
- Ensuring compatibility with various simulation and analysis tools

## Troubleshooting

### Common Issues

1. **Import Error**: `numpy-stl` not installed
   ```bash
   pip install numpy-stl
   ```

2. **Permission Denied**: Output directory not writable
   ```bash
   chmod 755 output_directory
   ```

3. **File Not Found**: Check input file paths
   ```bash
   ls -la *.stl
   ```

4. **Corrupted STL**: The tool will report specific errors for malformed files

### Performance Tips

- For large batches, use SSD storage for faster I/O
- Binary STL files process faster than ASCII
- Use `--verbose` mode only when needed (it's slower due to file size calculations) 