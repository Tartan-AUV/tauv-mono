# Lucid Driver - New Features

## Overview
The Lucid driver has been enhanced with two new features:
1. **Full-resolution image saving** - Save debayered images to disk
2. **CUDA-accelerated downsampling** - Downsample images before publishing to reduce bandwidth

## New ROS Parameters

### `save_folder` (string, default: "")
- Path to folder where full-resolution debayered images will be saved
- If empty or not set, no images will be saved
- The folder will be created automatically if it doesn't exist
- Images are saved as PNG files with timestamp and counter in filename format: `image_YYYYMMDD_HHMMSS_mmm_NNNNNN.png`

### `publish_downsampling_factor` (int, default: 1)
- Factor by which to downsample images before publishing
- Value of 1 means no downsampling (publish full resolution)
- Value of 2 means publish images at 1/2 resolution (1/4 total pixels)
- Value of 4 means publish images at 1/4 resolution (1/16 total pixels)
- Uses CUDA-accelerated bilinear interpolation for fast downsampling

## Example Usage

### Example 1: Save full-resolution images only
```yaml
lucid:
  ros__parameters:
    camera_ip: "10.0.2.11"
    topic_name: "/image_raw"
    save_folder: "/home/user/camera_images"
    publish_downsampling_factor: 1
```

### Example 2: Downsample by 2x for publishing, no saving
```yaml
lucid:
  ros__parameters:
    camera_ip: "10.0.2.11"
    topic_name: "/image_raw"
    save_folder: ""
    publish_downsampling_factor: 2
```

### Example 3: Save full-resolution and publish downsampled
```yaml
lucid:
  ros__parameters:
    camera_ip: "10.0.2.11"
    topic_name: "/image_raw"
    save_folder: "/home/user/camera_images"
    publish_downsampling_factor: 4
```

## Implementation Details

### Image Pipeline
1. Camera captures Bayer pattern image (BayerRG8)
2. CUDA-accelerated debayering converts to BGR8 (full resolution)
3. If `save_folder` is set, full-resolution BGR image is saved to disk
4. If `publish_downsampling_factor` > 1, CUDA-accelerated resize is applied
5. Downsampled (or full-resolution if factor=1) image is published to ROS topic

### Performance Considerations
- All image processing (debayering and downsampling) uses CUDA for maximum performance
- Saving images to disk happens on CPU and may impact performance if disk I/O is slow
- Downsampling reduces network bandwidth and processing load on subscribers
- Consider using SSD storage for `save_folder` if saving at high frame rates

### File Naming Convention
Saved images use the format: `image_YYYYMMDD_HHMMSS_mmm_NNNNNN.png`
- `YYYYMMDD`: Year, month, day
- `HHMMSS`: Hour, minute, second
- `mmm`: Milliseconds (3 digits)
- `NNNNNN`: Sequential counter (6 digits, zero-padded)

This ensures unique filenames and chronological ordering.
