#!/bin/bash

# Check if folder argument is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <folder_path>"
    exit 1
fi

FOLDER="$1"

# Find and count matching files
count=$(find "$FOLDER" -type f -name "video_*_success.mp4" | wc -l)

# Print result
echo "Total count of video_*_success.mp4 files in $FOLDER: $count"
