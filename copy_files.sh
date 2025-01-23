#!/bin/bash

# Usage check
if [ $# -ne 2 ]; then
    echo "Usage: $0 <source_directory> <destination_directory>"
    exit 1
fi

# Define source and destination directories
SOURCE_DIR="$1"
DEST_DIR="$2"

# Check if source directory exists
if [ ! -d "$SOURCE_DIR" ]; then
    echo "Source directory '$SOURCE_DIR' does not exist."
    exit 1
fi

# Create the destination directory if it doesn't exist
mkdir -p "$DEST_DIR"

# Find all files (not directories) under the source directory
find "$SOURCE_DIR" -type f | while read -r file; do
    # Calculate relative path from the source directory
    relative_path="${file#$SOURCE_DIR/}"
    
    # Determine the full destination path
    dest_file="$DEST_DIR/$relative_path"
    
    # Create the necessary directories in the destination if they don't exist
    mkdir -p "$(dirname "$dest_file")"
    
    # Copy the file to the destination
    cp "$file" "$dest_file"
    
    echo "Copied: $file to $dest_file"
done

echo "File copying complete."

