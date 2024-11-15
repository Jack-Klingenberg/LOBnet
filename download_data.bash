#!/bin/bash

# Define download URL
URL=$1

# Create "data" directory if it doesn't exist
mkdir -p data

# Download the file to the "data" directory
cd data
wget --content-disposition "$URL"

# Find the downloaded zip file name
ZIP_FILE=$(ls -t *.zip | head -n 1)

# Check if the file exists and unzip it
if [[ -f "$ZIP_FILE" ]]; then
    unzip "$ZIP_FILE"
else
    echo "Error: File not downloaded or found."
    exit 1
fi

# Navigate into the "published" directory (or the extracted directory)
PUBLISHED_DIR=$(ls -d */ | grep -i "published" | head -n 1)
if [[ -d "$PUBLISHED_DIR" ]]; then
    cd "$PUBLISHED_DIR"
else
    echo "Error: 'published' directory not found."
    exit 1
fi

# Navigate to the BenchmarkDatasets subdirectory
if [[ -d "BenchmarkDatasets" ]]; then
    cd BenchmarkDatasets
else
    echo "Error: 'BenchmarkDatasets' directory not found."
    exit 1
fi

# Unzip the BenchmarkDatasets.zip
if [[ -f "BenchmarkDatasets.zip" ]]; then
    unzip "BenchmarkDatasets.zip"
else
    echo "Error: BenchmarkDatasets.zip not found in BenchmarkDatasets subdirectory."
    exit 1
fi

echo "File downloaded, directories created, and files unzipped successfully."