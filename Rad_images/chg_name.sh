#!/bin/bash

# Path to the directory containing the folders to rename
directory="."

# Check if the directory exists
if [ ! -d "$directory" ]; then
  echo "The directory $directory does not exist."
  exit 1
fi

# Loop through subdirectories of the directory
for old_dir in "$directory"/*_*_*/; do
    # Extract the directory name
    base=$(basename "$old_dir")
    
    # Check if the directory name matches the format DDMMYYYY_HH_HH
    if [[ "$base" =~ ^[0-9]{8}_[0-9]{2}_[0-9]{2}$ ]]; then
        # Extract parts of the date
        day=${base:0:2}
        month=${base:2:2}
        year=${base:4:4}
        
        # Build the new directory name in the format YYYYMMDD_HH_HH
        new_base="${year}${month}${day}_${base:9}"
        
        # Full path to the new directory
        new_dir="${directory}/${new_base}"
        
        # Rename the directory
        mv "$old_dir" "$new_dir"
        
        echo "Renamed $old_dir to $new_dir"
    else
        echo "Directory $old_dir does not match the expected format."
    fi
done

