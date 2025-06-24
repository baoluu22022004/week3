#!/bin/bash

# Check if an argument is provided
if [ $# -eq 0 ]; then
    echo "Error: No argument provided."
    echo "Usage: $0 DDMMYYYY Start_HH Stop_HH"
    exit 1
fi

# Extract arguments
date_arg=$1

# Check the length of the date argument
if [ ${#date_arg} -ne 8 ]; then
    echo "Error: The date argument must be in the format DDMMYYYY."
    exit 1
fi

# Set default hours if not all arguments are provided
if [ $# -eq 1 ]; then
    Start_HH=00
    Stop_HH=23
fi

if [ $# -eq 2 ]; then
    Start_HH=$2
    Stop_HH=23
fi

if [ $# -eq 3 ]; then
    Start_HH=$2
    Stop_HH=$3
fi

# Split the date argument into DD, MM, YYYY
DD=${date_arg:6:2}
MM=${date_arg:4:2}
YY=${date_arg:0:4}

# Base URL
base_url="https://kttvnb.info/kttvnb-admin/public/products/RADAR_NEW/"$YY"/"$MM"/"$DD

# Create and enter the output directory
mkdir $date_arg"_"$Start_HH"_"$Stop_HH
cd $date_arg"_"$Start_HH"_"$Stop_HH

# Function to download images for a specific hour
download_images_for_hour() {
    local hour=$1
    for minute in 00 10 20 30 40 50; do
        # Format the hour and minute to two digits
        hour_formatted=$hour
        minute_formatted=$minute

        # Construct the full URL
        url="${base_url}/${hour_formatted}_${minute_formatted}.png"
        
        # Check if the file already exists
        if [ -f "${hour_formatted}_${minute_formatted}.png" ]; then 
            echo "${hour_formatted}_${minute_formatted}.png already exists"
            continue
        fi
        
        # Download the image
        wget -q "$url" -P .

        # Check if the download was successful
        if [ $? -eq 0 ]; then
            echo "Downloaded: $url"
        else
            echo "Failed to download: $url"
        fi
    done
}

# Loop from start hour to stop hour
current_HH=$Start_HH
while [ $((10#$current_HH)) -le $((10#$Stop_HH)) ]; do
    download_images_for_hour $current_HH
    current_HH=$(printf "%02d" $((10#$current_HH + 1)))
done

# Move back to the parent directory
cd ..
