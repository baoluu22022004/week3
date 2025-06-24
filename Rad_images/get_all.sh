#!/bin/bash

# Check if the file exists
input_file="../RS_SAIGON/list_ev_sup5mm_up_170624.txt"
if [ ! -f "$input_file" ]; then
    echo "Error: The file $input_file does not exist."
    exit 1
fi

# Read the file line by line
while IFS= read -r line; do
    # Extract start and stop date and time
    start_date_time=$(echo "$line" | awk '{print $1, $2}')
    stop_date_time=$(echo "$line" | awk '{print $3, $4}')

    # Extract start date in the format DDMMYYYY
    start_date=$(echo "$start_date_time" | awk '{print $1}')
    DD=$(echo "$start_date" | cut -d'-' -f3)
    MM=$(echo "$start_date" | cut -d'-' -f2)
    YYYY=$(echo "$start_date" | cut -d'-' -f1)
    formatted_date="${YYYY}${MM}${DD}"
#    echo "$start_date_time"
#    echo "$formatted_date"

    # Extract the rounded start hour
    start_HH=$(echo "$start_date_time" | awk '{print $2}' | cut -d':' -f1)

    # Extract the rounded stop hour
    stop_HH=$(echo "$stop_date_time" | awk '{print $2}' | cut -d':' -f1)

    # Display the results
    ./get_images.sh $formatted_date $start_HH $stop_HH
done < "$input_file"
