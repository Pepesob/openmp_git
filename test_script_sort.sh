#!/bin/bash

# Check if base output name is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <base_output_name>"
    exit 1
fi

# Define output filenames
base_name="$1"
csv_vals_file="${base_name}_csv_vals.csv"
buckets_json_file="${base_name}_buckets.json"

# Write headers
echo "calculated_action,schedule,chunk_size,array_size,threads,execution_time_sec,buckets_amount" > "$csv_vals_file"
echo '{ "buckets": [' > "$buckets_json_file"  # Start JSON array

# Parameters
schedules=("static")
chunk_sizes=(2)

# Thread range=
thread_start=1
thread_end=48
thread_step=1
thread_numbers=($(eval echo {$thread_start..$thread_end..$thread_step}))

problem_sizes=(1e7 2e7 3e7 4e7 5e7 6e7 7e7 8e7 9e7 10e7)
bucket_amounts=(1000)

# Path to compiled program
program="./bucket_sort"

first_entry=true

# Loop through all combinations
for schedule in "${schedules[@]}"; do
    for chunk_size in "${chunk_sizes[@]}"; do
        for thread_number in "${thread_numbers[@]}"; do
            for problem_size in "${problem_sizes[@]}"; do
                for bucket_amount in "${bucket_amounts[@]}"; do
                    echo "Running: $schedule, chunk_size: $chunk_size, tab_size: $problem_size, thread_number: $thread_number, bucket_amount: $bucket_amount"

                    # Run program and capture output
                    output=$($program "$schedule" "$chunk_size" "$problem_size" "$thread_number" "$bucket_amount")

                    # Extract and append CSV values
                    echo "$output" | jq -r '.csv_vals[]' >> "$csv_vals_file"

                    # Extract bucket array as JSON
                    bucket_line=$(echo "$output" | jq -c '.buckets')

                    # Append comma if not the first JSON array
                    if [ "$first_entry" = true ]; then
                        first_entry=false
                    else
                        echo "," >> "$buckets_json_file"
                    fi

                    # Write the bucket array to JSON file
                    echo "  $bucket_line" >> "$buckets_json_file"
                done
            done
        done
    done
done

# Close the JSON array
echo "]}" >> "$buckets_json_file"
