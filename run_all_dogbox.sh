#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Optional: Enable debugging by uncommenting the following line
# set -x

# Define the project root directory (assumes the script is run from the root)
PROJECT_ROOT="$(pwd)"

# Define directories
DATA_DIR="${PROJECT_ROOT}/data"
CONFIGS_DIR="${PROJECT_ROOT}/configs"
SCRIPTS_DIR="${PROJECT_ROOT}/scripts"
RESULTS_DIR="${PROJECT_ROOT}/results"

# Define the default config file
DEFAULT_CONFIG_FILE="${CONFIGS_DIR}/config_dogbox.yaml"

# Ensure the results directory exists
mkdir -p "${RESULTS_DIR}"

# Loop through all basin CSV files in the data directory
for file in "${DATA_DIR}"/basin_*.csv
do
  # Check if any files match the pattern
  if [ ! -e "$file" ]; then
    echo "No files matching basin_*.csv found in ${DATA_DIR}."
    exit 1
  fi

  # Extract the basin name by removing the directory and file extension
  basin_name=$(basename "$file" .csv)

  # Define the output directory for the current basin
  basin_output_dir="${RESULTS_DIR}/${basin_name}"

  # Create the basin's results directory if it doesn't exist
  mkdir -p "${basin_output_dir}"

  # Echo the processing status
  echo "---------------------------------------------"
  echo "Processing basin: ${basin_name}"
  echo "Using default config file: ${DEFAULT_CONFIG_FILE}"
  echo "Output will be saved to: ${basin_output_dir}"
  echo "---------------------------------------------"

  # Execute the fit_model.py script with the appropriate arguments
  python "${SCRIPTS_DIR}/fit_model.py" \
    --input_file "${file}" \
    --config_file "${DEFAULT_CONFIG_FILE}" \
    --output_dir "${basin_output_dir}"

  # Check if the script executed successfully
  if [ $? -eq 0 ]; then
    echo "Successfully processed basin: ${basin_name}"
  else
    echo "Error processing basin: ${basin_name}. Check logs for details."
  fi

done

echo "All basins have been processed successfully."

