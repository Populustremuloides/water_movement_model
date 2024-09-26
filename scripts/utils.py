# scripts/utils.py

import numpy as np
import pandas as pd
import os
import yaml
import logging

def load_data(file_path):
    """
    Load and preprocess data from a CSV file.

    Parameters:
        file_path (str): Path to the CSV file.

    Returns:
        dict: Dictionary containing preprocessed data arrays.
    """
    try:
        df = pd.read_csv(file_path, parse_dates=['Date'])
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logging.error(f"Error reading {file_path}: {e}")
        raise

    # Extract columns
    flow = df['flow'].values
    temp = df['average temperature (C)'].values
    precip = df['precipitation'].values
    ET = df['ET'].values
    PET = df['PET'].values

    # Preprocess data
    PET = -PET  # Negate PET to ensure positivity
    temp = temp  # Already in Celsius
    # Normalize by mean
    PET /= np.mean(PET)
    temp /= np.mean(temp)
    precip /= np.mean(precip)
    flow /= np.mean(flow)
    ET /= np.mean(ET)

    data = {
        'flow': flow,
        'temp': temp,
        'precip': precip,
        'ET': ET,
        'PET': PET
    }

    return data

def load_config(config_path):
    """
    Load configuration from a YAML file.

    Parameters:
        config_path (str): Path to the YAML config file.

    Returns:
        dict: Configuration dictionary.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

