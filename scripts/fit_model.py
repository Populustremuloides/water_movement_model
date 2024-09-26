# scripts/fit_model.py

import argparse
import os
import sys
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import yaml
import logging
from utils import load_data, load_config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Determine the absolute path of the project root
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))

def getWPrime(thetas, index, w, m, precip, temp, PET):
    """
    Calculate the derivative of w (wPrime) based on the current state and parameters.
    """
    # Ensure thetas has at least 7 elements
    if len(thetas) < 7:
        raise IndexError(f"Expected at least 7 thetas_model parameters, got {len(thetas)}")

    precip_in_rate = precip[index] * thetas[5]
    denominator = 1 + np.exp((-thetas[2] * temp[index]) - thetas[3])
    proportion_groundwater = np.exp(-thetas[1]) / denominator
    flow_out_rate = thetas[0] * w * proportion_groundwater
    evapotranspiration_out_rate = thetas[6] * PET[index] * (1 + np.exp(thetas[4]) * m)

    return precip_in_rate - flow_out_rate - evapotranspiration_out_rate

def getWs(thetas, precip, PET, temp, m, w0):
    """
    Generate the series of w values over time based on model parameters.
    """
    ws = [w0]
    for i in range(len(precip)):
        w_prime = getWPrime(thetas, i, ws[-1], m, precip, temp, PET)
        new_w = max(0, ws[-1] + w_prime)
        ws.append(new_w)
    return np.array(ws[1:])

def getQs(thetas, m, ws, temps):
    """
    Calculate Q values based on model parameters and w series.
    """
    denominator = 1 + np.exp((-thetas[2] * temps) - thetas[3])
    proportion_groundwater = np.exp(-thetas[1]) / denominator
    return thetas[0] * ws * proportion_groundwater

def simulateStream(thetas, m, precip, PET, temp, w0):
    """
    Simulate Q and W series for a single basin.
    """
    ws = getWs(thetas, precip, PET, temp, m, w0)
    qs = getQs(thetas, m, ws, temp)
    return qs, ws

def getModelMap(thetas_model, w0, m, precip, PET, temp):
    """
    Generate the model's Q and W series for a single basin.
    """
    qs, ws = simulateStream(thetas_model, m, precip, PET, temp, w0)
    return qs, ws

def getMeasuredQs(flow):
    """
    Retrieve the measured Q values.
    """
    return flow

def getResidual(thetas_model, m, precip, PET, temp, w0, flow):
    """
    Calculate the residual between measured and modeled Q values.
    """
    qs_pred, _ = getModelMap(thetas_model, w0, m, precip, PET, temp)
    residual = flow - qs_pred
    return residual

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Fit water movement model to drainage basin data.")
    parser.add_argument('--input_file', type=str, required=True, help="Path to the basin CSV data file.")
    parser.add_argument('--config_file', type=str, default="configs/default_config.yaml", help="Path to the config YAML file.")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save the results.")
    args = parser.parse_args()

    # Load configuration
    config_path = os.path.join(project_root, args.config_file)
    config = load_config(config_path)
    optimization_config = config.get('optimization', {})
    plot_config = config.get('plot', {})
    output_dir = args.output_dir  # Use the provided output directory

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Extract optimization settings
    L1_weight = optimization_config.get('loss_function', {}).get('L1_weight', 1.0)
    L2_weight = optimization_config.get('loss_function', {}).get('L2_weight', 0.1)
    bounds = optimization_config.get('bounds', {})
    lower_bounds = bounds.get('lower', [0]*7)
    upper_bounds = bounds.get('upper', [100]*7)
    initial_guess_mean = optimization_config.get('initial_guess', {}).get('mean', 0)
    initial_guess_std = optimization_config.get('initial_guess', {}).get('std', 1)

    # Load and preprocess data
    input_file_path = os.path.join(project_root, args.input_file)
    data = load_data(input_file_path)
    flow = data['flow']
    temp = data['temp']
    precip = data['precip']
    ET = data['ET']
    PET = data['PET']

    # Convert lower and upper bounds to numpy arrays for vectorized operations
    lower_bounds = np.array(lower_bounds)
    upper_bounds = np.array(upper_bounds)

    # Set to always estimate w0 and m
    # Total parameters: thetas_model (7) + w0 (1) + m (1) = 9
    n_thetas_model = 7
    n_additional_params = 2  # w0 and m
    n_params = n_thetas_model + n_additional_params

    # Validate that bounds match the number of parameters
    if len(lower_bounds) != n_params or len(upper_bounds) != n_params:
        logging.error(f"Bounds length {len(lower_bounds)} or {len(upper_bounds)} do not match n_params {n_params}")
        sys.exit(1)

    # Initial guess: sample uniformly within bounds to ensure feasibility
    theta_0 = np.random.uniform(low=lower_bounds, high=upper_bounds, size=n_params)

    # Log the initial guess
    logging.info(f"Initial guess (theta_0): {theta_0}")

    # Define bounds as a tuple for least_squares
    bounds_tuple = (lower_bounds, upper_bounds)

    # Define residual function for least_squares
    def residual_func(thetas):
        # Extract thetas_model, w0, and m from thetas
        thetas_model = thetas[0:n_thetas_model]
        w0 = thetas[n_thetas_model]
        m = thetas[n_thetas_model + 1]
        logging.debug(f"thetas_model: {thetas_model}, w0: {w0}, m: {m}")
        logging.debug(f"thetas_model size: {thetas_model.size}, w0 size: {w0}, m size: {m}")
        return getResidual(thetas_model, m, precip, PET, temp, w0, flow)

    # Perform optimization
    logging.info("Starting optimization...")
    try:
        result = least_squares(
            residual_func,
            theta_0,
            method='trf',  # Trust Region Reflective algorithm
            loss=optimization_config.get('method', 'huber'),
            bounds=bounds_tuple
        )
    except Exception as e:
        logging.error(f"Optimization failed: {e}")
        sys.exit(1)

    # Check if optimization was successful
    if not result.success:
        logging.error(f"Optimization failed: {result.message}")
        sys.exit(1)

    # Extract fitted parameters
    fitted_thetas_all = result.x
    thetas_all = np.exp(fitted_thetas_all)  # Exponentiate to get actual parameters

    # Extract thetas_model, w0, and m
    thetas_model = thetas_all[0:n_thetas_model]
    w0 = thetas_all[n_thetas_model]
    m = thetas_all[n_thetas_model + 1]

    # Log the fitted parameters
    logging.info(f"Fitted parameters (thetas_all): {thetas_all}")

    # Simulate the model with fitted parameters
    qs_pred, ws_pred = simulateStream(thetas_model, m, precip, PET, temp, w0)

    # Log stats of predicted and measured Q
    logging.info(f"qs_pred stats: min={qs_pred.min()}, max={qs_pred.max()}, mean={qs_pred.mean()}")
    logging.info(f"flow stats: min={flow.min()}, max={flow.max()}, mean={flow.mean()}")

    # Calculate correlation
    if np.std(qs_pred) == 0 or np.std(flow) == 0:
        logging.warning("One of the input arrays for correlation is constant. Pearson correlation is not defined.")
        cc, p = (np.nan, np.nan)
    else:
        cc, p = pearsonr(qs_pred, flow)
    logging.info(f"Pearson correlation coefficient: {cc}, p-value: {p}")

    # Save fitted parameters
    np.save(os.path.join(output_dir, "thetas.npy"), thetas_all)
    np.save(os.path.join(output_dir, "correlationCoefficients.npy"), np.array([cc]))

    # Plot actual vs predicted Q
    if plot_config.get('save', True):
        plt.figure(figsize=(10,6))
        plt.plot(flow, label='Measured Q')
        plt.plot(qs_pred, label='Predicted Q')
        plt.title(f'Basin: {os.path.basename(output_dir)} | Pearson r: {cc:.4f}')
        plt.xlabel('Time')
        plt.ylabel('Flow (Normalized)')
        plt.legend()
        plot_format = plot_config.get('format', 'png')
        plot_path = os.path.join(output_dir, f"Q_comparison.{plot_format}")
        plt.savefig(plot_path)
        plt.close()
        logging.info(f"Plot saved to {plot_path}")

    logging.info(f"Fitting completed for basin: {os.path.basename(output_dir)}")

if __name__ == "__main__":
    main()

