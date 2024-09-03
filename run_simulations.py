import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from typing import Dict, List, Tuple, NamedTuple
import pandas as pd
import glob
import os
from collections import defaultdict
import cvxpy as cp
from joblib import Parallel, delayed
from tqdm import tqdm
from skimage.filters import threshold_otsu
import csv
import logging
import time
import psutil
from dataclasses import dataclass

SIMULATION_STUDY = "101"
NUMBER_CHANEGPOINTS = 1
NUMBER_NODES = 10
NUMBER_ORIGINAL_SAMPLES = 1000
GAUSSIAN_SIGMA = NUMBER_ORIGINAL_SAMPLES / 100

try:
    CURRENT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
except NameError:
    CURRENT_DIRECTORY = os.getcwd()

CURRENT_DIRECTORY = '/home/jsc123/summer_project/simulation_studies' # '/Users/jamiecuin/Documents/University/Imperial/MSc Statistics/Summer Project/EGM Changepoint Detection/Code/Simulation Studies'
SIMULATION_DIRECTORY = os.path.join(CURRENT_DIRECTORY, f'simulation_study_{SIMULATION_STUDY}', f'number_changepoints_{NUMBER_CHANEGPOINTS}', f'number_nodes_{NUMBER_NODES}', f'number_samples_{NUMBER_ORIGINAL_SAMPLES}')
CSV_DIRECTORY = os.path.join(SIMULATION_DIRECTORY, 'csv_files')
PLOT_DIRECTORY = os.path.join(SIMULATION_DIRECTORY, 'plots')
RESULTS_DIRECTORY = os.path.join(SIMULATION_DIRECTORY, 'results')

# Set up logging
log_file_path = os.path.join(SIMULATION_DIRECTORY, 'simulation_log.txt')
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename=log_file_path,
                    filemode='w')
logger = logging.getLogger(__name__)

@dataclass
class SimulationParams:
    gaussian_sigma: float
    total_original_samples: int

def log_memory_usage():
    process = psutil.Process()
    memory_info = process.memory_info()
    logger.info(f"Memory usage: {memory_info.rss / 1024 / 1024:.2f} MB")

def load_data(csv_directory_path: str, results_directory_path: str) -> Tuple[Dict[Tuple[int, int], Dict[int, pd.DataFrame]], Dict[int, List[int]]]:
    logger.info(f"Loading data from directory: {csv_directory_path}")
    csv_files = glob.glob(os.path.join(csv_directory_path, '*.csv'))
    dataframes = defaultdict(dict)
    for file in csv_files:
        if 'master_changepoint_locations' in file:
            continue
        df = pd.read_csv(file)
        filename = os.path.basename(file)
        parts = filename.split('_')
        
        if len(parts) == 7:
            simulation = int(parts[4][10:])
            permutation = int(parts[5][11:])
            node = int(parts[6].split('.')[0][4:])
            dataframes[(simulation, permutation)][node] = df
        else:
            logger.info(f"Warning: Unexpected filename format {filename}. Skipping this file.")
    
    changepoint_file = os.path.join(results_directory_path, 'master_changepoint_locations.csv')
    if not os.path.exists(changepoint_file):
        logger.info(f"Warning: master_changepoint_locations.csv not found in {results_directory_path}")
        changepoints = {}
    else:
        changepoints_df = pd.read_csv(changepoint_file)
        changepoints = {}
        for _, row in changepoints_df.iterrows():
            sim = row['simulation']
            cp_indices = row['changepoint_indices']
            if pd.isna(cp_indices) or cp_indices == '':
                changepoints[sim] = []
            else:
                try:
                    changepoints[sim] = [int(cp_indices)]
                except ValueError:
                    changepoints[sim] = [int(x) for x in cp_indices.split(';') if x.strip()]

    logger.info(f"Loaded {len(dataframes)} simulations and {len(changepoints)} changepoints")
    
    return dataframes, changepoints

def fused_lasso(data, m, lambda1_value, lambda2_value, return_all: bool = False):
    n, p = data.shape
    y = data[:, m]
    X = np.delete(data.copy(), m, axis=1).T
    
    lambda1 = cp.Parameter(nonneg=True)
    lambda2 = cp.Parameter(nonneg=True)
    beta = cp.Variable((p-1, n))
    
    lasso_penalty = cp.norm1(beta)
    fusion_penalty = cp.sum([cp.norm2(beta[:, i] - beta[:, i-1]) for i in range(1, n)])
    loss = cp.sum_squares(y - cp.sum(cp.multiply(X, beta), axis=0))
    
    objective = cp.Minimize(loss + (2 * lambda1 * fusion_penalty) + (2 * lambda2 * lasso_penalty))
    problem = cp.Problem(objective)
    
    lambda1.value = lambda1_value
    lambda2.value = lambda2_value
    
    problem.solve()
    
    beta_estimated = beta.value
    loss_value = loss.value
    penalty = (2 * lambda1.value * fusion_penalty.value) + (2 * lambda2.value * lasso_penalty.value)
    
    if return_all:
        return beta_estimated, loss_value, penalty
    return beta_estimated

def calculate_differences(beta_estimated):
    return np.abs(np.diff(beta_estimated, axis=1))

def calculate_BIC(loss, beta_estimated, n_samples, method='BIC'):
    first_differences = calculate_differences(beta_estimated)
    bic = n_samples * np.log(loss / n_samples) + np.sum(np.abs(first_differences) > 1e-6) * np.log(n_samples)
    return bic

def evaluate_params(lambda1_value, lambda2_value, transformed_data, n_transformed_samples, d, bic_method):
    total_BIC = 0
    for m in range(d-1):
        beta_estimated, loss, penalty = fused_lasso(transformed_data, m, lambda1_value, lambda2_value, return_all=True)
        bic = calculate_BIC(loss, beta_estimated, n_transformed_samples, method=bic_method)
        total_BIC += bic
    return lambda1_value, lambda2_value, total_BIC

def get_lambda_range(lambda_theoretical: float, n_points: int = 10) -> list:
    """
    TODO
    """
    lower_bound_lambda = (1/3) * lambda_theoretical
    upper_bound_lambda = 3 * lambda_theoretical

    lambda_range = np.linspace(lower_bound_lambda, upper_bound_lambda, num=n_points)

    return lambda_range


def get_regularization_params(transformed_data, n_transformed_samples):
    """
    TODO
    """
    d = transformed_data.shape[1] + 1
    lambda1_theoretical = 1 * n_transformed_samples ** (1/2)
    lambda2_theoretical = 2 * np.sqrt(np.log(d-1) / n_transformed_samples)

    lambda1_range = get_lambda_range(lambda1_theoretical, n_points=20)
    lambda2_range = get_lambda_range(lambda2_theoretical, n_points=20)

    lambda_combinations = [(l1, l2) for l1 in lambda1_range for l2 in lambda2_range]
    
    start_time = time.time()
    results = Parallel(n_jobs=-1)(
        delayed(evaluate_params)(l1, l2, transformed_data, n_transformed_samples, d)
        for l1, l2 in tqdm(lambda_combinations, desc="Evaluating lambda pairs")
    )
    end_time = time.time()
    logger.info(f"Parameter evaluation completed in {end_time - start_time:.2f} seconds")
    
    best_lambda1, best_lambda2, best_BIC_sum = min(results, key=lambda x: x[2])
    
    logger.info(f"Optimal lambda1: {best_lambda1}, lambda2: {best_lambda2}")
    return best_lambda1, best_lambda2

def plot_with_changepoints(data, changepoints, title, xlabel: str = '', ylabel: str = '', filename: str = None):
    plt.figure(figsize=(10, 6))
    plt.plot(data)
    for cp in changepoints:
        if isinstance(cp, int):
            index, color, linestyle = cp, 'r', '--'
        elif isinstance(cp, tuple):
            index, color, linestyle = cp
        plt.axvline(x=index, color=color, linestyle=linestyle, label=f'Changepoint at index {index}')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    if filename:
        plt.savefig(filename)
        plt.close()
        logger.info(f"Plot saved as '{filename}'")
    else:
        plt.show()

def compute_dot_S(tilde_S):
    dot_S = np.zeros(next(iter(tilde_S.values())).shape[0])
    for tilde_S_k in tilde_S.values():
        dot_S += tilde_S_k
    return dot_S

def estimate_changepoints(dot_S: np.ndarray, params: SimulationParams, min_threshold: float = 0) -> Tuple[List[int], np.ndarray, float]:
    logger.info("Estimating changepoints")
    smoothed_data = gaussian_filter1d(dot_S, params.gaussian_sigma)
    threshold = threshold_otsu(smoothed_data)
    threshold = max(threshold, min_threshold)
    peaks, _ = find_peaks(smoothed_data, height=threshold)
    logger.info(f"Estimated {len(peaks)} changepoints")
    return peaks.tolist(), smoothed_data, threshold

def plot_changepoint_estimation(dot_S: np.ndarray, smoothed_data: np.ndarray, 
                                peaks: List[int], threshold: float,
                                sim: int, perm: int, plot_directory: str,
                                true_changepoints: List[int]):
    plt.figure(figsize=(15, 6))
    plt.plot(dot_S, label='Non-Smoothed')
    plt.plot(smoothed_data, label='Smoothed', linewidth=2)
    plt.scatter(peaks, smoothed_data[peaks], color='red', label='Estimated Peaks')
    plt.axhline(y=threshold, color='gray', linestyle='--', label='Threshold')
    
    # Plot true changepoints
    for cp in true_changepoints:
        plt.axvline(x=cp, color='red', linestyle=':', linewidth=2, label='True Changepoint')
    
    # Remove duplicate labels
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    
    plt.title(f'Changepoint Estimation - Simulation {sim}, Permutation {perm}')
    plt.xlabel('Time')
    plt.ylabel('Value')
    
    filename = os.path.join(plot_directory, f'changepoint_estimation_sim{sim}_perm{perm}.png')
    plt.savefig(filename)
    plt.close()
    logger.info(f"Changepoint estimation plot saved as '{filename}'")

def compute_tilde_S(dataframe: pd.DataFrame, params: SimulationParams) -> np.ndarray:
    logger.info(f"Computing tilde_S for dataframe with shape: {dataframe.shape}")
    transformed_data = dataframe.iloc[:, 1:].values
    original_index = dataframe.iloc[:, 0].values - 1
    d = transformed_data.shape[1] + 1
    n_transformed_samples = transformed_data.shape[0]
    
    lambda1_k, lambda2_k = get_regularization_params(
        transformed_data,
        n_transformed_samples
    )
    
    beta_hat_k = {}
    beta_hat_k_differences = {}
    for m in range(d-1):
        beta_hat_k[m] = fused_lasso(transformed_data, m, lambda1_k, lambda2_k)
        beta_hat_k_differences[m] = calculate_differences(beta_hat_k[m])
        beta_hat_k_differences[m] = np.insert(beta_hat_k_differences[m], 0, 0, axis=1)
    
    beta_differences_original_index = defaultdict(lambda: np.zeros(params.total_original_samples))
    for m in range(d-1):
        for i, difference in enumerate(beta_hat_k_differences[m].T):
            beta_differences_original_index[m][original_index[i]] = np.sum(difference)
    
    tilde_S = np.zeros(params.total_original_samples)
    for differences_array in beta_differences_original_index.values():
        tilde_S += differences_array
    
    return tilde_S, lambda1_k, lambda2_k

def update_results_csv(filename, sim, perm, lambda_pairs, threshold, predicted_changepoints):
    file_exists = os.path.isfile(filename)
    
    with open(filename, 'a', newline='') as csvfile:
        fieldnames = ['simulation', 'permutation', 'lambda_pairs', 'threshold', 'predicted_changepoints']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow({
            'simulation': sim,
            'permutation': perm,
            'lambda_pairs': ';'.join([f"{l1},{l2}" for l1, l2 in lambda_pairs]),
            'threshold': threshold,
            'predicted_changepoints': ';'.join(map(str, predicted_changepoints))
        })
    logger.info(f"Results updated in CSV: {filename}")

def run_simulation(
        sim: int, 
        perm: int, 
        dataframes: Dict[int, pd.DataFrame], 
        changepoints: Dict[int, List[int]], 
        params: SimulationParams,
        results_directory: str,
        plot_directory: str
    ) -> None:
    logger.info(f"Running Simulation {sim}, Permutation {perm}")
    
    tilde_S_dict = {}
    lambda_pairs = []

    d = len(dataframes)
    for k, df in dataframes.items():
        logger.info(f"Analyzing Dataset {k}")
        tilde_S_k, lambda1_k, lambda2_k = compute_tilde_S(df, params)
        tilde_S_dict[k] = tilde_S_k
        lambda_pairs.append((lambda1_k, lambda2_k))
    
    true_changepoints = changepoints.get(sim, [])
    
    dot_S = compute_dot_S(tilde_S_dict)
    
    min_threshold = 2.5e-4 * d * (d-1) # \tau_min

    estimated_changepoints, smoothed_data, threshold = estimate_changepoints(dot_S, params, min_threshold=min_threshold)
    
    # Save only the final smoothed estimation plot
    plot_changepoint_estimation(dot_S, smoothed_data, estimated_changepoints, threshold, 
                                sim, perm, plot_directory, true_changepoints)
    
    csv_results_filename = os.path.join(results_directory, 'simulation_results.csv')
    update_results_csv(csv_results_filename, sim, perm, lambda_pairs, threshold, estimated_changepoints)


def main():
    csv_results_filename = os.path.join(RESULTS_DIRECTORY, 'simulation_results.csv')

    if os.path.exists(csv_results_filename):
        logger.warning(f"The results.csv file: {csv_results_filename} already exists. Renaming the existing file.")
        os.rename(csv_results_filename, f"{csv_results_filename}.bak")
    
    logger.info(f"Starting simulation in directory: {SIMULATION_DIRECTORY}")
    all_simulations_dataframes, changepoints = load_data(csv_directory_path=CSV_DIRECTORY, results_directory_path=RESULTS_DIRECTORY)
    
    params = SimulationParams(
        gaussian_sigma=GAUSSIAN_SIGMA,
        total_original_samples=NUMBER_ORIGINAL_SAMPLES
    )
    
    logger.info(f"Simulation parameters: {params}")
    
    for (sim, perm), dataframes in tqdm(all_simulations_dataframes.items(), desc="Processing simulations"):
        try:
            run_simulation(sim, perm, dataframes, changepoints, params, results_directory=RESULTS_DIRECTORY, plot_directory=PLOT_DIRECTORY)
        except Exception as e:
            logger.error(f"Error in simulation {sim}, permutation {perm}: {str(e)}", exc_info=True)
    
    logger.info("All simulations completed")
    log_memory_usage()

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    logger.info(f"Total execution time: {end_time - start_time:.2f} seconds")