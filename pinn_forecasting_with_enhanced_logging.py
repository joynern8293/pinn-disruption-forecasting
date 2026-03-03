
# -*- coding: utf-8 -*-
"""
This script implements a Physics-Informed Neural Network (PINN) to 
model and forecast stock prices.

MODIFICATION: This version has been updated to use a simplified loss function
for better training performance. It retains all modern enhancements like user
input, logging, progress bars, dynamic parameter guessing, and LaTeX reporting.
It now also prompts the user for training hyperparameters and zips all output
files at the end.
"""
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR
import numpy as np
import logging
from datetime import datetime
import os
from tqdm import trange # Import tqdm for the progress bar

# Import functions from the utility file
from pinn_utilities_10_9_25 import (
    create_strided_kernel,
    create_gaussian_kernel,
    plot_forecast_results,
    plot_loss
)

# Import the new report generator and the zipping function
from report_generator import generate_latex_report, zip_output_files

# region: Class Definitions

class StockDataHandler:
    """
    Handles loading, preprocessing, and splitting of stock data.
    """
    def __init__(self, filepath, z_window_size, y_window_size, y_stride, w_window_size, w_std):
        self.filepath = filepath
        self.z_window_size = z_window_size
        self.y_window_size = y_window_size
        self.y_stride = y_stride
        self.w_window_size = w_window_size
        self.w_std = w_std
        self.df_full = None

    def load_and_process_data(self):
        """
        Loads the stock data for the full date range and calculates temporal variables.
        """
        df = pd.read_csv(
            self.filepath,
            usecols=["<DATE>", "<CLOSE>"],
            parse_dates=["<DATE>"],
            index_col="<DATE>",
        )
        df.rename(columns={"<CLOSE>": "Close"}, inplace=True)
        df.index.name = "Date"

        self.df_full = df.copy()

        self.df_full["P_Data"] = self.df_full["Close"].rolling(window=30).mean()
        self.df_full["Z_Data"] = self.df_full["P_Data"].rolling(window=self.z_window_size).mean()
        def strided_mean(x): return x[::self.y_stride].mean()
        self.df_full["Y_Data"] = self.df_full["P_Data"].rolling(window=self.y_window_size).apply(strided_mean, raw=True)
        self.df_full["W_Data"] = self.df_full["P_Data"].rolling(window=self.w_window_size, win_type='gaussian', center=True).mean(std=self.w_std)

        self.df_full.dropna(inplace=True)
        logging.info(f"Full data processed successfully. Shape: {self.df_full.shape}")
        return self.df_full

    def get_split_tensors(self, test_split_ratio):
        """
        Splits the processed data into training and testing sets based on a ratio
        and converts them to tensors.
        """
        if self.df_full is None:
            raise ValueError("Data has not been loaded. Call load_and_process_data() first.")

        split_index = int(len(self.df_full) * (1 - test_split_ratio))
        df_train = self.df_full.iloc[:split_index].copy()
        df_test = self.df_full.iloc[split_index:].copy()

        t_train = np.arange(len(df_train)).reshape(-1, 1)
        p_train = df_train["P_Data"].values.reshape(-1, 1)

        t_test = np.arange(len(df_train), len(self.df_full)).reshape(-1, 1)
        p_test = df_test["P_Data"].values.reshape(-1, 1)

        t_train_tensor = torch.tensor(t_train, dtype=torch.float32, requires_grad=True)
        p_train_tensor = torch.tensor(p_train, dtype=torch.float32)
        t_test_tensor = torch.tensor(t_test, dtype=torch.float32, requires_grad=True)
        p_test_tensor = torch.tensor(p_test, dtype=torch.float32)

        return (t_train_tensor, p_train_tensor), (t_test_tensor, p_test_tensor), (df_train, df_test)

class ForcingNet(nn.Module):
    """
    Neural network to model the unknown forcing function f(t).
    """
    def __init__(self, input_size=1, hidden_size=32, output_size=1):
        super(ForcingNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, t):
        return self.net(t)

class PINN(nn.Module):
    """
    Main PINN model combining exponential growth with a learned forcing function.
    """
    def __init__(self, P0_initial, k_initial, t1_initial, t2_initial, s_initial=0.1):
        super(PINN, self).__init__()
        self.forcing_net = ForcingNet()

        # All key parameters are learnable
        self.P0 = nn.Parameter(torch.tensor([float(P0_initial)]))
        self.k = nn.Parameter(torch.tensor([float(k_initial)]))
        self.t1 = nn.Parameter(torch.tensor([float(t1_initial)]))
        self.t2 = nn.Parameter(torch.tensor([float(t2_initial)]))
        self.s = nn.Parameter(torch.tensor([float(s_initial)]))

    def forward(self, t):
        p_base = self.P0 * torch.exp(self.k * t)
        sigmoid1 = torch.sigmoid(self.s * (t - self.t1))
        sigmoid2 = torch.sigmoid(self.s * (t - self.t2))
        S = sigmoid1 * (1 - sigmoid2)
        f_t = self.forcing_net(t)
        p_pred = p_base + S * f_t
        return p_pred, S, f_t

class PINNTrainer:
    """
    Manages the training and forecasting process for the PINN model.
    """
    def __init__(self, pinn_model, learning_rate, gamma):
        self.model = pinn_model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = ExponentialLR(self.optimizer, gamma=gamma)
        self.loss_history, self.data_loss_history, self.pde_loss_history = [], [], []
        # Temporal losses are no longer part of the main loss function
        self.temporal_losses_history = {}

    def _calculate_pde_loss(self, t, p_pred, S, f_t):
        dp_dt = torch.autograd.grad(p_pred, t, grad_outputs=torch.ones_like(p_pred), create_graph=True)[0]
        pde_residual = dp_dt - self.model.k * p_pred - S * f_t
        return torch.mean(pde_residual**2)

    def train(self, t_data, p_data, epochs, lambda_pde=1.0):
        logging.info("Starting training...")
        self.model.train()
        epoch_iterator = trange(epochs, desc="Training Progress", unit="epoch")

        for epoch in epoch_iterator:
            self.optimizer.zero_grad()
            p_pred, S, f_t = self.model(t_data)

            # Data-fit loss
            loss_data = nn.MSELoss()(p_pred, p_data)

            # PDE loss for P(t)
            loss_pde = self._calculate_pde_loss(t_data, p_pred, S, f_t)

            # Simplified total loss
            total_loss = loss_data + lambda_pde * loss_pde

            total_loss.backward()
            self.optimizer.step()
            self.loss_history.append(total_loss.item())
            self.data_loss_history.append(loss_data.item())
            self.pde_loss_history.append(loss_pde.item())

            epoch_iterator.set_postfix({"Total Loss": f"{total_loss.item():.4f}", "Data Loss": f"{loss_data.item():.4f}"})

            if (epoch + 1) % 1000 == 0:
                self.scheduler.step()
                logging.info(f"Epoch [{epoch+1}/{epochs}], Total: {total_loss.item():.4f}, Data: {loss_data.item():.4f}, PDE: {loss_pde.item():.4f}")

                params = {
                    'P0': self.model.P0.item(), 'k': self.model.k.item(),
                    't1': self.model.t1.item(), 't2': self.model.t2.item(),
                    's': self.model.s.item()
                }
                logging.info(f"Params @ Epoch [{epoch+1}]: P0={params['P0']:.2f}, k={params['k']:.6f}, t1={params['t1']:.2f}, t2={params['t2']:.2f}, s={params['s']:.4f}")

        logging.info("Training complete.")

    def predict(self, t_data):
        self.model.eval()
        with torch.no_grad():
            p_pred, _, _ = self.model(t_data)
        return p_pred.cpu().numpy()

# endregion

def get_user_input_files():
    """Gets and validates user input for file path and test split ratio."""
    while True:
        file_path = input("Please enter the name of the data file (e.g., spy.us.txt): ")
        if os.path.exists(file_path): break
        else: print(f"Error: File '{file_path}' not found. Please try again.")
    while True:
        try:
            test_split_str = input("Enter the percent of data for testing (e.g., 0.20 for 20%): ")
            test_split_ratio = float(test_split_str)
            if 0 < test_split_ratio < 1: break
            else: print("Error: Please enter a decimal value between 0 and 1.")
        except ValueError: print("Error: Invalid input. Please enter a decimal value (e.g., 0.20).")
    return file_path, test_split_ratio

def get_initial_parameter_guesses(p_train_tensor):
    """
    Prompts for initial guesses for all learnable parameters: P0, k, t1, t2, and s.
    """
    while True:
        choice = input("Use automatic initial parameter guesses? (yes/no): ").lower().strip()
        if choice in ['yes', 'y', 'no', 'n']: break
        print("Invalid input. Please enter 'yes' or 'no'.")

    p_train_np = p_train_tensor.cpu().numpy()
    train_len = len(p_train_np)

    if choice in ['yes', 'y']:
        print("Generating automatic initial guesses...")
        P0_guess = p_train_np[0, 0]
        t1_guess = train_len * 0.33
        t2_guess = train_len * 0.66
        s_guess = 0.1

        P_start = p_train_np[0, 0]
        P_end = p_train_np[-1, 0]
        T = train_len
        k_guess = np.log(max(1e-8, P_end) / max(1e-8, P_start)) / T

        print(f"  - P0 (Initial Value): {P0_guess:.2f}")
        print(f"  - k (Growth Rate): {k_guess:.6f}")
        print(f"  - t1 (Disruption Start): {t1_guess:.2f}")
        print(f"  - t2 (Disruption End): {t2_guess:.2f}")
        print(f"  - s (Transition Steepness): {s_guess:.4f}")
        return P0_guess, k_guess, t1_guess, t2_guess, s_guess
    else:
        print("Please enter your initial guesses for the learnable parameters.")
        while True:
            try:
                P0_guess = float(input(f"Enter guess for P0 (e.g., {p_train_np[0,0]:.2f}): "))
                k_guess = float(input("Enter guess for k (e.g., 0.001): "))
                t1_guess = float(input(f"Enter guess for t1 (e.g., {int(train_len * 0.33)}): "))
                t2_guess = float(input(f"Enter guess for t2 (e.g., {int(train_len * 0.66)}): "))
                s_guess = float(input("Enter guess for s (e.g., 0.1): "))
                return P0_guess, k_guess, t1_guess, t2_guess, s_guess
            except ValueError:
                print("Invalid input. Please ensure all values are numbers.")

def get_training_hyperparameters():
    """Gets and validates user input for training hyperparameters."""
    print("\nPlease enter the training hyperparameters.")
    while True:
        try:
            epochs = int(input("Enter the number of training epochs (e.g., 20000): "))
            if epochs > 0: break
            else: print("Error: Epochs must be a positive integer.")
        except ValueError: print("Error: Invalid input. Please enter an integer.")
    while True:
        try:
            learning_rate = float(input("Enter the learning rate (e.g., 0.001): "))
            if learning_rate > 0: break
            else: print("Error: Learning rate must be a positive number.")
        except ValueError: print("Error: Invalid input. Please enter a decimal value.")
    while True:
        try:
            lambda_pde = float(input("Enter the PDE loss weight (lambda_pde, e.g., 0.01): "))
            if lambda_pde >= 0: break
            else: print("Error: Lambda PDE must be a non-negative number.")
        except ValueError: print("Error: Invalid input. Please enter a decimal value.")
    while True:
        try:
            scheduler_gamma = float(input("Enter the learning rate scheduler gamma (e.g., 0.95): "))
            if 0 < scheduler_gamma <= 1: break
            else: print("Error: Gamma must be between 0 and 1.")
        except ValueError: print("Error: Invalid input. Please enter a decimal value.")
    return epochs, learning_rate, lambda_pde, scheduler_gamma

def main():
    run_timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_filename = f"pinn_run_{run_timestamp}.log"
    forecast_plot_filename = f"forecast_plot_{run_timestamp}.png"
    loss_plot_filename = f"loss_plot_{run_timestamp}.png"
    report_filename = f"pinn_report_{run_timestamp}.tex"
    zip_filename = f"pinn_output_{run_timestamp}.zip"

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(log_filename, 'w', 'utf-8'), logging.StreamHandler()], force=True)

    try:
        logging.info("--- Starting PINN Run ---")
        file_path, test_split_ratio = get_user_input_files()

        # --- Get Training Hyperparameters from User ---
        EPOCHS, LEARNING_RATE, LAMBDA_PDE, SCHEDULER_GAMMA = get_training_hyperparameters()

        logging.info(f"Data Configuration: File='{file_path}', Test Split Ratio='{test_split_ratio}'")
        logging.info(f"Training Hyperparameters: LR={LEARNING_RATE}, Gamma={SCHEDULER_GAMMA}, Epochs={EPOCHS}, Lambda_PDE={LAMBDA_PDE}")

        Z_WINDOW_SIZE, Y_WINDOW_SIZE, Y_STRIDE, W_WINDOW_SIZE, W_STD = 21, 41, 2, 41, 10.0

        data_handler = StockDataHandler(
            filepath=file_path, z_window_size=Z_WINDOW_SIZE, y_window_size=Y_WINDOW_SIZE, y_stride=Y_STRIDE,
            w_window_size=W_WINDOW_SIZE, w_std=W_STD
        )
        data_handler.load_and_process_data()
        (t_train, p_train), (t_test, p_test), (df_train, df_test) = data_handler.get_split_tensors(test_split_ratio)
        logging.info(f"Data split complete. Training size: {len(t_train)}, Test size: {len(t_test)}")

        P0_initial, k_initial, t1_initial, t2_initial, s_initial = get_initial_parameter_guesses(p_train)

        initial_params = {'P0': P0_initial, 'k': k_initial, 't1': t1_initial, 't2': t2_initial, 's': s_initial}
        logging.info(f"Using Initial Parameter Guesses: {initial_params}")

        pinn_model = PINN(P0_initial=P0_initial, k_initial=k_initial, t1_initial=t1_initial,
                          t2_initial=t2_initial, s_initial=s_initial)

        trainer = PINNTrainer(pinn_model=pinn_model, learning_rate=LEARNING_RATE, gamma=SCHEDULER_GAMMA)
        trainer.train(t_train, p_train, epochs=EPOCHS, lambda_pde=LAMBDA_PDE)

        p_train_pred = trainer.predict(t_train)
        forecast = trainer.predict(t_test)

        logging.info("--- Final Model Parameters ---")
        final_params = {
            'P0': trainer.model.P0.item(), 'k': trainer.model.k.item(),
            't1': trainer.model.t1.item(), 't2': trainer.model.t2.item(),
            's': trainer.model.s.item()
        }
        for name, val in final_params.items(): logging.info(f"Learned {name}: {val:.6f}")

        logging.info(f"Saving forecast plot to {forecast_plot_filename}...")
        plot_forecast_results(df_train, df_test, p_train.numpy(), p_test.numpy(), p_train_pred, forecast, save_path=forecast_plot_filename)
        logging.info(f"Saving loss plot to {loss_plot_filename}...")
        plot_loss(EPOCHS, trainer, save_path=loss_plot_filename)

        logging.info(f"Generating LaTeX report at {report_filename}...")

        scripts_used = [
            'pinn_forecasting_with_enhanced_logging.py',
            'pinn_utils2.py',
            'report_generator.py'
        ]

        config_params = {
            'Data File': file_path, 'Test Split Ratio': f"{test_split_ratio:.2f}",
            'Epochs': EPOCHS, 'Learning Rate': f"{LEARNING_RATE:.4f}",
            'Lambda PDE': f"{LAMBDA_PDE:.4f}", 'Scheduler Gamma': f"{SCHEDULER_GAMMA:.2f}",
            'Scripts Used': ', '.join(scripts_used)
        }

        generate_latex_report(
            report_filename=report_filename, log_filename=log_filename,
            forecast_plot_filename=forecast_plot_filename, loss_plot_filename=loss_plot_filename,
            final_params=final_params, config=config_params
        )
        logging.info("Report generation complete.")

        # --- Zip all output files ---
        logging.info(f"Zipping output files to {zip_filename}...")
        files_to_zip = [
            log_filename,
            forecast_plot_filename,
            loss_plot_filename,
            report_filename
        ]
        zip_output_files(zip_filename, files_to_zip)
        logging.info("Zipping complete.")

        logging.info("--- PINN Run Finished Successfully ---")

    except Exception as e:
        logging.error("An error occurred during the run:", exc_info=True)
    finally:
        logging.shutdown()

if __name__ == "__main__":
    main()
