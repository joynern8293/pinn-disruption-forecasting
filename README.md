# pinn-disruption-forecasting

# PINN for Time Series Forecasting with Disruption Modeling

A Physics-Informed Neural Network (PINN) designed to model and forecast time series data that exhibits underlying exponential growth combined with a temporary disruption. The model learns the parameters of the growth and the disruption window directly from the data.

---

## Features

- **Synthetic Data Generation**: Includes a script (`synthetic_data_generator.py`) to create realistic time series data with tunable exponential growth, noise, and complex, phased-in disruptions.
- **PINN Core Model**: Uses a PyTorch-based neural network to learn the governing differential equation of the system.
- **Learnable Disruption Window**: The model automatically learns the start ($t_1$), end ($t_2$), and steepness ($s$) of a disruption period using a sigmoid window function $S(t) = \sigma(s(t-t_1)) \cdot (1 - \sigma(s(t-t_2)))$.
- **Interactive Execution**: Prompts the user for file paths, hyperparameters, and initial parameter guesses for a customized run.
- **Automated Reporting**: Automatically generates a comprehensive LaTeX report (`.tex`) and a `.zip` archive containing the run log, forecast plots, and the report itself.

---

## Project Files

- `pinn_forecasting_with_enhanced_logging.py`: The main executable script that orchestrates the data handling, model training, and reporting.

- `pinn_utilities_10_9_25.py`: Contains helper functions for plotting results and losses.

- `report_generator.py`: Contains functions to generate the final LaTeX report and zip the output files.

- `synthetic_data_generator.py`: A standalone script to generate sample time series data for testing the model.

- `test_pinn.py`: Unit tests for key components of the data handling and input functions.

---

## Requirements

The model is built using Python 3. The main dependencies are:

- PyTorch
- Pandas
- NumPy
- Matplotlib
- TQDM

You can install all the required packages by running:
```bash
pip install torch pandas numpy matplotlib tqdm
