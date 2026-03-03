# -*- coding: utf-8 -*-
"""
Synthetic Data Generator (with Phasing)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def generate_data(
    filename="synthetic_data.csv",
    n_points=400,
    P0=100.0,
    k=0.01,
    disruption_start=150,
    disruption_end=280,
    disruption_factor=0.3,
    disruption_components=[('sin', 1.0)],
    transition_steepness=0.1, # New parameter to control phase-in/out
    noise_level=2.0,
    plot_data=False
):
    """
    Generates synthetic time series data with an exponential growth trend and a temporary,
    additive disruption F(t) that smoothly phases in and out using a sigmoid window.

    Args:
        transition_steepness (float): Controls the sharpness of the phase-in/out.
            Analogous to the 's' parameter in the PINN model. Higher is steeper.
    """
    t = np.arange(n_points)

    # Base exponential growth y = P0 * exp(k*t)
    p_base = P0 * np.exp(k * t)

    # Create the composite disruption signal F(t)
    # This raw shape is defined as if it were active instantly
    raw_disruption = np.zeros(n_points)
    disruption_len = disruption_end - disruption_start

    if disruption_len > 0:
        base_magnitude = p_base[disruption_start] * disruption_factor

        for dtype, factor in disruption_components:
            if dtype == 'sin':
                x = np.linspace(0, np.pi, disruption_len)
                component_shape = np.sin(x)
            elif dtype == 'cos':
                x = np.linspace(-np.pi / 2, np.pi / 2, disruption_len)
                component_shape = np.cos(x)
            elif dtype == 'dip':
                x = np.linspace(0, np.pi, disruption_len)
                component_shape = -np.sin(x)
            elif dtype == 'sin2x':
                x = np.linspace(0, 2 * np.pi, disruption_len)
                component_shape = np.sin(x)
            elif dtype == 'cos2x':
                x = np.linspace(0, 2 * np.pi, disruption_len)
                component_shape = np.cos(x)
            elif dtype == 'linear':
                component_shape = np.linspace(0, 1, disruption_len)
            else:
                raise ValueError(f"Unknown disruption_type: '{dtype}'")

            raw_disruption[disruption_start:disruption_end] += (base_magnitude * factor * component_shape)

    # Create the smooth sigmoid window S(t)
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    sigmoid1 = sigmoid(transition_steepness * (t - disruption_start))
    sigmoid2 = sigmoid(transition_steepness * (t - disruption_end))
    window = sigmoid1 * (1 - sigmoid2)

    # Apply the window to the raw disruption: S(t) * F(t)
    final_disruption = raw_disruption * window

    # Add Gaussian noise
    noise = np.random.normal(0, noise_level, n_points)

    # Combine components: y = ke^(rt) + S(t)*F(t) + noise
    p_final = p_base + final_disruption + noise

    # Create DataFrame
    dates = pd.to_datetime(pd.date_range(start='2020-01-01', periods=n_points))
    df = pd.DataFrame({'<DATE>': dates, '<CLOSE>': p_final})

    # Save to CSV
    df.to_csv(filename, index=False)

    if plot_data:
        fig, ax1 = plt.subplots(figsize=(14, 8))

        # Plot the data on the primary axis
        ax1.plot(dates, p_base, label='Base Exponential Growth', linestyle='--', color='gray')
        ax1.plot(dates, p_final, label=f'Final Data', color='navy')
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Value")
        ax1.grid(True, linestyle='--', alpha=0.6)

        # Plot the window on the secondary axis
        ax2 = ax1.twinx()
        ax2.plot(dates, window, label='Sigmoid Window S(t)', color='red', linestyle=':')
        ax2.set_ylabel("Window Activation", color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        ax2.set_ylim(-0.05, 1.05)

        disruption_str = " + ".join([f"{factor}*{dtype}" for dtype, factor in disruption_components])
        plt.title(f"Synthetic Data with Phased Disruption (F(t) = {disruption_str})")

        # Combine legends from both axes
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='upper left')

        fig.tight_layout()
        plt.show()

    return filename

if __name__ == '__main__':
    print("Generating and plotting sample synthetic data with a smooth phase-in/out...")

    complex_disruption = [('sin2x', 1.0), ('linear', 0.5)]

    generate_data(
        disruption_components=complex_disruption,
        transition_steepness=0.08, # Using a smoother transition for demonstration
        plot_data=True,
        filename='synthetic_phased.csv'
    )
    print("Saved phased dataset.")

