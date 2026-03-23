import numpy as np
import matplotlib.pyplot as plt
import os
import argparse


def analyze_energy_stability(filepath):
    if not os.path.exists(filepath):
        print(f"Error: File not found at '{filepath}'")
        return

    print(f"Loading data from: {filepath}")

    try:
        # Load the data (allow_pickle=True just in case)
        data = np.load(filepath, allow_pickle=True)

        # Handle the case where the file contains the model object instead of data
        if data.ndim == 0:
            print("Error: This appears to be the trained model file, not the raw data.")
            print("Please use one of your 'station_fingerprints.npy' files.")
            return

        # Ensure 2D array [Rows=Snapshots, Cols=Frequencies]
        if data.ndim != 2:
            print(f"Error: Expected 2D array, got {data.ndim} dimensions.")
            return

        # --- THE METHOD ---
        # Instead of looking at frequencies, we sum the magnitude of all frequencies
        # for each row. This gives us the "Total Energy" (approximate volume) for that 0.25s snapshot.

        # 1. Calculate Energy per snapshot
        energy_profile = np.sum(data, axis=1)

        # 2. Normalize it (scale between 0 and 1) so we can compare quiet stations to loud ones fairly
        if np.max(energy_profile) > 0:
            energy_profile = energy_profile / np.max(energy_profile)

        # 3. Calculate "Stability" (Standard Deviation)
        # Low Std Dev = Steady volume (likely Music)
        # High Std Dev = Spikey volume (likely Talk/Sports with pauses)
        stability_score = np.std(energy_profile)

        print(f"\n--- Results for {os.path.basename(filepath)} ---")
        print(f"Energy Variance Score: {stability_score:.4f}")

        if stability_score < 0.15:
            print("Prediction: Likely MUSIC (Steady 'Wall of Sound')")
        else:
            print("Prediction: Likely TALK/SPORTS (Pauses and bursts of speech)")

        # --- PLOT ---
        plt.figure(figsize=(12, 6))

        # Plot the volume over time
        plt.plot(energy_profile, color='orange', label='Volume/Energy Level')

        # Draw a line for the average
        plt.axhline(y=np.mean(energy_profile), color='blue', linestyle='--', label='Average Volume')

        plt.title(f'Volume Stability Analysis: {os.path.basename(filepath)}')
        plt.xlabel('Time (Snapshots)')
        plt.ylabel('Normalized Energy (0.0 - 1.0)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Analyze the volume stability of an .npy file.")
    parser.add_argument('--file', type=str, required=True, help='Path to the .npy data file')
    args = parser.parse_args()

    analyze_energy_stability(args.file)

