import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

# Configuration
SAMPLE_RATE = 44100
RECORD_SECONDS = 0.25 # 4 snapshots taken per 1sec of audio recording
BLOCK_SIZE = int(SAMPLE_RATE * RECORD_SECONDS)
NYQUIST_FREQUENCY = SAMPLE_RATE / 2  # The maximum frequency we can see


def load_and_plot_fingerprint(filepath):
    if not os.path.exists(filepath):
        print(f"Error: File not found at '{filepath}'")
        print("Please check the path and try again.")
        return

    print(f"Loading data from: {filepath}")

    try:
        all_fingerprints = np.load(filepath, allow_pickle=True)

        if all_fingerprints.dtype == object:
            print("\nWarning: The loaded file appears to contain Python objects. Attempting to extract numerical data.")
            fingerprint_data = [
                row[1] for row in all_fingerprints
                if isinstance(row, (list, np.ndarray)) and len(row) > 1 and isinstance(row[1], np.ndarray)
            ]

            if not fingerprint_data:
                print("Error: Could not extract valid numerical fingerprint arrays from the object data structure.")
                print("The file structure might be different from expected.")
                return

            all_fingerprints = np.array(fingerprint_data)

        if all_fingerprints.ndim != 2:
            print(f"Error:Expected a 2D array, but got {all_fingerprints.ndim} dimensions :(.")
            return

        # Calculate the average fingerprint across all collected snapshots
        average_fingerprint = np.mean(all_fingerprints, axis=0)

        # Create the frequency axis for the plot
        num_features = len(average_fingerprint)
        frequencies = np.linspace(0, NYQUIST_FREQUENCY, num_features)

        # Create the plots
        plt.figure(figsize=(12, 6))
        plt.plot(frequencies, average_fingerprint)

        plt.title(f'Average FFT Fingerprint for: {os.path.basename(filepath)}')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude (Strength of Signal)')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.xlim(0, 15000)  # Zoom in to 15kHz
        plt.show()

        print("\nPlot displayed. Close the graph window to exit.")

    except Exception as e:
        print(f"An error occurred while processing the file: {e}")
        print("This usually means the data structure inside the file is unexpected.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Visualizes the average FFT fingerprint from a collected .npy file."
    )
    parser.add_argument(
        '--file',
        type=str,
        required=True,
        help='The path to the .npy data file (e.g., radio_data/WHMD_1071_Country_fingerprints.npy)'
    )

    args = parser.parse_args()

    load_and_plot_fingerprint(args.file)