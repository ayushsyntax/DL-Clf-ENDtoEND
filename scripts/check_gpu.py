"""
Diagnostic script for GPU hardware detection.
Verifies if TensorFlow can access NVIDIA acceleration.
Displays the count and details of available GPUs.
"""

import tensorflow


def run_hardware_diagnostic() -> None:
    """Scan the system for physical GPU devices.

    Prints the list of detected hardware to the console.
    """
    try:
        physical_gpus = tensorflow.config.list_physical_devices("GPU")

        if physical_gpus:
            print(f"GPUs detected: {len(physical_gpus)}")
            for device in physical_gpus:
                print(f"Device info: {device}")
        else:
            print("No physical GPU devices found by TensorFlow.")

    except Exception as diagnostic_error:
        print(f"Hardware scan failed: {str(diagnostic_error)}")


if __name__ == "__main__":
    run_hardware_diagnostic()
