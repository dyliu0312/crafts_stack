from typing import List, Tuple

import numpy as np


def calculate_signal_background_stats(
    data: List[np.ndarray], mask: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate statistics for signal and background regions.

    Parameters:
        data: List of input data arrays.
        mask: Boolean mask.

    Returns:
        signal_means: List of signal means.
        signal_stds: List of signal standard deviations.
        background_means: List of background means.
        background_stds: List of background standard deviations.
    """

    signal_means = []
    signal_stds = []
    background_means = []
    background_stds = []

    for arr in data:
        if arr.shape != mask.shape:
            raise ValueError(
                f"Data shape {arr.shape} does not match mask shape {mask.shape}"
            )

        # Calculate signal statistics
        signal_data = arr[mask]
        signal_means.append(np.mean(signal_data))
        signal_stds.append(np.std(signal_data))

        # Calculate background statistics
        background_data = arr[~mask]
        background_means.append(np.mean(background_data))
        background_stds.append(np.std(background_data))

    return (
        np.array(signal_means),
        np.array(signal_stds),
        np.array(background_means),
        np.array(background_stds),
    )
