import os
from itertools import product
from typing import Dict, Iterable, List, Optional, Tuple, Union

import h5py
import numpy as np
from numpy.typing import NDArray


def get_filepath_format(base: str, pattern: str, *args) -> List[str]:
    """Generates file paths from a base directory and a pattern.

    This function creates all possible file path combinations from a base
    directory, a format pattern, and a variable number of arguments. Arguments
    can be single values or space-separated strings, which will be expanded
    into all combinations.

    Args:
        base (str): The base directory for the file paths.
        pattern (str): A format string with placeholders (e.g., '{}').
        *args: A variable number of arguments to fill the placeholders in the
            pattern. If an argument is a string containing spaces, it will be
            split to create multiple values for combination.

    Returns:
        List[str]: A list of all generated file path strings.

    Example:
        >>> base = '/home/user/data'
        >>> pattern = "{}_df{}_rmfg{}.h5"
        >>> args = ('tng', '120k', '0 1')
        >>> get_filepath_format(base, pattern, *args)
        ['/home/user/data/tng_df120k_rmfg0.h5', '/home/user/data/tng_df120k_rmfg1.h5']

    Note:
        The number of placeholders in the pattern must match the number of
        arguments provided in `*args`.
    """
    # Process arguments: split strings with spaces into lists, convert others to lists
    processed_args = []
    for arg in args:
        if isinstance(arg, str) and " " in arg:
            processed_args.append(arg.split())
        else:
            processed_args.append([str(arg)])

    # Generate all possible combinations of arguments
    filepaths = []
    for combination in product(*processed_args):
        filepath = os.path.join(base, pattern.format(*combination))
        filepaths.append(filepath)

    return filepaths


def load_groups(h5file: str, prefix: str = "500_") -> List[str]:
    """Loads group names from an HDF5 file that match a specific pattern.

    This function opens an HDF5 file and returns a list of root-level group
    names that start with the prefix.

    Args:
        h5file (str): Path to the input HDF5 file.
        prefix (str): Prefix of the group names, defaults to '500_'.

    Returns:
        List[str]: A list of group names matching the pattern.
    """
    with h5py.File(h5file, "r") as f:
        return [k for k in f.keys() if k.startswith(prefix)]


def load_stack(
    f: h5py.File,
    gname: str,
    unit_factor: float = 1.0,
    s_key: str = "Signal",
    m_key: str = "Mask",
) -> np.ma.MaskedArray:
    """Loads and processes a stacked signal from a group in an HDF5 file.

    This function reads a 'Signal' and a 'Mask' dataset from a specified
    group within an open HDF5 file. It creates a masked array, computes the
    mean along the first axis (axis=0), and scales the result by a unit
    factor.

    Args:
        f (h5py.File): An open HDF5 file object.
        gname (str): The name of the group to load data from.
        unit_factor (float, optional): A scaling factor to apply to the
            signal data. Defaults to 1.0.
        s_key (str, optional): The name of the signal dataset within the
            group. Defaults to "Signal".
        m_key (str, optional): The name of the mask dataset within the
            group. Defaults to "Mask".

    Returns:
        np.ma.MaskedArray: The processed (averaged and scaled) masked signal
            array.
    """
    si = f[f"{gname}/{s_key}"][:]  # pyright: ignore[reportIndexIssue]
    mi = f[f"{gname}/{m_key}"][:]  # pyright: ignore[reportIndexIssue]
    s = np.ma.array(si, mask=mi)
    s_mean = s.mean(axis=0) * unit_factor
    return s_mean


def load_stack_all_groups(
    h5file: str,
    unit_factor: float = 1.0,
    s_key: str = "Signal",
    m_key: str = "Mask",
) -> Tuple[np.ma.MaskedArray, List[np.ma.MaskedArray]]:
    """Loads and stacks signals from all relevant groups in an HDF5 file.

    This function identifies all groups starting with "500_" in the given
    HDF5 file, loads the stacked signal from each one, and then computes
    the mean of all group signals.

    Args:
        h5file (str): Path to the input HDF5 file.
        unit_factor (float, optional): A scaling factor to apply to the
            signal data in each group. Defaults to 1.0.
        s_key (str, optional): The name of the signal dataset within each
            group. Defaults to "Signal".
        m_key (str, optional): The name of the mask dataset within each
            group. Defaults to "Mask".

    Returns:
        Tuple[np.ma.MaskedArray, List[np.ma.MaskedArray]]: A tuple containing:
            - The final mean-stacked signal across all groups.
            - A list of the individual stacked signals from each group.
    """
    groups = load_groups(h5file)
    sig_list = []
    with h5py.File(h5file, "r") as f:
        for g in groups:
            s = load_stack(f, g, unit_factor=unit_factor, s_key=s_key, m_key=m_key)
            sig_list.append(s)
    stack_sig = np.ma.mean(sig_list, axis=0)
    return stack_sig, sig_list


def read_h5_data_with_mask(
    file_path: str,
    group_keys: Iterable[str],
    dataset_keys: Union[str, List[str]] = ["stack", "fitted", "residual", "jk_std"],
    mask_dataset_key: Optional[str] = "peak_mask",
    fixed_mask_data: Optional[NDArray[np.bool_]] = None,
) -> Dict[str, Dict[str, np.ma.MaskedArray]]:
    """
    Reads data from specified groups and datasets in an HDF5 file and applies masking.

    This function iterates through the specified HDF5 groups, reads the corresponding
    datasets, and applies a mask based on the following priority:
    1. If `fixed_mask_data` is provided, it is used.
    2. Otherwise, if `mask_dataset_key` exists in the HDF5 group, that dataset is used as the mask.
    3. Otherwise, the data is not masked (mask=False).

    Args:
        file_path (str): Path to the input HDF5 file.
        group_keys (Iterable[str]): The group keys (paths) to read data from.
        dataset_keys (Union[str, List[str]]): The dataset key(s) (names) to read data from.
            Defaults to ["stack", "fitted", "residual", "jk_std"].
        mask_dataset_key (Optional[str], optional): The key inside each HDF5 group that
            contains the mask data. If None, masking only occurs if `fixed_mask_data`
            is supplied. Defaults to "peak_mask".
        fixed_mask_data (Optional[NDArray[np.bool_]], optional): A fixed NumPy boolean array
            to apply as a mask to all read datasets. If provided, it takes precedence over
            `mask_dataset_key`.

    Returns:
        Dict(str, Dict[str, np.ma.MaskedArray]):
            A nested dictionary where outer keys are **group names**, inner keys are
            **dataset names**, and values are `numpy.ma.MaskedArray` containing the data.

    Raises:
        FileNotFoundError: If the `file_path` does not exist.

    Examples:
        Assume 'results.h5' is an HDF5 file containing groups 'Sample_A' and 'Sample_B'.
        'Sample_A' contains datasets 'stack' and 'peak_mask'.
        'Sample_B' contains only the dataset 'stack'.

        Example 1: Using the default mask "peak_mask" from the HDF5 file (Priority 2):

        >>> results_1 = read_hdf5_data_with_mask(
        ...     file_path="results.h5",
        ...     group_keys=["Sample_A", "Sample_B"],
        ...     dataset_keys=["stack"]
        ... )
        >>> # The 'stack' data in 'Sample_A' will be masked by 'peak_mask' (if present).
        >>> # The 'stack' data in 'Sample_B' will be unmasked (mask=False).
        >>> print(results_1.keys())
        dict_keys(['Sample_A', 'Sample_B'])

        Example 2: Overriding with a fixed external mask (Priority 1):

        >>> # Create a dummy fixed mask of the appropriate shape (e.g., a 2x5 array)
        >>> fixed_mask = np.full((2, 5), True)
        >>> results_2 = read_hdf5_data_with_mask(
        ...     file_path="results.h5",
        ...     group_keys=["Sample_A"],
        ...     dataset_keys="stack",
        ...     fixed_mask_data=fixed_mask # This mask is applied regardless of 'peak_mask' existence
        ... )
        >>> # The 'stack' data in 'Sample_A' will be masked entirely by 'fixed_mask_data'.
        >>> print(results_2["Sample_A"]["stack"].mask.all())
        True

        Example 3: Reading multiple datasets:

        >>> results_3 = read_hdf5_data_with_mask(
        ...     file_path="results.h5",
        ...     group_keys=["Sample_A"],
        ...     dataset_keys=["stack", "fitted"] # Read both datasets
        ... )
        >>> print(results_3["Sample_A"].keys())
        dict_keys(['stack', 'fitted'])
    """
    # Ensure dataset_keys is a list for iteration
    if isinstance(dataset_keys, str):
        dset_keys_list = [dataset_keys]
    else:
        dset_keys_list = dataset_keys

    results: Dict[str, Dict[str, np.ma.MaskedArray]] = {}

    try:
        # Use h5py.File context manager to safely open and close the file
        with h5py.File(file_path, "r") as hf:
            for group_name in group_keys:
                if group_name not in hf:
                    print(
                        f"Warning: Group not found in HDF5 file: '{group_name}', skipping."
                    )
                    continue

                group = hf[group_name]
                group_results: Dict[str, np.ma.MaskedArray] = {}

                # Determine the mask data to be used based on priority
                mask_data: Union[bool, NDArray]
                if fixed_mask_data is not None:
                    # Priority 1: Use the externally provided fixed mask
                    mask_data = fixed_mask_data
                elif mask_dataset_key is not None and mask_dataset_key in group:  # pyright: ignore[reportOperatorIssue]
                    # Priority 2: Read the mask from the HDF5 group
                    mask_data = group[mask_dataset_key][()]  # pyright: ignore[reportAssignmentType, reportIndexIssue]
                else:
                    # Priority 3: No mask applied (False indicates all data is valid)
                    mask_data = False

                for dataset_name in dset_keys_list:
                    if dataset_name in group:  # pyright: ignore[reportOperatorIssue]
                        # Read dataset data
                        data = group[dataset_name][()]  # pyright: ignore[reportIndexIssue]

                        # Create MaskedArray
                        masked_data = np.ma.array(data, mask=mask_data)
                        group_results[dataset_name] = masked_data
                    else:
                        print(
                            f"Warning: Dataset '{dataset_name}' not found in group '{group_name}', skipping."
                        )

                if group_results:
                    results[group_name] = group_results

    except FileNotFoundError as e:
        # Re-raise with a more informative message
        raise FileNotFoundError(f"File not found: {file_path}") from e

    return results
