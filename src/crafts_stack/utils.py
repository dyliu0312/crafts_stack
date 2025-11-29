import os
from itertools import product
from typing import List, Tuple

import h5py
import numpy as np


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
