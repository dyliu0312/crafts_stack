"""
A script to construct a galaxy catalog corresponding to the mock HIIM map.
It select galaxies with the target frequency range and also satisfy \
given selection (e.g. Main Sample), both given by the flag dataset in the HDF5 file.
It can optionally adding a frequency column and shifting coordinates.
"""

import os
from typing import Optional, Sequence

import h5py as h5
import numpy as np
import pandas as pd
from mytools.constant import HI_REST_FREQ
from mytools.data import read_h5


def get_flag(file, keys=None, repeat: int = 1):
    """
    Reads boolean flag datasets from an HDF5 file, combines them with logical AND,
    and optionally repeats the flag array.
    """
    if keys is None:
        with h5.File(file, "r") as f:
            keys = list(f.keys())
    elif isinstance(keys, str):
        keys = [keys]
    print(f"Reading keys {keys} from file")

    flag_list = read_h5(file, keys)
    flag = np.logical_and.reduce(flag_list)

    print(f"Select {np.sum(flag)} out of {flag.size} data points")

    if repeat > 1:
        flag = np.tile(flag, repeat)
        print(f"Repeat {repeat} times, total {flag.size} data points")

    return flag


def construct_galcat_pd(
    file: str,
    key_galcat: str,
    cols: Sequence[str] = ["RA", "DEC", "z"],
    add_freq_col: bool = True,
    shift: bool = True,
    curr_filed_center: Sequence[float] = [90, 0],
    target_filed_center: Sequence[float] = [218, 42.5],
    freq0: Optional[float] = None,
) -> pd.DataFrame:
    """
    Constructs a pandas DataFrame from a galaxy catalog dataset,
    optionally adds a frequency column and shifts coordinates.

    Parameters:
        file (str): Path to the HDF5 file containing the galaxy catalog.
        key_galcat (str): Key of the galaxy catalog dataset in the HDF5 file.
        cols (list of str): Column names for the galaxy catalog dataset.
        add_freq_col (bool): Whether to add a frequency column based on redshift.
        shift (bool): Whether to shift RA/DEC coordinates from `curr_filed_center` to `target_filed_center`.
        curr_filed_center (list of float): Current field center coordinates.
        target_filed_center (list of float): Target field center coordinates.
        freq0 (float): Rest frequency of HI emission in MHz.
    Returns:
        pandas.DataFrame: DataFrame containing the galaxy catalog data.
    """
    if freq0 is None:
        freq0 = HI_REST_FREQ.to_value("MHz")  # pyright: ignore[reportAssignmentType]

    galcat = read_h5(file, key_galcat)
    galcat_pd = pd.DataFrame(galcat, columns=cols)  # pyright: ignore[reportArgumentType]
    if add_freq_col:
        galcat_pd["Freq"] = galcat_pd["z"].map(lambda z: freq0 / (1 + z))

    if shift:
        print(f"Shift RA/DEC by {curr_filed_center} to {target_filed_center}")
        galcat_pd["RA"] = (
            galcat_pd["RA"] - curr_filed_center[0] + target_filed_center[0]
        )
        galcat_pd["DEC"] = (
            galcat_pd["DEC"] - curr_filed_center[1] + target_filed_center[1]
        )
    return galcat_pd


def select_gal(
    df: pd.DataFrame,
    lim_ra: Sequence[float] = [180, 255],
    lim_dec: Sequence[float] = [40, 45],
    lim_freq: Sequence[float] = [1264, 1307],
):
    """
    Selects galaxies based on specified limits for RA, DEC, and frequency.
    """
    valid = (
        (df["RA"] > lim_ra[0])
        & (df["RA"] < lim_ra[1])
        & (df["DEC"] > lim_dec[0])
        & (df["DEC"] < lim_dec[1])
        & (df["Freq"] > lim_freq[0])
        & (df["Freq"] < lim_freq[1])
    )
    return df[valid]


def _get_env_vars():
    """
    Reads configuration arguments from environment variables.
    """
    # save
    save_file = os.environ.get("SAVE_FILE")
    if save_file is None:
        raise ValueError("Missing required argument: SAVE_FILE")

    # TNG100 MGS selection flag
    flag_file = os.environ.get("FLAG_FILE")
    if flag_file is None:
        raise ValueError("Missing required argument: FLAG_FILE")

    flag_key = os.environ.get("FLAG_KEY")
    if flag_key is None:
        raise ValueError("Missing required argument: FLAG_KEY")

    repeat = int(os.environ.get("REPEAT", 1))

    # Extended and meshed catalog file
    galcat_file = os.environ.get("GALCAT_FILE")
    if galcat_file is None:
        raise ValueError("Missing required argument: GALCAT_FILE")

    galcat_key = os.environ.get("GALCAT_KEY")
    if galcat_key is None:
        raise ValueError("Missing required argument: GALCAT_KEY")

    galcat_flag_key = os.environ.get("GALCAT_FLAG_KEY")
    if galcat_flag_key is None:
        raise ValueError("Missing required argument: GALCAT_FLAG_KEY")

    # Parameters for catalog construction
    cols = os.environ.get("GALCAT_COLS", "RA DEC z").split()
    add_freq_col = os.environ.get("ADD_FREQ_COL", "True").lower() in ("true", "1", "t")
    shift = os.environ.get("SHIFT", "True").lower() in ("true", "1", "t")
    curr_filed_center = [
        float(x) for x in os.environ.get("CURR_FILED_CENTER", "90 0").split()
    ]
    target_filed_center = [
        float(x) for x in os.environ.get("TARGET_FILED_CENTER", "218 42.5").split()
    ]
    freq0 = os.environ.get("FREQ0")
    freq0 = float(freq0) if freq0 is not None else None

    lim_ra = [float(x) for x in os.environ.get("LIM_RA", "180 255").split()]
    lim_dec = [float(x) for x in os.environ.get("LIM_DEC", "40 45").split()]
    lim_freq = [float(x) for x in os.environ.get("LIM_FREQ", "1264 1307").split()]

    args = dict(
        save_file=save_file,
        flag_file=flag_file,
        flag_key=flag_key,
        repeat=repeat,
        galcat_file=galcat_file,
        galcat_key=galcat_key,
        galcat_flag_key=galcat_flag_key,
        cols=cols,
        add_freq_col=add_freq_col,
        shift=shift,
        curr_filed_center=curr_filed_center,
        target_filed_center=target_filed_center,
        freq0=freq0,
        lim_ra=lim_ra,
        lim_dec=lim_dec,
        lim_freq=lim_freq,
    )
    return args


def main():
    """
    Main function to orchestrate the galaxy catalog processing and saving.
    """
    args = _get_env_vars()

    # 1. Read and combine boolean flags
    # 1.1 Get a boolean flag array from MGS-like selection
    flag = get_flag(args["flag_file"], args["flag_key"], args["repeat"])  # pyright: ignore[reportArgumentType]

    # 1.2 Get a boolean flag array from galaxy projection （ flag outside the freqency range）
    flag_freq = get_flag(args["galcat_file"], args["galcat_flag_key"])

    # 1.3 Combine the two boolean flag arrays using logical AND
    flag = np.logical_and(flag, flag_freq)

    # 2. Construct the pandas DataFrame from the galaxy catalog data
    galcat_pd = construct_galcat_pd(
        args["galcat_file"],  # pyright: ignore[reportArgumentType]
        args["galcat_key"],  # pyright: ignore[reportArgumentType]
        args["cols"],  # pyright: ignore[reportArgumentType]
        args["add_freq_col"],  # pyright: ignore[reportArgumentType]
        args["shift"],  # pyright: ignore[reportArgumentType]
        args["curr_filed_center"],  # pyright: ignore[reportArgumentType]
        args["target_filed_center"],  # pyright: ignore[reportArgumentType]
        args["freq0"],  # pyright: ignore[reportArgumentType]
    )

    # 3. Filter the DataFrame using the boolean flag
    galcat_pd_flaged = galcat_pd[flag]

    # 4. Apply additional cuts based on spatial and frequency limits
    galcat_pd_flaged_cut = select_gal(
        galcat_pd_flaged,  # pyright: ignore[reportArgumentType]
        args["lim_ra"],  # pyright: ignore[reportArgumentType]
        args["lim_dec"],  # pyright: ignore[reportArgumentType]
        args["lim_freq"],  # pyright: ignore[reportArgumentType]
    )

    # 5. Determine the file saving mode
    # If a new file is specified, use 'w' (write), otherwise use 'a' (append)
    if args["save_file"] is None or args["save_file"] == args["galcat_file"]:
        save_file_path = args["galcat_file"]
        save_mode = "a"
    else:
        save_file_path = args["save_file"]
        save_mode = "w"

    # 6. Save the processed data to the HDF5 file
    with h5.File(save_file_path, save_mode) as f:
        # Loop through each column of the processed DataFrame
        for col in galcat_pd_flaged_cut.columns:
            data_to_save = galcat_pd_flaged_cut[col].values
            dtype = data_to_save.dtype
            f.create_dataset(name=col, data=data_to_save, dtype=dtype)

    print(f"✅ Save to {save_file_path}")


if __name__ == "__main__":
    main()
