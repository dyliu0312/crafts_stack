"""
A script to prepare the cuboid map for further pairwise stacking, including zero padding (to make it equal length on last two axes), axis swapping (to make the frequency axis first) , and zero masking.
"""

import h5py as h5
import numpy as np


def center_to_edge_bin(bins):
    """
    Convert center bins to edge bins.
    """
    dd = bins[1] - bins[0]
    new_bins = [i for i in bins] + [bins[-1] + dd]
    return np.array(new_bins) - dd / 2


def extend_data_and_bins(data, bins):
    """
    Extend the data and bins by padding zeros on **both** sides to make it equal length on last two axes.
    Which means the shape of input data **(L, M, N)** became into **(L, N, N)**.
    The bins are extended assuming they are evenly spaced.

    Parameters:
    data (np.ndarray): The input data array with shape (L, M, N).
    bins (np.ndarray): The input bins array with shape (M,).

    Returns:
    np.ndarray: The extended data array with shape (L, N, N).
    np.ndarray: The extended bins array with shape (N,).
    """

    # Get the current and target size
    current_size = data.shape[1]
    target_size = data.shape[2]

    # Calculate how much padding is needed for both sides
    total_padding = target_size - current_size
    left_padding = total_padding // 2
    right_padding = total_padding - left_padding

    # Pad the data array with zeros on both sides
    extended_data = np.pad(
        data,
        ((0, 0), (left_padding, right_padding), (0, 0)),
        mode="constant",
        constant_values=0,
    )

    # Calculate the step size for bins (assuming they are evenly spaced)
    bin_step = bins[1] - bins[0]

    # Extend bins on both sides
    left_bins = np.arange(bins[0] - bin_step * left_padding, bins[0], bin_step)
    right_bins = np.arange(
        bins[-1] + bin_step, bins[-1] + bin_step * (right_padding + 1), bin_step
    )

    # Concatenate the extended bins
    extended_bins = np.concatenate([left_bins, bins, right_bins])

    return extended_data, extended_bins


def data_transform(data, bin_1, bin_2, padding=True, swap_axes=True, mask=True):
    """
    Transform the input data by applying zero padding, axis swapping, and zero masking if specified.

    This function processes the input data by optionally padding it with zeros, swapping its last two axes,
    and masking zero values. The bins corresponding to the data are also transformed accordingly.

    Parameters:
    - data (np.ndarray): The input data array with shape (L, M, N), where:
      - L is the number of samples,
      - M is the number of bins in the first dimension,
      - N is the number of bins in the second dimension.
    - bin_1 (np.ndarray): The bins array corresponding to the first dimension (shape: (M,)).
    - bin_2 (np.ndarray): The bins array corresponding to the second dimension (shape: (N,)).
    - padding (bool): Whether to pad the data with zeros to make its shape consistent.
                      Default is True. If True, `extend_data_and_bins` is called to pad the data and bins.
    - swap_axes (bool): Whether to swap the last two axes (i.e., swap the M and N dimensions).
                        Default is True. If True, the last two axes of the data are swapped.
    - mask (bool): Whether to mask the zero values in the data using a masked array.
                   Default is True. If True, zeros are masked in the data.

    Returns:
    - np.ndarray: The transformed data array with shape (L, N, N) after applying padding, axis swapping, and masking.
    - np.ndarray: The transformed bins array for the x-axis (shape: (N,)).
    - np.ndarray: The transformed bins array for the y-axis (shape: (N,)).

    Notes:
    - The `extend_data_and_bins` function is called if padding is enabled, which extends the data and bins
      to a consistent shape.
    - The `np.ma.masked_equal` function is used to create a masked array for data values equal to zero if masking is enabled.
    - If axis swapping is enabled, the last two axes of the data are swapped, and the corresponding bins are also swapped.
    - The function returns the transformed data and bin arrays, ready for further processing or saving.
    """

    if padding:
        data, bin_1 = extend_data_and_bins(data, bin_1)
        print("Data is padded with zeros.")

    if swap_axes:
        data = np.swapaxes(data, 1, 2)
        print("Last two axes were swapped.")
        xbin, ybin = bin_2, bin_1
    else:
        xbin, ybin = bin_1, bin_2

    if mask:
        data = np.ma.masked_equal(data, 0)
        print("Zeros are masked.")

    return data, xbin, ybin


def prepare_data_cube(
    input_path,
    output_path,
    input_keys=["freq", "lon_raster", "lat_raster", "values_raster"],
    output_keys=["T", "f_bin_edge", "x_bin_edge", "y_bin_edge"],
    padding=True,
    swap_axes=True,
    mask=True,
):
    """
    Prepares a data cube for stacking and saves the processed data into a new file.

    This function reads input data from a specified file, processes it by transforming the data and bins,
    and optionally applies padding, axis swapping, and masking. It then saves the transformed data to an output file.

    Parameters:
    - input_path (str): The path to the input file containing the raw data.
    - output_path (str): The path where the processed data will be saved.
    - input_keys (list[str]): List of keys corresponding to the data in the input file.
                               These include:
                               - "freq": Frequency bins (centered).
                               - "lon_raster": Longitude values (raster).
                               - "lat_raster": Latitude values (raster).
                               - "values_raster": Data values associated with latitude and longitude.
                               Default is ["freq", "lon_raster", "lat_raster", "values_raster"].
    - output_keys (list[str]): List of keys for the output file where the processed data will be stored.
                               These include:
                               - "T": Transformed data.
                               - "f_bin_edge": Frequency bin edges (calculated from centered bins).
                               - "x_bin_edge": Longitude bin edges.
                               - "y_bin_edge": Latitude bin edges.
                               Default is ["T", "f_bin_edge", "x_bin_edge", "y_bin_edge"].
    - padding (bool): If True, the data will be padded with zeros to ensure a consistent shape.
                      Default is True.
    - swap_axes (bool): If True, swaps the last two axes of the data (i.e., longitude and latitude axes).
                        This is useful for reshaping data into a specific format. Default is True.
    - mask (bool): If True, applies a mask to the data to exclude invalid or zero values.
                   Default is True.

    Returns:
    - None: The function does not return anything. It saves the processed data to the specified output file.

    Notes:
    - The function assumes that the input data is stored in an HDF5 file format.
    - The frequency bin edges are derived from the frequency centers using the `center_to_edge_bin` function.
    - The data is transformed by the `data_transform` function, which handles the application of padding, axis swapping, and masking.
    - The function creates the following datasets in the output file:
      - "T": The transformed data array.
      - "f_bin_edge": Frequency bin edges.
      - "x_bin_edge": Longitude bin edges.
      - "y_bin_edge": Latitude bin edges.
    - If the transformed data is a masked array, both the data and mask are stored separately.
    """

    with h5.File(input_path, "r") as fin:
        freq_key, lon_key, lat_key, val_key = input_keys
        r_lon = np.array(fin[lon_key][()])  # type: ignore
        r_lat = np.array(fin[lat_key][()])  # type: ignore
        r_val = np.array(fin[val_key][()])  # type: ignore
        fbin_center = np.array(fin[freq_key][()])  # type: ignore

    # Get edge bins from frequency center bins
    fbin = center_to_edge_bin(fbin_center)
    lon_bin = r_lon[0]
    lat_bin = r_lat[:, 0]

    # Transform the data (with optional padding, axis swapping, and masking)
    map_data, xbin, ybin = data_transform(
        r_val, lat_bin, lon_bin, padding, swap_axes, mask
    )

    # Save the transformed data and bins to an output file
    with h5.File(output_path, "w") as fout:
        data_key, fbin_key, xbin_key, ybin_key = output_keys
        if isinstance(map_data, np.ma.MaskedArray):
            fout.create_dataset(name=data_key, data=map_data.data, compression="gzip")
            fout.create_dataset(name="mask", data=map_data.mask, compression="gzip")
        else:
            fout.create_dataset(name=data_key, data=map_data, compression="gzip")
        fout.create_dataset(name=fbin_key, data=fbin)
        fout.create_dataset(name=xbin_key, data=xbin)
        fout.create_dataset(name=ybin_key, data=ybin)

    print(f"Data cube prepared and saved to {output_path}")


if __name__ == "__main__":
    import os

    INPUT_PATH = os.getenv("INPUT_PATH")
    OUTPUT_PATH = os.getenv("OUTPUT_PATH")

    if INPUT_PATH is None or OUTPUT_PATH is None:
        raise ValueError(
            "INPUT_PATH and OUTPUT_PATH must be set as environment variables"
        )

    INPUT_KEYS = os.getenv(
        "INPUT_KEYS", "freq,lon_raster,lat_raster,values_raster"
    ).split(",")
    OUTPUT_KEYS = os.getenv("OUTPUT_KEYS", "T,f_bin_edge,x_bin_edge,y_bin_edge").split(
        ","
    )
    PADDING = os.getenv("PADDING", "True").lower() in ["true", "1"]
    SWAP_AXES = os.getenv("SWAP_AXES", "True").lower() in ["true", "1"]
    MASK = os.getenv("MASK", "True").lower() in ["true", "1"]

    prepare_data_cube(
        INPUT_PATH,
        OUTPUT_PATH,
        input_keys=INPUT_KEYS,
        output_keys=OUTPUT_KEYS,
        padding=PADDING,
        swap_axes=SWAP_AXES,
        mask=MASK,
    )
