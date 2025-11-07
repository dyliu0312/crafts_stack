"""
This is a script for stacking signals of galaxies in a map using HEALPix pixellation.
"""

import os
from collections import defaultdict
from multiprocessing import Pool

import h5py as h5
import numpy as np

from crafts_stack.hpmap import find_pixels_within_radius, shift_pixel_to_target


# --- HEALPix Utilities ---
def find_frequency_index(freq, freq_bins):
    """
    Finds the closest index for a given frequency from a frequency bin.

    This function calculates the absolute difference between the target frequency and all frequency bins,
    and then returns the index of the minimum difference, thus ensuring that the closest frequency is always found.

    Parameters:
    - freq (float): The target frequency to look for.
    - freq_bins (numpy.ndarray): The array of available frequency bins.

    Returns:
    - int: The index of the closest frequency in `freq_bins`.
    """
    # Calculate the absolute difference between the target frequency and all frequency bins
    idx = np.abs(freq_bins - freq).argmin()
    return idx


# --- Map Stacking Functions ---
def get_stack_value(results, sort=False):
    """
    Aggregates and averages pixel values from parallel stacking results.

    Given a list of dictionaries, where each dictionary contains a galaxy's contribution of pixel indices and values,
    this function combines all results. For overlapping pixels (i.e., those appearing in multiple galaxy contributions), it calculates their average value.

    Parameters:
    - results (list): A list of dictionaries, each containing the keys 'pix' (pixel indices) and 'val' (corresponding values).

    Returns:
    - union_pix (numpy.ndarray): A sorted array of all unique pixel indices found.
    - avg_values (numpy.ndarray): An array of aggregated (or averaged) values corresponding to each pixel in `union_pix`.
    """
    pix_values = defaultdict(list)
    for result in results:
        for pix, val in zip(result["pix"], result["val"]):
            pix_values[pix].append(val)

    # Calculate average for overlapping pixels and collect results
    union_pix = []
    avg_values = []
    for pix, vals in pix_values.items():
        union_pix.append(pix)
        avg_values.append(np.ma.mean(vals))

    if sort:
        # Sort pixels and values to maintain a consistent order
        union_pix = np.array(sorted(union_pix))

        # Reorder avg_values to match the sorted union_pix
        pix_order = np.array(list(pix_values.keys()))
        pix_vals = np.array([np.mean(v) for v in pix_values.values()])

        # Create a mapping from original pix to sorted index
        map_dict = {p: i for i, p in enumerate(pix_order)}
        sorted_indices = [map_dict[p] for p in union_pix]

        avg_values = pix_vals[sorted_indices]

    return np.array(union_pix), np.ma.array(avg_values)


def stack_healpix_map(
    catalog,
    map_pix,
    map_value,
    freq_bins,
    radius_deg=5,
    nside=32,
    nfreq=None,
    lon0=90,
    lat0=0,
):
    """
    Processes a galaxy catalog to produce a stacked HEALPix map result.

    This function iterates through the galaxy catalog, finds the corresponding signals on the HEALPix map, normalizes the pixel indices relative to
    a common center, and returns the results for aggregation.

    Parameters:
    - catalog (list): A list of galaxy dictionaries containing 'ra', 'dec', and 'freq' keys.
    - map_pix (numpy.ndarray): An array of pixel indices for the sparse sky map.
    - map_value (numpy.ndarray): A 2D array of signal values, with shape `(nfreq, len(map_pix))`.
    - freq_bins (numpy.ndarray): A list of frequency bins for the sky map.
    - radius_deg (float, optional): The angular radius around each galaxy in degrees.
                                    Defaults to 5.
    - nside (int, optional): The HEALPix nside parameter. Defaults to 32.
    - nfreq (int, optional): The number of adjacent frequencies to average around the target frequency. If None, no averaging is performed.
    - lon0 (float, optional): The reference longitude for normalization. Defaults to 90.
    - lat0 (float, optional): The reference latitude for normalization. Defaults to 0.

    Returns:
    - list: A list of dictionaries, each containing the normalized pixel indices ('pix') and
            the corresponding signal values ('val') for one galaxy.
    """
    stack_res = []

    for galaxy in catalog:
        ra, dec, freq = galaxy["ra"], galaxy["dec"], galaxy["freq"]
        ra = np.mod(ra, 360)  # Ensure RA is in the range [0, 360)
        found_pix_ind = find_pixels_within_radius(nside, ra, dec, radius_deg)
        freq_ind = find_frequency_index(freq, freq_bins)

        # get the values of valid pix
        cut_ind = np.isin(map_pix, found_pix_ind)
        found_pix_ind = map_pix[cut_ind]

        if nfreq is not None:
            freq_slice = slice(
                max(freq_ind - nfreq, 0), min(freq_ind + nfreq + 1, len(freq_bins) - 1)
            )
            values = map_value[freq_slice, :][:, cut_ind].mean(axis=0)
        else:
            values = map_value[freq_ind, :][cut_ind]

        shifted_pixel = shift_pixel_to_target(
            nside, found_pix_ind, original_field=(ra, dec), new_field=(lon0, lat0)
        )

        stack_res.append({"pix": shifted_pixel, "val": values})
    return stack_res


# --- Data Loading and Saving Functions ---
def load_map(
    map_path,
    key_pix="map_pix",
    key_value="clean_map",
    key_freq="freq",
    key_nside="nside",
    swap_axis=False,
):
    """
    Loads a sparse HEALPix map and associated data from an HDF5 file.

    Parameters:
    - map_path (str): The path to the HDF5 file containing the map data.
    - key_pix (str, optional): The key for the pixel index dataset. Defaults to "map_pix".
    - key_value (str, optional): The key for the signal value dataset. Defaults to "clean_map".
    - key_nside (str, optional): The key for the HEALPix nside parameter. Defaults to "nside".
    - key_freq (str, optional): The key for the frequency bin dataset. Defaults to "freq".
    - swap_axis (bool): If True, swaps the frequency and pixel axes. Defaults to False.

    Returns:
    - tuple: A tuple containing:
        - nside (int): The HEALPix nside parameter.
        - map_pix (numpy.ndarray): The array of pixel indices.
        - map_value (numpy.ndarray): The 2D array of signal values (freq, pix).
        - freq_bins (numpy.ndarray): The array of frequency bins.

    """
    with h5.File(map_path, "r") as f:
        nside = f[key_nside][()]  # type: ignore
        map_pix = f[key_pix][()]  # type: ignore
        freq_bins = f[key_freq][()]  # type: ignore
        map_value = f[key_value][()]  # type: ignore
    if str(swap_axis).lower() in ("true", "1", "yes"):
        print("Swapping frequency and pixel axes.")
        map_value = np.swapaxes(map_value, 0, 1)  # type: ignore
    print(
        f"Loaded map with nside={nside}, {len(map_pix)} pixels, and {len(freq_bins)} frequency bins."
    )  # type: ignore
    return nside, map_pix, map_value, freq_bins


def load_catalog(cat_path, keys=["ra", "dec", "freq"], cut=None):
    """
    Loads a galaxy catalog from an HDF5 file.

    Parameters:
    - cat_path (str): The path to the HDF5 file containing the catalog.
    - keys (list, optional): A list of dataset keys to extract for each galaxy.
                             Defaults to ["ra", "dec", "freq"].
    - cut (int, optional): If provided, only load a slice of the catalog up to this many galaxies.

    Returns:
    - list: A list of dictionaries, where each dictionary represents a galaxy and contains the specified keys and their values.
    """

    with h5.File(cat_path, "r") as f:
        if cut is not None:
            # If a slice is provided, only load galaxies within the specified range

            catalog = [
                {key: f[key][i] for key in keys} for i in range(len(f[keys[0]]))[:cut]
            ]  # type: ignore

        else:
            catalog = [{key: f[key][i] for key in keys} for i in range(len(f[keys[0]]))]  # type: ignore

    print(f"Loaded catalog with {len(catalog)} galaxies.")

    return catalog


def save_result(pix, signal, path, nside):
    """
    Saves the stacked HEALPix pixel indices and values to an HDF5 file.

    Parameters:
    - pix (numpy.ndarray): The pixel indices of the stacked result.
    - signal (numpy.ndarray): The signal values of the stacked result.
    - path (str): The output file path.
    - nside (int): The HEALPix nside parameter.
    """
    with h5.File(path, "w") as f:
        f.create_dataset("stack_pix", data=pix, dtype="i4")
        f.create_dataset("stack_signal", data=signal, dtype="f2")
        f.create_dataset("nside", data=nside, dtype="i4")


# --- Main Function ---
def main(
    cat_path,
    map_path,
    map_keys={},
    cat_keys={},
    nworker: int = 4,
    radius_deg: float = 1,
    nfreq=None,
    lon0: float = 90,
    lat0: float = 0,
    ouput_path=None,
):
    """
    Main function to load data, stack galaxy signals, and save the results.

    This function orchestrates the entire process: loading the galaxy catalog and HEALPix map, parallelizing the stacking process,
    combining the results, and then saving the stacked pixels and values to an HDF5 file.

    Parameters:
    - cat_path (str): The path to the HDF5 file containing the galaxy catalog.
    - map_path (str): The path to the HDF5 file containing the sparse HEALPix map.
    - map_keys (dict, optional): A dictionary of keys for loading map data.
    - cat_keys (dict, optional): A dictionary of keys for loading catalog data.
    - nworker (int, optional): The number of worker processes for parallelization. Defaults to 4.
    - radius_deg (float, optional): The angular radius for stacking, in degrees. Defaults to 1.
    - nfreq (int, optional): The number of adjacent frequencies to average. Defaults to None.
    - lon0 (float, optional): The reference longitude for stacking. Defaults to 90.
    - lat0 (float, optional): The reference latitude for stacking. Defaults to 0.
    - ouput_path (str, optional): The path to save the output HDF5 file.
                                  If None, a default path is automatically generated.
    """
    catalog = load_catalog(
        cat_path,
        keys=cat_keys.get("keys", ["ra", "dec", "freq"]),
        cut=cat_keys.get("cut", None),
    )
    nside, map_pix, map_value, freq_bins = load_map(map_path, **map_keys)

    # Split the galaxy catalog into chunks for parallel processing
    chunk_size = len(catalog) // nworker
    galaxy_chunks = [
        catalog[i : i + chunk_size] for i in range(0, len(catalog), chunk_size)
    ]
    # Ensure all galaxies are included in the chunks
    if len(catalog) % nworker != 0:
        galaxy_chunks[-1].extend(catalog[-(len(catalog) % nworker) :])

    # Process galaxy stacking in parallel
    with Pool(nworker) as pool:
        results = pool.starmap(
            stack_healpix_map,
            [
                (
                    chunk,
                    map_pix,
                    map_value,
                    freq_bins,
                    radius_deg,
                    nside,
                    nfreq,
                    lon0,
                    lat0,
                )
                for chunk in galaxy_chunks
            ],
        )

    # Combine results from parallel processes
    all_results = [item for sublist in results for item in sublist]
    stack_map_pix, stack_map_value = get_stack_value(all_results)

    if ouput_path is None:
        ouput_path = os.path.join(os.path.dirname(map_path), "auto_stack_result.h5")

    save_result(stack_map_pix, stack_map_value, ouput_path, nside)
    print(f"Stacked results saved to: {ouput_path}")


if __name__ == "__main__":
    # Read parameters from environment variables
    cat_path = os.getenv("CAT_PATH")
    map_path = os.getenv("MAP_PATH")

    map_key_values_str = os.getenv("MAP_KEYS")
    map_key_names = ["key_pix", "key_value", "key_freq", "key_nside", "swap_axis"]
    if map_key_values_str is not None:
        try:
            # Split and strip whitespace
            map_key_values_list = [
                item.strip() for item in map_key_values_str.split(",")
            ]
            # Convert to dictionary
            map_key_dict = {k: v for k, v in zip(map_key_names, map_key_values_list)}
        except Exception as e:
            raise ValueError(
                f"Error parsing MAP_KEYS: {map_key_values_str} to {map_key_names}"
            ) from e
    else:
        map_key_dict = dict(
            key_pix="map_pix",
            key_value="clean_map",
            key_freq="freq",
            key_nside="nside",
            swap_axis=False,
        )

    cut = os.getenv("CUT")
    if cut is not None:
        cat_keys = {"keys": ["ra", "dec", "freq"], "cut": int(cut)}
    else:
        cat_keys = {"keys": ["ra", "dec", "freq"]}

    radius_deg = float(os.getenv("R_DEG", 1.0))
    nfreq = int(os.getenv("NFREQ", -1))  # -1 means no frequency averaging
    lon0 = float(os.getenv("LON0", 90.0))
    lat0 = float(os.getenv("LAT0", 0.0))
    out_path = os.getenv("OUT_PATH")
    nworker = int(os.getenv("NWORKER", 1))
    print(f"Using {nworker} workers for parallel processing.")

    # Basic validation for required paths
    if not cat_path or not map_path:
        raise ValueError("Please set the CAT_PATH and MAP_PATH environment variables.")

    # If not provided or set to a non-positive value, set nfreq to None
    nfreq = nfreq if nfreq > 0 else None

    # Call the main function
    main(
        cat_path,
        map_path,
        map_keys=map_key_dict,
        cat_keys=cat_keys,
        nworker=nworker,
        radius_deg=radius_deg,
        nfreq=nfreq,
        lon0=lon0,
        lat0=lat0,
        ouput_path=out_path,
    )
