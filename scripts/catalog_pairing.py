"""
A script to select galaxy pairs satifying a given angular separation and maximum radial separation
in a ra-dec-redshift space.
"""

import os
import sys
from functools import partial
from multiprocessing import Pool, cpu_count
from typing import Any, Dict, List, Optional, Tuple, Union

import astropy.units as u
import h5py as h5
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.cosmology import Planck18 as plk
from tqdm import tqdm


def load_catalog(
    file_path: str, dset_keys: List[str] = ["ra", "dec", "z", "freq"]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads galaxy data (RA, Dec, Redshift, Frequency) from an HDF5 file.

    Args:
        file_path (str): Path to the HDF5 catalog file.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            Arrays of RA, Dec, Redshift, and Frequency.
    """
    try:
        with h5.File(file_path, "r") as f:
            # Match keys with capital letter
            keys = [key for key in f.keys()]
            lowerkeys = [key.lower() for key in keys]
            new_dset_keys = []
            for dset_key in dset_keys:
                if dset_key not in f:
                    if dset_key in lowerkeys:
                        dset_key = keys[lowerkeys.index(dset_key)]
                    else:
                        raise KeyError(dset_key)
                new_dset_keys.append(dset_key)
            # Read data
            ra = np.array(
                f[new_dset_keys[0]], dtype="f4"
            )  # Ensure float type for calculations
            dec = np.array(f[new_dset_keys[1]], dtype="f4")
            redshift = np.array(f[new_dset_keys[2]], dtype="f4")
            frequency = np.array(f[new_dset_keys[3]], dtype="f4")

            # Adjust RA values to be within [0, 360) for consistency
            ra[ra < 0] += 360

            # Basic validation
            if not (len(ra) == len(dec) == len(redshift) == len(frequency)):
                raise ValueError("Mismatch in array lengths within the HDF5 catalog.")

            print(f"Loaded {len(ra)} entries from catalog: {file_path}")
            return ra, dec, redshift, frequency
    except FileNotFoundError:
        print(f"Error: Catalog file not found at '{file_path}'.")
        raise
    except KeyError as e:
        print(f"Error: Missing expected dataset '{e}' in HDF5 file '{file_path}'.")
        raise
    except Exception as e:
        print(f"An unexpected error occurred while loading catalog '{file_path}': {e}")
        raise


def radial_distance(z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Calculates the comoving radial distance for a given redshift.

    Args:
        z (Union[float, np.ndarray]): Redshift value(s).

    Returns:
        Union[float, np.ndarray]: Radial distance(s) in Mpc/h.
    """
    return (
        plk.comoving_distance(z).value * plk.h  # pyright: ignore[reportAttributeAccessIssue]
    )  # Convert to Mpc/h


def get_pair_info(
    ra1: float, ra2: float, dec1: float, dec2: float, z1: float, z2: float
) -> Tuple[float, float, bool]:
    """
    Calculates angular separation, radial separation, and checks if RA separation
    is dominant for a pair of galaxies.

    Args:
        ra1, ra2 (float): Right Ascension of galaxy 1 and 2 (degrees).
        dec1, dec2 (float): Declination of galaxy 1 and 2 (degrees).
        z1, z2 (float): Redshift of galaxy 1 and 2.

    Returns:
        Tuple[float, float, bool]:
            - angular_dist (float): Projected angular distance in Mpc.
            - radial_dist (float): Radial distance difference in Mpc.
            - is_ra_dominant (bool): True if RA separation is greater than DEC separation.
    """
    # Calculate angular distance
    coord1 = SkyCoord(ra=ra1 * u.deg, dec=dec1 * u.deg, frame="icrs")  # pyright: ignore[reportAttributeAccessIssue]
    coord2 = SkyCoord(ra=ra2 * u.deg, dec=dec2 * u.deg, frame="icrs")  # pyright: ignore[reportAttributeAccessIssue]
    # Get separation in radians
    sep_radian = coord1.separation(coord2).to(u.rad).value  # pyright: ignore[reportAttributeAccessIssue]

    # Calculate radial distances for each galaxy
    rad1 = radial_distance(z1)
    rad2 = radial_distance(z2)

    # Convert angular separation to a physical distance
    # This approximates the projected transverse separation by taking the mean distance
    angular_dist = np.mean([rad1, rad2]) * sep_radian  # pyright: ignore[reportCallIssue, reportArgumentType]

    # Calculate absolute difference in radial distances
    radial_dist = abs(rad1 - rad2)

    # Determine if separation is "longer" in RA direction
    is_ra_dominant = abs(ra2 - ra1) > abs(dec2 - dec1)

    return angular_dist, radial_dist, is_ra_dominant  # pyright: ignore[reportReturnType]


def process_single_index(
    i: int, ra, dec, redshift, frequency, min_ang_dist, max_ang_dist, max_radial_dist
) -> List[Dict[str, Any]]:
    """
    Processes a single galaxy index i to find valid pairs (i, j) for j > i.
    """
    n = len(ra)
    pairs = []

    for j in range(i + 1, n):
        ang_dist, rad_dist, is_ra_dominant = get_pair_info(
            ra[i], ra[j], dec[i], dec[j], redshift[i], redshift[j]
        )

        if min_ang_dist <= ang_dist <= max_ang_dist and rad_dist <= max_radial_dist:
            pairs.append(
                {
                    "ra1": ra[i],
                    "dec1": dec[i],
                    "frequency1": frequency[i],
                    "ra2": ra[j],
                    "dec2": dec[j],
                    "frequency2": frequency[j],
                    "angular_distance": ang_dist,
                    "radial_distance": rad_dist,
                    "is_ra_dominant": is_ra_dominant,
                }
            )

    return pairs


def find_galaxy_pairs(
    catalog_path: str,
    min_ang_dist: float,
    max_ang_dist: float,
    max_radial_dist: float,
    num_workers: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Finds galaxy pairs using multiprocessing within a catalog tile.
    """

    ra, dec, redshift, frequency = load_catalog(catalog_path)
    n = len(ra)

    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)

    with Pool(processes=num_workers) as pool:
        worker = partial(
            process_single_index,
            ra=ra,
            dec=dec,
            redshift=redshift,
            frequency=frequency,
            min_ang_dist=min_ang_dist,
            max_ang_dist=max_ang_dist,
            max_radial_dist=max_radial_dist,
        )

        results = list(
            tqdm(
                pool.imap(worker, range(n)),
                total=n,
                desc=f"Parallel processing {os.path.basename(catalog_path)}",
            )
        )

    # Flatten list of lists
    all_pairs = [pair for sublist in results for pair in sublist]
    print(f"Found {len(all_pairs)} galaxy pairs in {os.path.basename(catalog_path)}.")
    return all_pairs


def save_galaxy_pairs(file_path: str, pairs: List[Dict[str, Any]]) -> None:
    """
    Saves the identified galaxy pairs into an HDF5 file in the specified format.

    Args:
        file_path (str): Path to the output HDF5 file.
        pairs (List[Dict[str, Any]]): List of dictionaries representing the galaxy pairs.
    """
    if not pairs:
        print(f"No pairs to save for {file_path}. Skipping file creation.")
        return

    # Convert list of dicts to NumPy arrays in the specified format
    # 'pos' will be a (N, 6) array
    pos = np.array(
        [
            [p["ra1"], p["dec1"], p["frequency1"], p["ra2"], p["dec2"], p["frequency2"]]
            for p in pairs
        ],
        dtype="f4",
    )  # Use float32 for memory efficiency

    # 'dist_ang' and 'dist_rad' are 1D arrays
    dist_ang = np.array([p["angular_distance"] for p in pairs], dtype="f4")
    dist_rad = np.array([p["radial_distance"] for p in pairs], dtype="f4")

    # 'is_ra' is a 1D boolean array
    is_ra = np.array([p["is_ra_dominant"] for p in pairs], dtype=bool)

    # Ensure output directory exists
    output_dir = os.path.dirname(file_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    try:
        # Use 'w' mode to create a new file or overwrite if it exists.
        # This is generally safer for new output files per tile.
        with h5.File(file_path, "w") as f:
            f.create_dataset("pos", data=pos)
            f.create_dataset("dist_ang", data=dist_ang)
            f.create_dataset("dist_rad", data=dist_rad)
            f.create_dataset("is_ra", data=is_ra)

            # Add metadata as HDF5 attributes (optional, but good practice)
            f.attrs["description"] = "Galaxy pairs found with specified criteria"
            f["pos"].attrs["description"] = (
                "Positions of each galaxy pair. Saved as [ra1, dec1, freq1, ra2, dec2, freq2]. The units are degrees and MHz."
            )
            f["dist_ang"].attrs["description"] = (
                "Angular distance between the two galaxies in Mpc."
            )
            f["dist_rad"].attrs["description"] = (
                "Radial distance between the two galaxies in Mpc."
            )
            f["is_ra"].attrs["description"] = (
                "Boolean array indicating if the angular distance is along the RA axis."
            )

        print(f"Successfully saved {len(pairs)} galaxy pairs to '{file_path}'.")
    except Exception as e:
        print(f"Error saving galaxy pairs to '{file_path}': {e}", file=sys.stderr)
        raise


def process_galaxy_pairs(
    input_path: str,
    output_path: str,
    min_angular_dist: float,
    max_angular_dist: float,
    max_radial_dist: float,
    num_workers: Optional[int] = None,
) -> None:
    """
    Processes galaxy pairs for a single catalog tile. This function is designed
    to be called by a multiprocessing pool.

    Args:
        tile_index (int): The index of the current tile.
        input_path (str): Input catalog HDF5 file (e.g., 'split_galaxy_catalog.h5').
        output_path (str): Output pairs catalog HDF5 file (e.g., 'galaxy_pairs.h5').
        min_angular_dist (float): Minimum angular distance for pair selection (Mpc).
        max_angular_dist (float): Maximum angular distance for pair selection (Mpc).
        max_radial_dist (float): Maximum radial distance for pair selection (Mpc).
    """
    try:
        if not os.path.exists(input_path):
            print(f"Warning: Input catalog not found at '{input_path}'. Skipping.")
            return

        # Find and save pairs for this tile
        pairs = find_galaxy_pairs(
            input_path,
            min_angular_dist,
            max_angular_dist,
            max_radial_dist,
            num_workers,
        )
        if pairs:  # Only save if pairs are found
            save_galaxy_pairs(output_path, pairs)
        else:
            print(f"No pairs found. No output file created for '{output_path}'.")

    except Exception as e:
        # Log the error but don't re-raise, allowing other processes to continue.
        print(f"Error processing: {e}", file=sys.stderr)


# --- Main execution block ---
if __name__ == "__main__":
    # --- Read environment variables ---
    try:
        # Input catalog parameters (output from previous splitting step)
        INPUT_PATH = os.getenv("INPUT_PATH")
        if not INPUT_PATH:
            raise ValueError("Environment variable 'INPUT_PATH' is not set.")

        # Output pairs parameters
        OUTPUT_PATH = os.getenv("OUTPUT_PATH", "./galaxy_pair_catalog.h5")

        # Pair selection criteria
        MIN_ANGULAR_DIST = float(os.getenv("MIN_ANGULAR_DIST", "6"))  # Default 6 Mpc/h
        MAX_ANGULAR_DIST = float(
            os.getenv("MAX_ANGULAR_DIST", "14")
        )  # Default 14 Mpc/h
        MAX_RADIAL_DIST = float(os.getenv("MAX_RADIAL_DIST", "5"))  # Default 5 Mpc/h

        # Parallel processing configuration
        NUM_WORKERS = int(
            os.getenv("NUM_WORKERS", str(os.cpu_count()))
        )  # Default to CPU count
        if NUM_WORKERS <= 0:
            raise ValueError(
                "Environment variable 'NUM_WORKERS' must be a positive integer."
            )

    except ValueError as e:
        print(f"Error: Invalid or missing environment variable. {e}", file=sys.stderr)
        print(
            "Please ensure the following environment variables are correctly set:",
            file=sys.stderr,
        )
        print(
            "  - INPUT_PATH (path of the input galaxy catalog HDF5 file)",
            file=sys.stderr,
        )
        print(
            "  - OUTPUT_PATH (path of the output galaxy pairs HDF5 file)",
            file=sys.stderr,
        )
        print(
            "  - NUM_TILES (total number of galaxy catalog tiles to process)",
            file=sys.stderr,
        )
        print(
            "  - MIN_ANGULAR_DIST (optional, minimum angular separation in Mpc, default 8.9)",
            file=sys.stderr,
        )
        print(
            "  - MAX_ANGULAR_DIST (optional, maximum angular separation in Mpc, default 20.7)",
            file=sys.stderr,
        )
        print(
            "  - MAX_RADIAL_DIST (optional, maximum radial separation in Mpc, default 7.4)",
            file=sys.stderr,
        )
        print(
            "  - NUM_WORKERS (optional, number of CPU cores to use, default system CPU count)",
            file=sys.stderr,
        )
        sys.exit(1)
    except Exception as e:
        print(
            f"An unexpected error occurred while reading environment variables: {e}",
            file=sys.stderr,
        )
        sys.exit(1)

    # --- Main Workflow ---
    print("\n--- Starting Galaxy Pair Finding Workflow ---")
    process_galaxy_pairs(
        input_path=INPUT_PATH,
        output_path=OUTPUT_PATH,
        min_angular_dist=MIN_ANGULAR_DIST,
        max_angular_dist=MAX_ANGULAR_DIST,
        max_radial_dist=MAX_RADIAL_DIST,
        num_workers=NUM_WORKERS,
    )

    print("\n--- Galaxy Pair Finding Workflow Complete ---")
    print(f"Pair catalog saved to: {OUTPUT_PATH}")
