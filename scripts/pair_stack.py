"""
main script to finish galaxy pairwise stacking process.
"""

import gc
import multiprocessing as mp
import os
import sys
from typing import List, Sequence  # pyright: ignore[reportDeprecated]

import h5py as h5
import numpy as np
from mytools.bins import (  # type: ignore
    get_ids_edge,
    set_resbins,
)
from mytools.data import (  # type: ignore
    is_exist,
    read_h5,
    save_h5,
    split_data_generator,
)
from mytools.stack import (  # type: ignore
    cut_freq,
    hist_data_3d,
)
from tqdm import tqdm


def join_path(base: str, prefix: str, extension: str = ".h5") -> str:
    # Ensure extension starts with a dot
    if not extension.startswith("."):
        extension = "." + extension
    path = os.path.join(base, prefix)
    if not path.endswith(extension):
        return path + extension
    else:
        return path


def get_ipm(pair_catalog, is_pro_ra, map_bins, nfreqslice, hist_bins):
    """
    get indivual pair's map.

    params:
        paircat (list): the paircat of two sources.
        pro_ra (bool): whether to project on ra axis.
        mapbins (tuple): the map bins.
        nfs (int): the number of frequency slices to extract.
        histbins (tuple): the histogram bins.

    return: the indivual pair's map.
    """

    # get map index
    ira1, idec1, ifreq1 = get_ids_edge(pair_catalog[0], map_bins)
    ira2, idec2, ifreq2 = get_ids_edge(pair_catalog[1], map_bins)

    # dimensions
    nfreq, nra, ndec = (
        mask_map.shape
    )  # ignore the error of using before assinment, this is necessray for multiprocessing

    # Extracting individual pair cut
    if nfreqslice == 0:
        signal = np.ma.zeros([nra, ndec])
        flag = np.zeros([nra, ndec], dtype=bool)

        if ifreq1 == ifreq2:
            signal = mask_map[ifreq1]
        else:
            if is_pro_ra:
                ifids = ifreq1 + float(ifreq2 - ifreq1) / float(ira2 - ira1) * (
                    np.arange(nra) - ira1
                )
            else:
                ifids = ifreq1 + float(ifreq2 - ifreq1) / float(idec2 - idec1) * (
                    np.arange(ndec) - idec1
                )

            freq_indices = np.round(ifids).astype(int)
            cut_freq_indices, flag = cut_freq(
                freq_indices, flag, is_pro_ra, 0, nfreq - 1
            )
            signal = (
                mask_map[cut_freq_indices, range(nra)]
                if is_pro_ra
                else mask_map[cut_freq_indices, :, range(nra)]
            )

    else:
        total_fs = 2 * nfreqslice + 1
        signal = np.ma.zeros([total_fs, nra, ndec])
        flag = np.zeros([total_fs, nra, ndec], dtype=bool)

        if ifreq1 == ifreq2:
            freq_indices = np.arange(ifreq1 - nfreqslice, ifreq1 + nfreqslice + 1)
            vallid = (freq_indices >= 0) & (freq_indices <= nfreq - 1)
            if vallid.sum() != total_fs:
                np.clip(freq_indices, 0, nfreq - 1, out=freq_indices)
                flag[~vallid] = True

            signal = mask_map[freq_indices]

        else:
            if is_pro_ra:
                ifids = ifreq1 + float(ifreq2 - ifreq1) / float(ira2 - ira1) * (
                    np.arange(nra) - ira1
                )
            else:
                ifids = ifreq1 + float(ifreq2 - ifreq1) / float(idec2 - idec1) * (
                    np.arange(ndec) - idec1
                )

            freq_indices = np.round(ifids).astype(int)
            for i in range(total_fs):
                cut_freq_indices, flag[i] = cut_freq(
                    freq_indices - nfreqslice + i, flag[i], is_pro_ra, 0, nfreq - 1
                )
                signal[i] = (
                    mask_map[cut_freq_indices, range(nra)]
                    if is_pro_ra
                    else mask_map[cut_freq_indices, :, range(nra)]
                )

    # add up the flag mask
    signal.mask += flag

    # hist data
    p1 = [ira1, idec1]
    p2 = [ira2, idec2]

    s = hist_data_3d(signal, p1, p2, hist_bins)

    return s


def stack_mp(
    nworker, pair_catalog, is_pro_ra, map_bins, nfreqslice, hist_bins, random_flip=True
):
    """
    stacking in multiprocessing async pool
    """
    # multiprocess
    total_pairs = len(pair_catalog)

    p = mp.Pool(nworker)
    res = [
        p.apply_async(
            get_ipm, (pair_catalog[i], is_pro_ra[i], map_bins, nfreqslice, hist_bins)
        )
        for i in range(total_pairs)
    ]

    p.close()
    p.join()

    # get the result
    if random_flip:
        if nfreqslice == 0:
            signal = [
                np.flip(i.get(), axis=0) if np.random.choice([True, False]) else i.get()
                for i in res
            ]  # upside down
            signal = [
                np.flip(s, axis=1) if np.random.choice([True, False]) else s
                for s in signal
            ]  # left right
        else:
            signal = [
                np.flip(i.get(), axis=1) if np.random.choice([True, False]) else i.get()
                for i in res
            ]  # upside down
            signal = [
                np.flip(s, axis=2) if np.random.choice([True, False]) else s
                for s in signal
            ]  # left right
    else:
        signal = [i.get() for i in res]

    # average those result.
    s = np.ma.array(signal).mean(axis=0)
    return s


def stack_run(
    output: str,
    pair_catalog: np.ndarray,
    is_pro_ra: np.ndarray,
    map_bins: Sequence[np.ndarray],
    hist_bins: Sequence[np.ndarray],
    nfreqslice: int,
    split_size: int = 500,
    nworker: int = 24,
    random_flip: bool = True,
    savekeys: List[str] = ["Signal", "Mask"],
    compression: str = "gzip",
    skip_exist: bool = True,
):
    """
    split data to stack and save seprately.

    Parameters
    ----------
    output : str
        output file name
    pair_catalog : np.ndarray
        pair catalog
    is_pro_ra : np.ndarray
        boolean array, indicating if the pair stacking is gona projection along ra (i.e. ra longer than dec)
    map_bins : list or tuple
        list of center bins of the map (ra, dec, freq)
    hist_bins : list or tuple
        list of center bins of the histogram (x, y)
    nfreqslice : int
        number of frequency slice to extract during stack
    split_size : int, optional
        number of pairs to process in one split, by default 500
    nworker : int, optional
        number of workers to use, by default 24
    random_flip : bool, optional
        if random flip the signal of pairs, by default True
    savekeys : list, optional
        list of keys of output dataset to save, by default ["Signal", "Mask"]
    compression : str, optional
        compression method of h5 file, by default "gzip"
    skip_exist : bool, optional
        if skip the split that already exist in the output file, by default True

    Returns
    -------
    None
    """

    # splite data for multiprocessing

    pbar = tqdm(
        enumerate(split_data_generator(split_size, pair_catalog, is_pro_ra)),
        total=len(pair_catalog) // split_size + 1,
        desc="Processing splits",
    )

    if skip_exist:
        print("--- skip_exist enabled, checking existing results ---")
        if is_exist(output):
            with h5.File(output, "r") as f:
                exist_keys = [i for i in f.keys()]
            if exist_keys == []:
                skip_exist = False
                print("--- No existing result found, disable skipping ---")
            else:
                print("--- Found existing result in output file, enable skipping ---")
        else:
            skip_exist = False
            print("--- Output file not found, disable skipping ---")

    for i, (ipair, ipra) in pbar:
        pbar.set_postfix({"split": f"{i}"})
        groupname = str(split_size) + "_" + str(i)
        if skip_exist and groupname in exist_keys:  # pyright: ignore[reportPossiblyUnboundVariable]
            continue

        s = stack_mp(nworker, ipair, ipra, map_bins, nfreqslice, hist_bins, random_flip)

        # save result
        save_h5(output, savekeys, [s.data, s.mask], groupname, compression=compression)


if __name__ == "__main__":
    print("---- Starting Stacking Script ----")
    print("--- Initializing ---")
    # --- Read configuration from environment variables ---
    try:
        # Required parameters
        ## files
        INPUT_MAP_BASE = os.getenv("INPUT_MAP_BASE")
        if INPUT_MAP_BASE is None:
            raise ValueError("Environment variable 'INPUT_MAP_BASE' is not set.")

        INPUT_MAP_PREFIX = os.getenv("INPUT_MAP_PREFIX")
        if INPUT_MAP_PREFIX is None:
            raise ValueError("Environment variable 'INPUT_MAP_PREFIX' is not set.")

        INPUT_PAIECAT_BASE = os.getenv("INPUT_PAIRCAT_BASE")
        if INPUT_PAIECAT_BASE is None:
            raise ValueError("Environment variable 'INPUT_PAIRCAT_BASE' is not set.")

        INPUT_PAIRCAT_PREFIX = os.getenv("INPUT_PAIRCAT_PREFIX")
        if INPUT_PAIRCAT_PREFIX is None:
            raise ValueError("Environment variable 'INPUT_PAIRCAT_PREFIX' is not set.")

        OUTPUT_STACK_BASE = os.getenv("OUTPUT_STACK_BASE")
        if OUTPUT_STACK_BASE is None:
            raise ValueError("Environment variable 'OUTPUT_STACK_BASE' is not set.")

        OUTPUT_STACK_PREFIX = os.getenv("OUTPUT_STACK_PREFIX")
        if OUTPUT_STACK_PREFIX is None:
            raise ValueError("Environment variable 'OUTPUT_STACK_PREFIX' is not set.")

        OUTPUT_STACK_DATA_KEYS = os.getenv("OUTPUT_STACK_KEYS", "Signal,Mask").split(
            ","
        )
        if len(OUTPUT_STACK_DATA_KEYS) != 2:
            raise ValueError(
                "Environment variable 'OUTPUT_STACK_KEYS' must contain exactly two keys."
            )

        ## stacking
        NFS_STR = os.getenv("NFS")
        if NFS_STR is None:
            raise ValueError(
                "Environment variable 'NFS' (number of frequency slices) is not set."
            )
        NFS = int(NFS_STR)

        SSIZE_STR = os.getenv("SSIZE")
        if SSIZE_STR is None:
            raise ValueError("Environment variable 'SSIZE' (split size) is not set.")
        SSIZE = int(SSIZE_STR)

        # Optional parameters with default values
        INPUT_MAP_MASKED = os.getenv("INPUT_MAP_MASKED", "True").lower() == "true"
        INPUT_MAP_KEYS = os.getenv(
            "INPUT_MAP_KEYS", "T,mask,f_bin_edge,x_bin_edge,y_bin_edge"
        ).split(",")
        INPUT_PAIECAT_KEYS = os.getenv("INPUT_PAIRCAT_KEYS", "is_ra,pos").split(",")

        NWORKER_STR = os.getenv("NWORKER")
        if NWORKER_STR is None or NWORKER_STR.lower() == "none":
            nworker = mp.cpu_count()
            print(
                f"Warning: Environment variable 'NWORKER' not set. Defaulting to {nworker} workers.",
                file=sys.stderr,
            )
        else:
            nworker = int(NWORKER_STR)

        RANDOM_FLIP_STR = os.getenv("RANDOM_FLIP", "True")
        RANDOM_FLIP = RANDOM_FLIP_STR.lower() == "true"

        HALFWIDTH_STR = os.getenv(
            "HALFWIDTH", "3.0"
        )  # Changed to float for more flexibility
        HALFWIDTH = float(HALFWIDTH_STR)

        NPIX_X_STR = os.getenv("NPIX_X", "120")
        NPIX_X = int(NPIX_X_STR)

        NPIX_Y_STR = os.getenv("NPIX_Y", "120")
        NPIX_Y = int(NPIX_Y_STR)

        # skip existing stacks
        SKIP_EXIST_STR = os.getenv("SKIP_EXIST", "False")
        SKIP_EXIST = SKIP_EXIST_STR.lower() == "true"

        # compression
        COMPRESSION = os.getenv("COMPRESSION", "gzip")

    except ValueError as e:
        print(f"Configuration Error: {e}", file=sys.stderr)
        print("\nPlease set the following environment variables:", file=sys.stderr)
        print("  Required:", file=sys.stderr)
        print("    INPUT_MAP_BASE       (e.g., '/data/my_input_map_base_path')")
        print("    INPUT_MAP_PREFIX     (e.g., 'my_input_map_prefix')")
        print("    INPUT_PAIRCAT_BASE   (e.g., '/data/my_input_paircat_base_path')")
        print("    INPUT_PAIRCAT_PREFIX (e.g., 'my_input_paircat_prefix')")
        print("    OUTPUT_STACK_BASE    (e.g., '/data/my_output_stack_base_path')")
        print("    OUTPUT_STACK_PREFIX  (e.g., 'my_output_stack_prefix')")
        print("    NFS                  (e.g., '10')", file=sys.stderr)
        print("    SSIZE                (e.g., '1000')", file=sys.stderr)
        print("  Optional:", file=sys.stderr)
        print("    COMPRESSION          (e.g., 'gzip' or 'lzf', default 'gzip')")
        print("    SKIP_EXIST           (e.g., 'True' or 'False', default 'False')")
        print(
            "    INPUT_MAP_MASK       (e.g., 'True' or 'False', default 'True')",
            file=sys.stderr,
        )
        print(
            "    INPUT_MAP_KEYS       (e.g., 'T,mask,f_bin_edge,x_bin_edge,y_bin_edge', default 'T,mask,f_bin_edge,x_bin_edge,y_bin_edge')",
            file=sys.stderr,
        )
        print(
            "    INPUT_PAIRCAT_KEYS   (e.g., 'is_ra,pos', default 'is_ra,pos')",
            file=sys.stderr,
        )
        print("    OUTPUT_STACK_KEYS    (e.g., 'Signal,Mask')", file=sys.stderr)
        print(
            "    NWORKER              (e.g., '4', default to CPU count)",
            file=sys.stderr,
        )
        print(
            "    RANDOM_FLIP          (e.g., 'True' or 'False', default 'True')",
            file=sys.stderr,
        )
        print("    HALFWIDTH            (e.g., '3.0', default '3.0')", file=sys.stderr)
        print("    NPIX_X               (e.g., '120', default '120')", file=sys.stderr)
        print("    NPIX_Y               (e.g., '120', default '120')", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(
            f"An unexpected error occurred while reading environment variables: {e}",
            file=sys.stderr,
        )
        sys.exit(1)

    # --- Construct tile-specific file paths ---

    map_path = join_path(INPUT_MAP_BASE, INPUT_MAP_PREFIX)
    paircat_path = join_path(INPUT_PAIECAT_BASE, INPUT_PAIRCAT_PREFIX)
    output_path = join_path(OUTPUT_STACK_BASE, OUTPUT_STACK_PREFIX)

    # --- Validate Input Files ---
    if not is_exist(map_path):
        print(f"Error: Map file not found at '{map_path}'. Exiting.", file=sys.stderr)
        sys.exit(1)

    if not is_exist(paircat_path):
        print(
            f"Error: Pair catalog not found at '{paircat_path}'. Exiting.",
            file=sys.stderr,
        )
        sys.exit(1)

    # --- Validate Output Path (Check if already exists, prevent overwrite) ---
    if is_exist(output_path):
        if SKIP_EXIST:
            print(
                "┌" + "-" * 62 + "┐",
                "│" + " " * 27 + "WARNING" + " " * 28 + "│",
                "├" + "-" * 62 + "┤",
                "| This job is currently trying to add new stacking results to  |",
                "| the output file that already exists.                         |",
                "└" + "-" * 62 + "┘",
                sep="\n",
            )
        else:
            print(
                "┌" + "-" * 62 + "┐",
                "│" + " " * 28 + "ERROR" + " " * 29 + "│",
                "├" + "-" * 62 + "┤",
                "| Output file already exists. To prevent accidental overwrite, |",
                "| this operation has been stopped.                             |",
                "|" + " " * 62 + "|",
                "| If you want to add new results to the existing file, instead |",
                "| of creating a new one, please set SKIP_EXIST = True.         |",
                "└" + "-" * 62 + "┘",
                sep="\n",
                file=sys.stderr,
            )
            sys.exit(1)

    # --- Print loaded configuration ---
    print("Processing with the following configuration:")
    print("--------------")
    print(f" Map: {map_path}")
    print(f" Map Masked: {INPUT_MAP_MASKED}")
    print(f" Pair Catalog: {paircat_path}")
    print(f" Output Stack: {output_path}")
    print(f" Output Stack Data Keys: {OUTPUT_STACK_DATA_KEYS}")
    print(f" NFS (frequency slices to stack): {NFS}")
    print(f" NWORKER (number of workers): {nworker}")
    print(f" SSIZE (catalog split size for processing): {SSIZE}")
    print(f" RANDOM_FLIP (randomly flip individual pair map): {RANDOM_FLIP}")
    print(f" COMPRESSION (compression method): {COMPRESSION}")
    print(f" SKIP_EXIST (skip existing stacks): {SKIP_EXIST}")
    print(f" HALFWIDTH (stack result map half-width): {HALFWIDTH}")
    print(f" NPIX_X (stack result map X pixels): {NPIX_X}")
    print(f" NPIX_Y (stack result map Y pixels): {NPIX_Y}")
    print(
        f" Stacked result map size: {2 * HALFWIDTH}x{2 * HALFWIDTH} with {NPIX_X}x{NPIX_Y} pixels"
    )
    print("--------------")

    # --- Loading Data ---
    print("\n--- Loading Data ---")

    ## paircat prepare
    # The pair catalog from the previous step saves 'is_ra' and 'pos'.
    # Ensure read_h5 correctly returns these.
    try:
        is_pra, pos = read_h5(paircat_path, INPUT_PAIECAT_KEYS)
    except Exception as e:
        print(f"Failed to load pair catalog from {paircat_path}: {e}", file=sys.stderr)
        sys.exit(1)

    # Shuffle the data to get more randomization
    # Concatenate is_pra (boolean) and pos_data (float) for shuffling.
    # We need to ensure 'is_pra' is numeric for column_stack, then cast back.
    pro_pos = np.column_stack(
        [is_pra.astype(np.int8), pos]
    )  # Convert bool to int8 for shuffling
    np.random.shuffle(pro_pos)

    is_pra = pro_pos[:, 0].astype(bool)  # Convert back to bool
    paircat_raw = pro_pos[:, 1:]  # Remaining columns are the position data

    # Split paircat to match the function input format: [[gal1_info], [gal2_info]]
    # Each inner list has [ra, dec, frequency]
    # 'pos' from previous script is [ra1, dec1, freq1, ra2, dec2, freq2]
    # So we need to reshape it into (N, 2, 3) where N is number of pairs.
    paircat = paircat_raw.reshape(
        -1, 2, 3
    )  # Reshape to N rows, 2 elements (galaxy), 3 values (ra, dec, freq)

    print(f"Loaded {len(is_pra)} galaxy pairs.")
    print(f"Pair catalog reshaped to: {paircat.shape}")

    ## mapfile prepare
    try:
        key_map, key_mask, key_fbin, key_xbin, key_ybin = INPUT_MAP_KEYS
        # load mapbins
        mapbin_keys_choice = [key_xbin, key_ybin, key_fbin]
        mapbins = read_h5(map_path, mapbin_keys_choice)
        # load map
        if INPUT_MAP_MASKED:
            map_array, mask_array = read_h5(map_path, [key_map, key_mask])
            mask_map = np.ma.masked_array(map_array, mask=mask_array, dtype=np.float32)
        else:
            map_array = read_h5(map_path, key_map)
            mask_map = np.ma.masked_array(
                map_array, mask=map_array == 0, dtype=np.float32
            )
            print("Masked map with zeros.")
        print(f"Loaded map from {map_path} with shape {map_array.shape}")  # pyright: ignore[reportAttributeAccessIssue]
    except Exception as e:
        print(f"Failed to load map data from {map_path}: {e}", file=sys.stderr)
        sys.exit(1)

    # Calculate resbins based on default args or env vars
    resbins = set_resbins(HALFWIDTH, NPIX_X, NPIX_Y, 2)

    # --- Stack ---
    print("\n---- Processing Stacking ----")
    stack_run(
        output=output_path,
        pair_catalog=paircat,
        is_pro_ra=is_pra,
        map_bins=mapbins,  # pyright: ignore[reportArgumentType]
        hist_bins=resbins,
        nfreqslice=NFS,
        split_size=SSIZE,
        nworker=nworker,
        random_flip=RANDOM_FLIP,
        savekeys=OUTPUT_STACK_DATA_KEYS,
        compression=COMPRESSION,
        skip_exist=SKIP_EXIST,
    )

    gc.collect()  # Trigger garbage collection
    print("\n---- Done ----")
