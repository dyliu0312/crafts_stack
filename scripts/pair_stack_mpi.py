"""
main script to finish galaxy pairwise stacking process with MPI acceleration.
"""

import gc
import os
import sys
from typing import List, Sequence

import numpy as np
from mpi4py import MPI
from hifigps.bins import (
    get_ids_edge,
    set_resbins,
)
from hifigps.data import (
    is_exist,
    read_h5,
    save_h5,
)
from hifigps.stack import (
    cut_freq,
    hist_data_3d,
)


def join_path(base: str, prefix: str, extension: str = ".h5") -> str:
    if not extension.startswith("."):
        extension = "." + extension
    path = os.path.join(base, prefix)
    if not path.endswith(extension):
        return path + extension
    else:
        return path


def get_ipm(pair_catalog, is_pro_ra, map_bins, nfreqslice, hist_bins, mask_map):
    """
    get indivual pair's map.

    params:
        paircat (list): the paircat of two sources.
        pro_ra (bool): whether to project on ra axis.
        mapbins (tuple): the map bins.
        nfs (int): the number of frequency slices to extract.
        histbins (tuple): the histogram bins.
        mask_map (np.ma.MaskedArray): the masked map data.

    return: the indivual pair's map.
    """

    # get map index
    ira1, idec1, ifreq1 = get_ids_edge(pair_catalog[0], map_bins)
    ira2, idec2, ifreq2 = get_ids_edge(pair_catalog[1], map_bins)

    # dimensions
    nfreq, nra, ndec = mask_map.shape

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


def stack_mpi(
    pair_catalog, is_pro_ra, map_bins, nfreqslice, hist_bins, mask_map, random_flip=True
):
    """
    stacking with MPI
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    total_pairs = len(pair_catalog)
    local_n = total_pairs // size
    local_start = rank * local_n
    if rank == size - 1:
        local_end = total_pairs
    else:
        local_end = local_start + local_n

    local_pair_catalog = pair_catalog[local_start:local_end]
    local_is_pro_ra = is_pro_ra[local_start:local_end]

    local_signal = []
    for i in range(len(local_pair_catalog)):
        s = get_ipm(
            local_pair_catalog[i],
            local_is_pro_ra[i],
            map_bins,
            nfreqslice,
            hist_bins,
            mask_map,
        )
        if random_flip:
            if nfreqslice == 0:
                if np.random.choice([True, False]):
                    s = np.flip(s, axis=0)
                if np.random.choice([True, False]):
                    s = np.flip(s, axis=1)
            else:
                if np.random.choice([True, False]):
                    s = np.flip(s, axis=1)
                if np.random.choice([True, False]):
                    s = np.flip(s, axis=2)
        local_signal.append(s)

    local_signal_sum = np.ma.array(local_signal).sum(axis=0)

    global_signal_sum = np.ma.empty_like(local_signal_sum)
    comm.Allreduce(local_signal_sum, global_signal_sum, op=MPI.SUM)

    return global_signal_sum / total_pairs


def stack_run(
    output: str,
    pair_catalog: np.ndarray,
    is_pro_ra: np.ndarray,
    map_bins: Sequence[np.ndarray],
    hist_bins: Sequence[np.ndarray],
    nfreqslice: int,
    mask_map: np.ma.MaskedArray,
    random_flip: bool = True,
    savekeys: List[str] = ["Signal", "Mask"],
    compression: str = "gzip",
):
    """
    Run stacking with MPI and save the result.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    s = stack_mpi(
        pair_catalog,
        is_pro_ra,
        map_bins,
        nfreqslice,
        hist_bins,
        mask_map,
        random_flip,
    )

    if rank == 0:
        save_h5(output, savekeys, [s.data, s.mask], compression=compression)


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
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

        # Optional parameters with default values
        INPUT_MAP_MASKED = os.getenv("INPUT_MAP_MASKED", "True").lower() == "true"
        INPUT_MAP_KEYS = os.getenv(
            "INPUT_MAP_KEYS", "T,mask,f_bin_edge,x_bin_edge,y_bin_edge"
        ).split(",")
        INPUT_PAIECAT_KEYS = os.getenv("INPUT_PAIRCAT_KEYS", "is_ra,pos").split(",")

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

        # compression
        COMPRESSION = os.getenv("COMPRESSION", "gzip")

    except ValueError as e:
        if rank == 0:
            print(f"Configuration Error: {e}", file=sys.stderr)
            print("\nPlease set the following environment variables:", file=sys.stderr)
            # ... (rest of the error message)
        sys.exit(1)
    except Exception as e:
        if rank == 0:
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
    if rank == 0:
        if not is_exist(map_path):
            print(
                f"Error: Map file not found at '{map_path}'. Exiting.", file=sys.stderr
            )
            sys.exit(1)

        if not is_exist(paircat_path):
            print(
                f"Error: Pair catalog not found at '{paircat_path}'. Exiting.",
                file=sys.stderr,
            )
            sys.exit(1)

        if is_exist(output_path):
            print(
                f"Error: Output file already exists at '{output_path}'. Exiting to prevent overwrite.",
                file=sys.stderr,
            )
            sys.exit(1)

    # --- Print loaded configuration ---
    if rank == 0:
        print("Processing with the following configuration:")
        print("--------------")
        print(f" Map: {map_path}")
        print(f" Map Masked: {INPUT_MAP_MASKED}")
        print(f" Pair Catalog: {paircat_path}")
        print(f" Output Stack: {output_path}")
        print(f" Output Stack Data Keys: {OUTPUT_STACK_DATA_KEYS}")
        print(f" NFS (frequency slices to stack): {NFS}")
        print(f" RANDOM_FLIP (randomly flip individual pair map): {RANDOM_FLIP}")
        print(f" COMPRESSION (compression method): {COMPRESSION}")
        print(f" HALFWIDTH (stack result map half-width): {HALFWIDTH}")
        print(f" NPIX_X (stack result map X pixels): {NPIX_X}")
        print(f" NPIX_Y (stack result map Y pixels): {NPIX_Y}")
        print(
            f" Stacked result map size: {2 * HALFWIDTH}x{2 * HALFWIDTH} with {NPIX_X}x{NPIX_Y} pixels"
        )
        print("--------------")

    # --- Loading Data ---
    if rank == 0:
        print("\n--- Loading Data ---")

    ## paircat prepare
    if rank == 0:
        try:
            is_pra, pos = read_h5(paircat_path, INPUT_PAIECAT_KEYS)
        except Exception as e:
            print(
                f"Failed to load pair catalog from {paircat_path}: {e}", file=sys.stderr
            )
            sys.exit(1)

        pro_pos = np.column_stack([is_pra.astype(np.int8), pos])
        np.random.shuffle(pro_pos)

        is_pra = pro_pos[:, 0].astype(bool)
        paircat_raw = pro_pos[:, 1:]
        paircat = paircat_raw.reshape(-1, 2, 3)

        print(f"Loaded {len(is_pra)} galaxy pairs.")
        print(f"Pair catalog reshaped to: {paircat.shape}")
    else:
        is_pra = None
        paircat = None

    is_pra = comm.bcast(is_pra, root=0)
    paircat = comm.bcast(paircat, root=0)

    ## mapfile prepare
    if rank == 0:
        try:
            key_map, key_mask, key_fbin, key_xbin, key_ybin = INPUT_MAP_KEYS
            mapbin_keys_choice = [key_xbin, key_ybin, key_fbin]
            mapbins = read_h5(map_path, mapbin_keys_choice)
            if INPUT_MAP_MASKED:
                map_array, mask_array = read_h5(map_path, [key_map, key_mask])
                mask_map = np.ma.masked_array(
                    map_array, mask=mask_array, dtype=np.float32
                )
            else:
                map_array = read_h5(map_path, key_map)
                mask_map = np.ma.masked_array(
                    map_array, mask=map_array == 0, dtype=np.float32
                )
                print("Masked map with zeros.")
            print(f"Loaded map from {map_path} with shape {mask_map.shape}")
        except Exception as e:
            print(f"Failed to load map data from {map_path}: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        mask_map = None
        mapbins = None

    mask_map = comm.bcast(mask_map, root=0)
    mapbins = comm.bcast(mapbins, root=0)

    resbins = set_resbins(HALFWIDTH, NPIX_X, NPIX_Y, 2)

    # --- Stack ---
    if rank == 0:
        print("\n---- Processing Stacking ----")
    stack_run(
        output=output_path,
        pair_catalog=paircat,
        is_pro_ra=is_pra,
        map_bins=mapbins,
        hist_bins=resbins,
        nfreqslice=NFS,
        mask_map=mask_map,
        random_flip=RANDOM_FLIP,
        savekeys=OUTPUT_STACK_DATA_KEYS,
        compression=COMPRESSION,
    )

    if rank == 0:
        gc.collect()
        print("\n---- Done ----")
