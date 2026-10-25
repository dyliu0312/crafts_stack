"""
This script is a modification of pair_stack_mpi.py to calculate the number of
contributing pixels for each point in the final stacked map using MPI.
"""

import gc
import os
import sys
from typing import Sequence

import h5py as h5
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


def get_ipm_for_count(
    pair_catalog, is_pro_ra, map_bins, nfreqslice, hist_bins, mask_map
):
    """
    Get indivual pair's map and convert it to a binary map for counting.
    Non-zero, unmasked pixels are set to 1.
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

    # --- MODIFICATION: Convert signal to binary for counting ---
    # Get data where not masked and not zero
    signal_data = signal.filled(0)
    signal_data[signal_data != 0] = 1
    signal = np.ma.masked_array(signal_data, mask=signal.mask)
    # --- END MODIFICATION ---

    # hist data
    p1 = [ira1, idec1]
    p2 = [ira2, idec2]

    s = hist_data_3d(signal, p1, p2, hist_bins)

    return s


def stack_mpi_for_count(
    pair_catalog, is_pro_ra, map_bins, nfreqslice, hist_bins, mask_map, random_flip=True
):
    """
    stacking with MPI for pixel counting
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
        s = get_ipm_for_count(
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

    return global_signal_sum


def stack_run_for_count(
    output: str,
    pair_catalog: np.ndarray,
    is_pro_ra: np.ndarray,
    map_bins: Sequence[np.ndarray],
    hist_bins: Sequence[np.ndarray],
    nfreqslice: int,
    mask_map: np.ma.MaskedArray,
    random_flip: bool = True,
    savekey: str = "Pixel_Counts",
    compression: str = "gzip",
):
    """
    Run stacking with MPI for pixel counting and save the result.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    s = stack_mpi_for_count(
        pair_catalog,
        is_pro_ra,
        map_bins,
        nfreqslice,
        hist_bins,
        mask_map,
        random_flip,
    )

    if rank == 0:
        save_h5(output, [savekey], [s.data.astype(np.int32)], compression=compression)


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        print("---- Starting Pixel Count Stacking Script (MPI) ----")
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

        OUTPUT_STACK_DATA_KEY = os.getenv("OUTPUT_STACK_KEY", "Pixel_Counts")

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
        print(f" Output Stack Data Key: {OUTPUT_STACK_DATA_KEY}")
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
    # Use shared memory for the large map to save memory on each node.
    # Each node's leader (node_rank 0) reads the data into a shared buffer.
    node_comm = comm.Split_type(MPI.COMM_TYPE_SHARED)
    node_rank = node_comm.Get_rank()

    if rank == 0:
        print("\n--- Loading Data into Shared Memory ---")

    # Step 1: Rank 0 reads map metadata and broadcasts it to all processes.
    if rank == 0:
        try:
            key_map, key_mask, key_fbin, key_xbin, key_ybin = INPUT_MAP_KEYS
            with h5.File(map_path, "r") as f:
                map_shape = f[key_map].shape
                map_dtype = f[key_map].dtype
                if INPUT_MAP_MASKED:
                    mask_shape = f[key_mask].shape
                    mask_dtype = f[key_mask].dtype
                else:
                    mask_shape = map_shape
                    mask_dtype = np.bool_

            map_meta = {
                "map_shape": map_shape,
                "map_dtype": map_dtype,
                "mask_shape": mask_shape,
                "mask_dtype": mask_dtype,
                "masked": INPUT_MAP_MASKED,
            }
            mapbin_keys_choice = [key_xbin, key_ybin, key_fbin]
            mapbins = read_h5(map_path, mapbin_keys_choice)

        except Exception as e:
            print(
                f"Rank 0 Error: Failed to read map metadata from {map_path}: {e}",
                file=sys.stderr,
            )
            # Use exceptions to signal errors to other ranks
            map_meta = e
            mapbins = e
    else:
        map_meta = None
        mapbins = None

    map_meta = comm.bcast(map_meta, root=0)
    mapbins = comm.bcast(mapbins, root=0)

    # Check if an error occurred during broadcast
    if isinstance(map_meta, Exception):
        if rank > 0:  # Prevent rank 0 from printing duplicate error
            print(f"Rank {rank} received error from rank 0. Exiting.", file=sys.stderr)
        sys.exit(1)
    if isinstance(mapbins, Exception):
        if rank > 0:
            print(
                f"Rank {rank} received error on mapbins from rank 0. Exiting.",
                file=sys.stderr,
            )
        sys.exit(1)

    # Step 2: Allocate shared memory windows on each node.
    win_map, win_mask = None, None
    try:
        map_nbytes = (
            int(np.prod(map_meta["map_shape"]))
            * np.dtype(map_meta["map_dtype"]).itemsize
        )
        win_map = MPI.Win.Allocate_shared(
            map_nbytes if node_rank == 0 else 0,
            np.dtype(map_meta["map_dtype"]).itemsize,
            comm=node_comm,
        )
        buf_map, _ = win_map.Shared_query(0)
        shared_map_array = np.ndarray(
            buffer=buf_map, dtype=map_meta["map_dtype"], shape=map_meta["map_shape"]
        )

        mask_nbytes = (
            int(np.prod(map_meta["mask_shape"]))
            * np.dtype(map_meta["mask_dtype"]).itemsize
        )
        win_mask = MPI.Win.Allocate_shared(
            mask_nbytes if node_rank == 0 else 0,
            np.dtype(map_meta["mask_dtype"]).itemsize,
            comm=node_comm,
        )
        buf_mask, _ = win_mask.Shared_query(0)
        shared_mask_array = np.ndarray(
            buffer=buf_mask, dtype=map_meta["mask_dtype"], shape=map_meta["mask_shape"]
        )
    except Exception as e:
        print(
            f"Rank {rank} Error: Failed to allocate shared memory: {e}", file=sys.stderr
        )
        comm.Abort(1)

    # Step 3: The leader process on each node (node_rank == 0) reads the data from the file.
    if node_rank == 0:
        try:
            key_map, key_mask, _, _, _ = INPUT_MAP_KEYS
            if map_meta["masked"]:
                map_data, mask_data = read_h5(map_path, [key_map, key_mask])
                shared_map_array[:] = map_data
                shared_mask_array[:] = mask_data
            else:
                map_data = read_h5(map_path, key_map)
                shared_map_array[:] = map_data
                shared_mask_array[:] = map_data == 0

            if rank == 0:  # Global rank 0 provides more detailed logging
                print(f"Loaded map from {map_path} with shape {shared_map_array.shape}")
                if not map_meta["masked"]:
                    print("Masked map with zeros.")

        except Exception as e:
            print(
                f"Rank {rank} (node leader) Error: Failed to read map data into shared memory: {e}",
                file=sys.stderr,
            )
            comm.Abort(1)

    # Step 4: Synchronize all processes on the node.
    node_comm.Barrier()

    # Step 5: All processes create a masked array view of the data in shared memory.
    mask_map = np.ma.masked_array(
        shared_map_array, mask=shared_mask_array, dtype=np.float32
    )

    resbins = set_resbins(HALFWIDTH, NPIX_X, NPIX_Y, 2)

    # --- Stack ---
    if rank == 0:
        print("\n---- Processing Stacking for Pixel Counts ----")
    stack_run_for_count(
        output=output_path,
        pair_catalog=paircat,
        is_pro_ra=is_pra,
        map_bins=mapbins,
        hist_bins=resbins,
        nfreqslice=NFS,
        mask_map=mask_map,
        random_flip=RANDOM_FLIP,
        savekey=OUTPUT_STACK_DATA_KEY,
        compression=COMPRESSION,
    )

    # Free shared memory windows
    if win_map:
        win_map.Free()
    if win_mask:
        win_mask.Free()

    if rank == 0:
        gc.collect()
        print("\n---- Done ----")
