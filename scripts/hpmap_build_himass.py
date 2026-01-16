"""
A script to build the HI mass HEALPix map from (TNG100) simulation.
"""
import gc
import os
import shutil
import logging

import h5py
import healpy as hp
import numpy as np
from astropy import units as u
from astropy.constants import c as speed_of_light # type: ignore
from astropy.cosmology import Planck18 as cosmo
from mpi4py import MPI
from scipy.interpolate import interp1d


# ---------------- Logging Setup ----------------
def setup_logger(rank):
    """
    Configure logging for each MPI rank.
    Each rank writes to stdout with its rank ID in the message.
    """
    logger = logging.getLogger(f"Rank{rank}")
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt=f"[Rank {rank:3d}] %(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )
    handler.setFormatter(formatter)

    if logger.hasHandlers():
        logger.handlers.clear()
    logger.addHandler(handler)
    return logger


# ---------------- Constants ----------------
u_mhz = u.Unit("MHz")
u_mpc = u.Unit("Mpc")
u_kms = u.Unit("km/s")

default_unit_mass = 1e10 * u.Unit("Msun")
rest_freq = 1420.40575177  # MHz, 21cm rest frequency
nu_HI = rest_freq * u_mhz

# Pre-compute comoving distance and redshift lookup tables
z_grid = np.linspace(0, 5, 5000)
d_grid = cosmo.comoving_distance(z_grid).value # type: ignore
dist2z = interp1d(
    d_grid, z_grid, kind="cubic", bounds_error=False, fill_value="extrapolate" # type: ignore
)

# ---------------- MPI Init ----------------
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
logger = setup_logger(rank)


# ---------------- Functions ----------------
def box_local_to_radec(xyz_local_mpc, ra0_deg=0.0, dec0_deg=0.0, z0=None, R0_mpc=None):
    xyz_local_mpc = np.asarray(xyz_local_mpc, dtype="f4")
    if xyz_local_mpc.ndim != 2 or xyz_local_mpc.shape[1] != 3:
        raise ValueError("xyz_local_mpc must be an array of shape (N,3).")

    if (z0 is None) and (R0_mpc is None):
        raise ValueError("Provide either z0 or R0_mpc.")
    if R0_mpc is None:
        R0_mpc = cosmo.comoving_distance(z0).to_value(u_mpc) # type: ignore

    ra0, dec0 = np.deg2rad(ra0_deg), np.deg2rad(dec0_deg)
    r_hat = np.array(
        [np.cos(dec0) * np.cos(ra0), np.cos(dec0) * np.sin(ra0), np.sin(dec0)]
    )
    e_ra = np.array([-np.sin(ra0), np.cos(ra0), 0.0])
    e_dec = np.array(
        [-np.sin(dec0) * np.cos(ra0), -np.sin(dec0) * np.sin(ra0), np.cos(dec0)]
    )

    x, y, z = xyz_local_mpc[:, 0], xyz_local_mpc[:, 1], xyz_local_mpc[:, 2]
    R = (
        (R0_mpc + z)[:, None] * r_hat[None, :]
        + x[:, None] * e_ra[None, :]
        + y[:, None] * e_dec[None, :]
    )
    D = np.linalg.norm(R, axis=1)
    nhat = R / D[:, None]
    ra_rad = np.mod(np.arctan2(nhat[:, 1], nhat[:, 0]), 2 * np.pi)
    dec_rad = np.arcsin(np.clip(nhat[:, 2], -1.0, 1.0))
    return ra_rad, dec_rad, D


def distance_to_redshift(D_mpc):
    return dist2z(D_mpc)


def apply_rsd(z_cos, vlos_kms):
    return z_cos + (vlos_kms / (speed_of_light.to_value(u_kms))) * (1 + z_cos)


def select_pixels_in_region(nside, ra_range=None, dec_range=None):
    npix = hp.nside2npix(nside)
    ra, dec = hp.pix2ang(nside, np.arange(npix), lonlat=True)  # degrees
    mask = np.ones(npix, dtype=bool)
    if ra_range is not None:
        ra_min, ra_max = ra_range
        mask &= (ra >= ra_min) & (ra <= ra_max)
    if dec_range is not None:
        dec_min, dec_max = dec_range
        mask &= (dec >= dec_min) & (dec <= dec_max)
    return np.where(mask)[0]


def _tile_data_generator(pos_chunk, vlos_chunk, mass_chunk, Lbox_mpc, tiling):
    nx, ny, nz = tiling
    if nx == 1 and ny == 1 and nz == 1:
        yield pos_chunk, vlos_chunk, mass_chunk
        return

    for i in range(-(nx // 2), nx - nx // 2):
        for j in range(-(ny // 2), ny - ny // 2):
            for k in range(-(nz // 2), nz - nz // 2):
                shift = np.array([i * Lbox_mpc, j * Lbox_mpc, k * Lbox_mpc])
                pos_tiled = pos_chunk + shift
                yield pos_tiled, vlos_chunk, mass_chunk


def _read_and_broadcast_params(comm, rank):
    params = None
    if rank == 0:
        input_cat_filename = os.getenv("INPUT_CAT_FILE")
        if not input_cat_filename:
            raise ValueError("Error: 'INPUT_CAT_FILE' env var is not set.")

        keys = {
            "POS_KEY_PAR": "par_pos", "VLOS_KEY_PAR": "par_vlos", "MASS_KEY_PAR": "hi_mass",
            "POS_KEY_GAL": "gal_pos", "VLOS_KEY_GAL": "gal_vlos", 
            "OUTPUT_MASS_FILE": "output_mass_cube.h5", "TMP_DIR": None, 
            "Z0": "0.1", "RA0": "90", "DEC0": "0", "NSIDE": "128",
            "NFREQ": "5", "CHUNK_SIZE": "1000000", "GC_EVERY": "4",
            "RA_MIN": "0", "RA_MAX": "360", "DEC_MIN": "-90", "DEC_MAX": "90",
            "TILING": "1,1,1", "LBOX_MPC": None, "UNIT_MASS": None,
            "NU_MIN": None, "NU_MAX": None, "SKIP_PAR": True
        }

        env_vars = {key: os.getenv(key, default) for key, default in keys.items()}

        with h5py.File(input_cat_filename, "r") as f:

            N_orig_par = f[env_vars["POS_KEY_PAR"]].shape[0] if env_vars["POS_KEY_PAR"] in f else 0 # type: ignore
            N_orig_gal = f[env_vars["POS_KEY_GAL"]].shape[0] if env_vars["POS_KEY_GAL"] in f else 0 # type: ignore

            if env_vars["LBOX_MPC"] is None:
                env_vars["LBOX_MPC"] = f.attrs.get("Lbox_mpc")
            if env_vars["LBOX_MPC"] is None:
                raise ValueError("Error: 'LBOX_MPC' not set in env or HDF5 file.")

        nu_min_str, nu_max_str = env_vars["NU_MIN"], env_vars["NU_MAX"]
        if nu_min_str is None or nu_max_str is None:
            raise ValueError("Error: 'NU_MIN' or 'NU_MAX' must be set as environment variables.")
        nu_min, nu_max = float(nu_min_str), float(nu_max_str)

        freq_edges = np.linspace(nu_min, nu_max, int(env_vars["NFREQ"]) + 1)
        freqs = 0.5 * (freq_edges[:-1] + freq_edges[1:])
        logger.info(f"Frequency edges: {freq_edges[:2]} ... {freq_edges[-2:]} MHz")

        ra_range = (float(env_vars["RA_MIN"]), float(env_vars["RA_MAX"]))
        dec_range = (float(env_vars["DEC_MIN"]), float(env_vars["DEC_MAX"]))
        nside = int(env_vars["NSIDE"])
        pix_idx = select_pixels_in_region(nside, ra_range, dec_range)
        logger.info(f"Selected {len(pix_idx)} pixels for this sky region.")

        if env_vars["TMP_DIR"] is None:
            out_dir = os.path.dirname(os.path.abspath(env_vars["OUTPUT_MASS_FILE"])) or "."
            base_outname = os.path.splitext(os.path.basename(env_vars["OUTPUT_MASS_FILE"]))[0]
            env_vars["TMP_DIR"] = os.path.join(out_dir, f"tmp_{base_outname}_{os.getpid()}")
        os.makedirs(env_vars["TMP_DIR"], exist_ok=True)

        params = {
            "input_cat_filename": input_cat_filename,
            "output_mass_filename": env_vars["OUTPUT_MASS_FILE"],
            "tmp_dir": env_vars["TMP_DIR"],
            "z0": float(env_vars["Z0"]),
            "ra0_deg": float(env_vars["RA0"]),
            "dec0_deg": float(env_vars["DEC0"]),
            "nside": nside,
            "nfreq": int(env_vars["NFREQ"]),
            "pos_key_par": env_vars["POS_KEY_PAR"],
            "vlos_key_par": env_vars["VLOS_KEY_PAR"],
            "mass_key_par": env_vars["MASS_KEY_PAR"],
            "skip_par": any(env_vars["SKIP_PAR"].lower() == x for x in ("true", "1", "yes")),
            "pos_key_gal": env_vars["POS_KEY_GAL"],
            "vlos_key_gal": env_vars["VLOS_KEY_GAL"],
            "chunk_size": int(env_vars["CHUNK_SIZE"]),
            "nu_min": nu_min,
            "nu_max": nu_max,
            "freqs": freqs,
            "freq_edges": freq_edges,
            "ra_range": ra_range,
            "dec_range": dec_range,
            "tiling": tuple(map(int, env_vars["TILING"].split(","))),
            "gc_every": int(env_vars["GC_EVERY"]),
            "Lbox_mpc": float(env_vars["LBOX_MPC"]),
            "unit_mass": env_vars["UNIT_MASS"],
            "N_orig_par": N_orig_par,
            "N_orig_gal": N_orig_gal,
            "pix_idx": pix_idx
        }
    return comm.bcast(params, root=0)

def _process_data_chunk(comm, rank, params):
    """
    Performs the main data processing logic for a given rank.
    Reads data in chunks, applies transformations, bins it, and
    writes the local results to a temporary file.
    """
    
    # Unpack parameters
    (input_cat_filename, pos_key_par, vlos_key_par, mass_key_par, pos_key_gal, vlos_key_gal,
    chunk_size, tiling, Lbox_mpc, z0, ra0_deg, dec0_deg, nside, nfreq,
    freqs, freq_edges, pix_idx, N_orig_par, N_orig_gal, tmp_dir, skip_par) = (
        params["input_cat_filename"],
        params["pos_key_par"],params["vlos_key_par"],params["mass_key_par"],params["pos_key_gal"],params["vlos_key_gal"],
        params["chunk_size"],params["tiling"], params["Lbox_mpc"], params["z0"], 
        params["ra0_deg"],params["dec0_deg"], params["nside"], params["nfreq"], 
        params["freqs"], params["freq_edges"], params["pix_idx"],
        params["N_orig_par"], params["N_orig_gal"], params["tmp_dir"], params["skip_par"]
    )

    # Calculate local chunk distribution
    if not skip_par:
        counts_par = [(N_orig_par // comm.Get_size()) + (1 if i < N_orig_par % comm.Get_size() else 0) for i in range(comm.Get_size())]
        offsets_par = np.cumsum([0] + counts_par[:-1])
        start_par, end_par = offsets_par[rank], offsets_par[rank] + counts_par[rank]

    counts_gal = [(N_orig_gal // comm.Get_size()) + (1 if i < N_orig_gal % comm.Get_size() else 0) for i in range(comm.Get_size())]
    offsets_gal = np.cumsum([0] + counts_gal[:-1])
    start_gal, end_gal = offsets_gal[rank], offsets_gal[rank] + counts_gal[rank]

    # Initialize local data structures
    npix_sub = len(pix_idx)
    pix_map = np.full(hp.nside2npix(nside), 0, dtype=int)
    pix_map[pix_idx] = np.ones_like(npix_sub)

    mass_cube_local = np.zeros((nfreq, npix_sub), dtype=np.float32)
    gal_cat_local = []
    gal_cat_local_flag = []

    with h5py.File(input_cat_filename, "r", driver="mpio", comm=comm) as f:
        if not skip_par:
            # ---- Process Particle Data ----
            if pos_key_par in f and vlos_key_par in f and mass_key_par in f:
                dset_pos_par = f[pos_key_par]
                dset_vlos_par = f[vlos_key_par]
                dset_mass_par = f[mass_key_par]

                for i in range(start_par, end_par, chunk_size): # type: ignore
                    j = min(i + chunk_size, end_par) # type: ignore
                    pos_chunk = dset_pos_par[i:j] # type: ignore
                    vlos_chunk = dset_vlos_par[i:j] # type: ignore
                    mass_chunk = dset_mass_par[i:j] # type: ignore

                    for pos_tiled, vlos_tiled, mass_tiled in _tile_data_generator(
                        pos_chunk, vlos_chunk, mass_chunk, Lbox_mpc, tiling
                    ):
                        lon, lat, D = box_local_to_radec(pos_tiled, ra0_deg, dec0_deg, z0)
                        z_cos = distance_to_redshift(D)
                        z_obs = apply_rsd(z_cos, vlos_tiled)
                        nu = rest_freq / (1 + z_obs)
                        freq_idx = np.searchsorted(freq_edges, nu, side="right") - 1
                        valid = (freq_idx >= 0) & (freq_idx < nfreq)
                        pix = hp.ang2pix(nside, np.degrees(lon), np.degrees(lat), lonlat=True)
                        pix_sub = pix_map[pix]
                        valid_sub = valid & (pix_sub == 1.)

                        for k in range(nfreq):
                            mask = valid_sub & (freq_idx == k)
                            if not np.any(mask):
                                continue
                            mass_cube_local[k] += np.bincount(
                                pix_sub[mask], weights=mass_tiled[mask], minlength=npix_sub # type: ignore
                            )
                    logger.info(f"processed particle chunk {j}/{end_par}") # type: ignore

        # ---- Process Galaxy Data ----
        if pos_key_gal in f:
            dset_pos_gal = f[pos_key_gal]
            if vlos_key_gal in f:
                vlos_corr = True
                dset_vlos_gal = f[vlos_key_gal]
            else:
                vlos_corr = False
                dset_vlos_gal = np.zeros(dset_pos_gal.shape[0]) # type: ignore
            for i in range(start_gal, end_gal, chunk_size):
                j = min(i + chunk_size, end_gal)
                pos_chunk = dset_pos_gal[i:j] # type: ignore
                vlos_chunk = dset_vlos_gal[i:j] # type: ignore

                for pos_gal_tiled, vlos_chunk_tiled, _ in _tile_data_generator(
                    pos_chunk, vlos_chunk, np.zeros(pos_chunk.shape[0]), Lbox_mpc, tiling # type: ignore
                ):
                    lon, lat, D = box_local_to_radec(pos_gal_tiled, ra0_deg, dec0_deg, z0)
                    z_cos = distance_to_redshift(D)
                    z_obs = apply_rsd(z_cos, vlos_chunk_tiled) if vlos_corr else z_cos
                    nu = rest_freq / (1 + z_obs)
                    freq_idx = np.searchsorted(freq_edges, nu,  side="right") - 1
                    valid = (freq_idx >= 0) & (freq_idx < nfreq)
                    pix = hp.ang2pix(nside, np.degrees(lon), np.degrees(lat), lonlat=True)
                    pix_sub = pix_map[pix]
                    valid_sub = valid & (pix_sub == 1.)
                    gal_cat_local_flag.append(valid_sub)
                    gal_pos_new = np.column_stack([np.degrees(lon), np.degrees(lat), z_cos])
                    gal_cat_local.append(gal_pos_new)
                logger.info(f"processed galaxy chunk {j}/{end_gal}")

    # --- Write local results to a temporary file ---
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_file = os.path.join(tmp_dir, f"rank_{rank:04d}.h5")

    if len(gal_cat_local) > 0:
        gal_cat_local_arr = np.concatenate(gal_cat_local, axis=0).astype(np.float32)
    else:
        gal_cat_local_arr = np.zeros((0, 3), dtype=np.float32)
    if len(gal_cat_local_flag) > 0:
        gal_cat_local_flag_arr = np.concatenate(gal_cat_local_flag, axis=0).astype(np.bool)
    else:
        gal_cat_local_flag_arr = np.zeros((0,), dtype=np.bool)

    with h5py.File(tmp_file, "w") as tf:
        if not skip_par:
            tf.create_dataset("mass_cube", data=mass_cube_local, dtype="f4", compression="gzip")
        tf.create_dataset("galaxy_catalog", data=gal_cat_local_arr, dtype="f4", compression="gzip")
        tf.create_dataset("galaxy_catalog_flag", data=gal_cat_local_flag_arr, dtype="bool", compression="gzip")
        tf.create_dataset("pix_idx", data=pix_idx, dtype="i8")
        tf.create_dataset("freqs", data=freqs, dtype="f8")
        tf.attrs["nside"] = nside
        tf.attrs["z0"] = z0
        tf.attrs["field_center_deg"] = [ra0_deg, dec0_deg]
        tf.attrs["Lbox_mpc"] = Lbox_mpc
        tf.attrs["tiling"] = tiling
        tf.attrs["rank"] = rank
    
    logger.info(f"wrote temporary file: {tmp_file}")

    # Clean up memory
    del mass_cube_local
    del gal_cat_local
    del gal_cat_local_arr
    gc.collect()
    
    # return tmp_file

def _aggregate_and_finalize(rank, size, params):
    """
    On the root rank, aggregates all temporary files, combines the data,
    and writes the final output file.
    """
    (output_mass_filename, tmp_dir, unit_mass, Lbox_mpc, tiling, nside, z0,
    ra0_deg, dec0_deg, freqs, pix_idx, skip_par) = (
        params["output_mass_filename"], params["tmp_dir"], params["unit_mass"],
        params["Lbox_mpc"], params["tiling"], params["nside"], params["z0"],
        params["ra0_deg"], params["dec0_deg"], params["freqs"], params["pix_idx"],
        params["skip_par"]
    )

    npix_sub = len(pix_idx)
    nfreq = len(freqs)
    
    if rank == 0:
        logger.info("Aggregating temporary files...")

        if not skip_par:
            mass_cube_global = np.zeros((nfreq, npix_sub), dtype="f4")
        gal_list = []
        gal_flag_list = []
        
        # Read and merge data from each rank's temp file
        for r in range(size):
            tmp_file_r = os.path.join(tmp_dir, f"rank_{r:04d}.h5")
            if not os.path.exists(tmp_file_r):
                logger.info(f"Warning: missing temporary file for rank {r}: {tmp_file_r}")
                continue
            with h5py.File(tmp_file_r, "r") as tf:
                if not skip_par:
                    mass_cube_global += tf["mass_cube"][:] # type: ignore
                gal = tf["galaxy_catalog"][:] # type: ignore
                gal_flag = tf["galaxy_catalog_flag"][:] # type: ignore
                if len(gal) > 0: # type: ignore
                    gal_list.append(gal)
                    gal_flag_list.append(gal_flag)
            logger.info(f"merged rank {r} data.")

        # Concatenate galaxy catalog
        if len(gal_list) > 0:
            gal_catalog = np.concatenate(gal_list, axis=0).astype(np.float32)
        else:
            gal_catalog = np.zeros((0, 3), dtype=np.float32)
        if len(gal_flag_list) > 0:
            gal_catalog_flag = np.concatenate(gal_flag_list, axis=0).astype(np.bool)
        else:
            gal_catalog_flag = np.zeros((0,), dtype=np.bool)

        # Write final output file
        logger.info("Writing final output file...")
        with h5py.File(output_mass_filename, "w") as fout:
            if not skip_par:
                fout.create_dataset(
                    "mass_cube", data=mass_cube_global, dtype="f4", compression="gzip" # type: ignore
                )
            fout.create_dataset("freqs", data=freqs)
            fout.create_dataset("pix_idx", data=pix_idx)
            fout.create_dataset(
                "galaxy_catalog", data=gal_catalog, dtype="f4", compression="gzip"
            )
            fout.create_dataset(
                "galaxy_catalog_flag", data=gal_catalog_flag, dtype="bool", compression="gzip"
            )
            fout.attrs["nside"] = nside
            fout.attrs["field_center_deg"] = [ra0_deg, dec0_deg]
            fout.attrs["z0"] = z0
            fout.attrs["unit_mass"] = (str(default_unit_mass) if unit_mass is None else unit_mass)
            fout.attrs["unit_freqs"] = "MHz"
            fout.attrs["Lbox_mpc"] = Lbox_mpc
            fout.attrs["tiling"] = tiling

        logger.info(f"Done! Output file saved to: {output_mass_filename}")
        
        # Clean up temporary directory
        shutil.rmtree(tmp_dir)
        logger.info(f"Cleaned up temporary directory: {tmp_dir}")


def process_and_save_data_mpi():
    if rank == 0:
        logger.info(f"[MPI] Running with {size} ranks.")

    try:
        params = _read_and_broadcast_params(comm, rank)
    except Exception as e:
        logger.error(f"Error during parameter setup: {e}")
        comm.Abort(1)

    try:
        logger.info("Starting data processing...")
        _process_data_chunk(comm, rank, params)
    except Exception as e:
        logger.error(f"Error during data processing: {e}")
        comm.Abort(1)

    comm.Barrier()

    try:
        _aggregate_and_finalize(rank, size, params)
    except Exception as e:
        logger.error(f"Error during aggregation: {e}")
        comm.Abort(1)

    comm.Barrier()


if __name__ == "__main__":
    process_and_save_data_mpi()
