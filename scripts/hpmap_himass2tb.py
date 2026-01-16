"""
A script to convert the HI mass HEALPIX map into a brightness temperature one.
"""

import gc
import os
from multiprocessing import Pool, cpu_count
from typing import Optional, Sequence, Tuple, Union

import h5py
import healpy as hp
import numpy as np
from astropy import units as u
from astropy.cosmology import Planck18 as cosmo
from mytools.constant import A_HI, C_BOL, C_PLK, HI_MASS, HI_REST_FREQ, LIGHT_SPEED
from tqdm import tqdm

# ------------------ Constants ------------------
u_mhz = u.Unit("MHz")
u_s = u.Unit("s")
u_mk = u.Unit("mK")
u_mass = 1e10 * u.Unit("Msun")

nu_HI = HI_REST_FREQ.to(u_mhz)


# ------------------ Functions ------------------
def compute_temp_channel(args) -> Tuple[int, np.ndarray]:
    """Compute brightness temperature for one channel (for Pool.map)."""
    i, mass_slice, z, input_mass_unit, output_temp_unit, omega_pix_sr, delta_nu = args
    D_L = cosmo.luminosity_distance(z)  # pyright: ignore[reportAttributeAccessIssue]
    numerator = (1 + z) ** 3 * 3 * LIGHT_SPEED**2 * C_PLK * A_HI
    denominator = (
        32 * np.pi * C_BOL * HI_MASS * nu_HI * D_L**2 * omega_pix_sr * delta_nu
    )
    tb_k = (numerator / denominator) * (mass_slice * input_mass_unit)
    return i, tb_k.to_value(output_temp_unit)


def convert_mass_cube_to_brightness_temp(
    mass_cube: np.ndarray,
    freqs: Sequence[float],
    nside: int,
    input_mass_unit: Union[str, u.Unit] = u_mass,
    output_temp_unit: Union[str, u.Unit] = u_mk,
    nworker: Optional[int] = None,
    delta_nu: Optional[Union[float, str, u.Quantity]] = None,
) -> np.ndarray:
    """Convert entire cube using multiprocessing."""
    if isinstance(input_mass_unit, str):
        input_mass_unit = u.Unit(input_mass_unit)
    if isinstance(output_temp_unit, str):
        output_temp_unit = u.Unit(output_temp_unit)

    omega_pix_sr = hp.nside2pixarea(nside, degrees=False)
    z_channels = nu_HI / (freqs * u_mhz) - 1

    if delta_nu is None:
        delta_nu = freqs[1] - freqs[0]
    elif isinstance(delta_nu, str):
        delta_nu = float(delta_nu)

    if not isinstance(delta_nu, u.Quantity):
        delta_nu = delta_nu * u_mhz

    nfreq = len(freqs)
    temp_cube = np.zeros_like(mass_cube, dtype="f4")

    tasks = [
        (
            i,
            mass_cube[i],
            z_channels[i],
            input_mass_unit,
            output_temp_unit,
            omega_pix_sr,
            delta_nu,
        )
        for i in range(nfreq)
    ]

    if nworker is None:
        nworker = min(cpu_count(), nfreq)

    with Pool(processes=nworker) as pool:
        # Use tqdm to show progress
        results = list(
            tqdm(
                pool.imap(compute_temp_channel, tasks),
                total=nfreq,
                desc="Converting mass to temperature",
            )
        )

    for i, tb in results:
        temp_cube[i] = tb

    return temp_cube


def smooth_one_channel(args) -> Tuple[int, np.ndarray]:
    """Apply beam smoothing for one channel."""
    i, temp_slice, nside, pix_idx, beam_fwhm_arcmin = args
    fwhm_rad = np.radians(float(beam_fwhm_arcmin) / 60.0)
    npix = hp.nside2npix(nside)

    full_map = np.full(npix, hp.UNSEEN, dtype="f4")
    full_map[pix_idx] = temp_slice
    smoothed_full = hp.smoothing(full_map, fwhm=fwhm_rad)
    return i, smoothed_full[pix_idx]


def apply_beam_smoothing(
    temp_cube: np.ndarray,
    nside: int,
    pix_idx: np.ndarray,
    beam_fwhm_arcmin: str,
    nworker: Optional[int] = None,
) -> np.ndarray:
    """Apply smoothing with multiprocessing."""
    nfreq = temp_cube.shape[0]
    convolved_temp_cube = np.zeros_like(temp_cube, dtype="f4")

    tasks = [(i, temp_cube[i], nside, pix_idx, beam_fwhm_arcmin) for i in range(nfreq)]

    if nworker is None:
        nworker = min(cpu_count(), nfreq)

    with Pool(processes=nworker) as pool:
        # Use tqdm to show progress
        results = list(
            tqdm(
                pool.imap(smooth_one_channel, tasks),
                total=nfreq,
                desc="Applying beam smoothing",
            )
        )

    for i, smoothed in results:
        convolved_temp_cube[i] = smoothed

    return convolved_temp_cube


# ------------------ Main ------------------
if __name__ == "__main__":
    # --- Read parameters from env vars ---
    input_masss_filename = os.getenv("INPUT_MASS_FILE", "output_mass_cube.h5")
    output_temp_filename = os.getenv("OUTPUT_TEMP_FILE", "final_temp_cube.h5")
    beam_fwhm_arcmin = os.getenv("BEAM_FWHM_ARCMIN", "5")
    input_mass_unit = os.getenv("UNIT_MASS")
    output_temp_unit = os.getenv("UNIT_TEMP", "mK")
    nworker = os.getenv("NWORKER", None)
    delta_nu = os.getenv("DELTA_NU", None)
    smooth_only = os.getenv("SMOOTH_ONLY", "False").lower() == "true"
    key_dset = os.getenv("KEY_DSET", "nside,pix_idx,mass_cube,temp,temp_conv,freqs")
    key_nside, key_pix, key_mass, key_t, key_tc, key_freq = key_dset.split(",")

    if nworker is not None:
        nworker = int(nworker)

    if not smooth_only:
        if not os.path.exists(input_masss_filename):
            raise FileNotFoundError(f"Input file not found: {input_masss_filename}")

        print(f"Loading mass cube from {input_masss_filename}...")
        with h5py.File(input_masss_filename, "r") as f:
            mass_cube = f[key_mass][:]  # pyright: ignore[reportIndexIssue]
            pix_idx = f[key_pix][:]  # pyright: ignore[reportIndexIssue]
            freqs = f[key_freq][:]  # pyright: ignore[reportIndexIssue]

            input_attrs = dict(f.attrs.items())

            try:
                nside = f[key_nside][()]  # pyright: ignore[reportIndexIssue]
            except KeyError:
                print("nside not found in input file, trying to get it from attributes...")
                nside = f.attrs.get(key_nside, None)
            if nside is None:
                raise ValueError("nside not found in input file")

            if input_mass_unit is None:
                input_mass_unit = f.attrs.get("unit_mass", u_mass)

        print("Converting mass cube to brightness temperature...")
        temp_cube = convert_mass_cube_to_brightness_temp(
            mass_cube,  # pyright: ignore[reportArgumentType]
            freqs,  # pyright: ignore[reportArgumentType]
            nside,  # pyright: ignore[reportArgumentType]
            input_mass_unit,
            output_temp_unit,
            nworker,
            delta_nu,
        )

        print(f"Saving final cube to {output_temp_filename}...")
        with h5py.File(output_temp_filename, "w") as fout:
            for key, value in input_attrs.items():
                fout.attrs[key] = value

            fout.create_dataset(key_t, data=temp_cube, dtype="f4", compression="gzip")
            fout.create_dataset(key_freq, data=freqs)
            fout.create_dataset(key_pix, data=pix_idx)

            fout.attrs["unit_temp"] = str(output_temp_unit)

        del mass_cube, freqs
        gc.collect()

        if beam_fwhm_arcmin is not None:
            print(f"Applying beam smoothing with FWHM = {beam_fwhm_arcmin} arcmin...")
            final_cube = apply_beam_smoothing(
                temp_cube,
                nside,  # pyright: ignore[reportArgumentType]
                pix_idx,  # pyright: ignore[reportArgumentType]
                beam_fwhm_arcmin,
                nworker,
            )

            with h5py.File(output_temp_filename, "a") as fout:
                fout.create_dataset(
                    key_tc, data=final_cube, dtype="f4", compression="gzip"
                )
                fout.attrs["beam_fwhm_arcmin"] = float(beam_fwhm_arcmin)

            print("Post-processing complete!")

    else:
        if not os.path.exists(output_temp_filename):
            raise FileNotFoundError(f"Input file not found: {output_temp_filename}")

        if beam_fwhm_arcmin is None:
            raise ValueError(
                "BEAM_FWHM_ARCMIN must be provided for smoothing only mode."
            )

        print(f"Loading temp cube from {output_temp_filename}...")
        with h5py.File(output_temp_filename, "r") as f:
            pix_idx = f[key_pix][:]  # pyright: ignore[reportIndexIssue]
            temp_cube = f[key_t][:]  # pyright: ignore[reportIndexIssue]
            nside = f.attrs[key_nside]
        print(f"Applying beam smoothing with FWHM = {beam_fwhm_arcmin} arcmin...")
        final_cube = apply_beam_smoothing(
            temp_cube,  # pyright: ignore[reportArgumentType]
            nside,  # pyright: ignore[reportArgumentType]
            pix_idx,  # pyright: ignore[reportArgumentType]
            beam_fwhm_arcmin,
            nworker,
        )

        with h5py.File(output_temp_filename, "a") as fout:
            fout.create_dataset(key_tc, data=final_cube, dtype="f4", compression="gzip")
            fout.attrs["beam_fwhm_arcmin"] = float(beam_fwhm_arcmin)

    print("Post-processing complete!")
