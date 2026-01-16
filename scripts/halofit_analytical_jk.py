import os

import h5py
import numpy as np
from mytools.data import read_h5
from mytools.utils import get_coord, get_mask_sector
from tqdm import tqdm

from crafts_stack.halo_fitting import halofit
from crafts_stack.halo_models import get_default_model
from crafts_stack.utils import get_filepath_format, load_stack_all_groups


def jackknife(sig_list, coord, model, fit_mask, weights):
    """Jackknife return mean_res and pixel-wise jk std."""
    n = len(sig_list)
    jk_residuals = []

    for i in tqdm(range(n)):
        subsig = [sig_list[j] for j in range(n) if j != i]
        stack = np.ma.mean(subsig, axis=0)

        data_list, _, _ = halofit(
            coord,
            stack,
            model=model,
            mask=fit_mask,
            weights=weights[fit_mask],
            print_model=False,
            info_fit=False,
        )
        residual = data_list[2]
        jk_residuals.append(residual)

    jk_residuals = np.array(jk_residuals)

    mean_res = jk_residuals.mean(axis=0)
    jk_var = (n - 1) * np.var(jk_residuals, axis=0, ddof=0)
    jk_std = np.sqrt(jk_var)

    return mean_res, jk_std


def main(h5_inputs, h5_output, h5_weight, h5_weight_key, unit_factor):
    coord = get_coord()
    model = get_default_model()

    # masks
    fit_mask = get_mask_sector([0.1, 5])
    halo_peak_mask = get_mask_sector([0.0, 0.1], [-np.pi / 2, np.pi / 2])

    # load weights
    weights = read_h5(h5_weight, h5_weight_key)

    with h5py.File(h5_output, "a") as f:
        for h5_input in h5_inputs:
            if not os.path.exists(h5_input):
                print(f"File not found: {h5_input}, skipping.")
                continue

            print(f"Processing {h5_input}...")

            # load data and stack, applying unit_factor
            grp_mean, grp_sig_list = load_stack_all_groups(
                h5_input, unit_factor=unit_factor
            )

            # apply halo_peak_mask to data
            grp_mean.mask = grp_mean.mask | halo_peak_mask
            for s in grp_sig_list:
                s.mask = s.mask | halo_peak_mask

            # direct fit on full stack
            data_list, fit_model, fit_info = halofit(
                coord,  # pyright: ignore[reportArgumentType]
                grp_mean,
                model=model,
                mask=fit_mask,
                weights=weights[fit_mask],
            )

            # jackknife
            mean_res, jk_std = jackknife(grp_sig_list, coord, model, fit_mask, weights)

            group_name = os.path.splitext(os.path.basename(h5_input))[0]

            g = f.create_group(group_name)
            g["stack"] = grp_mean
            g["fitted"] = data_list[1]
            g["residual"] = data_list[2]
            g["fit_mask"] = fit_mask
            g["peak_mask"] = halo_peak_mask
            g["param_cov"] = fit_info["param_cov"]
            g["jk_mean_residual"] = mean_res
            g["jk_std"] = jk_std


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    # Arguments for get_filepath_format
    parser.add_argument(
        "--base", default=os.getenv("BASE", "."), help="Base directory for input files"
    )
    parser.add_argument(
        "--pattern",
        default=os.getenv("PATTERN"),
        required=os.getenv("PATTERN") is None,
        help="Filename pattern for input files",
    )
    parser.add_argument(
        "--args",
        default=os.getenv("ARGS"),
        required=os.getenv("ARGS") is None,
        help="Comma-separated arguments for the pattern",
    )

    # Other arguments
    parser.add_argument(
        "--weight",
        default=os.getenv("WEIGHT"),
        required=os.getenv("WEIGHT") is None,
    )
    parser.add_argument(
        "--weight_key",
        default=os.getenv("WEIGHT_KEY"),
        required=os.getenv("WEIGHT_KEY") is None,
    )
    parser.add_argument(
        "--output",
        default=os.getenv("OUTPUT"),
        required=os.getenv("OUTPUT") is None,
    )
    parser.add_argument(
        "--unit_factor",
        type=float,
        default=float(os.getenv("UNIT_FACTOR", 1.0)),
        help="Multiply signal by this factor (e.g., convert to microK)",
    )

    args = parser.parse_args()

    # Generate file list
    h5_inputs = get_filepath_format(args.base, args.pattern, *args.args.split(","))

    main(h5_inputs, args.output, args.weight, args.weight_key, args.unit_factor)
