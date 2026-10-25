import argparse

# import sys
# import os
import numpy as np

# # Add project root to path
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from crafts_stack.halo_fitting import build_model, gen_test_data, halofit
from hifigps.plot import plot_stack_fit_res, plt


def main():
    parser = argparse.ArgumentParser(description="Test halo model fitting.")
    parser.add_argument(
        "model_name",
        type=str,
        choices=["gaussian", "nfw", "lorentz"],
        help="Name of the halo model to test.",
    )
    parser.add_argument("nstd", type=float, help="The std of noise level.")
    args = parser.parse_args()

    model_name = args.model_name
    nstd = args.nstd

    model_configs = {
        "gaussian": {
            "kw_h1": dict(
                amplitude=60,
                x_mean=-1,
                x_stddev=0.51,
                y_mean=0,
                y_stddev=0.5,
            ),
            "bounds": {
                "amplitude_0": (0.0, None),
                "x_stddev_0": (0.0, None),
                "y_stddev_0": (0.0, None),
                "amplitude_1": (0.0, None),
                "x_stddev_1": (0.0, None),
                "y_stddev_1": (0.0, None),
            },
            "constraints": {},
        },
        "nfw": {
            "kw_h1": dict(
                amplitude=60,
                x_mean=-1,
                r_s=0.51,
                y_mean=0,
                ellipticity=0.2,
                theta=np.pi / 4,
            ),
            "bounds": {
                "amplitude_0": (0, None),
                "r_s_0": (0, None),
                "ellipticity_0": (0, 0.9),
                "amplitude_1": (0, None),
                "r_s_1": (0, None),
                "ellipticity_1": (0, 0.9),
            },
            "constraints": {
                "fixed": [
                    "x_mean_0",
                    "y_mean_0",
                    "x_mean_1",
                    "y_mean_1",
                ],
                "tied": [("amplitude_1", "amplitude_0")],
            },
        },
        "lorentz": {
            "kw_h1": dict(
                amplitude=60,
                x_mean=-1,
                fwhm=1.0,
                y_mean=0,
                ellipticity=0.2,
                theta=np.pi / 4,
            ),
            "bounds": {
                "amplitude_0": (0, None),
                "fwhm_0": (0, None),
                "ellipticity_0": (0, 0.9),
                "amplitude_1": (0, None),
                "fwhm_1": (0, None),
                "ellipticity_1": (0, 0.9),
            },
            "constraints": {
                "fixed": [
                    "x_mean_0",
                    "y_mean_0",
                    "x_mean_1",
                    "y_mean_1",
                ],
                "tied": [("amplitude_1", "amplitude_0")],
            },
        },
    }

    config = model_configs[model_name]
    kw_h1 = config["kw_h1"]
    bounds = config["bounds"]
    constraints = config["constraints"]

    print(f"--- Testing {model_name.upper()} model ---")

    data = gen_test_data(model_name, kw_h1=kw_h1, noise_std=nstd)

    compound_model = build_model(model_name, bounds=bounds, constraints=constraints)

    data_list, fit_info = halofit(data, model=compound_model, model_name=model_name)

    plot_stack_fit_res(data_list)
    plt.show()


if __name__ == "__main__":
    main()
