import warnings

import numpy as np
from astropy.modeling import Fittable2DModel, Parameter, fitting, models
from astropy.utils.exceptions import AstropyUserWarning
from mytools.halo_new import get_coord


class NFW2D(Fittable2DModel):
    r"""
    Elliptical 2D NFW profile.

    $$\rho(r) = \frac{\rho_0}{\frac{r}{R_s} \left( 1 + \frac{r}{R_s} \right)^2}$$
    """

    amplitude = Parameter(default=1.0, min=0)  # ρ₀ - characteristic density
    x_mean = Parameter(default=0.0)  # x center position
    y_mean = Parameter(default=0.0)  # y center position
    r_s = Parameter(default=1.0, min=0.1)  # scale radius
    ellipticity = Parameter(default=0.0, min=0, max=0.9)  # 1 - b/a
    theta = Parameter(default=0.0)  # position angle in radians

    @staticmethod
    def evaluate(x, y, amplitude, x_mean, y_mean, r_s, ellipticity, theta):
        """
        Evaluate the NFW density profile on 2D xy grid
        """
        q = 1.0 - ellipticity
        # Add a small epsilon to avoid division by zero for q=0 (ellipticity=1)
        q = np.maximum(q, 1e-8)

        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        x_c = x - x_mean
        y_c = y - y_mean

        x_rot = x_c * cos_theta + y_c * sin_theta
        y_rot = -x_c * sin_theta + y_c * cos_theta

        r = np.sqrt(x_rot**2 + (y_rot / q) ** 2)

        # Dimensionless radius, ensure it's not exactly zero
        x_val = np.maximum(r / r_s, 1e-8)

        # 3D NFW density formula
        density = amplitude / (x_val * (1 + x_val) ** 2)

        return density


def build_model(
    init_h1=None,
    init_h2=None,
    init_const=None,
    bounds=None,
    constraints=None,
):
    """Construct a compound model with two 2D NFW profiles and a constant background."""

    init_h1 = init_h1 or {
        "amplitude": 50,
        "x_mean": -1,
        "y_mean": 0,
        "r_s": 0.5,
        "ellipticity": 0.0,
        "theta": 0.0,
    }
    init_h2 = init_h2 or {
        "amplitude": 50,
        "x_mean": 1,
        "y_mean": 0,
        "r_s": 0.5,
        "ellipticity": 0.0,
        "theta": 0.0,
    }
    init_const = init_const or {"amplitude": 0}

    # create independent submodels
    h1 = NFW2D(**init_h1)
    h2 = NFW2D(**init_h2)
    const = models.Const2D(**init_const)

    # Apply parameter bounds (if provided)
    if bounds:
        for name, (low, high) in bounds.items():
            for sub in (h1, h2):
                if hasattr(sub, name):
                    getattr(sub, name).bounds = (low, high)

    model = h1 + h2 + const

    if constraints:
        if "fixed" in constraints:
            for param_name in constraints["fixed"]:
                if hasattr(model, param_name):
                    getattr(model, param_name).fixed = True
        if "tied" in constraints:
            for target_param, source_param in constraints["tied"]:
                if hasattr(model, target_param) and hasattr(model, source_param):

                    def make_tie_func(source):
                        return lambda m: getattr(m, source)

                    getattr(model, target_param).tied = make_tie_func(source_param)

    return model


def gen_test_data(
    x=None,
    y=None,
    kw_h1=None,
    kw_h2=None,
    kw_const=None,
    noise_std=None,
    seed=42,
):
    """Generate test data composed of two 2D NFW profiles plus a constant background."""

    # Generate coordinates if not provided
    if x is None and y is None:
        x, y = get_coord()
    elif x is None or y is None:
        y = x if y is None else y
        x = y if x is None else x

    model = build_model(kw_h1, kw_h2, kw_const)
    data = model(x, y)

    # Combine models and add Gaussian noise
    np.random.seed(seed)
    noise_std = noise_std or 0.1 * np.max(data)
    data_noisy = data + np.random.normal(0, noise_std, data.shape)

    return data_noisy


def halofit(*args, model=None, mask=None, print_model=True):
    """Fit 2D data using a provided or auto-generated double NFW model with constant background."""

    # Parse input arguments
    if len(args) == 1:
        data = args[0]
        x, y = get_coord()
    elif len(args) == 2:
        (x, y), data = args
    elif len(args) == 3:
        x, y, data = args
    else:
        raise ValueError(
            "Invalid number of arguments. Use (data), ((x, y), data), or (x, y, data)."
        )

    # Build model if not provided
    if model is None:
        model = build_model()

    # Apply mask if available
    if mask is not None:
        x_fit, y_fit, data_fit = x[mask], y[mask], data[mask]
    else:
        x_fit, y_fit, data_fit = x, y, data

    # Perform fitting
    fitter = fitting.TRFLSQFitter()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=AstropyUserWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        fit_model = fitter(model, x_fit, y_fit, data_fit)

    if print_model:
        print("\nFitted double NFW model:")
        print(fit_model)

    # Compute fitted data and residuals
    fit_data = fit_model(x, y)
    residuals = data - fit_data
    chi2 = np.sum(residuals**2)
    rms = np.sqrt(np.mean(residuals**2))

    print("\nGoodness of fit:")
    print(f"Chi-squared: {chi2:.3f}")
    print(f"RMS residual: {rms:.3f}")

    return [data, fit_data, residuals], fitter.fit_info


if __name__ == "__main__":
    from mytools.plot import plot_stack_fit_res, plt

    kw_h1 = dict(
        amplitude=60,
        x_mean=-1,
        r_s=0.51,
        y_mean=0,
        ellipticity=0.2,
        theta=np.pi / 4,
    )
    data = gen_test_data(kw_h1=kw_h1)
    bounds = {"amplitude": (0, None), "r_s": (0, None), "ellipticity": (0, 0.9)}
    compound_model = build_model(bounds=bounds)

    data_list, fit_info = halofit(data, model=compound_model)
    plot_stack_fit_res(data_list)
    plt.show()
