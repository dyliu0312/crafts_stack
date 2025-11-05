import warnings
import numpy as np
from astropy.modeling import fitting, models
from astropy.utils.exceptions import AstropyUserWarning
from mytools.halo_new import get_coord


def build_model(init_h1=None, init_h2=None, init_const=None, bounds=None, fix_theta=0):
    """Construct a compound model with two 2D Gaussians and a constant background.

    Parameters
    ----------
    init_h1, init_h2, init_const : dict, optional
        Initialization dictionaries for the two Gaussians and the constant.
    bounds : dict, optional
        Bounds for parameters in format {param_name: (min, max)}.
    fix_theta : float, optional
        Fix the theta parameter of both Gaussians to this value. If None, theta is not fixed. Default is fixed to 0.
    """

    init_h1 = init_h1 or {
        "amplitude": 50,
        "x_mean": -1,
        "y_mean": 0,
        "x_stddev": 0.5,
        "y_stddev": 0.5,
    }
    init_h2 = init_h2 or {
        "amplitude": 50,
        "x_mean": 1,
        "y_mean": 0,
        "x_stddev": 0.5,
        "y_stddev": 0.5,
    }
    init_const = init_const or {"amplitude": 0}

    # create independent submodels
    h1 = models.Gaussian2D(**init_h1)
    h2 = models.Gaussian2D(**init_h2)
    const = models.Const2D(**init_const)

    # Fix theta to zero if requested
    if fix_theta is not None:
        h1.theta.fixed = True
        h2.theta.fixed = True
        h1.theta.value = fix_theta
        h2.theta.value = fix_theta

    # Combine submodels into a single model
    model = h1 + h2 + const  # pyright: ignore[reportOperatorIssue]

    # Apply parameter bounds (if provided)
    if bounds:
        for name, bound in bounds.items():
            getattr(model, name).bounds = bound

    return model


def gen_test_data(
    x=None, y=None, kw_h1=None, kw_h2=None, kw_const=None, noise_std=None, seed=42
):
    """Generate test data composed of two 2D Gaussian profiles plus a constant background."""

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
    """Fit 2D data using a provided or auto-generated double Gaussian model with constant background."""

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
        print("\nFitted double Gaussian model:")
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
        x_stddev=0.51,
        y_mean=0,
        y_stddev=0.5,
    )
    data = gen_test_data(kw_h1=kw_h1)
    bounds = {"amplitude": (0, None), "x_stddev": (0, None), "y_stddev": (0, None)}
    compound_model = build_model(bounds=bounds)

    data_list, fit_info = halofit(data, model=compound_model)
    plot_stack_fit_res(data_list)
    plt.show()
