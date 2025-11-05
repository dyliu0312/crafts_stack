from typing import Any, List, Optional, Tuple, Union, Mapping
import warnings
import numpy as np
from astropy.modeling import fitting, models
from astropy.utils.exceptions import AstropyUserWarning
from mytools.halo_new import get_coord, info_fitness


def build_model(
    init_h1: Optional[Mapping[str, float]] = None,
    init_h2: Optional[Mapping[str, float]] = None,
    init_const: Optional[Mapping[str, float]] = None,
    bounds: Optional[
        Mapping[str, Tuple[Union[float, None], Union[float, None]]]
    ] = None,
    constraints: Optional[Mapping[str, Union[List[str], List[Tuple[str, str]]]]] = None,
):
    """
    Construct a compound model with two 2D Gaussians and a constant background.

    Parameters
    ----------
    init_h1, init_h2, init_const : dict, optional
        Initialization dictionaries for the two Gaussians and the constant. Defaults parameters are:
        - init_h1: {"amplitude": 50, "x_mean": -1, "y_mean": 0, "x_stddev": 0.5, "y_stddev": 0.5}
        - init_h2: {"amplitude": 50, "x_mean": 1, "y_mean": 0, "x_stddev": 0.5, "y_stddev": 0.5}
        - init_const: {"amplitude": 0}
    bounds : dict, optional
        Bounds for parameters in format {param_name: (min, max)}.
    fix_theta : float, optional
        Fix the theta parameter of both Gaussians to this value. If None, theta is not fixed. Default is fixed to 0.

    Returns
    -------
    model : CompoundModel
        The constructed compound model.
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

    # Combine submodels into a single model
    model = h1 + h2 + const  # pyright: ignore[reportOperatorIssue]

    # Apply parameter bounds (if provided)
    if bounds:
        for name, bound in bounds.items():
            getattr(model, name).bounds = bound

    if constraints:
        if "fixed" in constraints:
            for param_name in constraints["fixed"]:
                if hasattr(model, param_name):  # pyright: ignore[reportArgumentType]
                    getattr(model, param_name).fixed = True  # pyright: ignore[reportArgumentType]
        if "tied" in constraints:
            for target_param, source_param in constraints["tied"]:
                if hasattr(model, target_param) and hasattr(model, source_param):

                    def make_tie_func(source):
                        return lambda m: getattr(m, source)

                    getattr(model, target_param).tied = make_tie_func(source_param)

    return model


def gen_test_data(
    x: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
    kw_h1: Optional[Mapping[str, float]] = None,
    kw_h2: Optional[Mapping[str, float]] = None,
    kw_const: Optional[Mapping[str, float]] = None,
    noise_std: Optional[float] = None,
    seed: float = 42,
) -> np.ndarray:
    """
    Generate test data composed of two 2D Gaussian profiles plus a constant background.

    Parameters
    ----------
    x, y : np.ndarray, optional
        Arrays of x and y coordinates. If not provided, they will be generated using `get_coord()`.
    kw_h1, kw_h2, kw_const : dict, optional
        Keyword arguments for the two Gaussians and the constant, respectively. Defaults parameters are:
        - kw_h1: {"amplitude": 50, "x_mean": -1, "y_mean": 0, "x_stddev": 0.5, "y_stddev": 0.5}
        - kw_h2: {"amplitude": 50, "x_mean": 1, "y_mean": 0, "x_stddev": 0.5, "y_stddev": 0.5}
        - kw_const: {"amplitude": 0}
    noise_std : float, optional
        Standard deviation of the Gaussian noise to add to the data. Default is None, in which case no noise is added.
    seed : int, optional
        Seed for the random number generator. Default is 42.
    """

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
    if noise_std is not None:
        data += np.random.normal(0, noise_std, data.shape)

    return data


def halofit(
    *args: np.ndarray,
    model=None,
    mask: Optional[np.ndarray] = None,
    print_model: bool = True,
) -> Tuple[List[np.ndarray], Any]:
    """
    Fit 2D data using a provided or auto-generated double Gaussian model with constant background.

    Parameters
    ----------
    args : tuple
        Either (data), ((x, y), data), or (x, y, data).
    model : astropy.modeling.models.Model, optional
        Model to use for fitting. If not provided, a default model will be generated.
    mask : np.ndarray, optional
        Mask to apply to the data before fitting.
    print_model : bool, optional
        Whether to print the fitted model. Default is True.

    Returns
    -------
    Tuple[List[np.ndarray], Any]
    """

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
    res = info_fitness(data, fit_data)

    return [data, fit_data, res], fitter.fit_info  # pyright: ignore[reportReturnType]


if __name__ == "__main__":
    from mytools.plot import plot_stack_fit_res, plt

    kw_h1 = dict(
        amplitude=60,
        x_mean=-1,
        x_stddev=0.51,
        y_mean=0,
        y_stddev=0.5,
    )
    nstd = 10
    data = gen_test_data(kw_h1=kw_h1, noise_std=nstd)
    bounds = {
        "amplitude_0": (0.0, None),
        "x_stddev_0": (0.0, None),
        "y_stddev_0": (0.0, None),
        "amplitude_1": (0.0, None),
        "x_stddev_1": (0.0, None),
        "y_stddev_1": (0.0, None),
    }
    compound_model = build_model(bounds=bounds)

    data_list, fit_info = halofit(data, model=compound_model)
    plot_stack_fit_res(data_list)
    plt.show()
