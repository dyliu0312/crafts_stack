from typing import Any, List, Mapping, Optional, Tuple, Union
import warnings

import numpy as np
from astropy.modeling import Fittable2DModel, Parameter, fitting, models
from astropy.utils.exceptions import AstropyUserWarning
from mytools.halo_new import get_coord, info_fitness


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
    init_h1: Optional[Mapping[str, float]] = None,
    init_h2: Optional[Mapping[str, float]] = None,
    init_const: Optional[Mapping[str, float]] = None,
    bounds: Optional[
        Mapping[str, Tuple[Union[float, None], Union[float, None]]]
    ] = None,
    constraints: Optional[Mapping[str, Union[List[str], List[Tuple[str, str]]]]] = None,
):
    """
    Construct a compound model with two 2D NFW profiles and a constant background.

    Parameters
    ----------
    init_h1, init_h2, init_const : dict, optional
        Initialization dictionaries for the NFW profiles and the constant. Defaults parameters are:
        - init_h1: {"amplitude": 50, "x_mean": -1, "y_mean": 0, "r_s": 0.5, "ellipticity": 0.0, "theta": 0.0}
        - init_h2: {"amplitude": 50, "x_mean": 1, "y_mean": 0, "r_s": 0.5, "ellipticity": 0.0, "theta": 0.0}
        - init_const: {"amplitude": 0}
    bounds : dict, optional
        Bounds for parameters in format {param_name: (min, max)}.
    constraints: dict, optional
        Parameter constraints for the model. Currently supports:
        - "fixed": list of parameter names to fix
        - "tied": list of tuples (target_param, source_param) to tie parameters

    Returns:
    --------
    model: CompoundModel
        The constructed compound model.
    """

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

    model = h1 + h2 + const  # pyright: ignore[reportOperatorIssue]

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
    Generate test data composed of two 2D NFW profiles plus a constant background.

    Parameters
    ----------
    x, y : np.ndarray, optional
        Coordinates of the data points. If not provided, they will be generated using `get_coord()`.
    kw_h1, kw_h2, kw_const : dict, optional
        Keyword arguments for the NFW profiles and the constant. Defaults are:
        - kw_h1: {"amplitude": 50, "x_mean": -1, "y_mean": 0, "r_s": 0.5, "ellipticity": 0.0, "theta": 0.0}
        - kw_h2: {"amplitude": 50, "x_mean": 1, "y_mean": 0, "r_s": 0.5, "ellipticity": 0.0, "theta": 0.0}
        - kw_const: {"amplitude": 0}
    noise_std : float, optional
        Standard deviation of the Gaussian noise to be added to the data. If None, no noise is added.
    seed : int, optional
        Seed for the random number generator. Default is 42.

    Returns
    -------
    data : np.ndarray
        The generated test data.
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
    Fit 2D data using a provided or auto-generated double NFW model with constant background.

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
        print("\nFitted double NFW model:")
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
        r_s=0.51,
        y_mean=0,
        ellipticity=0.2,
        theta=np.pi / 4,
    )
    nstd = 10
    data = gen_test_data(kw_h1=kw_h1, noise_std=nstd)
    bounds = {"amplitude": (0, None), "r_s": (0, None), "ellipticity": (0, 0.9)}
    constraints = {
        "fixed": [
            "x_mean_0",
            "y_mean_0",
            "x_mean_1",
            "y_mean_1",
        ],  # fix the center of the two NFW profiles
        "tied": [
            ("amplitude_1", "amplitude_0")  # amplitude_1 = amplitude_0
        ],
    }
    compound_model = build_model(bounds=bounds, constraints=constraints)

    data_list, fit_info = halofit(data, model=compound_model)
    plot_stack_fit_res(data_list)
    plt.show()
