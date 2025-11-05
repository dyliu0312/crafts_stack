"""
Utility functions for fitting 2D halo models to data.
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from astropy.modeling import fitting, models
from astropy.utils.exceptions import AstropyUserWarning
from mytools.halo_new import get_coord, info_fitness

from crafts_stack.halo_models import get_model


def build_model(
    model_name: str = "nfw",
    init_h1: Optional[Dict[str, float]] = None,
    init_h2: Optional[Dict[str, float]] = None,
    init_const: Optional[Dict[str, float]] = None,
    bounds: Optional[Dict[str, Tuple[Union[float, None], Union[float, None]]]] = None,
    constraints: Optional[Dict[str, Union[List[str], List[Tuple[str, str]]]]] = None,
):
    """
    Construct a compound model with two halo profiles and a constant background.

    Parameters
    ----------
    model_name : str, optional
        Name of the halo model to use, by default "nfw"
    init_h1, init_h2 : dict, optional
        Initial parameters for the first halo profile, the second halo profile, and the constant background, by default None to use defaults based on model_name. The defaults are:
        - nfw: {"amplitude": 50, "r_s": 0.5, "ellipticity": 0, "theta": 0}
        - gaussian: {"amplitude": 50, "x_stddev": 0.5, "y_stddev": 0.5, "theta": 0}
        - lorentz: {"amplitude": 50, "fwhm": 1.0, "ellipticity": 0, "theta": 0}
        - other: {"amplitude": 50}
    init_const : dict, optional
        Initial parameters for the constant background, by default None to use default {"amplitude": 0}
    bounds : dict, optional
        Bounds for parameters in format {param_name: (min, max)}. By default None
    constraints : dict, optional
        Parameter constraints for the model, by default None. Currently supports:
        - "fixed": list of parameter names to fix
        - "tied": list of tuples (target_param, source_param) to tie parameters

    Returns
    -------
    astropy.modeling.CompoundModel
        Compound model with two halo profiles and a constant background
    """
    model_class = get_model(model_name)

    # Set default initial parameters based on model type
    if model_name == "nfw":
        default_init = {"amplitude": 50, "r_s": 0.5, "ellipticity": 0, "theta": 0}
    elif model_name == "gaussian":
        default_init = {"amplitude": 50, "x_stddev": 0.5, "y_stddev": 0.5, "theta": 0}
    elif model_name == "lorentz":
        default_init = {"amplitude": 50, "fwhm": 1.0, "ellipticity": 0, "theta": 0}
    else:
        default_init = {"amplitude": 50}

    h1_init = default_init.copy()
    h1_init.update({"x_mean": -1, "y_mean": 0})
    if init_h1 is not None:
        h1_init.update(init_h1)  # pyright: ignore[reportArgumentType, reportCallIssue]

    h2_init = default_init.copy()
    h2_init.update({"x_mean": 1, "y_mean": 0})
    if init_h2 is not None:
        h2_init.update(init_h2)  # pyright: ignore[reportArgumentType, reportCallIssue]

    init_const = init_const or {"amplitude": 0}

    # create independent submodels
    h1 = model_class(**h1_init)
    h2 = model_class(**h2_init)
    const = models.Const2D(**init_const)

    model = h1 + h2 + const  # pyright: ignore[reportOperatorIssue]

    # Apply parameter bounds (if provided)
    if bounds:
        for name, bound in bounds.items():
            if hasattr(model, name):
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
    model_name: str = "nfw",
    x: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
    kw_h1: Optional[Dict[str, float]] = None,
    kw_h2: Optional[Dict[str, float]] = None,
    kw_const: Optional[Dict[str, float]] = None,
    noise_std: Optional[float] = None,
    seed: float = 42,
) -> np.ndarray:
    """
    Generate test data based on a specified model.

    Parameters
    ----------
    model_name : str, optional
        Name of the model to use for generating the test data. Default is "nfw".
    x, y : np.ndarray, optional
        Coordinates of the data points. If not provided, they will be generated using `get_coord()`.
    kw_h1, kw_h2, kw_const : dict, optional
        Keyword arguments for the model. Default is None to use defaults based on model_name.
    noise_std : float, optional
        Standard deviation of the Gaussian noise to be added to the data. If None, no noise is added.
    seed : int, optional
        Seed for the random number generator. Default is 42.

    Returns
    -------
    data : np.ndarray
        The generated test data.
    """
    if x is None and y is None:
        x, y = get_coord()
    elif x is None or y is None:
        y = x if y is None else y
        x = y if x is None else x

    model = build_model(model_name, kw_h1, kw_h2, kw_const)
    data = model(x, y)

    np.random.seed(seed)
    if noise_std is not None:
        data += np.random.normal(0, noise_std, data.shape)

    return data


def halofit(
    *args: np.ndarray,
    model: Optional[Any] = None,
    model_name: Optional[str] = "nfw",
    mask: Optional[np.ndarray] = None,
    print_model: bool = True,
) -> Tuple[List[np.ndarray], Any]:
    """
    Fit 2D data using a provided or auto-generated model.

    Parameters
    ----------
    args : tuple
        Either (data), ((x, y), data), or (x, y, data).
    model : astropy.modeling.models.Model, optional
        Model to use for fitting. If not provided, a default model will be generated based on the model_name.
    model_name : str, optional
        Name of the model to use for fitting. Default is "nfw".
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
        if model_name is None:
            raise ValueError("Either 'model' or 'model_name' must be provided.")
        model = build_model(model_name)

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
        if model_name:
            print(f"\nFitted double {model_name.upper()} model:")
        else:
            print("\nFitted model:")
        print(fit_model)

    # Compute fitted data and residuals
    fit_data = fit_model(x, y)
    res = info_fitness(data, fit_data)

    return [data, fit_data, res], fitter.fit_info  # pyright: ignore[reportReturnType]
