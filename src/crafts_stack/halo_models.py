"""
Analytical models for the halo fitting.

This module contains the definitions of the analytical models, implemented as classes that inherit from `astropy.modeling.Fittable2DModel`.
Currently included signal models are:
- `NFW2D`: Elliptical 2D NFW profile.
- `Lorentz2D`: Elliptical 2D Lorentzian profile.
- `Gaussian2D`: 2D Gaussian profile.

Background models are:
- `Const2D`: Constant background model.
- `Poly2D`: Polynomial background model
"""

from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
from astropy.modeling import Fittable2DModel, Parameter, models


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

        # Dimensionless radius, ensure it\'s not exactly zero
        x_val = np.maximum(r / r_s, 1e-8)

        # 3D NFW density formula
        density = amplitude / (x_val * (1 + x_val) ** 2)

        return density


class Lorentz2D(Fittable2DModel):
    r"""
    Elliptical 2D Lorentzian profile.
    """

    amplitude = Parameter(default=1.0, min=0)
    x_mean = Parameter(default=0.0)
    y_mean = Parameter(default=0.0)
    fwhm = Parameter(default=1.0, min=0.1)  # Full width at half maximum
    ellipticity = Parameter(default=0.0, min=0, max=0.9)
    theta = Parameter(default=0.0)

    @staticmethod
    def evaluate(x, y, amplitude, x_mean, y_mean, fwhm, ellipticity, theta):
        """
        Evaluate the Lorentzian profile on 2D xy grid
        """
        q = 1.0 - ellipticity
        q = np.maximum(q, 1e-8)

        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        x_c = x - x_mean
        y_c = y - y_mean

        x_rot = x_c * cos_theta + y_c * sin_theta
        y_rot = -x_c * sin_theta + y_c * cos_theta

        r_sq = x_rot**2 + (y_rot / q) ** 2

        # gamma is HWHM
        gamma = fwhm / 2.0

        density = amplitude / (1 + r_sq / gamma**2)

        return density


# Dictionary of available models
AVAILABLE_MODELS = {
    "nfw": NFW2D,
    "gaussian": models.Gaussian2D,
    "lorentz": Lorentz2D,
}


def get_model(model_name: str):
    """
    Get a halo model class by its name.
    """
    model_name = model_name.lower()
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(
            f"Model '{model_name}' not available. "
            f"Choose from {list(AVAILABLE_MODELS.keys())}"
        )
    return AVAILABLE_MODELS[model_name]


# Dictionary of available background models
AVAILABLE_BACKGROUNDS = {
    "const": models.Const2D,
    "poly": models.Polynomial2D,
}


def get_background_model(model_name: str):
    """
    Get a background model class by its name.
    """
    model_name = model_name.lower()
    if model_name not in AVAILABLE_BACKGROUNDS:
        raise ValueError(
            f"Background model '{model_name}' not available. "
            f"Choose from {list(AVAILABLE_BACKGROUNDS.keys())}"
        )
    return AVAILABLE_BACKGROUNDS[model_name]


def build_model(
    halo_model: Literal["nfw", "gaussian", "lorentz"] = "nfw",
    bg_model: Literal["const", "poly", None] = None,
    param_h1: Optional[Dict[str, float]] = None,
    param_h2: Optional[Dict[str, float]] = None,
    param_bg: Optional[Dict[str, Any]] = None,
):
    """
    Construct a compound model with two halo profiles and a background.

    Parameters
    ----------
    halo_model : str, optional
        Name of the halo model to use, by default "nfw"
    bg_model : str, optional
        Name of the background model to use, by default "const"
    param_h1, param_h2 : dict, optional
        Parameters for the first halo profile, the second halo profile. The defaults are:
        - nfw: {"amplitude": 50, "r_s": 0.5, "ellipticity": 0, "theta": 0}
        - gaussian: {"amplitude": 50, "x_stddev": 0.5, "y_stddev": 0.5, "theta": 0}
        - lorentz: {"amplitude": 50, "fwhm": 1.0, "ellipticity": 0, "theta": 0}
    param_bg : dict, optional
        Parameters for the background model. The defaults are:
        - const: {"amplitude": 0}
        - poly: {"degree": 2}

    Returns
    -------
    astropy.modeling.CompoundModel
        Compound model with two halo profiles (and background)
    """
    model_class = get_model(halo_model)

    # Set default initial parameters based on model type
    if halo_model == "nfw":
        default_init = {"amplitude": 50, "r_s": 0.5, "ellipticity": 0, "theta": 0}
    elif halo_model == "gaussian":
        default_init = {"amplitude": 50, "x_stddev": 0.5, "y_stddev": 0.5, "theta": 0}
    elif halo_model == "lorentz":
        default_init = {"amplitude": 50, "fwhm": 1.0, "ellipticity": 0, "theta": 0}

    h1_init = default_init.copy()
    h1_init.update({"x_mean": -1, "y_mean": 0})
    if param_h1 is not None:
        h1_init.update(param_h1)  # pyright: ignore[reportArgumentType, reportCallIssue]

    h2_init = default_init.copy()
    h2_init.update({"x_mean": 1, "y_mean": 0})
    if param_h2 is not None:
        h2_init.update(param_h2)  # pyright: ignore[reportArgumentType, reportCallIssue]

    # create models of halos
    h1 = model_class(**h1_init)
    h2 = model_class(**h2_init)

    model = h1 + h2

    # background term
    if bg_model is not None:
        background_class = get_background_model(bg_model)
        if param_bg is None:
            if bg_model == "const":
                param_bg = {"amplitude": 0}
            elif bg_model == "poly":
                param_bg = {"degree": 1}

        bg = background_class(**param_bg)  # pyright: ignore[reportCallIssue]

        model += bg

    return model


def apply_prior(
    model:Fittable2DModel,
    bounds: Optional[Dict[str, Tuple[Union[float, None], Union[float, None]]]] = None,
    constraints: Optional[Dict[str, Union[List[str], List[Tuple[str, str]]]]] = None,
)-> Fittable2DModel:
    """
    Apply parameter bounds and constraints to a model.

    Parameters
    ----------
    model : astropy.modeling.Model
        Model to apply bounds and constraints to    
    bounds : dict, optional
        Bounds for parameters in format {param_name: (min, max)}. By default None
    constraints : dict, optional
        Parameter constraints for the model, by default None. Currently supports:
        - "fixed": list of parameter names to fix
        - "tied": list of tuples (target_param, source_param) to tie parameters
    
    Returns
    -------
    astropy.modeling.Model
    """
    # Apply parameter bounds (if provided)
    if bounds:
        for name, bound in bounds.items():
            if hasattr(model, name):
                getattr(model, name).bounds = bound
    # Apply parameter constraints (if provided)
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
