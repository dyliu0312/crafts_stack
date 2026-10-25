## [unreleased]

### 💼 Other

- Imporve type hint
- Merge gaussian and nfw model, add lorentz model, provide uniform halo_fitting
- Imporve func docs
- Add ploly2d background model
- Change poly2d to two polynomial2d composed model
- Imporve func type hints
- Remove poly2d, refine structure
- Add error map generation func, cov_to_error_map
- Improve docstring
- Rm test halofit
- Add decompose model to get contributions of submodels
- Remove ellipticity param of NFW model, and set default model NFW+Gaussian+Const
- Add bootstrap error estimation
- Constraint parameters of gaussian model due to symmetry, and refine lorentz support
- Remove fitting error estimation, add params' error based on params_cov
- Perform halo fitting with jackknife std estimation
- Add pair stack slurm job scritps
- Remove shuffle given it's lack of randomness
- Add tng stack results for pcaloss correction
- Add numerical halofit as comparison
- Remove upper bounds of the amplitudes of the NFW model
- Add read_h5_data_with_mask to handle halofit_jk.py output
- Add multiple hpfunc kwargs support
- Ruff format
- Add example results of TNG with analytical halo model
- Update module name
