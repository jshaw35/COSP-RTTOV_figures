"""
This script is used to preprocess the CESM2 data for comparison with AIRS L1C tile data.
"""
# %%

import numpy as np
import xarray as xr
import os
import glob
import scipy.io
from scipy.stats import pearsonr

wnum_data = "/glade/u/home/jonahshaw/Scripts/git_repos/COSP-RTTOV_diageval/SCAM_validation/sarta_333profiles_vmr.mat"

# AIRS L1C channel indices used in COSP-RTTOV
l1c_chan_indices = np.array([
    41,
    54,
    72,
    202,
    234,
    294,
    336,
    378,
    485,
    489,
    572,
    794,
    960,
    961,
    967,
    1055,
    1075,
    1089,
    1110,
    1130,
    1158,
    1325,
    1511,
    1520,
    1697,
    1723,
    1805,
    1852,
    1862,
    1866,
    1937,
    2088,
    2138,
    2164,
    2165,
    2166,
    2176,
    2186,
    2383,
    2411,
    2588,
    2592,
    2600,
    2606,
    2620,
])


def preprocess_coords(ds):
    for coord in ds.coords:
        if "RTTOV_CHAN_I00" in coord:
            ds = ds.rename({coord: "wnum"})
    return ds


def load_CESM2(
    case_2000_2014: str,
    case_2015_2021: str,
    var: str,
):
    """
    Flexibly preprocess arbitrary CESM2 output variables to match AIRS tile regions and compute weighted spatial averages.

    Parameters:
        case_str (str): The string identifier for the CESM2 case.
        case_2000_2014 (str): The string identifier for the CESM2 case from 2000 to 2014.
        case_2015_2021 (str): The string identifier for the CESM2 case from 2015 to 2021.
        var (str): The variable name to process.

    Returns:
        xarray.Dataset: The weighted spatial average of the CESM2 data for the specified AIRS tile region.
    """
    case_paths = [case_2000_2014, case_2015_2021]

    # Collect filea paths
    file_paths = []
    for _case in case_paths:
        file_paths += glob.glob(f"/glade/campaign/univ/ucuc0007/COSP_RTTOV_output/{_case}/atm/proc/tseries/*h0.{var}.*.nc")

    # Open files
    cesm2_data = xr.open_mfdataset(file_paths)
    cesm2_data["time"] = xr.CFTimeIndex(cesm2_data["time"].values).shift(-1, "MS").shift(14, "1D") # Shift to the 15th of each month

    return cesm2_data


def load_CESM2_RTTOV_AIRS(
    case_2000_2014: str,
    case_2015_2021: str,
):
    """
    Flexibly preprocess arbitrary CESM2 output variables to match AIRS tile regions and compute weighted spatial averages.

    Parameters:
        case_str (str): The string identifier for the CESM2 case.
        case_2000_2014 (str): The string identifier for the CESM2 case from 2000 to 2014.
        case_2015_2021 (str): The string identifier for the CESM2 case from 2015 to 2021.
        var (str): The variable name to process.

    Returns:
        xarray.Dataset: The weighted spatial average of the CESM2 data for the specified AIRS tile region.
    """
    case_paths = [case_2000_2014, case_2015_2021]

    wnum_ds = scipy.io.loadmat(wnum_data)

    # Use monthly data where AIRS has better statistics
    var_dict = {
        "total": "rttov_rad_total_inst",
        "clear": "rttov_rad_clear_inst",
        "total_bt": "rttov_bt_total_inst",
        "clear_bt": "rttov_bt_clear_inst",
    }

    # Collect file paths
    file_paths = []
    for _scene_type in var_dict:
        _var = var_dict[_scene_type]
        for _case in case_paths:
            file_paths += glob.glob(f"/glade/campaign/univ/ucuc0007/COSP_RTTOV_output/{_case}/atm/proc/tseries/*h0.{_var}*.nc")

    # Open while renaming spectral coordinate to "wnum". Wouldn't work with multiple instruments or channel selections.
    cesm2_data = xr.open_mfdataset(file_paths, preprocess=preprocess_coords)
    cesm2_data["time"] = xr.CFTimeIndex(cesm2_data["time"].values).shift(-1, "MS").shift(14, "1D") # Shift to the 15th of each month
    cesm2_data["wnum"] = xr.DataArray(wnum_ds["fx"][:, 0], dims="wnum").isel(wnum=l1c_chan_indices - 1)

    return cesm2_data


def to_png(file, filename, loc='/glade/u/home/jonahshaw/figures/', dpi=200, ext='png', **kwargs):
    '''
    Simple function for one-line saving.
    Saves to "/glade/u/home/jonahshaw/figures" by default
    '''
    output_dir = loc
    full_path = '%s%s.%s' % (output_dir, filename, ext)

    if not os.path.exists(output_dir + filename):
        file.savefig(
            full_path,
            format=ext,
            dpi=dpi,
            **kwargs,
        )


def m(x, w):
    """Weighted Mean"""
    return np.sum(x * w) / np.sum(w)


def cov(x, y, w):
    """Weighted Covariance"""
    return np.sum(w * (x - m(x, w)) * (y - m(y, w))) / np.sum(w)


def corr(x, y, w):
    """Weighted Correlation"""
    return cov(x, y, w) / np.sqrt(cov(x, x, w) * cov(y, y, w))


def correlate_data(
    dataA: xr.DataArray,
    dataB: xr.DataArray,
):
    """
    Calculate the weighted spatial correlation coefficient between two datasets.

    Parameters:
        dataA (xr.DataArray): The first dataset.
        dataB (xr.DataArray): The second dataset.
        weights (xr.DataArray): The weights to apply to the correlation calculation.

    Returns:
        float: The weighted spatial correlation coefficient.
    """
    # Flatten the data arrays and remove NaNs
    weights = np.cos(np.deg2rad(dataA["lat"]))
    dataA_flat = dataA.values.flatten()
    dataB_flat = dataB.values.flatten()
    weights_flat = weights.broadcast_like(dataA).values.flatten()
    mask = ~np.isnan(dataA_flat) & ~np.isnan(dataB_flat) & (weights_flat > 0)

    # Calculate the Pearson correlation coefficient
    correlation_coefficient, _ = pearsonr(dataA_flat[mask], dataB_flat[mask])
    correlation_coefficient_weighted = corr(dataA_flat[mask], dataB_flat[mask], weights_flat[mask])

    print(f"Spatial correlation coefficient: {correlation_coefficient}")
    print(f"Weighted spatial correlation coefficient: {correlation_coefficient_weighted}")

    return correlation_coefficient, correlation_coefficient_weighted


# %%

if __name__ == "__main__":

    # Load model data
    case_str = "nudged_swathed"
    case_2000_2014 = "20241023_100107.FHIST.f09_f09_mg17.cesm2.1.5_port"
    case_2015_2021 = "20241028_100834.FHIST.f09_f09_mg17.cesm2.1.5_port_SSPbranch"

    cesm2_rttov_outs = load_CESM2_RTTOV_AIRS(
        case_2000_2014,
        case_2015_2021,
    ).isel(time=0)
    
    cesm2_rttov_outs.to_netcdf("./data_dir/fig4_cesm2_rttov_outs.nc")

    cesm2_vars = ["T", "TS", "CLDTOT"]
    cesm2_extra_vars = ["hyam", "hybm", "P0", "PS"]
    cesm2_out_dict = {}
    for i,_var in enumerate(cesm2_vars):
        print(_var)
        cesm2_out_var = load_CESM2(
            case_2000_2014,
            case_2015_2021,
            _var,
        )
        for i in cesm2_extra_vars:
            if i in cesm2_out_var.data_vars:
                cesm2_out_dict[i] = cesm2_out_var[i]
                cesm2_extra_vars.remove(i)

        cesm2_out_dict[_var] = cesm2_out_var[_var]
        cesm2_out_dict[_var].name = _var
    cesm_out_ds = xr.merge(cesm2_out_dict.values()).isel(time=0)
    cesm_out_ds.to_netcdf("./data_dir/fig4_cesm2_outs.nc")

# %%
