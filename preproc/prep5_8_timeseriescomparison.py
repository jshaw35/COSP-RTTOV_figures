"""
File: AIRS_CESM2_spectralcomparison.py
Author: Jonah Shaw

This script contains functions for plotting spectral timeseries data from AIRS and CESM2. The functions are used to plot the spectral timeseries data for a given tile, wavenumber, scene type, and branch type. The functions are also used to plot a 2x2 grid of spectral timeseries plots for different scene and branch types. The functions are also used to plot the average spectra for a given tile, scene type, and branch type.
The functions are also used to plot the average spectra for a given tile, scene type, and branch type for each month of the year.

"""
# %%
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import os
import pandas as pd

month_dict = {
    1: "January",
    2: "February",
    3: "March",
    4: "April",
    5: "May",
    6: "June",
    7: "July",
    8: "August",
    9: "September",
    10: "October",
    11: "November",
    12: "December",
}


def load_timeseries_data(
    tile_str: str,
    wnum: float,
    scene_type: str,
    branch_type: str,
    i_wnum: int = None,
):

    processed_airs_path = f"/glade/campaign/univ/ucuc0007/AIRS_data/{tile_str}/proc/{tile_str}_tseries_deg_0_5.nc"
    processed_cesm2_path = f"/glade/campaign/univ/ucuc0007/AIRS_data/{tile_str}/proc/CESM2_nudged_swathed_{tile_str}.nc"

    if scene_type not in ["total", "clear"]:
        raise ValueError(f"Invalid scene type: {scene_type}. Must be 'total' or 'clear'.")

    if branch_type == "asc":
        airs_var = f"rad_{scene_type}_{branch_type}"
        cesm2_var = f"rttov_rad_{scene_type}_inst001"
    elif branch_type == "des":
        airs_var = f"rad_{scene_type}_{branch_type}"    
        cesm2_var = f"rttov_rad_{scene_type}_inst002"
    else:
        raise ValueError(f"Invalid branch type: {branch_type}. Must be 'asc' or 'des'.")

    if not os.path.exists(processed_airs_path):
        raise FileNotFoundError(f"{processed_airs_path} does not exist.")
    if not os.path.exists(processed_cesm2_path):
        raise FileNotFoundError(f"{processed_cesm2_path} does not exist.")

    airs_ds = xr.open_dataset(processed_airs_path)[airs_var]
    cesm2_ds = xr.open_dataset(processed_cesm2_path)[cesm2_var]

    if i_wnum is None:
        airs_tsubset = airs_ds.sel(wnum=wnum, time=slice("2002-10-01", "2021-12-31"))
        cesm2_tsubset = cesm2_ds.sel(wnum=wnum, time=slice("2002-10-01", "2021-12-31"))
    else:
        airs_tsubset = airs_ds.sel(time=slice("2002-10-01", "2021-12-31")).isel(wnum=i_wnum)
        cesm2_tsubset = cesm2_ds.sel(time=slice("2002-10-01", "2021-12-31")).isel(wnum=i_wnum)

    # Make time axises the same
    cesm2_tsubset["time"] = airs_tsubset["time"]

    return airs_tsubset, cesm2_tsubset


def spectral_timeseries_plot(
    tile_str: str,
    wnum: float,
    scene_type: str,
    branch_type: str,
    i_wnum: int = None,
    ax=None,
):
    """
    Plots the spectral timeseries for AIRS and CESM2 data.

    Parameters:
        tile_str (str): The tile string indicating the geographical region.
        wnum (float): The wavenumber to select.
        scene_type (str): The scene type, either 'total' or 'clear'.
        branch_type (str): The branch type, either 'asc' or 'des'.
        i_wnum (int, optional): The index of the wavenumber to select. Defaults to None.

    Raises:
        ValueError: If an invalid scene type or branch type is provided.
        FileNotFoundError: If the processed AIRS or CESM2 file does not exist.

    Returns:
        None
    """

    processed_airs_path = f"/glade/campaign/univ/ucuc0007/AIRS_data/{tile_str}/proc/{tile_str}_tseries_deg_0_5.nc"
    processed_cesm2_path = f"/glade/campaign/univ/ucuc0007/AIRS_data/{tile_str}/proc/CESM2_nudged_swathed_{tile_str}.nc"

    if scene_type not in ["total", "clear"]:
        raise ValueError(f"Invalid scene type: {scene_type}. Must be 'total' or 'clear'.")

    if branch_type == "asc":
        airs_var = f"rad_{scene_type}_{branch_type}"
        cesm2_var = f"rttov_rad_{scene_type}_inst001"
    elif branch_type == "des":
        airs_var = f"rad_{scene_type}_{branch_type}"    
        cesm2_var = f"rttov_rad_{scene_type}_inst002"
    else:
        raise ValueError(f"Invalid branch type: {branch_type}. Must be 'asc' or 'des'.")

    if not os.path.exists(processed_airs_path):
        raise FileNotFoundError(f"{processed_airs_path} does not exist.")
    if not os.path.exists(processed_cesm2_path):
        raise FileNotFoundError(f"{processed_cesm2_path} does not exist.")

    airs_ds = xr.open_dataset(processed_airs_path)
    cesm2_ds = xr.open_dataset(processed_cesm2_path)

    if i_wnum is None:
        airs_tsubset = airs_ds.sel(wnum=wnum, time=slice("2002-10-01", "2021-12-31"))
        cesm2_tsubset = cesm2_ds.sel(wnum=wnum, time=slice("2002-10-01", "2021-12-31"))
    else:
        airs_tsubset = airs_ds.sel(time=slice("2002-10-01", "2021-12-31")).isel(wnum=i_wnum)
        cesm2_tsubset = cesm2_ds.sel(time=slice("2002-10-01", "2021-12-31")).isel(wnum=i_wnum)

    # Make time axises the same
    cesm2_tsubset["time"] = airs_tsubset["time"]
    # Plot
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    # ax = axs[0]
    ax.plot(
        airs_tsubset["time"],
        airs_tsubset[airs_var],
        label=f"AIRS {scene_type} {branch_type}",
    )
    ax.plot(
        cesm2_tsubset["time"],
        cesm2_tsubset[cesm2_var],
        label=f"CESM2 {scene_type} {branch_type}",
    )
    ax.set_xlabel("Year")
    ax.set_ylabel("Radiance (mW/m^2/sr/cm^-1)")
    ax.set_title(f"{scene_type}-sky {str(np.round(airs_tsubset.wnum.values, 2))} cm^-1")
    ax.legend()


def spectral_timeseries_4plot(
    tile_str: str,
    wnum: float,
    i_wnum: int = None,
):
    """
    Plots a 2x2 grid of spectral timeseries plots for different scene and branch types.

    Parameters:
        tile_str (str): The tile identifier string.
        wnum (float): The wavenumber for the spectral timeseries.
        i_wnum (int, optional): The index of the wavenumber. Defaults to None.

    Returns:
        None
    """

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs = axs.flat

    for i, scene_type in enumerate(["clear", "total"]):
        for j, branch_type in enumerate(["asc", "des"]):
            spectral_timeseries_plot(
                tile_str,
                wnum,
                scene_type,
                branch_type,
                i_wnum=i_wnum,
                ax=axs[i*2 + j],
            )

    plt.tight_layout()

    return fig, axs


def spectral_summary_plot(
    tile_str: str,
    scene_type: str,
    branch_type: str,
    ax=None,
    ufunc=None,
):

    processed_airs_path = f"/glade/campaign/univ/ucuc0007/AIRS_data/{tile_str}/proc/{tile_str}_tseries_deg_0_5.nc"
    processed_cesm2_path = f"/glade/campaign/univ/ucuc0007/AIRS_data/{tile_str}/proc/CESM2_nudged_swathed_{tile_str}.nc"

    if scene_type not in ["total", "clear"]:
        raise ValueError(f"Invalid scene type: {scene_type}. Must be 'total' or 'clear'.")

    if branch_type == "asc":
        airs_var = f"rad_{scene_type}_{branch_type}"
        cesm2_var = f"rttov_rad_{scene_type}_inst001"
    elif branch_type == "des":
        airs_var = f"rad_{scene_type}_{branch_type}"    
        cesm2_var = f"rttov_rad_{scene_type}_inst002"
    else:
        raise ValueError(f"Invalid branch type: {branch_type}. Must be 'asc' or 'des'.")

    if not os.path.exists(processed_airs_path):
        raise FileNotFoundError(f"{processed_airs_path} does not exist.")
    if not os.path.exists(processed_cesm2_path):
        raise FileNotFoundError(f"{processed_cesm2_path} does not exist.")

    airs_ds = xr.open_dataset(processed_airs_path)
    cesm2_ds = xr.open_dataset(processed_cesm2_path)

    airs_tsubset = airs_ds.sel(time=slice("2002-10-01", "2021-12-31"))
    cesm2_tsubset = cesm2_ds.sel(time=slice("2002-10-01", "2021-12-31"))
    # Make time axises the same
    cesm2_tsubset["time"] = airs_tsubset["time"]

    # Apply a function to the data
    if ufunc is not None:
        airs_tsubset = ufunc(airs_tsubset)
        cesm2_tsubset = ufunc(cesm2_tsubset)

    # Plot the average spectra
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    ax.plot(
        airs_tsubset["wnum"],
        airs_tsubset[airs_var].mean("time"),
        label=f"AIRS {scene_type} {branch_type}",
    )
    ax.plot(
        cesm2_tsubset["wnum"],
        cesm2_tsubset[cesm2_var].mean("time"),
        label=f"CESM2 {scene_type} {branch_type}",
    )
    ax.set_xlabel("Wavenumber (cm^-1)")
    ax.set_ylabel("Radiance (mW/m^2/sr/cm^-1)")
    ax.set_title(f"{scene_type}-sky {branch_type}")
    ax.legend()


def sel_month(ds, month):
    return ds.sel(time=ds["time.month"] == month)


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


# %%

# Produce average values for the window and stratosphere channels by averaging over scene types.
scene_types = ["clear", "total"]
branch_types = ["asc", "des"]

data_dict_tropical = {}
wnum_i_list = [1, 6, 23] # 1 is the stratosphere channel, 6 is upper-troposphere, 23 is the window channel
scene_type_selector = ["clear", "clear", "clear"]
tile_strs = ["N00p00_E160p00", "N44p00_W100p00"]

tile_dict = {}
for tile_str in tile_strs:
    wnum_dict = {}
    for wnum_i, scene_str in zip(wnum_i_list, scene_type_selector):
        data_dict = {}
        for branch_type in branch_types:
            airs_ds, cesm2_ds = load_timeseries_data(
                tile_str,
                1231.3276,
                scene_str,
                branch_type,
                i_wnum=wnum_i,
            )
            data_dict[f"airs_clear_{branch_type}"] = airs_ds
            data_dict[f"cesm2_clear_{branch_type}"] = cesm2_ds

        # Create the cloud fraction proxy directly from data_dict
        airs_clear_avg = 0.5 * (
            data_dict["airs_clear_asc"] +
            data_dict["airs_clear_des"]
        )

        cesm2_clear_avg = 0.5 * (
            data_dict["cesm2_clear_asc"] +
            data_dict["cesm2_clear_des"]
        )
        airs_clear_avg.to_netcdf(f"./data_dir/{tile_str}_{str(wnum_i)}_AIRS_clear.nc")
        cesm2_clear_avg.to_netcdf(f"./data_dir/{tile_str}_{str(wnum_i)}_CESM2_clear.nc")
        wnum_dict[str(wnum_i)] = [airs_clear_avg, cesm2_clear_avg]
    tile_dict[tile_str] = wnum_dict

# Recreate the simple cloud fraction estimate from the window channel
scene_types = ["clear", "total"]
branch_types = ["asc", "des"]

data_dict_tropical = {}
tile_strs = ["N00p00_E160p00", "N44p00_W100p00"]

tile_cloud_dict = {}
for tile_str in tile_strs:
    data_dict = {}
    for scene_type in scene_types:
        for branch_type in branch_types:
            airs_ds, cesm2_ds = load_timeseries_data(
                tile_str,
                1231.3276,
                scene_type,
                branch_type,
                i_wnum=23,
            )
            data_dict[f"airs_{scene_type}_{branch_type}"] = airs_ds
            data_dict[f"cesm2_{scene_type}_{branch_type}"] = cesm2_ds

    # Create the cloud fraction proxy directly from data_dict
    airs_clear_minus_total = 0.5 * (
        data_dict["airs_clear_asc"] +
        data_dict["airs_clear_des"] -
        data_dict["airs_total_asc"] -
        data_dict["airs_total_des"]
    )

    cesm2_clear_minus_total = 0.5 * (
        data_dict["cesm2_clear_asc"] +
        data_dict["cesm2_clear_des"] -
        data_dict["cesm2_total_asc"] -
        data_dict["cesm2_total_des"]
    )
    airs_clear_minus_total.to_netcdf(f"./data_dir/{tile_str}_AIRS_cloud.nc")
    cesm2_clear_minus_total.to_netcdf(f"./data_dir/{tile_str}_CESM2_cloud.nc")
    tile_cloud_dict[tile_str] = {
        "airs": airs_clear_minus_total,
        "cesm2": cesm2_clear_minus_total,
    }

# %%