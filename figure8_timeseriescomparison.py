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


def spectral_timeseries_plot(
    tile_str: str,
    wnum: float,
    scene_type: str,
    branch_type: str,
    processed_airs_path: str = None,
    processed_cesm2_path: str = None,
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

    if processed_airs_path is None:
        processed_airs_path = f"/glade/campaign/univ/ucuc0007/AIRS_data/{tile_str}/proc/{tile_str}_tseries_deg_0_5.nc"
    if processed_cesm2_path is None:
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


def spectral_annualtimeseries_plot(
    tile_str: str,
    wnum: float,
    scene_type: str,
    branch_type: str,
    processed_airs_path: str = None,
    processed_cesm2_path: str = None,
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

    if processed_airs_path is None:
        processed_airs_path = f"/glade/campaign/univ/ucuc0007/AIRS_data/{tile_str}/proc/{tile_str}_tseries_deg_0_5.nc"
    if processed_cesm2_path is None:
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
        
    airs_tsubset = airs_tsubset.groupby("time.month").mean("time")
    cesm2_tsubset = cesm2_tsubset.groupby("time.month").mean("time")

    # Make time axises the same
    cesm2_tsubset["month"] = airs_tsubset["month"]
    # Plot
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    ax.plot(
        airs_tsubset["month"],
        airs_tsubset[airs_var],
        label=f"AIRS {scene_type} {branch_type}",
    )
    ax.plot(
        cesm2_tsubset["month"],
        cesm2_tsubset[cesm2_var],
        label=f"CESM2 {scene_type} {branch_type}",
    )
    ax.set_xlabel("Month")
    ax.set_ylabel("Radiance (mW/m^2/sr/cm^-1)")
    ax.set_title(f"{scene_type}-sky {str(np.round(airs_tsubset.wnum.values, 2))} cm^-1")
    ax.legend()


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

# Create simple figure showing the use of swathing by satellite orbit branch. 
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
axs = axs.flat

tile_str = "N44p00_W100p00"
processed_airs_path = f"./data_dir/{tile_str}_tseries_deg_0_5.nc"
processed_cesm2_path = f"./data_dir/CESM2_nudged_swathed_{tile_str}.nc"
ax=axs[0]
spectral_timeseries_plot(
    tile_str,
    1231.3276,
    "clear",
    "asc",
    processed_airs_path=processed_airs_path,
    processed_cesm2_path=processed_cesm2_path,
    ax=ax,
)
ax.set_title("Daytime Orbits", fontsize=15)

ax=axs[1]
spectral_timeseries_plot(
    tile_str,
    1231.3276,
    "clear",
    "des",
    processed_airs_path=processed_airs_path,
    processed_cesm2_path=processed_cesm2_path,
    ax=ax,
)
ax.set_title("Nighttime Orbits", fontsize=15)

labels = ['a', 'b']
for ax, label in zip(axs, labels):
    ax.legend()
    ax.set_xlabel("Year", fontsize=15)
    ax.set_ylabel("Radiance (mW/m$^2$/sr/cm$^{-1}$)", fontsize=15)
    ax.set_ylim(20, 85)
    ax.tick_params(axis='both', labelsize=14)
    ax.text(0.02, 0.98, f'{label}.', transform=ax.transAxes, fontsize=14, verticalalignment='top', weight='bold')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, ["CESM2", "AIRS"], fontsize=14)
axs[0].get_legend().remove()

to_png(fig, "midlat_window_daynight", dpi=200, ext="pdf", bbox_inches='tight')

# %%

# Create simple figure showing the use of swathing by satellite orbit branch. 
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
axs = axs.flat

tile_str = "N44p00_W100p00"
processed_airs_path = f"./data_dir/{tile_str}_tseries_deg_0_5.nc"
processed_cesm2_path = f"./data_dir/CESM2_nudged_swathed_{tile_str}.nc"
ax=axs[0]
spectral_annualtimeseries_plot(
    tile_str,
    1231.3276,
    "clear",
    "asc",
    processed_airs_path=processed_airs_path,
    processed_cesm2_path=processed_cesm2_path,
    ax=ax,
)
ax.set_title("Daytime Orbits", fontsize=15)

ax=axs[1]
spectral_annualtimeseries_plot(
    tile_str,
    1231.3276,
    "clear",
    "des",
    processed_airs_path=processed_airs_path,
    processed_cesm2_path=processed_cesm2_path,
    ax=ax,
)
ax.set_title("Nighttime Orbits", fontsize=15)

labels = ['a', 'b']
for ax, label in zip(axs, labels):
    ax.legend()
    ax.set_xlabel("Month", fontsize=15)
    ax.set_ylabel("Radiance (mW/m$^2$/sr/cm$^{-1}$)", fontsize=15)
    ax.set_ylim(20, 85)
    ax.set_xlim(1, 12)
    ax.set_xticks(np.arange(1, 13, 1))
    ax.tick_params(axis='both', labelsize=14)
    ax.text(0.02, 0.98, f'{label}.', transform=ax.transAxes, fontsize=14, verticalalignment='top', weight='bold')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, ["CESM2", "AIRS"], fontsize=14)
axs[0].get_legend().remove()

to_png(fig, "midlat_window_daynight_annual", dpi=200, ext="pdf", bbox_inches='tight')

# %%

# Combined plot
fig, axs = plt.subplots(2, 2, figsize=(12, 12))
axs = axs.flat

tile_str = "N44p00_W100p00"
processed_airs_path = f"./data_dir/{tile_str}_tseries_deg_0_5.nc"
processed_cesm2_path = f"./data_dir/CESM2_nudged_swathed_{tile_str}.nc"
ax=axs[0]
spectral_timeseries_plot(
    tile_str,
    1231.3276,
    "clear",
    "asc",
    processed_airs_path=processed_airs_path,
    processed_cesm2_path=processed_cesm2_path,
    ax=ax,
)
ax.set_title("Daytime Orbits", fontsize=15)

ax=axs[1]
spectral_timeseries_plot(
    tile_str,
    1231.3276,
    "clear",
    "des",
    processed_airs_path=processed_airs_path,
    processed_cesm2_path=processed_cesm2_path,
    ax=ax,
)
ax.set_title("Nighttime Orbits", fontsize=15)

labels = ['a', 'b']
for ax, label in zip(axs, labels):
    ax.legend()
    ax.set_xlabel("Year", fontsize=15)
    ax.set_ylabel("Radiance (mW/m$^2$/sr/cm$^{-1}$)", fontsize=15)
    ax.set_ylim(20, 85)
    ax.tick_params(axis='both', labelsize=14)
    ax.text(0.02, 0.98, f'{label}.', transform=ax.transAxes, fontsize=14, verticalalignment='top', weight='bold')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, ["CESM2", "AIRS"], fontsize=14)

ax=axs[2]
spectral_annualtimeseries_plot(
    tile_str,
    1231.3276,
    "clear",
    "asc",
    processed_airs_path=processed_airs_path,
    processed_cesm2_path=processed_cesm2_path,
    ax=ax,
)
ax.set_title("Daytime Orbits", fontsize=15)

ax=axs[3]
spectral_annualtimeseries_plot(
    tile_str,
    1231.3276,
    "clear",
    "des",
    processed_airs_path=processed_airs_path,
    processed_cesm2_path=processed_cesm2_path,
    ax=ax,
)
ax.set_title("Nighttime Orbits", fontsize=15)

labels = ['c', 'd']
for ax, label in zip(axs[2:], labels):
    ax.legend()
    ax.set_xlabel("Month", fontsize=15)
    ax.set_ylabel("Radiance (mW/m$^2$/sr/cm$^{-1}$)", fontsize=15)
    ax.set_ylim(20, 85)
    ax.set_xlim(1, 12)
    ax.set_xticks(np.arange(1, 13, 1))
    ax.tick_params(axis='both', labelsize=14)
    ax.text(0.02, 0.98, f'{label}.', transform=ax.transAxes, fontsize=14, verticalalignment='top', weight='bold')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, ["CESM2", "AIRS"], fontsize=14)

axs[0].get_legend().remove()
axs[2].get_legend().remove()
axs[3].get_legend().remove()

to_png(fig, "midlat_window_daynight_both", dpi=200, ext="pdf", bbox_inches='tight')

# %%
