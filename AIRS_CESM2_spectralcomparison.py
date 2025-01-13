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
    # axs = axs.flat

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
    tile_cloud_dict[tile_str] = {
        "airs": airs_clear_minus_total,
        "cesm2": cesm2_clear_minus_total,
    }

# %%

# Timeseries summary plot with surface, stratosphere, and cloud fraction channels.
fig, axs = plt.subplots(3, 2, figsize=(12, 18))
axs = axs.flat

# Tropical window channel
airs_data = tile_dict["N00p00_E160p00"]["23"][0]
cesm_data = tile_dict["N00p00_E160p00"]["23"][1]
ax = axs[0]
ax.plot(
    airs_data.time,
    airs_data,
    label="AIRS",
    color="tab:blue",
    alpha=0.75,
)
ax.plot(
    cesm_data.time,
    cesm_data,
    label="CESM2",
    color="tab:orange",
    alpha=0.65,
)
# Plot annual data that is masked appropriately.
nan_mask = ~np.isnan(airs_data.sel(time=slice("2003", "2021")))

airs_annual_data = airs_data.sel(time=slice("2003", "2021")).resample(time="YE").mean()
airs_annual_data["time"] = pd.DatetimeIndex(airs_annual_data["time"]) - pd.Timedelta(days=183)
cesm_annual_data = cesm_data.sel(time=slice("2003", "2021")).where(nan_mask).resample(time="YE").mean()
cesm_annual_data["time"] = pd.DatetimeIndex(cesm_annual_data["time"]) - pd.Timedelta(days=183)

ax.plot(
    airs_annual_data.time,
    airs_annual_data,
    color="darkblue",
    linestyle="--",
)
ax.plot(
    cesm_annual_data.time,
    cesm_annual_data,
    color="darkred",
    linestyle="--",
)

ax.set_title("Tropical Surface Temperature Channel")

# Mid-latitude window channel
airs_data = tile_dict["N44p00_W100p00"]["23"][0]
cesm_data = tile_dict["N44p00_W100p00"]["23"][1]
ax = axs[1]
ax.plot(
    airs_data.time,
    airs_data,
    label="AIRS",
)
ax.plot(
    cesm_data.time,
    cesm_data,
    label="CESM2",
)

nan_mask = ~np.isnan(airs_data.sel(time=slice("2003", "2021")))

airs_annual_data = airs_data.sel(time=slice("2003", "2021")).resample(time="YE").mean()
airs_annual_data["time"] = pd.DatetimeIndex(airs_annual_data["time"]) - pd.Timedelta(days=183)
cesm_annual_data = cesm_data.sel(time=slice("2003", "2021")).where(nan_mask).resample(time="YE").mean()
cesm_annual_data["time"] = pd.DatetimeIndex(cesm_annual_data["time"]) - pd.Timedelta(days=183)

ax.plot(
    airs_annual_data.time,
    airs_annual_data,
    color="darkblue",
    linestyle="--",
)
ax.plot(
    cesm_annual_data.time,
    cesm_annual_data,
    color="darkred",
    linestyle="--",
)

ax.set_title("Mid-latitude Surface Temperature Channel")

# Tropical stratosphere channel
airs_data = tile_dict["N00p00_E160p00"]["1"][0]
cesm_data = tile_dict["N00p00_E160p00"]["1"][1]
ax = axs[2]
ax.plot(
    airs_data.time,
    airs_data,
    label="AIRS",
)
ax.plot(
    cesm_data.time,
    cesm_data,
    label="CESM2",
)

nan_mask = ~np.isnan(airs_data.sel(time=slice("2003", "2021")))

airs_annual_data = airs_data.sel(time=slice("2003", "2021")).resample(time="YE").mean()
airs_annual_data["time"] = pd.DatetimeIndex(airs_annual_data["time"]) - pd.Timedelta(days=183)
cesm_annual_data = cesm_data.sel(time=slice("2003", "2021")).where(nan_mask).resample(time="YE").mean()
cesm_annual_data["time"] = pd.DatetimeIndex(cesm_annual_data["time"]) - pd.Timedelta(days=183)

ax.plot(
    airs_annual_data.time,
    airs_annual_data,
    color="darkblue",
    linestyle="--",
)
ax.plot(
    cesm_annual_data.time,
    cesm_annual_data,
    color="darkred",
    linestyle="--",
)
ax.set_title("Tropical Stratosphere Channel")

# Mid-latitude stratosphere channel
airs_data = tile_dict["N44p00_W100p00"]["1"][0]
cesm_data = tile_dict["N44p00_W100p00"]["1"][1]
ax = axs[3]
ax.plot(
    airs_data.time,
    airs_data,
    label="AIRS",
)
ax.plot(
    cesm_data.time,
    cesm_data,
    label="CESM2",
)

nan_mask = ~np.isnan(airs_data.sel(time=slice("2003", "2021")))

airs_annual_data = airs_data.sel(time=slice("2003", "2021")).resample(time="YE").mean()
airs_annual_data["time"] = pd.DatetimeIndex(airs_annual_data["time"]) - pd.Timedelta(days=183)
cesm_annual_data = cesm_data.sel(time=slice("2003", "2021")).where(nan_mask).resample(time="YE").mean()
cesm_annual_data["time"] = pd.DatetimeIndex(cesm_annual_data["time"]) - pd.Timedelta(days=183)

ax.plot(
    airs_annual_data.time,
    airs_annual_data,
    color="darkblue",
    linestyle="--",
)
ax.plot(
    cesm_annual_data.time,
    cesm_annual_data,
    color="darkred",
    linestyle="--",
)
ax.set_title("Mid-latitude Stratosphere Channel")

# Tropical tile
airs_data = tile_cloud_dict["N00p00_E160p00"]["airs"]
cesm_data = tile_cloud_dict["N00p00_E160p00"]["cesm2"]

ax = axs[4]
ax.plot(
    airs_data.time,
    airs_data,
    label="AIRS",
)

ax.plot(
    cesm_data.time,
    cesm_data,
    label="CESM2",
)

nan_mask = ~np.isnan(airs_data.sel(time=slice("2003", "2021")))

airs_annual_data = airs_data.sel(time=slice("2003", "2021")).resample(time="YE").mean()
airs_annual_data["time"] = pd.DatetimeIndex(airs_annual_data["time"]) - pd.Timedelta(days=183)
cesm_annual_data = cesm_data.sel(time=slice("2003", "2021")).where(nan_mask).resample(time="YE").mean()
cesm_annual_data["time"] = pd.DatetimeIndex(cesm_annual_data["time"]) - pd.Timedelta(days=183)

ax.plot(
    airs_annual_data.time,
    airs_annual_data,
    color="darkblue",
    linestyle="--",
)
ax.plot(
    cesm_annual_data.time,
    cesm_annual_data,
    color="darkred",
    linestyle="--",
)
ax.set_title("Clear-sky minus Total-sky Window Channel Radiance")

# Mid-latitude tile
airs_data = tile_cloud_dict["N44p00_W100p00"]["airs"]
cesm_data = tile_cloud_dict["N44p00_W100p00"]["cesm2"]

ax = axs[5]
ax.plot(
    airs_data.time,
    airs_data,
    label="AIRS",
)

ax.plot(
    cesm_data.time,
    cesm_data,
    label="CESM2",
)

nan_mask = ~np.isnan(airs_data.sel(time=slice("2003", "2021")))

airs_annual_data = airs_data.sel(time=slice("2003", "2021")).resample(time="YE").mean()
airs_annual_data["time"] = pd.DatetimeIndex(airs_annual_data["time"]) - pd.Timedelta(days=183)
cesm_annual_data = cesm_data.sel(time=slice("2003", "2021")).where(nan_mask).resample(time="YE").mean()
cesm_annual_data["time"] = pd.DatetimeIndex(cesm_annual_data["time"]) - pd.Timedelta(days=183)

ax.plot(
    airs_annual_data.time,
    airs_annual_data,
    color="darkblue",
    linestyle="--",
)
ax.plot(
    cesm_annual_data.time,
    cesm_annual_data,
    color="darkred",
    linestyle="--",
)
ax.set_title("Clear-sky minus Total-sky Window Channel Radiance")

for ax in axs:
    ax.legend()
    ax.set_xlabel("Time")
    ax.set_ylabel("Radiance (mW/m$^2$/sr/cm$^{-1}$)")

# to_png(fig, "timeseries_summary_strato", loc="/glade/u/home/jonahshaw/figures/")

# %%

# Timeseries summary plot with surface, troposphere, and cloud fraction channels.
fig, axs = plt.subplots(3, 2, figsize=(12, 14))
axs = axs.flat

# Tropical window channel
airs_data = tile_dict["N00p00_E160p00"]["23"][0]
cesm_data = tile_dict["N00p00_E160p00"]["23"][1]
ax = axs[0]
ax.plot(
    airs_data.time,
    airs_data,
    label="AIRS",
    color="tab:blue",
    alpha=0.75,
)
ax.plot(
    cesm_data.time,
    cesm_data,
    label="CESM2",
    color="tab:orange",
    alpha=0.65,
)
# Plot annual data that is masked appropriately.
nan_mask = ~np.isnan(airs_data.sel(time=slice("2003", "2021")))

airs_annual_data = airs_data.sel(time=slice("2003", "2021")).resample(time="YE").mean()
airs_annual_data["time"] = pd.DatetimeIndex(airs_annual_data["time"]) - pd.Timedelta(days=183)
cesm_annual_data = cesm_data.sel(time=slice("2003", "2021")).where(nan_mask).resample(time="YE").mean()
cesm_annual_data["time"] = pd.DatetimeIndex(cesm_annual_data["time"]) - pd.Timedelta(days=183)

ax.plot(
    airs_annual_data.time,
    airs_annual_data,
    color="darkblue",
    linestyle="--",
)
ax.plot(
    cesm_annual_data.time,
    cesm_annual_data,
    color="darkred",
    linestyle="--",
)

ax.set_title("Tropical Surface Temperature Channel")

# Mid-latitude window channel
airs_data = tile_dict["N44p00_W100p00"]["23"][0]
cesm_data = tile_dict["N44p00_W100p00"]["23"][1]
ax = axs[1]
ax.plot(
    airs_data.time,
    airs_data,
    label="AIRS",
)
ax.plot(
    cesm_data.time,
    cesm_data,
    label="CESM2",
)

nan_mask = ~np.isnan(airs_data.sel(time=slice("2003", "2021")))

airs_annual_data = airs_data.sel(time=slice("2003", "2021")).resample(time="YE").mean()
airs_annual_data["time"] = pd.DatetimeIndex(airs_annual_data["time"]) - pd.Timedelta(days=183)
cesm_annual_data = cesm_data.sel(time=slice("2003", "2021")).where(nan_mask).resample(time="YE").mean()
cesm_annual_data["time"] = pd.DatetimeIndex(cesm_annual_data["time"]) - pd.Timedelta(days=183)

ax.plot(
    airs_annual_data.time,
    airs_annual_data,
    color="darkblue",
    linestyle="--",
)
ax.plot(
    cesm_annual_data.time,
    cesm_annual_data,
    color="darkred",
    linestyle="--",
)

ax.set_title("Mid-latitude Surface Temperature Channel")

# Tropical stratosphere channel
airs_data = tile_dict["N00p00_E160p00"]["6"][0]
cesm_data = tile_dict["N00p00_E160p00"]["6"][1]
ax = axs[2]
ax.plot(
    airs_data.time,
    airs_data,
    label="AIRS",
)
ax.plot(
    cesm_data.time,
    cesm_data,
    label="CESM2",
)

nan_mask = ~np.isnan(airs_data.sel(time=slice("2003", "2021")))

airs_annual_data = airs_data.sel(time=slice("2003", "2021")).resample(time="YE").mean()
airs_annual_data["time"] = pd.DatetimeIndex(airs_annual_data["time"]) - pd.Timedelta(days=183)
cesm_annual_data = cesm_data.sel(time=slice("2003", "2021")).where(nan_mask).resample(time="YE").mean()
cesm_annual_data["time"] = pd.DatetimeIndex(cesm_annual_data["time"]) - pd.Timedelta(days=183)

ax.plot(
    airs_annual_data.time,
    airs_annual_data,
    color="darkblue",
    linestyle="--",
)
ax.plot(
    cesm_annual_data.time,
    cesm_annual_data,
    color="darkred",
    linestyle="--",
)
ax.set_title("Tropical Upper-troposphere Channel")

# Mid-latitude stratosphere channel
airs_data = tile_dict["N44p00_W100p00"]["6"][0]
cesm_data = tile_dict["N44p00_W100p00"]["6"][1]
ax = axs[3]
ax.plot(
    airs_data.time,
    airs_data,
    label="AIRS",
)
ax.plot(
    cesm_data.time,
    cesm_data,
    label="CESM2",
)

nan_mask = ~np.isnan(airs_data.sel(time=slice("2003", "2021")))

airs_annual_data = airs_data.sel(time=slice("2003", "2021")).resample(time="YE").mean()
airs_annual_data["time"] = pd.DatetimeIndex(airs_annual_data["time"]) - pd.Timedelta(days=183)
cesm_annual_data = cesm_data.sel(time=slice("2003", "2021")).where(nan_mask).resample(time="YE").mean()
cesm_annual_data["time"] = pd.DatetimeIndex(cesm_annual_data["time"]) - pd.Timedelta(days=183)

ax.plot(
    airs_annual_data.time,
    airs_annual_data,
    color="darkblue",
    linestyle="--",
)
ax.plot(
    cesm_annual_data.time,
    cesm_annual_data,
    color="darkred",
    linestyle="--",
)
ax.set_title("Mid-latitude Upper-Troposphere Channel")

# Tropical tile
airs_data = tile_cloud_dict["N00p00_E160p00"]["airs"]
cesm_data = tile_cloud_dict["N00p00_E160p00"]["cesm2"]

ax = axs[4]
ax.plot(
    airs_data.time,
    airs_data,
    label="AIRS",
)

ax.plot(
    cesm_data.time,
    cesm_data,
    label="CESM2",
)

nan_mask = ~np.isnan(airs_data.sel(time=slice("2003", "2021")))

airs_annual_data = airs_data.sel(time=slice("2003", "2021")).resample(time="YE").mean()
airs_annual_data["time"] = pd.DatetimeIndex(airs_annual_data["time"]) - pd.Timedelta(days=183)
cesm_annual_data = cesm_data.sel(time=slice("2003", "2021")).where(nan_mask).resample(time="YE").mean()
cesm_annual_data["time"] = pd.DatetimeIndex(cesm_annual_data["time"]) - pd.Timedelta(days=183)

ax.plot(
    airs_annual_data.time,
    airs_annual_data,
    color="darkblue",
    linestyle="--",
)
ax.plot(
    cesm_annual_data.time,
    cesm_annual_data,
    color="darkred",
    linestyle="--",
)
ax.set_title("Clear-sky minus Total-sky Window Channel Radiance")

# Mid-latitude tile
airs_data = tile_cloud_dict["N44p00_W100p00"]["airs"]
cesm_data = tile_cloud_dict["N44p00_W100p00"]["cesm2"]

ax = axs[5]
ax.plot(
    airs_data.time,
    airs_data,
    label="AIRS",
)

ax.plot(
    cesm_data.time,
    cesm_data,
    label="CESM2",
)

nan_mask = ~np.isnan(airs_data.sel(time=slice("2003", "2021")))

airs_annual_data = airs_data.sel(time=slice("2003", "2021")).resample(time="YE").mean()
airs_annual_data["time"] = pd.DatetimeIndex(airs_annual_data["time"]) - pd.Timedelta(days=183)
cesm_annual_data = cesm_data.sel(time=slice("2003", "2021")).where(nan_mask).resample(time="YE").mean()
cesm_annual_data["time"] = pd.DatetimeIndex(cesm_annual_data["time"]) - pd.Timedelta(days=183)

ax.plot(
    airs_annual_data.time,
    airs_annual_data,
    # label="AIRS Annual",
    color="darkblue",
    linestyle="--",
)
ax.plot(
    cesm_annual_data.time,
    cesm_annual_data,
    # label="CESM Annual",
    color="darkred",
    linestyle="--",
)
ax.set_title("Clear-sky minus Total-sky Window Channel Radiance")

labels = ['a', 'b', 'c', 'd', 'e', 'f']
for ax, label in zip(axs, labels):
    ax.legend()
    ax.set_xlabel("Time")
    ax.set_ylabel("Radiance (mW/m$^2$/sr/cm$^{-1}$)")
    ax.text(0.02, 0.98, f'{label}.', transform=ax.transAxes, fontsize=14, verticalalignment='top', weight='bold')

# to_png(fig, "timeseries_summary_uppertropo", bbox_inches='tight', dpi=200, ext="pdf")

# %%

# Create simple figure showing the use of swathing by satellite orbit branch. 
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
axs = axs.flat

ax=axs[0]
spectral_timeseries_plot(
    "N44p00_W100p00",
    1231.3276,
    "clear",
    "asc",
    ax=ax,
)
ax.set_title("Daytime Orbits", fontsize=15)

ax=axs[1]
spectral_timeseries_plot(
    "N44p00_W100p00",
    1231.3276,
    "clear",
    "des",
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

airs_asc, cesm2_asc = load_timeseries_data(
    "N44p00_W100p00",
    1231.3276,
    "clear",
    "asc",
)
airs_des, cesm2_des = load_timeseries_data(
    "N44p00_W100p00",
    1231.3276,
    "clear",
    "des",
)
airs_diff = airs_asc - airs_des
cesm2_diff = cesm2_asc - cesm2_des

# Create simple figure showing the use of swathing by satellite orbit branch. 
fig, axs = plt.subplots(1, 3, figsize=(15, 6))
axs = axs.flat

ax=axs[0]
spectral_timeseries_plot(
    "N44p00_W100p00",
    1231.3276,
    "clear",
    "asc",
    ax=ax,
)
ax.set_title("Daytime Orbits", fontsize=15)

ax=axs[1]
spectral_timeseries_plot(
    "N44p00_W100p00",
    1231.3276,
    "clear",
    "des",
    ax=ax,
)
ax.set_title("Nighttime Orbits", fontsize=15)

ax = axs[2]
ax.plot(
    airs_diff.time,
    airs_diff,
    label="AIRS",
)
ax.plot(
    cesm2_diff.time,
    cesm2_diff,
    label="CESM2",
)

labels = ['a', 'b', 'c']
for ax, label in zip(axs, labels):
    ax.legend()
    ax.set_xlabel("Year", fontsize=15)
    ax.set_ylabel("Radiance (mW/m$^2$/sr/cm$^{-1}$)", fontsize=15)
    ax.set_ylim(20, 85)
    ax.tick_params(axis='both', labelsize=14)
    ax.text(0.02, 0.98, f'{label}.', transform=ax.transAxes, fontsize=14, verticalalignment='top', weight='bold')
    # handles, labels = ax.get_legend_handles_labels()
    # ax.get_legend().remove()
    # ax.legend(handles, ["CESM2", "AIRS"], fontsize=14)
# axs[0].get_legend().remove()
handles, labels = axs[2].get_legend_handles_labels()
axs[2].legend(handles, ["AIRS", "CESM2"], fontsize=14)
axs[2].set_ylim(-5, 40)

# to_png(fig, "midlat_window_daynight_3panel", dpi=200, ext="pdf", bbox_inches='tight')

# %%

# Other older but still useful plots.

# Chan 33 is 1596cm-1 WV channel
# From the Jacobians, this channel is sensitive to WV around 400mb and it is negative.
# At the same heights it is positively sensitive to atmospheric temperature.
# The clear-sky values from CESM2 are too low here, which the all-sky values are pretty good.
# I don't really understand exactly what this means.
# Too moist or too cold in the upper troposphere?
fig, axs = spectral_timeseries_4plot(
    "N00p00_E160p00",
    1231.3276,
    i_wnum=32,
)
axs[0].set_ylim(3.5, 8)
axs[1].set_ylim(3.5, 8)

axs[2].set_ylim(2.5, 6.5)
axs[3].set_ylim(2.5, 6.5)

# %%
# Channel 5 is 710.43cm-1 mid-tropospheric temperature/co2 channel 
# 350 mb weighting function peak for standard atmosphere.
# CESM2 is clearly to low in the clear-sky scenes and a bit to low in all-sky.
fig, axs = spectral_timeseries_4plot(
    "N00p00_E160p00",
    1231.3276,
    i_wnum=4,
)
axs[0].set_ylim(56, 61)
axs[1].set_ylim(56, 61)

axs[2].set_ylim(43, 60)
axs[3].set_ylim(43, 60)

# %%
# Channel 7 is a 740.97cm-1 mid-tropospheric temperature/co2 channel
# 416 mb weighting function peak for standard atmosphere. 
# CESM2 is clearly to low in the clear-sky scenes and a bit to low in all-sky.
fig, axs = spectral_timeseries_4plot(
    "N00p00_E160p00",
    1231.3276,
    i_wnum=6,
)

axs[0].set_ylim(56, 63)
axs[1].set_ylim(56, 63)

axs[2].set_ylim(43, 62)
axs[3].set_ylim(43, 62)

# %%
# Upper troposphere water vapor channel over a mid-latitude land region in the middle of N. America
# From the Jacobians, this channel is sensitive to WV around 400mb and it is negative.
# At the same heights it is positively sensitive to atmospheric temperature.
# CESM2 is too low everywhere, especially in the summer.
# So the mid-atmosphere is too moist and/or too cold in CESM2.
# It also seems to have a smaller seasonal cycle.
# Unlike the tropical tile, these biases don't resolve in the total-sky scenes.
fig, axs = spectral_timeseries_4plot(
    "N44p00_W100p00",
    1231.3276,
    i_wnum=32,
)

axs[0].set_ylim(3, 7)
axs[1].set_ylim(3, 7)

axs[2].set_ylim(2.75, 6.25)
axs[3].set_ylim(2.75, 6.25)

# %%
# Channel 7 is a 740.97cm-1 mid-tropospheric temperature/co2 channel
# 416 mb weighting function peak for standard atmosphere. 
# There is large seasonality here not seen at mid-latitudes, huh.
# CESM2 is clearly too cold.
fig, axs = spectral_timeseries_4plot(
    "N44p00_W100p00",
    1231.3276,
    i_wnum=6,
)

axs[0].set_ylim(47, 63)
axs[1].set_ylim(47, 63)

axs[2].set_ylim(45, 60)
axs[3].set_ylim(45, 60)

# %%
# Channel 24 is 1231.3276cm-1 window channel
# This shows good agreement in for all-sky and for clear-sky radiances in the descending branch.
# The ascending branch shows a larger difference between AIRS and CESM2.
# CESM2 underestimates the radiance, implying that the daytime temperatures are too low.
# In fact, there is almost no diurnal contrast in the clear-sky window channel radiances in CESM2, which makes sense given the ocean surface.
fig, axs = spectral_timeseries_4plot(
    "N00p00_E160p00",
    1231.3276,
    i_wnum=23,
)
axs[0].set_ylim(53, 62)
axs[1].set_ylim(53, 62)

axs[2].set_ylim(20, 60)
axs[3].set_ylim(20, 60)

# %%

# Window channel over a mid-latitude land region in the middle of N. America
# CESM2 is too cold in the winter (both branches).
# Clear-sky data: The nighttime branches show that CESM2 has too large of a seasonal cycle.
# All-sky data: There is more variability but in general the agreement is pretty good. 
# CESM2 is probably too warm in the summer but until if this is from the surface or not enough clouds.
fig, axs = spectral_timeseries_4plot(
    "N44p00_W100p00",
    1231.3276,
    i_wnum=23,
)
axs[0].set_ylim(20, 85)
axs[1].set_ylim(20, 85)

axs[2].set_ylim(15, 65)
axs[3].set_ylim(15, 65)

# %%

spectral_summary_plot(
    "N75p00_E000p00",
    "clear",
    "asc",
    # ufunc=lambda ds: sel_month(ds, 7),
)

fig, axs = plt.subplots(3, 4, figsize=(20, 15))
axs = axs.flat

for i_month, ax in zip(range(1, 13), axs):
    spectral_summary_plot(
        "N44p00_W100p00",
        "clear",
        "asc",
        ax=ax,
        ufunc=lambda ds: sel_month(ds, i_month),
    )
    ax.set_title(f"clear asc {month_dict[i_month]}")

# %%
