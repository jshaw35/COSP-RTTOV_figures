"""
This script is used to create cone plots demonstrating climate change detection with COSP-RTTOV and AIRS.
"""
# %%

import xarray as xr
import numpy as np
import os
import matplotlib.pyplot as plt


def load_timeseries_data(
    tile_str: str,
    wnum: float,
    scene_type: str,
    branch_type: str,
    airs_case_str: str,
    cesm_case_str: str,
    processed_airs_path: str = None,
    processed_cesm2_path: str = None,
    i_wnum: int = None,
):
    """
    Load and process time series data for AIRS and CESM2.

    Parameters:
    -----------
    tile_str : str
        The tile string identifier.
    wnum : float
        The wavenumber to select.
    scene_type : str
        The scene type, must be 'total' or 'clear'.
    branch_type : str
        The branch type, must be 'asc' or 'des'.
    airs_case_str : str
        The AIRS case string identifier. e.g. 'deg_0_5'.
    cesm_case_str : str
        The CESM2 case string identifier. e.g. 'nudged_swathed'.
    i_wnum : int, optional
        The index of the wavenumber to select. If None, wnum is used.

    Returns:
    --------
    tuple
        A tuple containing the AIRS and CESM2 time series subsets.

    Raises:
    -------
    ValueError
        If scene_type or branch_type is invalid.
    FileNotFoundError
        If the processed AIRS or CESM2 file does not exist.
    """

    if processed_airs_path is None:
        print("yikes")
        processed_airs_path = f"/glade/campaign/univ/ucuc0007/AIRS_data/{tile_str}/proc/{tile_str}_tseries_{airs_case_str}.nc"
    if processed_cesm2_path is None:
        print("yikes")
        processed_cesm2_path = f"/glade/campaign/univ/ucuc0007/AIRS_data/{tile_str}/proc/CESM2_{cesm_case_str}_{tile_str}.nc"

    if scene_type not in ["total", "clear"]:
        raise ValueError(f"Invalid scene type: {scene_type}. Must be 'total' or 'clear'.")

    if branch_type == "asc":
        airs_var = f"rad_{scene_type}_{branch_type}"
        cesm2_var = f"rttov_rad_{scene_type}_inst001"
    elif branch_type == "des":
        airs_var = f"rad_{scene_type}_{branch_type}"    
        cesm2_var = f"rttov_rad_{scene_type}_inst002"
    elif branch_type == "all":
        airs_var = [f"rad_{scene_type}_{branch_type}" for branch_type in ["asc", "des"]]
        cesm2_var = f"rttov_rad_{scene_type}_inst001"
    else:
        raise ValueError(f"Invalid branch type: {branch_type}. Must be 'asc' or 'des' or 'all'.")

    if not os.path.exists(processed_airs_path):
        raise FileNotFoundError(f"{processed_airs_path} does not exist.")
    if not os.path.exists(processed_cesm2_path):
        raise FileNotFoundError(f"{processed_cesm2_path} does not exist.")

    airs_ds = xr.open_dataset(processed_airs_path)[airs_var]
    cesm2_ds = xr.open_dataset(processed_cesm2_path)[cesm2_var]

    if i_wnum is None:
        airs_tsubset = airs_ds.sel(wnum=wnum, method="nearest")
        cesm2_tsubset = cesm2_ds.sel(wnum=wnum, method="nearest")
    else:
        airs_tsubset = airs_ds.isel(wnum=i_wnum)
        cesm2_tsubset = cesm2_ds.isel(wnum=i_wnum)

    return airs_tsubset, cesm2_tsubset


def compute_trends_wrapper(
    data,
    durations=np.arange(5,81,),
    time_dim="year",
    dask=False,
    startyears=None,
    **kwargs,
):
    '''
    Wrapper for running 'get_allvar_allstartyears' for different durations.
    Setup with correct startyears and concatenate at the end.
    '''

    first_year = data[time_dim][0]
    last_year = data[time_dim][-1]

    trends_allstartyear_allduration_list = []

    for duration in durations:
        print(duration, end=' ')
        if startyears is None:
            _startyears = np.arange(first_year, last_year+2-duration, 1)
        else:
            _startyears = startyears

        if dask:
            allvar_onedur_ds = dask.delayed(get_trends_allstartyears)(
                data,
                duration=duration,
                startyears=_startyears,
                dim=time_dim,
                **kwargs
            )
        else:
            allvar_onedur_ds = get_trends_allstartyears(
                data,
                duration=duration,
                startyears=_startyears,
                dim=time_dim,
                **kwargs
            )

        trends_allstartyear_allduration_list.append(allvar_onedur_ds)

    if dask:
        trends_allstartyear_allduration_list = dask.compute(*trends_allstartyear_allduration_list)
    trends_allstartyear_allduration_ds = xr.concat(trends_allstartyear_allduration_list, dim='duration')
    del trends_allstartyear_allduration_list

    return trends_allstartyear_allduration_ds


def get_trends_allstartyears(
    data,
    duration,
    startyears,
    dim='year',
    dask=False,
):
    '''
    Calculate: 
    a. trends of a given duration
    '''

    # Initialize list to save to
    trends_list = []

    for startyear in startyears:

        _startyr = startyear
        _endyr   = startyear + duration - 1

        _tsel = data.sel({dim:slice(_startyr,_endyr)})  # index differently here because the dates are different

        # Calculate the slope
        if dask:
            _tsel_polyfit = dask.delayed(xr.DataArray.polyfit)(_tsel, dim=dim, deg=1, skipna=True)['polyfit_coefficients'].sel(degree=1)
        else:
            _tsel_polyfit = xr.DataArray.polyfit(_tsel, dim=dim, deg=1, skipna=True)['polyfit_coefficients'].sel(degree=1)
        _tsel_slopes = _tsel_polyfit.drop_vars('degree')

        trends_list.append(_tsel_slopes.assign_coords({'startyear': startyear}).expand_dims('startyear'))

    if dask:
        trends_list = dask.compute(*trends_list)
    out = xr.concat(trends_list, dim='startyear').assign_coords({'duration': duration}).expand_dims('duration')

    return out


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

if __name__ == "__main__":

    data_dict = {}
    tile_strs = ["N00p00_E160p00", "N44p00_W100p00"]
    wnum = 0 # arbitrary
    i_wnum = 6 # Upper troposphere channel sounding ~300mb
    scene_type = "clear"
    branch_type = "all"
    airs_case_str = "deg_0_5"
    cesm_case_str = "PI_Control"

    for tile_str in tile_strs:
        processed_airs_path = f"./data_dir/{tile_str}_tseries_{airs_case_str}.nc"
        processed_cesm2_path = f"./data_dir/CESM2_{cesm_case_str}_{tile_str}.nc"
        airs_tsubset, cesm2_tsubset = load_timeseries_data(
            tile_str,
            wnum,
            scene_type,
            branch_type,
            airs_case_str,
            cesm_case_str,
            processed_airs_path=processed_airs_path,
            processed_cesm2_path=processed_cesm2_path,
            i_wnum=i_wnum,
        )
        data_dict[tile_str] = (airs_tsubset, cesm2_tsubset)


    cone_data_dict = {}
    for tile_str in tile_strs:

        # Compute trends from the PI-Control simulation
        pi_data = data_dict[tile_str][1].sel(time=slice("0501","0699")).groupby("time.year").mean()
        obs_data = 0.5 * (data_dict[tile_str][0]["rad_clear_asc"] + data_dict[tile_str][0]["rad_clear_des"]).sel(time=slice("2003-01-01", "2023-12-31")).groupby("time.year").mean()

        pi_trends = compute_trends_wrapper(
            pi_data,
            durations=np.arange(3, 22,),
        )

        obs_trends = compute_trends_wrapper(
            obs_data,
            durations=np.arange(3, 22,),
            startyears=[2003],
        )

        # Use the aboslute value of the trends. 
        # By assuming the trends are symmetric about zero, we can use the 95th percentile as a threshold 
        # instead of 2.5 and 97.5.
        pi_trends_abs = np.abs(pi_trends)
        pi_trends_95perc = pi_trends_abs.quantile(q=0.95, dim="startyear")
        cone_data_dict[tile_str] = (obs_trends, pi_trends_95perc, pi_trends)

    # Double cone plot
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    fig.subplots_adjust(wspace=0.35)

    startyear = 2003

    tile_name_dict = {"N00p00_E160p00": "Tropical Pacific", "N44p00_W100p00": "Mid-Latitude Land"}
    ylim_dict = {"N00p00_E160p00": (-0.25, 0.25), "N44p00_W100p00": (-0.35, 0.35)}
    labels = ['a', 'b', 'c', 'd', 'e', 'f']

    for tile_str, ax, label in zip(tile_strs, axs, labels):

        obs_trends = cone_data_dict[tile_str][0].squeeze()
        pi_trends_95perc = cone_data_dict[tile_str][1]
        pi_trends_all = cone_data_dict[tile_str][2]

        ax.plot(
            pi_trends_95perc["duration"],
            -1*pi_trends_95perc,
            color="black",
            alpha=1,
        )
        ax.plot(
            pi_trends_95perc["duration"],
            pi_trends_95perc,
            color="black",
            alpha=1,
        )
        ax.plot(
            pi_trends_95perc["duration"],
            -2*pi_trends_95perc,
            color="black",
            alpha=1,
            linestyle="--",
            linewidth=1,
        )
        ax.plot(
            pi_trends_95perc["duration"],
            2*pi_trends_95perc,
            color="black",
            alpha=1,
            linestyle="--",
            linewidth=1,
        )
        ax.fill_between(
            pi_trends_95perc["duration"],
            -1*pi_trends_95perc,
            pi_trends_95perc,
            color="black",
            alpha=0.5,
            label="Pre-Industrial Range",
        )
        ax.plot(
            obs_trends["duration"],
            obs_trends,
            color="blue",
            label="AIRS",
            linewidth=1,
        )
        ax.fill_between(
            obs_trends["duration"],
            obs_trends - 0.002,
            obs_trends + 0.002,
            color="blue",
            alpha=0.25,
        )

        ax.set_xlim(3, 21)
        ax.set_ylim(ylim_dict[tile_str])
        ax.xaxis.set_major_locator(plt.MultipleLocator(5))
        ax.xaxis.set_minor_locator(plt.MultipleLocator(1))
        ax.tick_params(axis='x', which='major', length=8)
        ax.tick_params(axis='x', which='minor', length=5)

        ax.set_xlabel('Trend Duration (years)', fontsize=16)
        ax.set_ylabel('Radiance Trend \n (mW/m$^2$/sr/cm$^{-1}$ per year)', fontsize=14)
        ax.tick_params(axis='both', labelsize=14)
        ax.legend()

        ax.text(0.02, 0.98, f'{label}.', transform=ax.transAxes, fontsize=14, verticalalignment='top', weight='bold')

        # Create a second x-axis on the top
        ax2 = ax.twiny()
        ax2.set_xlim(3, 21)
        ax2.set_xlabel('Year', fontsize=16)
        ax2.xaxis.set_major_locator(plt.MultipleLocator(5))
        ax2.xaxis.set_minor_locator(plt.MultipleLocator(1))
        ax2.tick_params(axis='both', labelsize=14)
        ax2.tick_params(axis='x', which='major', length=8)
        ax2.tick_params(axis='x', which='minor', length=5)
        ax2.set_xticks(np.arange(3, 21, 5))
        ax2.set_xticklabels(np.arange(2005, 2024, 5))

    # axs[0].set_title('Upper-troposphere Channel - Tropical', fontsize=18)
    # axs[1].set_title('Upper-troposphere Channel - Mid-latitude', fontsize=18)

    to_png(fig, f"AIRS_740cm-1_cone_plot", dpi=200, bbox_inches='tight', ext='pdf')

    # %%
