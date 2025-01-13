"""
Produce a set of polar maps for each month to later turn into a gif.
"""
# %%
import numpy as np
import xarray as xr
import glob


def repartition_prefire_channels(
    prefire_rad_ds: xr.Dataset,
    prefire_varname: str,
    prefire_dimname: str,
):
    """
    Function for reconstructing PREFIRE channel 7 from the RTTOV radiances.
    Because this channel had separate peaks in the SRF, the RTTOV team split the 
    coefficients into two channels. Channel 7 is reconstructed with a weighted average.
    """
    
    if (type(prefire_rad_ds) == xr.core.dataarray.Dataset): #xr.core.dataarray.DataArray
        prefire_rad_da = prefire_rad_ds[prefire_varname]
    else:
        prefire_rad_da = prefire_rad_ds
    prefire_rad_4_6 = prefire_rad_da.sel({prefire_dimname:[4, 5, 6]})

    prefire_rad_7_8 = prefire_rad_da.sel({prefire_dimname:[7, 8]})

    prefire_rad_11_64 = prefire_rad_da.sel({prefire_dimname:slice(11, None)})
    
    w7 = 0.2804943147197775
    w8 = 0.7195056852802225

    prefire_rad_7 = w7 * prefire_rad_7_8.sel({prefire_dimname:7}) + w8 * prefire_rad_7_8.sel({prefire_dimname:8})

    prefire_rad_7[prefire_dimname] = 7
    
    # Reassign to the instrument channels
    prefire_rad_11_64[prefire_dimname] = [10,11,12,13,14,15,16,19,20,21,22,23,24,25,26,27,28,29,
                                          30,31,32,33,34,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,
                                          52,53,54,55,56,57,58,59,60,61,62,63]

    prefire_srf_peak_wavelengths = np.array([ 
        0.421905,  0.421905,  0.421905,  4.548133,  5.45101 ,  5.577581,
        6.995181,  0.421905,  0.421905,  9.467543, 10.269162, 10.85139 ,
        11.771143, 12.404   , 13.171867, 14.150686,  0.421905,  0.421905,
        16.783371, 17.618743, 18.926648, 19.745143, 20.59739 , 21.373695,
        21.964362, 22.6816  , 23.483219, 24.723619, 25.525238, 26.174971,
        26.942838, 27.550381, 28.368876, 29.19581 ,  0.421905,  0.421905,
        32.4276  , 33.254533, 34.174286, 34.267105, 35.507505, 36.612895,
        36.764781, 37.853295, 39.085257, 39.262457, 40.612552, 41.566057,
        42.190476, 42.74739 , 43.878095, 44.198743, 45.633219, 45.835733,
        47.455848, 47.62461 , 49.4388  , 49.607562, 50.99141 , 51.835219,
        51.987105, 52.94061 , 54.383524,
    ])
    peak_wl_da = xr.DataArray(
        data=prefire_srf_peak_wavelengths,
        dims=[prefire_dimname],
        coords={prefire_dimname:np.arange(1, 64, 1)},
    )

    prefire_rad_inst_channels = xr.concat([prefire_rad_4_6, prefire_rad_7, prefire_rad_11_64], dim=prefire_dimname)
    prefire_rad_inst_channels = prefire_rad_inst_channels.assign_coords(
        {"peak_wavelength": peak_wl_da.sel({
            prefire_dimname:prefire_rad_inst_channels[prefire_dimname],
            }),
         },
    )
    
    return prefire_rad_inst_channels


def add_cyclic_point(xarray_obj, dim, period=None):
    if period is None:
        period = xarray_obj.sizes[dim] * xarray_obj.coords[dim][:2].diff(dim).item()
    first_point = xarray_obj.isel({dim: slice(1)})
    first_point.coords[dim] = first_point.coords[dim]+period
    return xr.concat([xarray_obj, first_point], dim=dim)


# %%

if __name__ == "__main__":
    # Compute Arctic-mean PREFIRE values for timeseries
    load_dir = "/glade/campaign/univ/ucuc0007/COSP_RTTOV_output/" #20241010_140608.FHIST.f09_f09_mg17.cesm2.1.5_port/atm/proc/tseries/"
    case_names = ["20241010_140608.FHIST.f09_f09_mg17.cesm2.1.5_port", "20241014_112607.FHIST.f09_f09_mg17.cesm2.1.5_port_SSPbranch"]

    # INST1 is PREFIRE.
    clear_rttov_var = "rttov_rad_clear_inst001"
    total_rttov_var = "rttov_rad_total_inst001"    
    rttov_dim = "RTTOV_CHAN_I001"

    monthly_clearsky_files = []
    monthly_totalsky_files = []
    for case in case_names:
        monthly_clearsky_files += glob.glob(f"{load_dir}/{case}/atm/proc/tseries/*h0.{clear_rttov_var}*.nc")
        monthly_totalsky_files += glob.glob(f"{load_dir}/{case}/atm/proc/tseries/*h0.{total_rttov_var}*.nc")

    clear_ds = xr.open_mfdataset(monthly_clearsky_files, chunks={"time":1})
    total_ds = xr.open_mfdataset(monthly_totalsky_files, chunks={"time":1})
    
    # Fix weird time stuff and shift time by one month to correct for CESM output.
    clear_ds["time"] = xr.CFTimeIndex(clear_ds["time"].values).shift(-1, "ME")
    total_ds["time"] = xr.CFTimeIndex(total_ds["time"].values).shift(-1, "ME")
    
    clear_ds = clear_ds[clear_rttov_var].assign_coords(cell_weight=np.cos(np.deg2rad(clear_ds.lat)))
    total_ds = total_ds[total_rttov_var].assign_coords(cell_weight=np.cos(np.deg2rad(total_ds.lat)))
        
    clear_repartition = repartition_prefire_channels(
        clear_ds, 
        clear_rttov_var, 
        rttov_dim,
    )
    clear_repartition = add_cyclic_point(clear_repartition, dim="lon", period=360)
    total_repartition = repartition_prefire_channels(
        total_ds, 
        total_rttov_var, 
        rttov_dim,
    )
    total_repartition = add_cyclic_point(total_repartition, dim="lon", period=360)

    clear_repartition.to_netcdf("./data_dir/PREFIRE_Polar_clearsky.nc")
    total_repartition.to_netcdf("./data_dir/PREFIRE_Polar_totalsky.nc")
    del clear_repartition, total_repartition

# %%
