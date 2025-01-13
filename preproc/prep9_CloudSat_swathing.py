"""
Evaluate differences in precipitation frequency between CloudSat ascending and descending branch nodes.
"""
# %%
import matplotlib.pyplot as plt   # for plotting
import numpy as np                # for arrays+math
import xarray as xr               # for netCDF data

import cartopy.crs as ccrs        # import projections
import glob

import matplotlib.colors as colors
import seaborn as sns


def sp_map(*nrs, projection = ccrs.PlateCarree(), **kwargs):
    return plt.subplots(*nrs, subplot_kw={'projection': projection}, **kwargs)


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

    # Load data!
    obsdata_dir = "/glade/u/home/jonahshaw/w/obs/CLOUDSAT/Kay2018_CloudSat_JGRA"
    all_dir = "CloudSat_precip_level3_v2"
    preproc_dir = "5x5deg_preprocess"

    modeldata_dir = "/glade/u/home/jonahshaw/w/obs/CLOUDSAT/COSP_RTTOV_swathing"

    all_obs_files = glob.glob(f"{obsdata_dir}/{all_dir}/{preproc_dir}/*.nc")
    all_obs_files.sort()

    model_alltime_files = glob.glob(f"{modeldata_dir}/*alltimes*.nc")
    model_daytime_files = glob.glob(f"{modeldata_dir}/*daytime*.nc")
    model_nighttime_files = glob.glob(f"{modeldata_dir}/*nighttime*.nc")

    # %%

    # Load preprocessed 5x5 deg. data
    obs_ds = xr.open_mfdataset(all_obs_files)

    # Only use pre-failure data
    # obs_ds = obs_ds.sel(time=slice(None, "2010-12-31"))
    obs_ds = obs_ds.sel(time=slice(None, "2010-05-31"))
    obs_ds["lon"] = obs_ds["lon"] % 360
    obs_ds = obs_ds.sortby("lon")

    # %%
    # Load CESM2 simulated data.
    model_alltime_ds = xr.open_mfdataset(model_alltime_files)
    model_daytime_ds = xr.open_mfdataset(model_daytime_files)
    model_nighttime_ds = xr.open_mfdataset(model_nighttime_files)

    # model_alltime_ds = model_alltime_ds.sel(time=slice("2006-06-01", "2011-03-31"))
    # model_daytime_ds = model_daytime_ds.sel(time=slice("2006-06-01", "2011-03-31"))
    # model_nighttime_ds = model_nighttime_ds.sel(time=slice("2006-06-01", "2011-03-31"))
    model_alltime_ds = model_alltime_ds.sel(time=slice("2006-06-01", "2010-05-31"))
    model_daytime_ds = model_daytime_ds.sel(time=slice("2006-06-01", "2010-05-31"))
    model_nighttime_ds = model_nighttime_ds.sel(time=slice("2006-06-01", "2010-05-31"))
    
    model_daytime_ds.to_netcdf("./data_dir/CESM2_CSdaytimeprecip_200606-201005.nc")
    model_nighttime_ds.to_netcdf("./data_dir/CESM2_CSnighttimeprecip_200606-201005.nc")
    
    

# %%

    ## Process observational data!
    # Example calculating precip frequency.
    precip_frac = obs_ds["precipcounts"].sum(dim="time") / obs_ds["counts"].sum(dim="time")
    precip_frac = precip_frac.compute()

    precip_frac_asc = obs_ds["ascprecipcounts"].sum(dim="time") / obs_ds["asccounts"].sum(dim="time")
    precip_frac_asc = precip_frac_asc.compute()
    precip_frac_asc.to_netcdf("./data_dir/CloudSat_ascprecip_200606-201005.nc")

    precip_frac_des = obs_ds["descprecipcounts"].sum(dim="time") / obs_ds["desccounts"].sum(dim="time")
    precip_frac_des = precip_frac_des.compute()
    precip_frac_des.to_netcdf("./data_dir/CloudSat_descprecip_200606-201005.nc")

    precip_frac_diff = precip_frac_asc - precip_frac_des

# %%