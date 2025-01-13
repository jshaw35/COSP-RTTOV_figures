"""
Evaluate differences in precipitation frequency between CloudSat ascending and descending branch nodes.
"""
# %%
import matplotlib.pyplot as plt   # for plotting
import numpy as np                # for arrays+math
import xarray as xr               # for netCDF data
import os
import cartopy.crs as ccrs        # import projections

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

    # Load preprocessed data
    # CESM2 with COSP-RTTOV
    model_daytime_ds = xr.open_dataset("./data_dir/CESM2_CSdaytimeprecip_200606-201005.nc")
    model_nighttime_ds = xr.open_dataset("./data_dir/CESM2_CSnighttimeprecip_200606-201005.nc")

    model_daytime_precipfrac = (1 - model_daytime_ds["CS_NOPRECIP"]).mean(dim="time").compute()
    model_nighttime_precipfrac = (1 - model_nighttime_ds["CS_NOPRECIP"]).mean(dim="time").compute()
    model_precip_frac_diff = model_daytime_precipfrac - model_nighttime_precipfrac

    # CloudSat observations
    precip_frac_asc = xr.open_dataarray("./data_dir/CloudSat_ascprecip_200606-201005.nc")
    precip_frac_des = xr.open_dataarray("./data_dir/CloudSat_descprecip_200606-201005.nc")

    precip_frac_diff = precip_frac_asc - precip_frac_des

    # Compute differences between models and observations.
    daytime_diff = model_daytime_precipfrac - precip_frac_asc
    nighttime_diff = model_nighttime_precipfrac - precip_frac_des
    diff_diff = model_precip_frac_diff - precip_frac_diff

    # Now combine the model and observations so that they can be compared!
    ## Plot model data!

    data = [
        [precip_frac_asc, precip_frac_des, precip_frac_diff],
        [model_daytime_precipfrac, model_nighttime_precipfrac, model_precip_frac_diff],
        [daytime_diff, nighttime_diff, diff_diff],
    ]

    row1_v = [[0.0, 0.25, 0.5], [-0.25, 0, 0.25]]
    row2_v = [[0.0, 0.5, 1], [-0.35, 0, 0.35]]
    row3_v = [[-0.1, 0, 1], [-0.35, 0, 0.35]]
    rows = [row1_v, row2_v, row3_v]

    fig, axs = sp_map(3, 3, projection=ccrs.Robinson(), figsize=(14, 9))
    fig.subplots_adjust(hspace=0.40)

    # Have a separate colorbar for each row
    length1 = 0.40
    length2 = 0.20
    height = 0.0125
    xstart1 = 0.175
    xstart2 = 0.685
    cax1 = plt.axes([xstart1, 0.65, length1, height])
    cax2 = plt.axes([xstart2, 0.65, length2, height])

    cax3 = plt.axes([xstart1, 0.37, length1, height])
    cax4 = plt.axes([xstart2, 0.37, length2, height])

    cax5 = plt.axes([xstart1, 0.09, length1, height])
    cax6 = plt.axes([xstart2, 0.09, length2, height])

    cbars = [
        [cax1, cax2, row1_v],
        [cax3, cax4, row2_v],
        [cax5, cax6, row3_v],
    ]
    error_cmap = "Spectral"
    # error_cmap = sns.diverging_palette(250, 30, l=65, center="dark", as_cmap=True)
    cmaps = [
        [None, None, "bwr"],
        [None, None, "bwr"],
        [error_cmap, error_cmap, "BrBG"],
    ]
    ims = []

    ziplist = [axs, rows, cmaps, data]
    for _axs, _row, _cmaps, _data in zip(*ziplist):

        print(_row)
        _divnorm1 = colors.TwoSlopeNorm(
            vmin=_row[0][0],
            vcenter=_row[0][1],
            vmax=_row[0][2],
        )
        _divnorm2 = colors.TwoSlopeNorm(
            vmin=_row[1][0],
            vcenter=_row[1][1],
            vmax=_row[1][2],
        )

        im0 = _axs[0].pcolormesh(
            _data[0].lon,
            _data[0].lat,
            _data[0],
            transform=ccrs.PlateCarree(),
            cmap=_cmaps[0],
            norm=_divnorm1,
        )
        im1 = _axs[1].pcolormesh(
            _data[1].lon,
            _data[1].lat,
            _data[1],
            transform=ccrs.PlateCarree(),
            cmap=_cmaps[1],
            norm=_divnorm1,
        )
        im2 = _axs[2].pcolormesh(
            _data[2].lon,
            _data[2].lat,
            _data[2],
            transform=ccrs.PlateCarree(),
            cmap=_cmaps[2],
            norm=_divnorm2,
        )
        ims.append([im0, im1, im2])

    axs = axs.flat
    axs[0].set_title("Daytime Orbits", fontsize=16)
    axs[1].set_title("Nighttime Orbits", fontsize=16)
    axs[2].set_title("Daytime minus Nighttime", fontsize=16)
    labels = ["a.", "b.", "c.", "d.", "e.", "f.", "g.", "h.", "i."]
    for _ax, _label in zip(axs, labels):
        _ax.coastlines()
        _ax.text(0.02, 1.0, _label, transform=_ax.transAxes, fontsize=14, verticalalignment='top', weight='bold')

    for _cbar, _im in zip(cbars, ims):
        cbar1 = fig.colorbar(
            _im[0],
            orientation='horizontal',
            ticks=np.arange(_cbar[2][0][0], _cbar[2][0][2] + 0.05, 0.1),
            cax=_cbar[0],
        )
        cbar1.ax.tick_params(labelsize=12)
        cbar2 = fig.colorbar(
            _im[2],
            orientation='horizontal',
            ticks=np.arange(_cbar[2][1][0] + 0.05, _cbar[2][1][2], 0.1),
            cax=_cbar[1],
        )
        cbar2.ax.tick_params(labelsize=12)

        cbar1.ax.set_xlabel("Precipitation Frequency (%)", fontsize=12)
        cbar2.ax.set_xlabel("Precipitation Frequency Difference (%)", fontsize=12)
    cbar1.ax.set_xlabel("Precipitation Frequency Difference (%)", fontsize=12)

    to_png(fig, "CloudSat_PREC_swathing", ext="pdf", dpi=200, bbox_inches='tight')
    to_png(fig, "CloudSat_PREC_swathing", ext="png", dpi=200, bbox_inches='tight')
# %%
