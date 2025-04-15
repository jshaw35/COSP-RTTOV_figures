"""
Produce a set of polar maps for each month to later turn into a gif.
"""
# %%
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import glob
import os
import matplotlib.gridspec as gridspec
import matplotlib.path as mpath
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from PIL import Image
import nc_time_axis


def polarCentral_set_latlim(lat_lims, ax):
    ax.set_extent([-180, 180, lat_lims[0], lat_lims[1]], ccrs.PlateCarree())
    # Compute a circle in axes coordinates, which we can use as a boundary
    # for the map. We can pan/zoom as much as we like - the boundary will be
    # permanently circular.
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)

    ax.set_boundary(circle, transform=ax.transAxes)


def add_map_features(ax):
    '''
    Single line command for xarray plots
    '''
    gl = ax.gridlines()
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5);
    ax.add_feature(cfeature.BORDERS, linewidth=0.5);
    gl.top_labels = False
    gl.right_labels = False


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


def add_cyclic_point(xarray_obj, dim, period=None):
    if period is None:
        period = xarray_obj.sizes[dim] * xarray_obj.coords[dim][:2].diff(dim).item()
    first_point = xarray_obj.isel({dim: slice(1)})
    first_point.coords[dim] = first_point.coords[dim]+period
    return xr.concat([xarray_obj, first_point], dim=dim)


def rttov_polar_panel_plot(
    data: xr.DataArray,
    rttov_channels: list,
    levels: np.ndarray = np.linspace(30, 120, 10),
    save_fig: bool = False,
):
    """
    This function creates a 4-panel plot of polar map of PREFIRE radiances.

    Args:
        data (xr.DataArray): Data to operate on and plot
        rttov_channels (list): List of integer indices to identify PREFIRE channels.
        levels (np.ndarray): list or array of levels for the contour plot.
        save_fig (bool, optional): Whether outputs should be save. Defaults to False.
    Returns:
        None
    """
    num_panels = len(rttov_channels)
    fig = plt.figure(figsize=[3.75 * num_panels, 4])

    gs = gridspec.GridSpec(1, num_panels, figure=fig)
    axs = []
    for i in range(num_panels):
        axs.append(fig.add_subplot(gs[0, i], projection=ccrs.NorthPolarStereo()))
    
    for ax, _channel in zip(axs, rttov_channels):
        
        panel_data = data.sel({rttov_dim:_channel})
        
        polarCentral_set_latlim([60, 90], ax)
        
        im = ax.contourf(
            panel_data['lon'],
            panel_data['lat'],
            panel_data,
            transform=ccrs.PlateCarree(),
            #  colors=palette,
            levels=levels,
            extend='max',
            vmin=0,
            vmax=120,
        )
        
        add_map_features(ax)
        
        label  = f"{panel_data['peak_wavelength'].values:.1f} $\mu$m"
        
        ax.set_title(label, fontsize=16)

    cbar_axh = fig.add_axes([0.27, 0.10, 0.50, 0.03])
    cbar1 = fig.colorbar(
        im,
        cax=cbar_axh,
        orientation='horizontal',
    )

    cbar1.set_label("PREFIRE Radiance (mWm$^{-2}$cm$^{-1}$sr$^{-1}$)", fontsize=15)

    fig.text(0.82, 0.00, data.time.dt.strftime("%Y-%m").values, fontsize=18)
    
    if save_fig:
        to_png(
            fig,
            f"PREFIRE_polar_{data.time.dt.strftime('%Y%m').values}",
            loc=save_dir,
            bbox_inches="tight",
        )
    plt.close(fig)


def rttov_polarplus_panel_plot(
    data: xr.DataArray,
    timeseries_data: xr.DataArray,
    timeseries_ann_data: xr.DataArray,
    rttov_channels: list,
    levels: np.ndarray = np.linspace(30, 120, 10),
    save_fig: bool = False,
):
    """
    This function creates a plot of polar map of PREFIRE radiances and time series 
    in a lower panel.

    Args:
        data (xr.DataArray): Data to operate on and plot (lat, lon, channel)
        timeseries_data (xr.DataArray): Timeseries data for lower panels
        timeseries_data (xr.DataArray): Annual timeseries data for lower panels
        rttov_channels (list): List of integer indices to identify PREFIRE channels.
        levels (np.ndarray): list or array of levels for the contour plot.
        save_fig (bool, optional): Whether outputs should be saved. Defaults to False.
    Returns:
        None
    """
    num_panels = len(rttov_channels)
    fig = plt.figure(figsize=[3.75 * num_panels, 4])

    gs = gridspec.GridSpec(3, num_panels, figure=fig)
    axs = []
    axs2 = []
    for i in range(num_panels):
        axs.append(fig.add_subplot(gs[:2, i], projection=ccrs.NorthPolarStereo()))
        axs2.append(fig.add_subplot(gs[2, i]))
    fig.subplots_adjust(hspace=0.20, wspace=0.3)
    
    for ax, ax2, _channel in zip(axs, axs2, rttov_channels):
        
        panel_data = data.sel({rttov_dim:_channel})
        
        polarCentral_set_latlim([60, 90], ax)
        
        im = ax.contourf(
            panel_data['lon'],
            panel_data['lat'],
            panel_data,
            transform=ccrs.PlateCarree(),
            #  colors=palette,
            levels=levels,
            extend='max',
            vmin=0,
            vmax=120,
        )
        
        add_map_features(ax)
        
        label = f"{panel_data['peak_wavelength'].values:.1f} $\mu$m"
        
        ax.set_title(label, fontsize=16)
        
        # Line plots of the average value
        _tsubset = timeseries_data.sel(time=slice(None, panel_data.time))

        ax2.plot(
            _tsubset.time,
            _tsubset.sel({rttov_dim:_channel}),
            color="grey",
        )

        _annual_tsubset = timeseries_ann_data.sel(year=slice(None, _tsubset["time.year"][-1] - 1))
        if len(_annual_tsubset) > 0:
            annual_tsubset_time = xr.cftime_range(
                str(_annual_tsubset.year[0].values),
                str(_annual_tsubset.year[-1].values + 1),
                freq="Y",
                closed=None,
            )
            ax2.plot(
                annual_tsubset_time,
                _annual_tsubset.sel({rttov_dim:_channel}),
                linestyle='--',
                marker='o',
                linewidth=1,
                markersize=5,
                color="black",
            )
        
        ax2.set_xlim(
            timeseries_data.time[0].values,
            timeseries_data.time[-1].values,
        )
        ax2.set_xticks(annual_tsubset_time[::5].values)
        ax2.set_xticklabels(annual_tsubset_time[::5].year)

        # if ~np.isnan(timeseries_data).all():
        #     ax2.set_ylim(
        #         timeseries_data.sel({rttov_dim:_channel}).min() * 0.98,
        #         timeseries_data.sel({rttov_dim:_channel}).max() * 1.02,
        #     )

        if np.isnan(timeseries_data).all():
            spread = timeseries_ann_data.sel({rttov_dim:_channel}).max() - timeseries_ann_data.sel({rttov_dim:_channel}).min()
            # mid = 0.5 * (timeseries_ann_data.sel({rttov_dim:_channel}).max() + timeseries_ann_data.sel({rttov_dim:_channel}).min())
            pass
            ax2.set_ylim(
                timeseries_ann_data.sel({rttov_dim:_channel}).min() - spread * 0.15,
                timeseries_ann_data.sel({rttov_dim:_channel}).max() + spread * 0.15,
            )
        else:
            ax2.set_ylim(
                timeseries_data.sel({rttov_dim:_channel}).min() * 0.98,
                timeseries_data.sel({rttov_dim:_channel}).max() * 1.02,
            )

        ax2.set_xlabel("Year", fontsize=15)
    axs2[0].set_ylabel("Radiance", fontsize=15)

    cbar_axh = fig.add_axes([0.27, -0.06, 0.50, 0.03])
    cbar1 = fig.colorbar(
        im,
        cax=cbar_axh,
        orientation='horizontal',
    )

    cbar1.set_label("PREFIRE Radiance (mWm$^{-2}$cm$^{-1}$sr$^{-1}$)", fontsize=15)
    if "time" in data.dims:
        fig.text(0.82, -0.15, data.time.dt.strftime("%Y-%m").values, fontsize=18)
    
    if save_fig:
        if (
            overwrite or
            not os.path.exists(os.path.join(save_dir, f"PREFIRE_polar_{data.time.dt.strftime('%Y%m').values}"))
        ):
            to_png(
                fig,
                f"PREFIRE_polar_{data.time.dt.strftime('%Y%m').values}",
                loc=save_dir,
                bbox_inches="tight",
                dpi=100,
            )
            plt.close(fig)
    else:
        return fig


def fix_fig_facecolor(
    fig: plt.Figure,
):
    """
    Adjusts the face color and transparency of a given Matplotlib figure and its axes.

    Parameters:
        fig (plt.Figure): The Matplotlib figure to be modified.
        This function sets the figure's face color to None and its alpha (transparency) to 0.
        It then iterates over all axes in the figure, setting each axis's face color to white
        and its alpha to 1.

    Returns:
        fig: The modified Matplotlib figure.
    """
    fig.patch.set_facecolor(None)
    fig.patch.set_alpha(0)

    axs = fig.get_axes()

    for ax in axs:
        ax.patch.set_facecolor('white')
        ax.patch.set_alpha(1)

    return fig


# %%

if __name__ == "__main__":

    rttov_channels = [14, 16, 23, 43]

    clear_repartition = xr.open_dataarray("./data_dir/PREFIRE_Polar_clearsky.nc", chunks={"time":1})

    clear_repartition = clear_repartition.sel(RTTOV_CHAN_I001=rttov_channels)

    # Compute timeseries.
    clear_arctic_avg = clear_repartition.sel(lat=slice(60, None)).weighted(clear_repartition.cell_weight).mean(dim=["lat", "lon"])
    clear_arctic_avg = clear_arctic_avg.compute()
    clear_annual_timeseries = clear_arctic_avg.groupby(clear_arctic_avg["time.year"]).mean(dim="time")
    clear_arctic_map_avg = clear_repartition.mean(dim="time").assign_coords(time=clear_repartition["time"][-1])
    del clear_repartition

    total_repartition = xr.open_dataarray("./data_dir/PREFIRE_Polar_totalsky.nc", chunks={"time":1})
    total_arctic_avg = total_repartition.sel(lat=slice(60, None)).weighted(total_repartition.cell_weight).mean(dim=["lat", "lon"])
    total_arctic_avg = total_arctic_avg.compute()
    total_annual_timeseries = total_arctic_avg.groupby(total_arctic_avg["time.year"]).mean(dim="time")
    total_arctic_map_avg = total_repartition.mean(dim="time").assign_coords(time=total_repartition["time"][-1])
    del total_repartition


    # %%
    # Pick more interesting channels:
    # 14: 12.4um peak radiance in the atmospheric window.
    # 16: 14.2um minimum radiance associated with CO2 and the stratosphere.
    # 23: 20.6um peak emission from the dirty window.
    # 43: 36.6um part of an H2O wiggle deep in the FIR? Dynamic range small though.
    rttov_channels = [14, 16, 23, 43]
    rttov_dim = "RTTOV_CHAN_I001"

    # Summary plots
    fig = rttov_polarplus_panel_plot(
        data=clear_arctic_map_avg,
        timeseries_data=clear_arctic_avg,
        timeseries_ann_data=clear_annual_timeseries,
        rttov_channels=rttov_channels,
        save_fig=False,
    )
    fig = fix_fig_facecolor(fig)
    to_png(fig, "PREFIRE_polar_2000_2014_clrsky_summary", dpi=200, ext="pdf", bbox_inches="tight")
    fig = rttov_polarplus_panel_plot(
        data=total_arctic_map_avg,
        timeseries_data=total_arctic_avg,
        timeseries_ann_data=total_annual_timeseries,
        rttov_channels=rttov_channels,
        save_fig=False,
    )
    fig = fix_fig_facecolor(fig)
    to_png(fig, "PREFIRE_polar_2000_2014_allsky_summary", dpi=200, ext="pdf", bbox_inches="tight")

    # %%
    # Only plot annual values for reviewer response.
    rttov_channels = [14, 16, 23, 43]
    rttov_dim = "RTTOV_CHAN_I001"

    # Summary plots
    fig = rttov_polarplus_panel_plot(
        data=clear_arctic_map_avg,
        timeseries_data=clear_arctic_avg.where(total_arctic_avg<0, np.nan),
        timeseries_ann_data=clear_annual_timeseries,
        rttov_channels=rttov_channels,
        save_fig=False,
    )
    fig = fix_fig_facecolor(fig)
    to_png(fig, "PREFIRE_polar_2000_2021_clrsky_annualsummary", dpi=200, ext="pdf", bbox_inches="tight")

    fig = rttov_polarplus_panel_plot(
        data=total_arctic_map_avg,
        timeseries_data=total_arctic_avg.where(total_arctic_avg<0, np.nan),
        timeseries_ann_data=total_annual_timeseries,
        rttov_channels=rttov_channels,
        save_fig=False,
    )
    fig = fix_fig_facecolor(fig)
    to_png(fig, "PREFIRE_polar_2000_2021_allsky_annualsummary", dpi=200, ext="pdf", bbox_inches="tight")

# %%
    # Produce plots for gif.
    for i in clear_repartition.time:
        clear_repartition_month = clear_repartition.sel(time=i)

        rttov_polarplus_panel_plot(
            data=clear_repartition_month,
            timeseries_data=clear_arctic_avg,
            timeseries_ann_data=clear_annual_timeseries,
            rttov_channels=rttov_channels,
            save_fig=False,
        )
        break

    for i in total_repartition.time:
        total_repartition_month = total_repartition.sel(time=i)

        rttov_polarplus_panel_plot(
            data=total_repartition_month,
            timeseries_data=total_arctic_avg,
            timeseries_ann_data=total_annual_timeseries,
            rttov_channels=rttov_channels,
            save_fig=False,
        )
        break
# %%

    # Combine the images as a gif.    
    img_files = glob.glob(f"{save_dir}/{'PREFIRE_polar_??????.png'}")
    img_files.sort()
    
    frames = []
    for i in img_files:
        new_frame = Image.open(i)
        frames.append(new_frame)
        
    frames[0].save(
        os.path.join(save_dir, 'PREFIRE_polar.gif'),
        format='GIF',
        append_images=frames[1:],
        save_all=True,
        duration=200,
        loop=0,
    )
# %%
