"""
Plot CESM2 output and simulated AIRS radiances to establish intuition.
"""
# %%

import numpy as np
import xarray as xr
import os
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib.colors import ListedColormap, BoundaryNorm


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

    # Load processed data
    cesm2_rttov_outs = xr.open_dataset("./data_dir/fig4_cesm2_rttov_outs.nc")
    cesm_out_ds = xr.open_dataset("./data_dir/fig4_cesm2_outs.nc")
    
    # Create summary figure.
    fig, axs = plt.subplots(3, 2, figsize=(12, 12), subplot_kw={'projection': ccrs.Robinson()})

    bt1231 = 0.5 * (cesm2_rttov_outs["rttov_bt_clear_inst001"].isel(wnum=23) + cesm2_rttov_outs["rttov_bt_clear_inst002"].isel(wnum=23))

    # AIRS 1231 cm-1
    ax = axs[0, 0]
    im = ax.pcolormesh(
        cesm2_rttov_outs["lon"],
        cesm2_rttov_outs["lat"],
        bt1231,
        transform=ccrs.PlateCarree(),
        cmap='viridis',
        vmin=215,
        vmax=310,
    )
    ax.set_title("1231cm$^{-1}$ Channel (clear-sky)", fontsize=16)
    cbar = fig.colorbar(im, ax=ax, orientation='horizontal')#, fraction=0.02, pad=0.1)
    cbar.set_label('Brightness Temperature (K)')

    # CESM2 TS
    ax = axs[0, 1]
    im = ax.pcolormesh(
        cesm_out_ds["lon"],
        cesm_out_ds["lat"],
        cesm_out_ds["TS"],
        transform=ccrs.PlateCarree(),
        cmap='viridis',
        vmin=215,
        vmax=310,
    )
    ax.set_title("Surface Temperature", fontsize=16)
    cbar = fig.colorbar(im, ax=ax, orientation='horizontal')#, fraction=0.02, pad=0.1)
    cbar.set_label('Surface Temperature (K)')

    # New Row 2 (mid-troposphere)
    # 741 cm-1 at (weight function peak 366mb (tropical) - 468mb (sub-arctic winter))
    # CO2 sensitivity peaks at 279 mb in the tropical atmosphere
    ax = axs[1, 0]
    im = ax.pcolormesh(
        cesm2_rttov_outs["lon"],
        cesm2_rttov_outs["lat"],
        cesm2_rttov_outs["rttov_bt_clear_inst001"].isel(wnum=6),
        transform=ccrs.PlateCarree(),
        cmap='viridis',
    )
    ax.set_title("741cm$^{-1}$ Channel (clear-sky)", fontsize=16)
    cbar = fig.colorbar(im, ax=ax, orientation='horizontal')#, fraction=0.02, pad=0.1)
    cbar.set_label('Brightness Temperature (K)')

    # CESM2 T at 273.9 hPa
    ax = axs[1, 1]
    im = ax.pcolormesh(
        cesm_out_ds["lon"],
        cesm_out_ds["lat"],
        cesm_out_ds["T"].isel(lev=16),
        transform=ccrs.PlateCarree(),
        cmap='viridis',
    )
    ax.set_title("Air Temperature at 274mb", fontsize=16)
    cbar = fig.colorbar(im, ax=ax, orientation='horizontal')#, fraction=0.02, pad=0.1)
    cbar.set_label('Temperature (K)')

    # Row 3
    # Simplistic cloud mask
    cmap_manual_colors =[
        [0.267004, 0.004874, 0.329415, 1.0,],
        [0.275191, 0.194905, 0.496005, 1.0,],
        [0.212395, 0.359683, 0.55171, 1.0,],
        [0.153364, 0.497, 0.557724, 1.0,],
        [0.122312, 0.633153, 0.530398, 1.0,],
        [0.288921, 0.758394, 0.428426, 1.0,],
        [0.993248, 0.906157, 0.143936, 1.0,],
    ]
    cmap_shared = ListedColormap(cmap_manual_colors)

    # Create separate norms for each variable
    levels_1231 = [-7, -2, 3, 8, 13, 18, 23]
    norm_1231 = BoundaryNorm(levels_1231, ncolors=len(cmap_manual_colors), clip=True)

    levels_cld = [0, 0.16, 0.32, 0.48, 0.64, 0.80, 1]
    norm_cld = BoundaryNorm(levels_cld, ncolors=len(cmap_manual_colors), clip=True)

    extent_limits = (-180, 180, -45, 45)

    simple_cloud = (cesm2_rttov_outs["rttov_bt_clear_inst001"] - cesm2_rttov_outs["rttov_bt_total_inst001"]).isel(wnum=23)
    simple_cloud = xr.concat([simple_cloud, simple_cloud.isel(lon=0)], dim="lon")
    simple_cloud["lon"] = np.append(cesm2_rttov_outs["lon"].values, 360)
    ax = axs[2, 0]
    im = ax.contourf(
        simple_cloud["lon"],
        simple_cloud["lat"],
        simple_cloud,
        transform=ccrs.PlateCarree(),
        cmap=cmap_shared,
        norm=norm_1231,
        levels=levels_1231,
        extend="both",
    )
    ax.set_title("1231cm$^{-1}$ Channel \n (Clear-sky minus All-sky)", fontsize=16)
    ax.set_extent(extent_limits, crs=ccrs.PlateCarree())
    cbar = fig.colorbar(im, ax=ax, orientation='horizontal')#, fraction=0.02, pad=0.1)
    cbar.set_label('Brightness Temperature Difference (K)')

    # CESM2's cloud
    cesm2_cldtot = cesm_out_ds["CLDTOT"]
    cesm2_cldtot = xr.concat([cesm2_cldtot, cesm2_cldtot.isel(lon=0)], dim="lon")
    cesm2_cldtot["lon"] = np.append(cesm_out_ds["lon"].values, 360)
    ax = axs[2, 1]
    im = ax.contourf(
        cesm2_cldtot["lon"],
        cesm2_cldtot["lat"],
        cesm2_cldtot,
        transform=ccrs.PlateCarree(),
        cmap=cmap_shared,
        norm=norm_cld,
        levels=levels_cld,
        extend="neither",
    )
    ax.set_title("Cloud Fraction", fontsize=16)
    ax.set_extent(extent_limits, crs=ccrs.PlateCarree())
    cbar = fig.colorbar(im, ax=ax, orientation='horizontal')#, fraction=0.02, pad=0.1)
    cbar.set_label('Cloud Fraction (0 - 1)')

    fig.tight_layout()

    # Adjust the layout to make the lowest row have the same width as the other axes
    for ax in axs[-1, :]:
        pos = ax.get_position()
        pos.y0 = pos.y0 + 0.02  # Adjust this value as needed
        ax.set_position(pos)
    # Add letter labels to each subplot
    labels = ['a', 'b', 'c', 'd']
    for ax, label in zip(axs.flat, labels):
        ax.text(0.02, 0.98, f'{label}.', transform=ax.transAxes, fontsize=14, verticalalignment='top', weight='bold')

    labels = ['e', 'f']
    for ax, label in zip(axs.flat[-2:], labels):
        ax.text(0.02, 1.2, f'{label}.', transform=ax.transAxes, fontsize=14, verticalalignment='top', weight='bold')
    
    # Add coastlines to all axes
    for ax in axs.flat:
        ax.coastlines()

    to_png(fig, "CESM2_maps_figure_alt", dpi=200, bbox_inches="tight", ext="png")

# %%
