"""
This file generates a figure comparing output from the COSP-RTTOV satellite simulator to top-of-atmosphere spectral irradiances produced by RRTMG-LW.

"""

# %%
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import os
import seaborn as sns

# %%

# Define functions


def to_png(
    file, filename, loc="/glade/u/home/jonahshaw/figures/", dpi=200, ext="png", **kwargs
):
    """
    Simple function for one-line saving.
    Saves to "/glade/u/home/jonahshaw/figures" by default
    """
    output_dir = loc
    # ext = 'png'
    full_path = "%s%s.%s" % (output_dir, filename, ext)

    if not os.path.exists(output_dir + filename):
        file.savefig(full_path, format=ext, dpi=dpi, **kwargs)
    #         file.clf()

    else:
        print("File already exists, rename or delete.")


def error_scatterplot(
    ax,
    error_data,
    label: str = None,
    color: str = "blue",
    marker: str = "o",
):

    ax.scatter(
        error_data.RRTMG_chan,
        error_data,
        color=color,
        marker=marker,
        label=label,
    )

    ax.set_xticks(error_data.RRTMG_chan)
    ax.set_xlabel("RRTMG-LW Channel", fontsize=15)


# %%

# List test cases for SCAM
scam_testcases = {
    "scam_arm97": [
        "20240906_150155.FSCAM.T42_T42.scam_arm97.cesm2.1.5_rttov",
        "Land convection",
    ],
    "scam_cgilsS6": [
        "20240906_151644.FSCAM.T42_T42.scam_cgilsS6.cesm2.1.5_rttov",
        "Shallow cumulus",
    ],
    "scam_cgilsS12": [
        "20240906_155245.FSCAM.T42_T42.scam_cgilsS12.cesm2.1.5_rttov",
        "Stratus",
    ],
    "scam_cgilsS11": [
        "20240906_170536.FSCAM.T42_T42.scam_cgilsS11.cesm2.1.5_rttov",
        "Stratocumulus",
    ],
    "scam_twp06": [
        "20240909_095510.FSCAM..scam_twp06.cesm2.1.5_rttov",
        "Tropical convection",
    ],
    "scam_mpace": [
        "20240909_144544.FSCAM.T42_T42.scam_mpace.cesm2.1.5_rttov",
        "Arctic",
    ],
    "scam_sparticus": [
        "20240909_145117.FSCAM.T42_T42.scam_sparticus.cesm2.1.5_rttov",
        "Cirrus",
    ],
}

# %%

# Create all-sky plots
save_dir = "./data_dir/"

fig, axs = plt.subplots(1, 2, figsize=(12, 5))
fig.subplots_adjust(wspace=0.3)

colors = sns.color_palette("colorblind")

for i, _color in zip(scam_testcases, colors):

    rrtmg_lw_fluxes = xr.open_dataset(f"{save_dir}/SCAM_rrtmg_lw_fluxes_{i}_214.nc")
    iasi_fluxes = xr.open_dataset(f"{save_dir}/SCAM_IASI_fluxes_{i}_214.nc")
    forum_fluxes = xr.open_dataset(f"{save_dir}/SCAM_FORUM_fluxes_{i}_214.nc")

    # Merge, taking FORUM for RRTMG-LW channels 2-4 and IASI for 5-15
    # Compute broadband values
    rrtmg_comp_fluxes = rrtmg_lw_fluxes.sel(RRTMG_chan=slice(2,15))
    rttov_joint_fluxes = xr.merge([forum_fluxes.sel(RRTMG_chan=slice(2,4)), iasi_fluxes])

    # Sum over all bands.
    all_chan_rrtmg = rrtmg_comp_fluxes.sum(dim="RRTMG_chan")
    rrtmg_comp_fluxes = xr.merge([rrtmg_comp_fluxes, all_chan_rrtmg.assign_coords(RRTMG_chan=0).expand_dims("RRTMG_chan")])
    all_chan_rttov = rttov_joint_fluxes.sum(dim="RRTMG_chan")
    rttov_joint_fluxes = xr.merge([rttov_joint_fluxes, all_chan_rttov.assign_coords(RRTMG_chan=0).expand_dims("RRTMG_chan")])

    # Compute error statistics. Compute mean error (more appropriate for clouds given the sampling), 
    # then normalize mean error by the average value
    joint_flux_error = rttov_joint_fluxes - rrtmg_comp_fluxes
    joint_flux_error_mean = joint_flux_error.mean(dim="time")
    joint_flux_error_mean_scaled = joint_flux_error_mean / rrtmg_comp_fluxes.mean(dim="time")

    # Plot the mean absolute and scaled errors for all-sky
    error_scatterplot(
        axs[0],
        joint_flux_error_mean["flux_allsky"].where(joint_flux_error_mean_scaled["RRTMG_chan"] != 14),
        i[5:],
        color=_color,
        marker="*",
    )
    axs[0].hlines(0, -0.5, 15.5, color="black", linestyle="--", zorder=0, linewidth=1, alpha=0.5)
    axs[0].set_ylabel("All-sky Irradiance Error (Wm$^{-2}$)", fontsize=15)
    axs[0].set_xlim(-0.5, 15.5)
    axs[0].legend()
    axs[0].set_xticklabels(["All"] + [str(i) for i in np.arange(2,16,1)])
    axs[0].tick_params(axis='both', labelsize=12)

    error_scatterplot(
        axs[1],
        joint_flux_error_mean_scaled["flux_allsky"].where(joint_flux_error_mean_scaled["RRTMG_chan"] != 14),
        i,
        color=_color,
        marker="*",
    )
    axs[1].hlines(0, -0.5, 15.5, color="black", linestyle="--", zorder=0, linewidth=1, alpha=0.5)
    axs[1].set_ylabel("Fractional All-sky Irradiance Error", fontsize=15)
    axs[1].set_ylim(-0.1, 0.1)
    axs[1].set_xlim(-0.5, 15.5)
    axs[1].set_xticklabels(["All"] + [str(i) for i in np.arange(2, 16, 1)])
    axs[1].tick_params(axis='both', labelsize=12)

to_png(fig, "SCAM_allsky_flux_error", ext="pdf", bbox_inches="tight")

# %%

# Create clear-sky plots
save_dir = "./data_dir/"

fig, axs = plt.subplots(1, 2, figsize=(12, 5))
fig.subplots_adjust(wspace=0.3)

colors = sns.color_palette("colorblind")

for i, _color in zip(scam_testcases, colors):

    rrtmg_lw_fluxes = xr.open_dataset(f"{save_dir}/SCAM_rrtmg_lw_fluxes_{i}_214.nc")
    iasi_fluxes = xr.open_dataset(f"{save_dir}/SCAM_IASI_fluxes_{i}_214.nc")
    forum_fluxes = xr.open_dataset(f"{save_dir}/SCAM_FORUM_fluxes_{i}_214.nc")

    # Merge, taking FORUM for RRTMG-LW channels 2-4 and IASI for 5-15
    # Compute broadband values
    rrtmg_comp_fluxes = rrtmg_lw_fluxes.sel(RRTMG_chan=slice(2,15))
    rttov_joint_fluxes = xr.merge([forum_fluxes.sel(RRTMG_chan=slice(2,4)), iasi_fluxes])

    # Sum over all bands.
    all_chan_rrtmg = rrtmg_comp_fluxes.sum(dim="RRTMG_chan")
    rrtmg_comp_fluxes = xr.merge([rrtmg_comp_fluxes, all_chan_rrtmg.assign_coords(RRTMG_chan=0).expand_dims("RRTMG_chan")])
    all_chan_rttov = rttov_joint_fluxes.sum(dim="RRTMG_chan")
    rttov_joint_fluxes = xr.merge([rttov_joint_fluxes, all_chan_rttov.assign_coords(RRTMG_chan=0).expand_dims("RRTMG_chan")])

    # Compute error statistics. Compute mean error (more appropriate for clouds given the sampling), 
    # then normalize mean error by the average value
    joint_flux_error = rttov_joint_fluxes - rrtmg_comp_fluxes
    joint_flux_error_mean = joint_flux_error.mean(dim="time")
    joint_flux_error_mean_scaled = joint_flux_error_mean / rrtmg_comp_fluxes.mean(dim="time")

    # Plot the mean absolute and scaled errors for all-sky
    error_scatterplot(
        axs[0],
        joint_flux_error_mean["flux_clrsky"].where(joint_flux_error_mean_scaled["RRTMG_chan"] != 14),
        i[5:],
        color=_color,
        marker="*",
    )
    axs[0].hlines(0, -0.5, 15.5, color="black", linestyle="--", zorder=0, linewidth=1, alpha=0.5)
    axs[0].set_ylabel("Clear-sky Irradiance Error (Wm$^{-2}$)", fontsize=15)
    axs[0].set_xlim(-0.5, 15.5)
    axs[0].legend()
    axs[0].set_xticklabels(["All"] + [str(i) for i in np.arange(2,16,1)])
    axs[0].tick_params(axis='both', labelsize=12)

    error_scatterplot(
        axs[1],
        joint_flux_error_mean_scaled["flux_clrsky"].where(joint_flux_error_mean_scaled["RRTMG_chan"] != 14),
        i,
        color=_color,
        marker="*",
    )
    axs[1].hlines(0, -0.5, 15.5, color="black", linestyle="--", zorder=0, linewidth=1, alpha=0.5)
    axs[1].set_ylabel("Fractional Clear-sky Irradiance Error", fontsize=15)
    axs[1].set_ylim(-0.1, 0.1)
    axs[1].set_xlim(-0.5, 15.5)
    axs[1].set_xticklabels(["All"] + [str(i) for i in np.arange(2, 16, 1)])
    axs[1].tick_params(axis='both', labelsize=12)
    
to_png(fig, "SCAM_clearsky_flux_error", ext="pdf", bbox_inches="tight")

# %%