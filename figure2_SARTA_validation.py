"""
Validate COSP-RTTOV against SARTA.    
    
"""

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import scipy.io
import os


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

# Load SARTA
sarta_out = "./data_dir/sarta_333profiles_vmr.mat"
sarta_hdf = scipy.io.loadmat(sarta_out)

# Write to xarray dataset
l1c_indices = np.arange(1, 2645 + 1)

wnum_da = xr.DataArray(sarta_hdf["fx"][:,0], dims=["wnum"], coords={"wnum": sarta_hdf["fx"][:,0]})
wnum_da.name="freq"

da_list = [wnum_da]
for _var in ["rx", "tx"]:
    da = xr.DataArray(sarta_hdf[_var][:,:], dims=["wnum", "time"], coords={"wnum": wnum_da, "time": np.arange(0, 333)})
    da.name = _var
    da_list.append(da)
    
sarta_ds = xr.merge(da_list)

# Load SCAM output
# Created data subset with:
# ncks -v rttov_bt_clear_inst001,rttov_rad_clear_inst001 /glade/derecho/scratch/jonahshaw/archive/20241028_112152.FSCAM.T42_T42.scam_arm95.cesm2.1.5_rttov/atm/hist/20241028_112152.FSCAM.T42_T42.scam_arm95.cesm2.1.5_rttov.cam.h0.1995-07-18-19800.nc data_dir/SCAM_sarta_subset.nc
scam_sarta_file = "./data_dir/SCAM_sarta_subset.nc"
scam_ds = xr.open_dataset(scam_sarta_file).squeeze()
# Select only timesteps when RTTOV is run (hourly, every 3 time steps)
rttov_outs_ds = scam_ds.isel(time=slice(4, None, 3)).rename({"RTTOV_CHAN_I001":"wnum"})
rttov_outs_ds["wnum"] = sarta_ds.wnum

# Select specific variables
data_rttov = rttov_outs_ds["rttov_bt_clear_inst001"]
data_sarta = sarta_ds["tx"]


# %%
# Create BT evaluation plotfig = plt.figure(figsize=(15, 5))
fig = plt.figure(figsize=(15, 5))

ax1 = plt.subplot(121)
ax2 = plt.subplot(222)
ax3 = plt.subplot(224)

ax = ax1

data_rttov = rttov_outs_ds["rttov_bt_clear_inst001"]
ax.plot(
    data_rttov.wnum,
    data_rttov.mean(dim="time"),
    label="COSP-RTTOV",
    linewidth=0.5,
)

data_sarta = sarta_ds["tx"]
data_sarta["time"] = data_rttov.time
ax.plot(
    data_sarta.wnum,
    data_sarta.mean(dim="time"),
    label="SARTA",
    linewidth=0.5,
)
ax.set_xlim(649, 2666)
ax.set_xlabel("Wavenumber (cm$^{-1}$)", fontsize=14)
ax.set_ylabel("Brightness Temperature (deg. K)", fontsize=14)
ax.tick_params(axis="both", labelsize=14)

ax = ax2

diff = data_rttov - data_sarta
ax.plot(
    diff.wnum,
    diff.mean(dim="time"),
    label="COSP-RTTOV minus SARTA",
)
ax.hlines(0, 649, 2666, linestyles="dashed", colors="black")
ax.set_xlim(649, 2666)
ax.set_ylim(-1.5, 5)
ax.set_xlabel("Wavenumber (cm$^{-1}$)", fontsize=14)
ax.set_ylabel("Difference \n (deg. K)", fontsize=14)
ax.tick_params(axis="both", labelsize=14)

ax = ax3
error_std = diff.std(dim="time")
ax.plot(
    error_std.wnum,
    error_std,
    # label="COSP-RTTOV minus SARTA",
)
ax.set_xlim(649, 2666)
# ax.set_ylim(-1.5, 5)
ax.set_xlabel("Wavenumber (cm$^{-1}$)", fontsize=14)
ax.set_ylabel("Difference $\sigma$ \n (deg. K)", fontsize=14)
ax.tick_params(axis="both", labelsize=14)

ax1.legend()
ax2.legend()

ax1.annotate("a.", xy=(0.01, 0.95), xycoords="axes fraction", fontsize=14)
ax2.annotate("b.", xy=(0.01, 0.88), xycoords="axes fraction", fontsize=14)
ax3.annotate("c.", xy=(0.01, 0.88), xycoords="axes fraction", fontsize=14)

to_png(fig, "sarta_rttov_comparison_BT", dpi=200, bbox_inches="tight", ext="pdf")


# %%
# Repeat for radiances
fig = plt.figure(figsize=(15, 5))

ax1 = plt.subplot(121)
ax2 = plt.subplot(222)
ax3 = plt.subplot(224)

ax = ax1

data_rttov = rttov_outs_ds["rttov_rad_clear_inst001"]
ax.plot(
    data_rttov.wnum,
    data_rttov.mean(dim="time"),
    label="COSP-RTTOV",
    linewidth=0.5,
)

data_sarta = sarta_ds["rx"]
data_sarta["time"] = data_rttov.time
ax.plot(
    data_sarta.wnum,
    data_sarta.mean(dim="time"),
    label="SARTA",
    linewidth=0.5,
)
ax.set_xlim(649, 2666)
ax.set_xlabel("Wavenumber (cm$^{-1}$)", fontsize=14)
ax.set_ylabel("Radiance (mW/cm-1/sr/m2)", fontsize=14)
ax.tick_params(axis="both", labelsize=14)

ax = ax2

diff = data_rttov - data_sarta
ax.plot(
    diff.wnum,
    diff.mean(dim="time"),
    label="COSP-RTTOV minus SARTA",
)
ax.hlines(0, 649, 2666, linestyles="dashed", colors="black")
ax.set_xlim(649, 2666)
ax.set_ylim(-1.5, 5)
ax.set_xlabel("Wavenumber (cm$^{-1}$)", fontsize=14)
ax.set_ylabel("Difference \n (mW/cm-1/sr/m2)", fontsize=14)
ax.tick_params(axis="both", labelsize=14)

ax = ax3
error_std = diff.std(dim="time")
ax.plot(
    error_std.wnum,
    error_std,
    # label="COSP-RTTOV minus SARTA",
)
ax.set_xlim(649, 2666)
# ax.set_ylim(-1.5, 5)
ax.set_xlabel("Wavenumber (cm$^{-1}$)", fontsize=14)
ax.set_ylabel("Difference $\sigma$ \n (mW/cm-1/sr/m2)", fontsize=14)
ax.tick_params(axis="both", labelsize=14)

ax1.legend()
ax2.legend()

ax1.annotate("a.", xy=(0.01, 0.95), xycoords="axes fraction", fontsize=14)
ax2.annotate("b.", xy=(0.02, 0.88), xycoords="axes fraction", fontsize=14)
ax3.annotate("c.", xy=(0.02, 0.88), xycoords="axes fraction", fontsize=14)

to_png(fig, "sarta_rttov_comparison_radiance", dpi=200, bbox_inches="tight", ext="pdf")