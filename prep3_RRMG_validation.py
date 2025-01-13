"""
This file generates a figure comparing output from the COSP-RTTOV satellite simulator to top-of-atmosphere spectral irradiances produced by RRTMG-LW.

"""

# %%
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import glob
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


def create_iasi_mask():
    """
    Create a boolean mask for which IASI channels fall in which RRTMG-LW bands.
    Also include a scaling ratio related to how many IASI channels we expect
    in the band versus how many there are

    Returns:
        iasi_chans_in_rrtmg_ds: xr.DataArray boolean mask index by IASI channels and RRTMG-LW bands.
    """
    # IASI covers the range of 645-2760 cm-1 using 8461 channels with 0.25cm-1 spacing.
    iasi_srf_midpoints = np.linspace(645, 2760, num=8461)  # ,endpoint=True)

    iasi_srf_midpoints_ds = xr.DataArray(
        data=iasi_srf_midpoints,
        dims=["IASI_channel_number"],
        coords={"IASI_channel_number": np.arange(1, 8461 + 1, 1)},
    )

    # Create a boolean mask for IASI channels being in RRTMG-LW bands
    bands_list = []
    for i, _chan in enumerate(RRTMG_LW_bnds2):

        if (iasi_srf_midpoints_ds[0] > _chan[0]) or (iasi_srf_midpoints_ds[-1] < _chan[1]):
            continue

        _iasi_chans_in_rrtmg_band = np.bitwise_and(
            (iasi_srf_midpoints_ds >= _chan[0]), (iasi_srf_midpoints_ds <= _chan[1])
        )

        _iasi_chans_in_rrtmg_band = (
            _iasi_chans_in_rrtmg_band.rename("IASI_chans")
            .assign_coords({"RRTMG_LW_band": i + 1})
            .expand_dims("RRTMG_LW_band")
        )

        # The scaling factor appears to be a <1% effect!
        _band_width = _chan[1] - _chan[0]
        _num_iasi_channels_in_rrtmg_band = _iasi_chans_in_rrtmg_band.sum()
        _iasi_chans_in_rrtmg_band = _iasi_chans_in_rrtmg_band.assign_attrs(
            scaling_ratio=float((_band_width / 0.25) / _num_iasi_channels_in_rrtmg_band)
        )  # how many iasi channels we expect in the band versus how many there are

        bands_list.append(_iasi_chans_in_rrtmg_band)

    iasi_chans_in_rrtmg_ds = xr.concat(bands_list, dim="RRTMG_LW_band")

    return iasi_chans_in_rrtmg_ds


def create_forum_mask():
    """
    Create a boolean mask for which FORUM channels fall in which RRTMG-LW bands.
    Also include a scaling ratio related to how many FORUM channels we expect
    in the band versus how many there are

    Returns:
        iasi_chans_in_rrtmg_ds: xr.DataArray boolean mask index by FORUM channels and RRTMG-LW bands.
    """

    # FORUM covers 100-1600 cm-1 with 5001 channels.
    forum_srf_midpoints = np.linspace(100, 1600, num=5001)

    forum_srf_midpoints_ds = xr.DataArray(
        data=forum_srf_midpoints,
        dims=["FORUM_channel_number"],
        coords={"FORUM_channel_number": np.arange(1, 5001 + 1, 1)},
    )

    # Create a boolean mask for FORUM channels being in RRTMG-LW bands
    bands_list = []
    for i, _chan in enumerate(RRTMG_LW_bnds2):

        if (forum_srf_midpoints_ds[0] > _chan[0]) or (
            forum_srf_midpoints_ds[-1] < _chan[1]
        ):
            continue

        _forum_chans_in_rrtmg_band = np.bitwise_and(
            (forum_srf_midpoints_ds >= _chan[0]), (forum_srf_midpoints_ds <= _chan[1])
        )

        _forum_chans_in_rrtmg_band = (
            _forum_chans_in_rrtmg_band.rename("FORUM_chans")
            .assign_coords({"RRTMG_LW_band": i + 1})
            .expand_dims("RRTMG_LW_band")
        )

        # The scaling factor appears to be a <1% effect!
        _band_width = _chan[1] - _chan[0]
        _num_forum_channels_in_rrtmg_band = _forum_chans_in_rrtmg_band.sum()
        _forum_chans_in_rrtmg_band = _forum_chans_in_rrtmg_band.assign_attrs(
            scaling_ratio=float((_band_width / 0.3) / _num_forum_channels_in_rrtmg_band)
        )  # how many forum channels we expect in the band versus how many there are

        bands_list.append(_forum_chans_in_rrtmg_band)

    forum_chans_in_rrtmg_ds = xr.concat(bands_list, dim="RRTMG_LW_band")

    return forum_chans_in_rrtmg_ds


def IASI_quadrature(
    scam_rad_output: xr.Dataset,
    quadrature_weights: xr.DataArray,
):

    iasi_allsky_vars = [i for i in iasi_vars if ("total" in i)]
    iasi_clrsky_vars = [i for i in iasi_vars if ("clear" in i)]
    iasi_chan_names = [
        "RTTOV_CHAN_I001",
        "RTTOV_CHAN_I002",
        "RTTOV_CHAN_I003",
        "RTTOV_CHAN_I004",
        "RTTOV_CHAN_I005",
        "RTTOV_CHAN_I006",
    ]

    all_chans = []

    _iasi_chan_width = 0.25  # Units: cm-1

    # Produce masks for aggregating IASI channels into FORUM masks
    iasi_chans_in_rrtmg_ds = create_iasi_mask()

    for _chan_i in iasi_chans_in_rrtmg_ds:

        print(_chan_i.RRTMG_LW_band.values)

        # Simulated IASI data
        _allsky_sum = []
        _clrsky_sum = []

        # Iterate over zenith viewing angles
        for j, (_allsky_name, _clrsky_name, _chan_name) in enumerate(
            zip(iasi_allsky_vars, iasi_clrsky_vars, iasi_chan_names)
        ):
            # Select only IASI channels in the current RRTMG-LW band
            _iasi_chan_all = (
                scam_rad_output_ds[_allsky_name][
                    :, _chan_i.rename({"IASI_channel_number": _chan_name})
                ]
                .squeeze()
                .drop_vars("RRTMG_LW_band")
            )
            _iasi_chan_clr = (
                scam_rad_output_ds[_clrsky_name][
                    :, _chan_i.rename({"IASI_channel_number": _chan_name})
                ]
                .squeeze()
                .drop_vars("RRTMG_LW_band")
            )

            # Weight the radiance values appropriately
            _allsky_sum.append(
                _iasi_chan_all.rename("IASI_rad")
                .assign_coords({"quadpt": j})
                .expand_dims("quadpt")
                .rename({_chan_name: "IASI_channel"})
            )
            _clrsky_sum.append(
                _iasi_chan_clr.rename("IASI_rad")
                .assign_coords({"quadpt": j})
                .expand_dims("quadpt")
                .rename({_chan_name: "IASI_channel"})
            )

        _allsky_rad_ds = xr.concat(_allsky_sum, dim="quadpt")
        _clrsky_rad_ds = xr.concat(_clrsky_sum, dim="quadpt")

        # Convert from radiance to irradiance (1e-3 for mW to W units)
        _rrtmglike_flux_all = (_allsky_rad_ds.sum(dim=["IASI_channel"]) * _iasi_chan_width * 1e-3 * np.pi)
        _rrtmglike_flux_clr = (_clrsky_rad_ds.sum(dim=["IASI_channel"]) * _iasi_chan_width * 1e-3 * np.pi)

        # Convert from radiance to irradiance
        _rrtmglike_flux_all.name = "flux_allsky"
        _rrtmglike_flux_clr.name = "flux_clrsky"

        _vals = xr.merge(
            [
                _rrtmglike_flux_all.assign_coords({"RRTMG_chan": int(_chan_i.RRTMG_LW_band)}).expand_dims("RRTMG_chan"),
                _rrtmglike_flux_clr.assign_coords({"RRTMG_chan": int(_chan_i.RRTMG_LW_band)}).expand_dims("RRTMG_chan"),
            ]
        )
        all_chans.append(_vals)

    all_chans_iasi_ds = xr.concat(all_chans, dim="RRTMG_chan")

    # Compute the cloud radiative effect.
    all_chans_iasi_ds["flux_cre"] = (
        all_chans_iasi_ds["flux_clrsky"] - all_chans_iasi_ds["flux_allsky"]
    )
    # Actually compute the quadrature over zenith viewing angle.
    quadrature = (all_chans_iasi_ds * quadrature_weights).sum(dim="quadpt")

    return quadrature, all_chans_iasi_ds


def FORUM_quadrature(
    scam_rad_output: xr.Dataset,
    quadrature_weights: xr.DataArray,
):

    _forum_chan_width = 0.3  # Units: cm-1

    forum_allsky_vars = [i for i in forum_vars if ("total" in i)]
    forum_clrsky_vars = [i for i in forum_vars if ("clear" in i)]
    forum_chan_names = [
        "RTTOV_CHAN_I007",
        "RTTOV_CHAN_I008",
        "RTTOV_CHAN_I009",
        "RTTOV_CHAN_I010",
        "RTTOV_CHAN_I011",
        "RTTOV_CHAN_I012",
    ]

    forum_chans_in_rrtmg_ds = create_forum_mask()

    all_chans = []
    for _chan_i in forum_chans_in_rrtmg_ds:

        print(_chan_i.RRTMG_LW_band.values)

        # Simulated FORUM data
        _allsky_sum = []
        _clrsky_sum = []
        for j, (_allsky_name, _clrsky_name, _chan_name) in enumerate(
            zip(forum_allsky_vars, forum_clrsky_vars, forum_chan_names)
        ):

            _forum_chan_all = (
                scam_rad_output[_allsky_name][:, _chan_i.rename({"FORUM_channel_number": _chan_name})]
                .squeeze()
                .drop_vars("RRTMG_LW_band")
            )
            _forum_chan_clr = (
                scam_rad_output[_clrsky_name][:, _chan_i.rename({"FORUM_channel_number": _chan_name})]
                .squeeze()
                .drop_vars("RRTMG_LW_band")
            )

            # Weight the radiance values appropriately
            _allsky_sum.append(
                _forum_chan_all.rename("FORUM_rad")
                .assign_coords({"quadpt": j})
                .expand_dims("quadpt")
                .rename({_chan_name: "FORUM_channel"})
            )
            _clrsky_sum.append(
                _forum_chan_clr.rename("FORUM_rad")
                .assign_coords({"quadpt": j})
                .expand_dims("quadpt")
                .rename({_chan_name: "FORUM_channel"})
            )

        _allsky_rad_ds = xr.concat(_allsky_sum, dim="quadpt")
        _clrsky_rad_ds = xr.concat(_clrsky_sum, dim="quadpt")

        # Convert from radiance to irradiance (1e-3 for mW to W units)
        _rrtmglike_flux_all = (_allsky_rad_ds.sum(dim=["FORUM_channel"]) * _forum_chan_width * 1e-3 * np.pi)
        _rrtmglike_flux_clr = (_clrsky_rad_ds.sum(dim=["FORUM_channel"]) * _forum_chan_width * 1e-3 * np.pi)

        # Convert from radiance to irradiance
        _rrtmglike_flux_all.name = "flux_allsky"
        _rrtmglike_flux_clr.name = "flux_clrsky"

        _vals = xr.merge(
            [
                _rrtmglike_flux_all.assign_coords({"RRTMG_chan": int(_chan_i.RRTMG_LW_band)}).expand_dims("RRTMG_chan"),
                _rrtmglike_flux_clr.assign_coords({"RRTMG_chan": int(_chan_i.RRTMG_LW_band)}).expand_dims("RRTMG_chan"),
            ]
        )

        all_chans.append(_vals)

    all_chans_forum_ds = xr.concat(all_chans, dim="RRTMG_chan")

    all_chans_forum_ds["flux_cre"] = (
        all_chans_forum_ds["flux_clrsky"]
        - all_chans_forum_ds["flux_allsky"]
    )

    quadrature = (all_chans_forum_ds * quadrature_weights).sum(dim="quadpt")

    return quadrature, all_chans_forum_ds


def get_RRTMG_LW_spectralflux(
    scam_rad_output: xr.Dataset,
):

    rrtmg_channel_indices = np.arange(1, 17, 1)
    _rrtmg_chan_all = (
        scam_rad_output_ds["LU"]
        .isel(ilev=0)
        .squeeze()
    )
    _rrtmg_chan_all["lw_band"] = rrtmg_channel_indices
    _rrtmg_chan_all = _rrtmg_chan_all.rename({"lw_band": "RRTMG_chan"}).rename("flux_allsky")

    _rrtmg_chan_clr = (
        scam_rad_output_ds["LUC"]
        .isel(ilev=0)
        .squeeze()
    )
    _rrtmg_chan_clr["lw_band"] = rrtmg_channel_indices
    _rrtmg_chan_clr = _rrtmg_chan_clr.rename({"lw_band": "RRTMG_chan"}).rename("flux_clrsky")

    rrtmg_chan_out = xr.merge([
        _rrtmg_chan_all,
        _rrtmg_chan_clr,
    ])
    rrtmg_chan_out["flux_cre"] = rrtmg_chan_out["flux_clrsky"] - rrtmg_chan_out["flux_allsky"]

    return rrtmg_chan_out


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

# The RRTMG band are:
# 16 RRTMG-LW bands (in cm-1): 10-350, 350-500, 500-630, 630-700, 700-820, 820-980, 980-1080,1080-1180, 1180-1390, 1390-1480, 1480-1800, 1800-2080, 2080-2250,2250-2380, 2380-2600, 2600-3000
# 14 RRTMG-SW bands (in cm-1): 2600-3250, 3250-4000, 4000-4650, 4650-5150, 5150-6150, 6150-7700, 7700-8050,8050-12850, 12850-16000, 16000-22650, 22650-29000, 29000-38000, 38000-50000,820-2600

RRTMG_LW_bnds2 = np.array(
    [
        [10, 350],
        [350, 500],
        [500, 630],
        [630, 700],
        [700, 820],
        [820, 980],
        [980, 1080],
        [1080, 1180],
        [1180, 1390],
        [1390, 1480],
        [1480, 1800],
        [1800, 2080],
        [2080, 2250],
        [2250, 2380],
        [2380, 2600],
        [2600, 3000],
    ]
)
RRTMG_LW_bnds = np.array([*RRTMG_LW_bnds2[:, 0], RRTMG_LW_bnds2[-1, -1]])

# Define quadrature weights and angles
viewing_zenith_angles = [7.19, 21.58, 35.97, 50.34, 64.71, 78.98]

quadrature_weights = [0.24915, 0.23349, 0.20317, 0.16008, 0.10694, 0.04718]
weights_da = xr.DataArray(
    data=quadrature_weights,
    dims=['quadpt'],
    coords={'quadpt': np.arange(0, 6, 1)},
)

rad_vars = ["FLUT", "FLDS", "FLUTC", "LU", "LUC", "LD", "LDC"]
rttov_vars = [
    "rttov_rad_clear_inst001",
    "rttov_rad_total_inst001",
    "rttov_rad_clear_inst002",
    "rttov_rad_total_inst002",
    "rttov_rad_clear_inst003",
    "rttov_rad_total_inst003",
    "rttov_rad_clear_inst004",
    "rttov_rad_total_inst004",
    "rttov_rad_clear_inst005",
    "rttov_rad_total_inst005",
    "rttov_rad_clear_inst006",
    "rttov_rad_total_inst006",
    "rttov_rad_clear_inst007",
    "rttov_rad_total_inst007",
    "rttov_rad_clear_inst008",
    "rttov_rad_total_inst008",
    "rttov_rad_clear_inst009",
    "rttov_rad_total_inst009",
    "rttov_rad_clear_inst010",
    "rttov_rad_total_inst010",
    "rttov_rad_clear_inst011",
    "rttov_rad_total_inst011",
    "rttov_rad_clear_inst012",
    "rttov_rad_total_inst012",
]

iasi_vars = [
    "rttov_rad_clear_inst001",
    "rttov_rad_total_inst001",
    "rttov_rad_clear_inst002",
    "rttov_rad_total_inst002",
    "rttov_rad_clear_inst003",
    "rttov_rad_total_inst003",
    "rttov_rad_clear_inst004",
    "rttov_rad_total_inst004",
    "rttov_rad_clear_inst005",
    "rttov_rad_total_inst005",
    "rttov_rad_clear_inst006",
    "rttov_rad_total_inst006",
]

forum_vars = [
    "rttov_rad_clear_inst007",
    "rttov_rad_total_inst007",
    "rttov_rad_clear_inst008",
    "rttov_rad_total_inst008",
    "rttov_rad_clear_inst009",
    "rttov_rad_total_inst009",
    "rttov_rad_clear_inst010",
    "rttov_rad_total_inst010",
    "rttov_rad_clear_inst011",
    "rttov_rad_total_inst011",
    "rttov_rad_clear_inst012",
    "rttov_rad_total_inst012",
]

microp_vars = [
    'CLOUD',
    'CLDTOT',
    'CLDLIQ',
    'CLDICE',
    'CLDTOT_CAL',
    'CLDTOT_CAL_LIQ',
    'CLDTOT_CAL_ICE',
    # 'ICIMR',
    # 'ICWMR',
    # 'RCM_CLUBB',
    # 'AREL',
    # 'AREI',
    'TMQ',
    'TS',
]

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

# Iterate over cases and compute the fluxes.
save_dir = "/glade/u/home/jonahshaw/Scripts/git_repos/COSP-RTTOV_diageval/data/"
rrtmg_lw_flux_list = []
iasi_flux_list = []
forum_flux_list = []

for i in scam_testcases:

    if (
        os.path.exists(f"{save_dir}/SCAM_rrtmg_lw_fluxes_{i}_214.nc") and
        os.path.exists(f"{save_dir}/SCAM_IASI_fluxes_{i}_214.nc") and
        os.path.exists(f"{save_dir}/SCAM_FORUM_fluxes_{i}_214.nc")):
        print("Data for %s already exists." % i)
        continue

    # Identify output filepaths
    scam_testcase_name = scam_testcases[i][0]

    # Is the first file always the longest?
    scam_output_dir = f"/glade/derecho/scratch/jonahshaw/archive/{scam_testcase_name}/atm/hist/"

    # Sorting and then taking the first file seems to consistently give the longest run.
    scam_output_files = glob.glob("%s/*.nc" % scam_output_dir)
    scam_output_files.sort()
    print(i)
    print(scam_output_files[0])
    scam_output_ds = xr.open_dataset(scam_output_files[0])

    scam_meteo_output_ds = scam_output_ds[microp_vars].isel(time=slice(4, None, 3)).squeeze()
    scam_meteo_output_ds.to_netcdf(f"{save_dir}/SCAM_meteo_{i}_214.nc")
    del scam_meteo_output_ds

    scam_rad_output_ds = (
        (scam_output_ds[rad_vars + rttov_vars]).isel(time=slice(4, None, 3)).squeeze()
    )
    del scam_output_ds

    # Get RRTMG-LW for comparison
    rrtmg_lw_fluxes = get_RRTMG_LW_spectralflux(
        scam_rad_output_ds,
    )
    rrtmg_lw_fluxes = rrtmg_lw_fluxes.assign_coords({"case": i}).expand_dims(i)

    # Perform IASI quadrature
    iasi_fluxes, iasi_angles = IASI_quadrature(
        scam_rad_output_ds,
        weights_da,
    )
    iasi_fluxes = iasi_fluxes.assign_coords({"case": i}).expand_dims(i)

    # Perform FORUM quadrature
    forum_fluxes, forum_angles = FORUM_quadrature(
        scam_rad_output_ds,
        weights_da,
    )
    forum_fluxes = forum_fluxes.assign_coords({"case": i}).expand_dims(i)

    rrtmg_lw_fluxes.to_netcdf(f"{save_dir}/SCAM_rrtmg_lw_fluxes_{i}_214.nc")
    iasi_fluxes.to_netcdf(f"{save_dir}/SCAM_IASI_fluxes_{i}_214.nc")
    forum_fluxes.to_netcdf(f"{save_dir}/SCAM_FORUM_fluxes_{i}_214.nc")

    del scam_rad_output_ds, rrtmg_lw_fluxes, iasi_fluxes, forum_fluxes

# %%