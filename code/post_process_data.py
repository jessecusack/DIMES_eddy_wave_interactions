#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# %% [markdown]
# # Process the raw mooring data
#
# Contents:
# * <a href=#raw>Raw data reprocessing.</a>
# * <a href=#corrected>Interpolated data processing.</a>
# * <a href=#ADCP>ADCP processing.</a>
# * <a href=#VMP>VMP processing.</a>
#
# Import the needed libraries.

# %%
import datetime
import glob
import os

import gsw
import numpy as np
import numpy.ma as ma
import scipy.integrate as igr
import scipy.interpolate as itpl
import scipy.io as io
import scipy.signal as sig
import seawater
import xarray as xr
from matplotlib import path
import munch

import load_data
import moorings as moo
import utils
from oceans.sw_extras import gamma_GP_from_SP_pt

# Data directory
data_in = os.path.expanduser("../data")
data_out = data_in


def esum(ea, eb):
    return np.sqrt(ea ** 2 + eb ** 2)


def emult(a, b, ea, eb):
    return np.abs(a * b) * np.sqrt((ea / a) ** 2 + (eb / b) ** 2)


# %% [markdown]
# <a id="raw"></a>

# %% [markdown]
# ## Process raw data into a more convenient format
#
# Parameters for raw processing.

# %%
# Corrected levels.
# heights = [-540., -1250., -2100., -3500.]
# Filter cut off (hours)
tc_hrs = 40.0
# Start of time series (matlab datetime)
t_start = 734494.0
# Length of time series
max_len = N_data = 42048
# Data file
raw_data_file = "moorings.mat"
# Index where NaNs start in u and v data from SW mooring
sw_vel_nans = 14027
# Sampling period (minutes)
dt_min = 15.0
# Window length for wave stress quantities and mesoscale strain quantities.
nperseg = 2 ** 9
# Spectra parameters
window = "hanning"
detrend = "constant"
# Extrapolation/interpolation limit above which data will be removed.
dzlim = 100.0
# Integration of spectra parameters. These multiple N and f respectively to set
# the integration limits.
fhi = 1.0
flo = 1.0
flov = 1.0  # When integrating spectra involved in vertical fluxes, get rid of
# the near inertial portion.
# When bandpass filtering windowed data use these params multiplied by f and N
filtlo = 0.9  # times f
filthi = 1.1  # times N

# Interpolation distance that raises flag (m)
zimax = 100.0

dt_sec = dt_min * 60.0  # Sample period in seconds.
dt_day = dt_sec / 86400.0  # Sample period in days.
N_per_day = int(1.0 / dt_day)  # Samples per day.

# %% ############### PROCESS RAW DATA #########################################
print("RAW DATA")
###############################################################################
# Load w data for cc mooring and chop from text files. I checked and all the
# data has the same start date and the same length
print("Loading vertical velocity data from text files.")
nortek_files = glob.glob(os.path.join(data_in, "cc_1_*.txt"))

depth = []
for file in nortek_files:
    with open(file, "r") as f:
        content = f.readlines()
        depth.append(int(content[3].split("=")[1].split()[0]))

idxs = np.argsort(depth)

w = np.empty((42573, 12))
datenum = np.empty((42573, 12))

for i in idxs:
    YY, MM, DD, hh, W = np.genfromtxt(
        nortek_files[i], skip_header=12, usecols=(0, 1, 2, 3, 8), unpack=True
    )
    YY = YY.astype(int)
    MM = MM.astype(int)
    DD = DD.astype(int)
    mm = (60 * (hh % 1)).astype(int)
    hh = np.floor(hh).astype(int)
    w[:, i] = W / 100
    dates = []
    for j in range(len(YY)):
        dates.append(datetime.datetime(YY[j], MM[j], DD[j], hh[j], mm[j]))
    dates = np.asarray(dates)
    datenum[:, i] = utils.datetime_to_datenum(dates)

idx_start = np.searchsorted(datenum[:, 0], t_start)
w = w[idx_start : idx_start + max_len]

# Start prepping raw data from the mat file.
print("Loading raw data file.")
data_path = os.path.join(data_in, raw_data_file)
ds = utils.loadmat(data_path)

cc = ds.pop("c")
nw = ds.pop("nw")
ne = ds.pop("ne")
se = ds.pop("se")
sw = ds.pop("sw")

cc["id"] = "cc"
nw["id"] = "nw"
ne["id"] = "ne"
se["id"] = "se"
sw["id"] = "sw"

moorings = [cc, nw, ne, se, sw]

# Useful information
dt_min = 15.0  # Sample period in minutes.
dt_sec = dt_min * 60.0  # Sample period in seconds.
dt_day = dt_sec / 86400.0  # Sample period in days.

print("Chopping time series.")
for m in moorings:
    m["idx_start"] = np.searchsorted(m["Dates"], t_start)

for m in moorings:
    m["N_data"] = max_len
    m["idx_end"] = m["idx_start"] + max_len

# Chop data to start and end dates.
varl = ["Dates", "Temp", "Sal", "u", "v", "Pres"]
for m in moorings:
    for var in varl:
        m[var] = m[var][m["idx_start"] : m["idx_end"], ...]


print("Renaming variables.")
print("Interpolating negative pressures.")
for m in moorings:
    __, N_levels = m["Pres"].shape
    m["N_levels"] = N_levels

    # Tile time and pressure
    m["t"] = np.tile(m.pop("Dates")[:, np.newaxis], (1, N_levels))

    # Fix negative pressures by interpolating nearby data.
    fix = m["Pres"] < 0.0
    if fix.any():
        levs = np.argwhere(np.any(fix, axis=0))[0]
        for lev in levs:
            x = m["t"][fix[:, lev], lev]
            xp = m["t"][~fix[:, lev], lev]
            fp = m["Pres"][~fix[:, lev], lev]
            m["Pres"][fix[:, lev], lev] = np.interp(x, xp, fp)

    # Rename variables
    m["P"] = m.pop("Pres")
    m["u"] = m["u"] / 100.0
    m["v"] = m["v"] / 100.0
    m["spd"] = np.sqrt(m["u"] ** 2 + m["v"] ** 2)
    m["angle"] = np.angle(m["u"] + 1j * m["v"])
    m["Sal"][(m["Sal"] < 33.5) | (m["Sal"] > 34.9)] = np.nan
    m["S"] = m.pop("Sal")
    m["Temp"][m["Temp"] < -2.0] = np.nan
    m["T"] = m.pop("Temp")

    # Dimensional quantities.
    m["f"] = gsw.f(m["lat"])
    m["ll"] = np.array([m["lon"], m["lat"]])
    m["z"] = gsw.z_from_p(m["P"], m["lat"])

    # Estimate thermodynamic quantities.
    m["SA"] = gsw.SA_from_SP(m["S"], m["P"], m["lon"], m["lat"])
    m["CT"] = gsw.CT_from_t(m["SA"], m["T"], m["P"])
    # specvol_anom = gsw.specvol_anom(m['SA'], m['CT'], m['P'])
    # m['sva'] = specvol_anom

cc["wr"] = w

print("Calculating thermodynamics.")
print("Excluding bad data using T-S funnel.")
# Chuck out data outside of TS funnel sensible range.
funnel = np.genfromtxt("funnel.txt")

for m in moorings:
    S = m["SA"].flatten()
    T = m["CT"].flatten()
    p = path.Path(funnel)
    in_funnel = p.contains_points(np.vstack((S, T)).T)
    fix = np.reshape(~in_funnel, m["SA"].shape)
    m["in_funnel"] = ~fix

    varl = ["S"]
    if fix.any():
        levs = np.squeeze(np.argwhere(np.any(fix, axis=0)))
        for lev in levs:
            x = m["t"][fix[:, lev], lev]
            xp = m["t"][~fix[:, lev], lev]
            for var in varl:
                fp = m[var][~fix[:, lev], lev]
                m[var][fix[:, lev], lev] = np.interp(x, xp, fp)

    # Re-estimate thermodynamic quantities.
    m["SA"] = gsw.SA_from_SP(m["S"], m["P"], m["lon"], m["lat"])
    m["CT"] = gsw.CT_from_t(m["SA"], m["T"], m["P"])

print("Calculating neutral density.")
# Estimate the neutral density
for m in moorings:
    # Compute potential temperature using the 1983 UNESCO EOS.
    m["PT0"] = seawater.ptmp(m["S"], m["T"], m["P"])
    # Flatten variables for analysis.
    lons = m["lon"] * np.ones_like(m["P"])
    lats = m["lat"] * np.ones_like(m["P"])
    S_ = m["S"].flatten()
    T_ = m["PT0"].flatten()
    P_ = m["P"].flatten()
    LO_ = lons.flatten()
    LA_ = lats.flatten()
    gamman = gamma_GP_from_SP_pt(S_, T_, P_, LO_, LA_)
    m["gamman"] = np.reshape(gamman, m["P"].shape) + 1000.0

print("Calculating slice gradients at C.")
# Want gradient of density/vel to be local, no large central differences.
slices = [slice(0, 4), slice(4, 6), slice(6, 10), slice(10, 12)]
cc["dgdz"] = np.empty((cc["N_data"], cc["N_levels"]))
cc["dTdz"] = np.empty((cc["N_data"], cc["N_levels"]))
cc["dudz"] = np.empty((cc["N_data"], cc["N_levels"]))
cc["dvdz"] = np.empty((cc["N_data"], cc["N_levels"]))
for sl in slices:
    z = cc["z"][:, sl]
    g = cc["gamman"][:, sl]
    T = cc["T"][:, sl]
    u = cc["u"][:, sl]
    v = cc["v"][:, sl]
    cc["dgdz"][:, sl] = np.gradient(g, axis=1) / np.gradient(z, axis=1)
    cc["dTdz"][:, sl] = np.gradient(T, axis=1) / np.gradient(z, axis=1)
    cc["dudz"][:, sl] = np.gradient(u, axis=1) / np.gradient(z, axis=1)
    cc["dvdz"][:, sl] = np.gradient(v, axis=1) / np.gradient(z, axis=1)

print("Filtering data.")
# Low pass filter data.
tc = tc_hrs * 60.0 * 60.0
fc = 1.0 / tc  # Cut off frequency.
normal_cutoff = fc * dt_sec * 2.0  # Nyquist frequency is half 1/dt.
b, a = sig.butter(4, normal_cutoff, btype="lowpass")

varl = [
    "z",
    "P",
    "S",
    "T",
    "u",
    "v",
    "wr",
    "SA",
    "CT",
    "gamman",
    "dgdz",
    "dTdz",
    "dudz",
    "dvdz",
]  # sva
for m in moorings:
    for var in varl:
        try:
            data = m[var].copy()
        except KeyError:
            continue

        m[var + "_m"] = np.nanmean(data, axis=0)

        # For the purpose of filtering set fill with 0 rather than nan (SW)
        nans = np.isnan(data)
        if nans.any():
            data[nans] = 0.0

        datalo = sig.filtfilt(b, a, data, axis=0)

        # Then put nans back...
        if nans.any():
            datalo[nans] = np.nan

        namelo = var + "_lo"
        m[namelo] = datalo
        namehi = var + "_hi"
        m[namehi] = m[var] - m[namelo]

    m["spd_lo"] = np.sqrt(m["u_lo"] ** 2 + m["v_lo"] ** 2)
    m["angle_lo"] = ma.angle(m["u_lo"] + 1j * m["v_lo"])
    m["spd_hi"] = np.sqrt(m["u_hi"] ** 2 + m["v_hi"] ** 2)
    m["angle_hi"] = ma.angle(m["u_hi"] + 1j * m["v_hi"])

# %% [markdown]
# Save the raw data.

# %% ##################### SAVE RAW DATA ######################################
io.savemat(os.path.join(data_out, "C_raw.mat"), cc)
io.savemat(os.path.join(data_out, "NW_raw.mat"), nw)
io.savemat(os.path.join(data_out, "NE_raw.mat"), ne)
io.savemat(os.path.join(data_out, "SE_raw.mat"), se)
io.savemat(os.path.join(data_out, "SW_raw.mat"), sw)

# %% [markdown]
# ## Create virtual mooring 'raw'.

# %%
print("VIRTUAL MOORING")
print("Determine maximum knockdown as a function of z.")

zms = np.hstack([m["z"].max(axis=0) for m in moorings if "se" not in m["id"]])
Dzs = np.hstack(
    [m["z"].min(axis=0) - m["z"].max(axis=0) for m in moorings if "se" not in m["id"]]
)

zmax_pfit = np.polyfit(zms, Dzs, 2)  # Second order polynomial for max knockdown

np.save(
    os.path.join(data_out, "zmax_pfit"), np.polyfit(zms, Dzs, 2), allow_pickle=False
)


# Define the knockdown model:
def zmodel(u, zmax, zmax_pfit):
    return zmax + np.polyval(zmax_pfit, zmax) * u ** 3


print("Load model data.")
mluv = xr.load_dataset("../data/mooring_locations_uv1.nc")
mluv = mluv.isel(
    t=slice(0, np.argwhere(mluv.u[:, 0, 0].data == 0)[0][0])
)  # Get rid of end zeros...
mluv = mluv.assign_coords(lon=mluv.lon)
mluv = mluv.assign_coords(id=["cc", "nw", "ne", "se", "sw"])
mluv["spd"] = (mluv.u ** 2 + mluv.v ** 2) ** 0.5


print("Create virtual mooring 'raw' dataset.")
savedict = {
    "cc": {"id": "cc"},
    "nw": {"id": "nw"},
    "ne": {"id": "ne"},
    "se": {"id": "se"},
    "sw": {"id": "sw"},
}
mids = ["cc", "nw", "ne", "se", "sw"]


def nearidx(a, v):
    return np.argmin(np.abs(np.asarray(a) - v))


for idx, mid in enumerate(mids):
    savedict[mid]["lon"] = mluv.lon[idx].data
    savedict[mid]["lat"] = mluv.lat[idx].data

    izs = []
    for i in range(moorings[idx]["N_levels"]):
        izs.append(nearidx(mluv.z, moorings[idx]["z"][:, i].max()))

    spdm = mluv.spd.isel(z=izs, index=idx).mean(dim="z")
    spdn = spdm / spdm.max()
    zmax = mluv.z[izs]

    zk = zmodel(spdn.data[:, np.newaxis], zmax.data[np.newaxis, :], zmax_pfit)
    savedict[mid]["z"] = zk
    savedict[mid]["t"] = np.tile(
        mluv.t.data[:, np.newaxis], (1, moorings[idx]["N_levels"])
    )

    fu = itpl.RectBivariateSpline(mluv.t.data, -mluv.z.data, mluv.u[..., idx].data)
    fv = itpl.RectBivariateSpline(mluv.t.data, -mluv.z.data, mluv.v[..., idx].data)

    uk = fu(mluv.t.data[:, np.newaxis], -zk, grid=False)
    vk = fv(mluv.t.data[:, np.newaxis], -zk, grid=False)
    savedict[mid]["u"] = uk
    savedict[mid]["v"] = vk

io.savemat("../data/virtual_mooring_raw.mat", savedict)

# %% [markdown]
# ## Create virtual mooring 'interpolated'.

# %%
# Corrected levels.
# heights = [-540., -1250., -2100., -3500.]
# Filter cut off (hours)
tc_hrs = 40.0
# Start of time series (matlab datetime)
# t_start = 734494.0
# Length of time series
# max_len = N_data = 42048
# Sampling period (minutes)
dt_min = 60.0
dt_sec = dt_min * 60.0  # Sample period in seconds.
dt_day = dt_sec / 86400.0  # Sample period in days.
N_per_day = int(1.0 / dt_day)  # Samples per day.
# Window length for wave stress quantities and mesoscale strain quantities.
nperseg = 2 ** 7
# Spectra parameters
window = "hanning"
detrend = "constant"
# Extrapolation/interpolation limit above which data will be removed.
dzlim = 100.0
# Integration of spectra parameters. These multiple N and f respectively to set
# the integration limits.
fhi = 1.0
flo = 1.0
flov = 1.0  # When integrating spectra involved in vertical fluxes, get rid of
# the near inertial portion.

# %%
moorings = utils.loadmat("../data/virtual_mooring_raw.mat")

cc = moorings.pop("cc")
nw = moorings.pop("nw")
ne = moorings.pop("ne")
se = moorings.pop("se")
sw = moorings.pop("sw")
moorings = [cc, nw, ne, se, sw]

N_data = cc["t"].shape[0]

# %% [markdown]
# Polynomial fits first.

# %%
print("**Generating corrected data**")
# Generate corrected moorings
z = np.concatenate([m["z"].flatten() for m in moorings])
u = np.concatenate([m["u"].flatten() for m in moorings])
v = np.concatenate([m["v"].flatten() for m in moorings])

print("Calculating polynomial coefficients.")
pzu = np.polyfit(z, u, 2)
pzv = np.polyfit(z, v, 2)

# %%
# Additional height in m to add to interpolation height.
hoffset = [-25.0, 50.0, -50.0, 100.0]

pi2 = np.pi * 2.0
nfft = nperseg
levis = [(0, 1, 2, 3), (4, 5), (6, 7, 8, 9), (10, 11)]
Nclevels = len(levis)
spec_kwargs = {
    "fs": 1.0 / dt_sec,
    "window": window,
    "nperseg": nperseg,
    "nfft": nfft,
    "detrend": detrend,
    "axis": 0,
}

idx1 = np.arange(nperseg, N_data, nperseg // 2)  # Window end index
idx0 = idx1 - nperseg  # Window start index
N_windows = len(idx0)

# Initialise the place holder dictionaries.
c12w = {"N_levels": 12}  # Dictionary for raw, windowed data from central mooring
c4w = {"N_levels": Nclevels}  # Dictionary for processed, windowed data
c4 = {"N_levels": Nclevels}  # Dictionary for processed data
# Dictionaries for raw, windowed data from outer moorings
nw5w, ne5w, se5w, sw5w = {"id": "nw"}, {"id": "ne"}, {"id": "se"}, {"id": "sw"}
moorings5w = [nw5w, ne5w, se5w, sw5w]
# Dictionaries for processed, windowed data from outer moorings
nw4w, ne4w, se4w, sw4w = {"id": "nw"}, {"id": "ne"}, {"id": "se"}, {"id": "sw"}
moorings4w = [nw4w, ne4w, se4w, sw4w]

# Initialised the arrays of windowed data
varr = ["t", "z", "u", "v"]
for var in varr:
    c12w[var] = np.zeros((nperseg, N_windows, 12))

var4 = [
    "t",
    "z",
    "u",
    "v",
    "dudx",
    "dvdx",
    "dudy",
    "dvdy",
    "dudz",
    "dvdz",
    "nstrain",
    "sstrain",
    "vort",
    "div",
]
for var in var4:
    c4w[var] = np.zeros((nperseg, N_windows, Nclevels))

for var in var4:
    c4[var] = np.zeros((N_windows, Nclevels))

# Initialised the arrays of windowed data for outer mooring
varro = ["z", "u", "v"]
for var in varro:
    for m5w in moorings5w:
        m5w[var] = np.zeros((nperseg, N_windows, 5))

var4o = ["z", "u", "v"]
for var in var4o:
    for m4w in moorings4w:
        m4w[var] = np.zeros((nperseg, N_windows, Nclevels))

# for var in var4o:
#     for m4 in moorings4:
#         m4[var] = np.zeros((N_windows, 4))

# Window the raw data.
for i in range(N_windows):
    idx = idx0[i]
    for var in varr:
        c12w[var][:, i, :] = cc[var][idx : idx + nperseg, :]

for i in range(N_windows):
    idx = idx0[i]
    for var in varro:
        for m5w, m in zip(moorings5w, moorings[1:]):
            m5w[var][:, i, :] = m[var][idx : idx + nperseg, :]

print("Interpolating properties.")
# Do the interpolation
for i in range(Nclevels):
    # THIS hoffset is important!!!
    c4["z"][:, i] = np.mean(c12w["z"][..., levis[i]], axis=(0, -1)) + hoffset[i]

    for j in range(N_windows):
        zr = c12w["z"][:, j, levis[i]]
        ur = c12w["u"][:, j, levis[i]]
        vr = c12w["v"][:, j, levis[i]]
        zi = c4["z"][j, i]
        c4w["z"][:, j, i] = np.mean(zr, axis=-1)
        c4w["t"][:, j, i] = c12w["t"][:, j, 0]
        c4w["u"][:, j, i] = moo.interp_quantity(zr, ur, zi, pzu)
        c4w["v"][:, j, i] = moo.interp_quantity(zr, vr, zi, pzv)

        dudzr = np.gradient(ur, axis=-1) / np.gradient(zr, axis=-1)
        dvdzr = np.gradient(vr, axis=-1) / np.gradient(zr, axis=-1)

        # Instead of mean, could moo.interp1d
        c4w["dudz"][:, j, i] = np.mean(dudzr, axis=-1)
        c4w["dvdz"][:, j, i] = np.mean(dvdzr, axis=-1)

        for m5w, m4w in zip(moorings5w, moorings4w):
            zr = m5w["z"][:, j, :]
            ur = m5w["u"][:, j, :]
            vr = m5w["v"][:, j, :]

            m4w["z"][:, j, i] = np.full((nperseg), zi)
            m4w["u"][:, j, i] = moo.interp_quantity(zr, ur, zi, pzu)
            m4w["v"][:, j, i] = moo.interp_quantity(zr, vr, zi, pzv)


print("Filtering windowed data.")
fcorcpd = np.abs(gsw.f(cc["lat"])) * 86400 / pi2

varl = ["u", "v"]
for var in varl:
    c4w[var + "_lo"] = utils.butter_filter(
        c4w[var], 24 / tc_hrs, fs=N_per_day, btype="low", axis=0
    )
    c4w[var + "_hi"] = c4w[var] - c4w[var + "_lo"]

varl = ["u", "v"]
for var in varl:
    for m4w in moorings4w:
        m4w[var + "_lo"] = utils.butter_filter(
            m4w[var], 24 / tc_hrs, fs=N_per_day, btype="low", axis=0
        )
        m4w[var + "_hi"] = m4w[var] - m4w[var + "_lo"]

c4w["zi"] = np.ones_like(c4w["z"]) * c4["z"]

print("Calculating horizontal gradients.")
# Calculate horizontal gradients
for j in range(N_windows):
    ll = np.stack(
        ([m["lon"] for m in moorings[1:]], [m["lat"] for m in moorings[1:]]), axis=1
    )
    uv = np.stack(
        (
            [m4w["u_lo"][:, j, :] for m4w in moorings4w],
            [m4w["v_lo"][:, j, :] for m4w in moorings4w],
        ),
        axis=1,
    )
    dudx, dudy, dvdx, dvdy, vort, div = moo.div_vort_4D(ll[:, 0], ll[:, 1], uv)
    nstrain = dudx - dvdy
    sstrain = dvdx + dudy
    c4w["dudx"][:, j, :] = dudx
    c4w["dudy"][:, j, :] = dudy
    c4w["dvdx"][:, j, :] = dvdx
    c4w["dvdy"][:, j, :] = dvdy
    c4w["nstrain"][:, j, :] = nstrain
    c4w["sstrain"][:, j, :] = sstrain
    c4w["vort"][:, j, :] = vort
    c4w["div"][:, j, :] = div

for var in var4:
    if var == "z":  # Keep z as modified by hoffset.
        continue
    c4[var] = np.mean(c4w[var], axis=0)

freq, c4w["Puu"] = sig.welch(c4w["u_hi"], **spec_kwargs)
_, c4w["Pvv"] = sig.welch(c4w["v_hi"], **spec_kwargs)
_, c4w["Cuv"] = sig.csd(c4w["u_hi"], c4w["v_hi"], **spec_kwargs)

c4w["freq"] = freq.copy()

# Get rid of annoying tiny values.
svarl = ["Puu", "Pvv", "Cuv"]
for var in svarl:
    c4w[var][0, ...] = 0.0
    c4[var + "_int"] = np.full((N_windows, 4), np.nan)

# Horizontal azimuth according to Jing 2018
c4w["theta"] = np.arctan2(2.0 * c4w["Cuv"].real, (c4w["Puu"] - c4w["Pvv"])) / 2

# Integration #############################################################
print("Integrating power spectra.")
for var in svarl:
    c4w[var + "_cint"] = np.full_like(c4w[var], fill_value=np.nan)

fcor = np.abs(gsw.f(cc["lat"])) / pi2
N_freq = len(freq)
freq_ = np.tile(freq[:, np.newaxis, np.newaxis], (1, N_windows, Nclevels))
# ulim = fhi * np.tile(c4["N"][np.newaxis, ...], (N_freq, 1, 1)) / pi2
ulim = 1e9  # Set a huge upper limit since we don't know what N is...
llim = fcor * flo
use = (freq_ < ulim) & (freq_ > llim)

svarl = ["Puu", "Pvv", "Cuv"]
for var in svarl:
    c4[var + "_int"] = igr.simps(use * c4w[var].real, freq, axis=0)
    c4w[var + "_cint"] = igr.cumtrapz(use * c4w[var].real, freq, axis=0, initial=0.0)

# Change lower integration limits for vertical components...
llim = fcor * flov
use = (freq_ < ulim) & (freq_ > llim)


# Usefull quantities
c4["nstress"] = c4["Puu_int"] - c4["Pvv_int"]
c4["sstress"] = -2.0 * c4["Cuv_int"]

c4["F_horiz"] = (
    -0.5 * (c4["Puu_int"] - c4["Pvv_int"]) * c4["nstrain"]
    - c4["Cuv_int"] * c4["sstrain"]
)


# ## Now we have to create the model 'truth'...
#
# Load the model data and estimate some gradients.

print("Estimating smoothed gradients (slow).")
mluv = xr.load_dataset("../data/mooring_locations_uv1.nc")
mluv = mluv.isel(
    t=slice(0, np.argwhere(mluv.u[:, 0, 0].data == 0)[0][0])
)  # Get rid of end zeros...
mluv = mluv.assign_coords(lon=mluv.lon)
mluv = mluv.assign_coords(id=["cc", "nw", "ne", "se", "sw"])
mluv["dudz"] = (["t", "z", "index"], np.gradient(mluv.u, mluv.z, axis=1))
mluv["dvdz"] = (["t", "z", "index"], np.gradient(mluv.v, mluv.z, axis=1))

uv = np.rollaxis(np.stack((mluv.u, mluv.v))[..., 1:], 3, 0)
dudx, dudy, dvdx, dvdy, vort, div = moo.div_vort_4D(mluv.lon[1:], mluv.lat[1:], uv)
nstrain = dudx - dvdy
sstrain = dvdx + dudy
mluv["dudx"] = (["t", "z"], dudx)
mluv["dudy"] = (["t", "z"], dudy)
mluv["dvdx"] = (["t", "z"], dvdx)
mluv["dvdy"] = (["t", "z"], dvdy)
mluv["nstrain"] = (["t", "z"], nstrain)
mluv["sstrain"] = (["t", "z"], sstrain)
mluv["vort"] = (["t", "z"], vort)
mluv["div"] = (["t", "z"], div)

# Smooth the model data in an equivalent way to the real mooring.
dudxs = (
    mluv.dudx.rolling(t=nperseg, center=True)
    .reduce(np.average, weights=sig.hann(nperseg))
    .dropna("t")
)
dvdxs = (
    mluv.dvdx.rolling(t=nperseg, center=True)
    .reduce(np.average, weights=sig.hann(nperseg))
    .dropna("t")
)
dudys = (
    mluv.dudy.rolling(t=nperseg, center=True)
    .reduce(np.average, weights=sig.hann(nperseg))
    .dropna("t")
)
dvdys = (
    mluv.dvdy.rolling(t=nperseg, center=True)
    .reduce(np.average, weights=sig.hann(nperseg))
    .dropna("t")
)
sstrains = (
    mluv.sstrain.rolling(t=nperseg, center=True)
    .reduce(np.average, weights=sig.hann(nperseg))
    .dropna("t")
)
nstrains = (
    mluv.nstrain.rolling(t=nperseg, center=True)
    .reduce(np.average, weights=sig.hann(nperseg))
    .dropna("t")
)
divs = (
    mluv.div.rolling(t=nperseg, center=True)
    .reduce(np.average, weights=sig.hann(nperseg))
    .dropna("t")
)
vorts = (
    mluv.vort.rolling(t=nperseg, center=True)
    .reduce(np.average, weights=sig.hann(nperseg))
    .dropna("t")
)
dudzs = (
    mluv.dudz.isel(index=0)
    .rolling(t=nperseg, center=True)
    .reduce(np.average, weights=sig.hann(nperseg))
    .dropna("t")
)
dvdzs = (
    mluv.dvdz.isel(index=0)
    .rolling(t=nperseg, center=True)
    .reduce(np.average, weights=sig.hann(nperseg))
    .dropna("t")
)

# Make spline fits.
fdudx = itpl.RectBivariateSpline(dudxs.t.data, -dudxs.z.data, dudxs.data)
fdvdx = itpl.RectBivariateSpline(dvdxs.t.data, -dvdxs.z.data, dvdxs.data)
fdudy = itpl.RectBivariateSpline(dudys.t.data, -dudys.z.data, dudys.data)
fdvdy = itpl.RectBivariateSpline(dvdys.t.data, -dvdys.z.data, dvdys.data)
fsstrain = itpl.RectBivariateSpline(sstrains.t.data, -sstrains.z.data, sstrains.data)
fnstrain = itpl.RectBivariateSpline(nstrains.t.data, -nstrains.z.data, nstrains.data)
fdiv = itpl.RectBivariateSpline(divs.t.data, -divs.z.data, divs.data)
fvort = itpl.RectBivariateSpline(vorts.t.data, -vorts.z.data, vorts.data)
fdudz = itpl.RectBivariateSpline(dudzs.t.data, -dudzs.z.data, dudzs.data)
fdvdz = itpl.RectBivariateSpline(dvdzs.t.data, -dvdzs.z.data, dvdzs.data)

# Interpolate using splines.
dudxt = fdudx(c4["t"], -c4["z"], grid=False)
dvdxt = fdvdx(c4["t"], -c4["z"], grid=False)
dudyt = fdudy(c4["t"], -c4["z"], grid=False)
dvdyt = fdvdy(c4["t"], -c4["z"], grid=False)
sstraint = fsstrain(c4["t"], -c4["z"], grid=False)
nstraint = fnstrain(c4["t"], -c4["z"], grid=False)
divt = fdiv(c4["t"], -c4["z"], grid=False)
vortt = fvort(c4["t"], -c4["z"], grid=False)
dudzt = fdudz(c4["t"], -c4["z"], grid=False)
dvdzt = fdvdz(c4["t"], -c4["z"], grid=False)

c4["dudxt"] = dudxt
c4["dvdxt"] = dvdxt
c4["dudyt"] = dudyt
c4["dvdyt"] = dvdyt
c4["sstraint"] = sstraint
c4["nstraint"] = nstraint
c4["divt"] = divt
c4["vortt"] = vortt
c4["dudzt"] = dudzt
c4["dvdzt"] = dvdzt

# %%
# %% ########################## SAVE CORRECTED FILES ##########################
io.savemat("../data/virtual_mooring_interpolated.mat", c4)
io.savemat("../data/virtual_mooring_interpolated_windowed.mat", c4w)

# %% [markdown]
# Signal to noise ratios.

# %%
print("Estimating signal to noise ratios.")

M = munch.munchify(utils.loadmat('../data/virtual_mooring_interpolated.mat'))

# shear strain
dsstrain = M.sstrain - M.sstraint
SNR_sstrain = M.sstrain.var(axis=0)/dsstrain.var(axis=0)
np.save('../data/SNR_sstrain', SNR_sstrain, allow_pickle=False)

# normal strain
dnstrain = M.nstrain - M.nstraint
SNR_nstrain = M.nstrain.var(axis=0)/dnstrain.var(axis=0)
np.save('../data/SNR_nstrain', SNR_nstrain, allow_pickle=False)

# zonal shear
ddudz = M.dudz - M.dudzt
SNR_dudz = M.dvdz.var(axis=0)/ddudz.var(axis=0)
np.save('../data/SNR_dudz', SNR_dudz, allow_pickle=False)

# meridional shear
ddvdz = M.dvdz - M.dvdzt
SNR_dvdz = M.dvdz.var(axis=0)/ddvdz.var(axis=0)
np.save('../data/SNR_dvdz', SNR_dvdz, allow_pickle=False)

# divergence
ddiv = M.div - M.divt
SNR_nstrain = M.div.var(axis=0)/ddiv.var(axis=0)
np.save('../data/SNR_div', SNR_nstrain, allow_pickle=False)

# %% [markdown]
# <a id="corrected"></a>

# %% [markdown]
# ## Generate interpolated data.
#
# Set parameters again.

# %%
# Corrected levels.
# heights = [-540., -1250., -2100., -3500.]
# Filter cut off (hours)
tc_hrs = 40.0
# Start of time series (matlab datetime)
t_start = 734494.0
# Length of time series
max_len = N_data = 42048
# Data file
raw_data_file = "moorings.mat"
# Index where NaNs start in u and v data from SW mooring
sw_vel_nans = 14027
# Sampling period (minutes)
dt_min = 15.0
# Window length for wave stress quantities and mesoscale strain quantities.
nperseg = 2 ** 9
# Spectra parameters
window = "hanning"
detrend = "constant"
# Extrapolation/interpolation limit above which data will be removed.
dzlim = 100.0
# Integration of spectra parameters. These multiple N and f respectively to set
# the integration limits.
fhi = 1.0
flo = 1.0
flov = 1.0  # When integrating spectra involved in vertical fluxes, get rid of
# the near inertial portion.
# When bandpass filtering windowed data use these params multiplied by f and N
filtlo = 0.9  # times f
filthi = 1.1  # times N

# Interpolation distance that raises flag (m)
zimax = 100.0

dt_sec = dt_min * 60.0  # Sample period in seconds.
dt_day = dt_sec / 86400.0  # Sample period in days.
N_per_day = int(1.0 / dt_day)  # Samples per day.

# %% [markdown]
# Polynomial fits first.

# %%
print("REAL MOORING INTERPOLATION")
print("**Generating corrected data**")

moorings = load_data.load_my_data()
cc, nw, ne, se, sw = moorings

# Generate corrected moorings
T = np.concatenate([m["T"].flatten() for m in moorings])
S = np.concatenate([m["S"].flatten() for m in moorings])
z = np.concatenate([m["z"].flatten() for m in moorings])
u = np.concatenate([m["u"].flatten() for m in moorings])
v = np.concatenate([m["v"].flatten() for m in moorings])
g = np.concatenate([m["gamman"].flatten() for m in moorings])

# SW problems...
nans = np.isnan(u) | np.isnan(v)

print("Calculating polynomial coefficients.")
pzT = np.polyfit(z[~nans], T[~nans], 3)
pzS = np.polyfit(z[~nans], S[~nans], 3)
pzg = np.polyfit(z[~nans], g[~nans], 3)
pzu = np.polyfit(z[~nans], u[~nans], 2)
pzv = np.polyfit(z[~nans], v[~nans], 2)

# %%
# Additional height in m to add to interpolation height.
hoffset = [-25.0, 50.0, -50.0, 100.0]

pi2 = np.pi * 2.0
nfft = nperseg
levis = [(0, 1, 2, 3), (4, 5), (6, 7, 8, 9), (10, 11)]
Nclevels = len(levis)
spec_kwargs = {
    "fs": 1.0 / dt_sec,
    "window": window,
    "nperseg": nperseg,
    "nfft": nfft,
    "detrend": detrend,
    "axis": 0,
}

idx1 = np.arange(nperseg, N_data, nperseg // 2)  # Window end index
idx0 = idx1 - nperseg  # Window start index
N_windows = len(idx0)

# Initialise the place holder dictionaries.
c12w = {"N_levels": 12}  # Dictionary for raw, windowed data from central mooring
c4w = {"N_levels": Nclevels}  # Dictionary for processed, windowed data
c4 = {"N_levels": Nclevels}  # Dictionary for processed data
# Dictionaries for raw, windowed data from outer moorings
nw5w, ne5w, se5w, sw5w = {"id": "nw"}, {"id": "ne"}, {"id": "se"}, {"id": "sw"}
moorings5w = [nw5w, ne5w, se5w, sw5w]
# Dictionaries for processed, windowed data from outer moorings
nw4w, ne4w, se4w, sw4w = {"id": "nw"}, {"id": "ne"}, {"id": "se"}, {"id": "sw"}
moorings4w = [nw4w, ne4w, se4w, sw4w]

# Initialised the arrays of windowed data
varr = ["t", "z", "u", "v", "gamman", "S", "T", "P"]
for var in varr:
    c12w[var] = np.zeros((nperseg, N_windows, cc["N_levels"]))

var4 = [
    "t",
    "z",
    "u",
    "v",
    "gamman",
    "dudx",
    "dvdx",
    "dudy",
    "dvdy",
    "dudz",
    "dvdz",
    "dgdz",
    "nstrain",
    "sstrain",
    "vort",
    "N2",
]
for var in var4:
    c4w[var] = np.zeros((nperseg, N_windows, Nclevels))

for var in var4:
    c4[var] = np.zeros((N_windows, Nclevels))

# Initialised the arrays of windowed data for outer mooring
varro = ["z", "u", "v"]
for var in varro:
    for m5w in moorings5w:
        m5w[var] = np.zeros((nperseg, N_windows, 5))

var4o = ["z", "u", "v"]
for var in var4o:
    for m4w in moorings4w:
        m4w[var] = np.zeros((nperseg, N_windows, Nclevels))

# for var in var4o:
#     for m4 in moorings4:
#         m4[var] = np.zeros((N_windows, 4))

# Window the raw data.
for i in range(N_windows):
    idx = idx0[i]
    for var in varr:
        c12w[var][:, i, :] = cc[var][idx : idx + nperseg, :]

for i in range(N_windows):
    idx = idx0[i]
    for var in varro:
        for m5w, m in zip(moorings5w, moorings[1:]):
            m5w[var][:, i, :] = m[var][idx : idx + nperseg, :]

c4["interp_far_flag"] = np.full_like(c4["u"], False, dtype=bool)

print("Interpolating properties.")
# Do the interpolation
for i in range(Nclevels):
    # THIS hoffset is important!!!
    c4["z"][:, i] = np.mean(c12w["z"][..., levis[i]], axis=(0, -1)) + hoffset[i]

    for j in range(N_windows):
        zr = c12w["z"][:, j, levis[i]]
        ur = c12w["u"][:, j, levis[i]]
        vr = c12w["v"][:, j, levis[i]]
        gr = c12w["gamman"][:, j, levis[i]]
        Sr = c12w["S"][:, j, levis[i]]
        Tr = c12w["T"][:, j, levis[i]]
        Pr = c12w["P"][:, j, levis[i]]
        zi = c4["z"][j, i]

        c4["interp_far_flag"][j, i] = np.any(np.min(np.abs(zr - zi), axis=-1) > zimax)

        c4w["z"][:, j, i] = np.mean(zr, axis=-1)
        c4w["t"][:, j, i] = c12w["t"][:, j, 0]
        c4w["u"][:, j, i] = moo.interp_quantity(zr, ur, zi, pzu)
        c4w["v"][:, j, i] = moo.interp_quantity(zr, vr, zi, pzv)
        c4w["gamman"][:, j, i] = moo.interp_quantity(zr, gr, zi, pzg)

        dudzr = np.gradient(ur, axis=-1) / np.gradient(zr, axis=-1)
        dvdzr = np.gradient(vr, axis=-1) / np.gradient(zr, axis=-1)
        dgdzr = np.gradient(gr, axis=-1) / np.gradient(zr, axis=-1)
        N2 = seawater.bfrq(Sr.T, Tr.T, Pr.T, cc["lat"])[0].T

        # Instead of mean, could moo.interp1d
        c4w["dudz"][:, j, i] = np.mean(dudzr, axis=-1)
        c4w["dvdz"][:, j, i] = np.mean(dvdzr, axis=-1)
        c4w["dgdz"][:, j, i] = np.mean(dgdzr, axis=-1)
        c4w["N2"][:, j, i] = np.mean(N2, axis=-1)

        for m5w, m4w in zip(moorings5w, moorings4w):
            if (m5w["id"] == "sw") & (
                idx1[j] > sw_vel_nans
            ):  # Skip this level because of NaNs
                zr = m5w["z"][:, j, (0, 1, 3, 4)]
                ur = m5w["u"][:, j, (0, 1, 3, 4)]
                vr = m5w["v"][:, j, (0, 1, 3, 4)]
            else:
                zr = m5w["z"][:, j, :]
                ur = m5w["u"][:, j, :]
                vr = m5w["v"][:, j, :]

            m4w["z"][:, j, i] = np.full((nperseg), zi)
            m4w["u"][:, j, i] = moo.interp_quantity(zr, ur, zi, pzu)
            m4w["v"][:, j, i] = moo.interp_quantity(zr, vr, zi, pzv)

print("Filtering windowed data.")
fcorcpd = np.abs(cc["f"]) * 86400 / pi2

Nmean = np.sqrt(np.average(c4w["N2"], weights=sig.hann(nperseg), axis=0))

varl = ["u", "v", "gamman"]
for var in varl:
    c4w[var + "_hib"] = np.zeros_like(c4w[var])
    c4w[var + "_lo"] = utils.butter_filter(
        c4w[var], 24 / tc_hrs, fs=N_per_day, btype="low", axis=0
    )
    c4w[var + "_hi"] = c4w[var] - c4w[var + "_lo"]

for i in range(Nclevels):
    for j in range(N_windows):
        Nmean_ = Nmean[j, i] * 86400 / pi2
        for var in varl:
            c4w[var + "_hib"][:, j, i] = utils.butter_filter(
                c4w[var][:, j, i],
                (filtlo * fcorcpd, filthi * Nmean_),
                fs=N_per_day,
                btype="band",
            )

varl = ["u", "v"]
for var in varl:
    for m4w in moorings4w:
        m4w[var + "_lo"] = utils.butter_filter(
            m4w[var], 24 / tc_hrs, fs=N_per_day, btype="low", axis=0
        )
        m4w[var + "_hi"] = m4w[var] - m4w[var + "_lo"]

c4w["zi"] = np.ones_like(c4w["z"]) * c4["z"]


print("Calculating horizontal gradients.")
# Calculate horizontal gradients
for j in range(N_windows):
    ll = np.stack(
        ([m["lon"] for m in moorings[1:]], [m["lat"] for m in moorings[1:]]), axis=1
    )
    uv = np.stack(
        (
            [m4w["u_lo"][:, j, :] for m4w in moorings4w],
            [m4w["v_lo"][:, j, :] for m4w in moorings4w],
        ),
        axis=1,
    )
    dudx, dudy, dvdx, dvdy, vort, _ = moo.div_vort_4D(ll[:, 0], ll[:, 1], uv)
    nstrain = dudx - dvdy
    sstrain = dvdx + dudy
    c4w["dudx"][:, j, :] = dudx
    c4w["dudy"][:, j, :] = dudy
    c4w["dvdx"][:, j, :] = dvdx
    c4w["dvdy"][:, j, :] = dvdy
    c4w["nstrain"][:, j, :] = nstrain
    c4w["sstrain"][:, j, :] = sstrain
    c4w["vort"][:, j, :] = vort

print("Calculating window averages.")
for var in var4 + ["u_lo", "v_lo", "gamman_lo"]:
    if var == "z":  # Keep z as modified by hoffset.
        continue
    c4[var] = np.average(c4w[var], weights=sig.hann(nperseg), axis=0)

print("Estimating w and b.")
om = np.fft.fftfreq(nperseg, 15 * 60)
c4w["w_hi"] = np.fft.ifft(
    1j
    * pi2
    * om[:, np.newaxis, np.newaxis]
    * np.fft.fft(-c4w["gamman_hi"] / c4["dgdz"], axis=0),
    axis=0,
).real
c4w["w_hib"] = np.fft.ifft(
    1j
    * pi2
    * om[:, np.newaxis, np.newaxis]
    * np.fft.fft(-c4w["gamman_hib"] / c4["dgdz"], axis=0),
    axis=0,
).real

# Estimate buoyancy variables
c4w["b_hi"] = -gsw.grav(-c4["z"], cc["lat"]) * c4w["gamman_hi"] / c4["gamman_lo"]
c4w["b_hib"] = -gsw.grav(-c4["z"], cc["lat"]) * c4w["gamman_hib"] / c4["gamman_lo"]
c4["N"] = np.sqrt(c4["N2"])

print("Estimating covariance spectra.")
freq, c4w["Puu"] = sig.welch(c4w["u_hi"], **spec_kwargs)
_, c4w["Pvv"] = sig.welch(c4w["v_hi"], **spec_kwargs)
_, c4w["Pww"] = sig.welch(c4w["w_hi"], **spec_kwargs)
_, c4w["Pwwg"] = sig.welch(c4w["gamman_hi"] / c4["dgdz"], **spec_kwargs)
c4w["Pwwg"] *= (pi2 * freq[:, np.newaxis, np.newaxis]) ** 2
_, c4w["Pbb"] = sig.welch(c4w["b_hi"], **spec_kwargs)
_, c4w["Cuv"] = sig.csd(c4w["u_hi"], c4w["v_hi"], **spec_kwargs)
_, c4w["Cuwg"] = sig.csd(c4w["u_hi"], c4w["gamman_hi"] / c4["dgdz"], **spec_kwargs)
c4w["Cuwg"] *= -1j * pi2 * freq[:, np.newaxis, np.newaxis]
_, c4w["Cvwg"] = sig.csd(c4w["v_hi"], c4w["gamman_hi"] / c4["dgdz"], **spec_kwargs)
c4w["Cvwg"] *= -1j * pi2 * freq[:, np.newaxis, np.newaxis]
_, c4w["Cub"] = sig.csd(c4w["u_hi"], c4w["b_hi"], **spec_kwargs)
_, c4w["Cvb"] = sig.csd(c4w["v_hi"], c4w["b_hi"], **spec_kwargs)

print("Estimating covariance matrices.")


def cov(x, y, axis=None):
    return np.mean((x - np.mean(x, axis=axis)) * (y - np.mean(y, axis=axis)), axis=axis)


c4["couu"] = cov(c4w["u_hib"], c4w["u_hib"], axis=0)
c4["covv"] = cov(c4w["v_hib"], c4w["v_hib"], axis=0)
c4["coww"] = cov(c4w["w_hib"], c4w["w_hib"], axis=0)
c4["cobb"] = cov(c4w["b_hib"], c4w["b_hib"], axis=0)
c4["couv"] = cov(c4w["u_hib"], c4w["v_hib"], axis=0)
c4["couw"] = cov(c4w["u_hib"], c4w["w_hib"], axis=0)
c4["covw"] = cov(c4w["v_hib"], c4w["w_hib"], axis=0)
c4["coub"] = cov(c4w["u_hib"], c4w["b_hib"], axis=0)
c4["covb"] = cov(c4w["v_hib"], c4w["b_hib"], axis=0)

c4w["freq"] = freq.copy()

# Get rid of annoying tiny values.
svarl = ["Puu", "Pvv", "Pbb", "Cuv", "Cub", "Cvb", "Pwwg", "Cuwg", "Cvwg"]
for var in svarl:
    c4w[var][0, ...] = 0.0
    c4[var + "_int"] = np.full((N_windows, 4), np.nan)

# Horizontal azimuth according to Jing 2018
c4w["theta"] = np.arctan2(2.0 * c4w["Cuv"].real, (c4w["Puu"] - c4w["Pvv"])) / 2

# Integration #############################################################
print("Integrating power spectra.")
for var in svarl:
    c4w[var + "_cint"] = np.full_like(c4w[var], fill_value=np.nan)

fcor = np.abs(cc["f"]) / pi2
N_freq = len(freq)
freq_ = np.tile(freq[:, np.newaxis, np.newaxis], (1, N_windows, Nclevels))
ulim = fhi * np.tile(c4["N"][np.newaxis, ...], (N_freq, 1, 1)) / pi2
llim = fcor * flo
use = (freq_ < ulim) & (freq_ > llim)

svarl = ["Puu", "Pvv", "Pbb", "Cuv", "Pwwg"]
for var in svarl:
    c4[var + "_int"] = igr.simps(use * c4w[var].real, freq, axis=0)
    c4w[var + "_cint"] = igr.cumtrapz(use * c4w[var].real, freq, axis=0, initial=0.0)

# Change lower integration limits for vertical components...
llim = fcor * flov
use = (freq_ < ulim) & (freq_ > llim)

svarl = ["Cub", "Cvb", "Cuwg", "Cvwg"]
for var in svarl:
    c4[var + "_int"] = igr.simps(use * c4w[var].real, freq, axis=0)
    c4w[var + "_cint"] = igr.cumtrapz(use * c4w[var].real, freq, axis=0, initial=0.0)

# Ruddic and Joyce effective stress
for var1, var2 in zip(["Tuwg", "Tvwg"], ["Cuwg", "Cvwg"]):
    func = use * c4w[var2].real * (1 - fcor ** 2 / freq_ ** 2)
    nans = np.isnan(func)
    func[nans] = 0.0
    c4[var1 + "_int"] = igr.simps(func, freq, axis=0)
    func = use * c4w[var2].real * (1 - fcor ** 2 / freq_ ** 2)
    nans = np.isnan(func)
    func[nans] = 0.0
    c4w[var1 + "_cint"] = igr.cumtrapz(func, freq, axis=0, initial=0.0)

# Usefull quantities
c4["nstress"] = c4["Puu_int"] - c4["Pvv_int"]
c4["sstress"] = -2.0 * c4["Cuv_int"]

c4["F_horiz"] = (
    -0.5 * (c4["Puu_int"] - c4["Pvv_int"]) * c4["nstrain"]
    - c4["Cuv_int"] * c4["sstrain"]
)
c4["F_vert"] = (
    -(c4["Cuwg_int"] - cc["f"] * c4["Cvb_int"] / c4["N"] ** 2) * c4["dudz"]
    - (c4["Cvwg_int"] + cc["f"] * c4["Cub_int"] / c4["N"] ** 2) * c4["dvdz"]
)
c4["F_vert_alt"] = -c4["Tuwg_int"] * c4["dudz"] - c4["Tvwg_int"] * c4["dvdz"]

c4["F_total"] = c4["F_horiz"] + c4["F_vert"]

c4["EPu"] = c4["Cuwg_int"] - cc["f"] * c4["Cvb_int"] / c4["N"] ** 2
c4["EPv"] = c4["Cvwg_int"] + cc["f"] * c4["Cub_int"] / c4["N"] ** 2

##

c4["nstress_cov"] = c4["couu"] - c4["covv"]
c4["sstress_cov"] = -2.0 * c4["couv"]

c4["F_horiz_cov"] = (
    -0.5 * (c4["couu"] - c4["covv"]) * c4["nstrain"] - c4["couv"] * c4["sstrain"]
)

c4["F_vert_cov"] = (
    -(c4["couw"] - cc["f"] * c4["covb"] / c4["N"] ** 2) * c4["dudz"]
    - (c4["covw"] + cc["f"] * c4["coub"] / c4["N"] ** 2) * c4["dvdz"]
)

c4["F_total_cov"] = c4["F_horiz_cov"] + c4["F_vert_cov"]

# %% [markdown]
# Estimate standard error on covariances.

# %%
bootnum = 1000
np.random.seed(12341555)
idxs = np.arange(nperseg, dtype="i2")

# def cov1(xy, axis=0):
#     x = xy[..., -1]
#     y = xy[..., -1]
#     return np.mean((x - np.mean(x, axis=axis))*(y - np.mean(y, axis=axis)), axis=axis)

print("Estimating error on covariance using bootstrap (slow).")

euu_ = np.zeros((bootnum, N_windows, Nclevels))
evv_ = np.zeros((bootnum, N_windows, Nclevels))
eww_ = np.zeros((bootnum, N_windows, Nclevels))
ebb_ = np.zeros((bootnum, N_windows, Nclevels))
euv_ = np.zeros((bootnum, N_windows, Nclevels))
euw_ = np.zeros((bootnum, N_windows, Nclevels))
evw_ = np.zeros((bootnum, N_windows, Nclevels))
eub_ = np.zeros((bootnum, N_windows, Nclevels))
evb_ = np.zeros((bootnum, N_windows, Nclevels))


for i in range(bootnum):
    idxs_ = np.random.choice(idxs, nperseg)

    u_ = c4w["u_hib"][idxs_, ...]
    v_ = c4w["v_hib"][idxs_, ...]
    w_ = c4w["w_hib"][idxs_, ...]
    b_ = c4w["b_hib"][idxs_, ...]

    euu_[i, ...] = cov(u_, u_, axis=0)
    evv_[i, ...] = cov(v_, v_, axis=0)
    eww_[i, ...] = cov(w_, w_, axis=0)
    ebb_[i, ...] = cov(b_, b_, axis=0)
    euv_[i, ...] = cov(u_, v_, axis=0)
    euw_[i, ...] = cov(u_, w_, axis=0)
    evw_[i, ...] = cov(v_, w_, axis=0)
    eub_[i, ...] = cov(u_, b_, axis=0)
    evb_[i, ...] = cov(v_, b_, axis=0)

c4["euu"] = euu_.std(axis=0)
c4["evv"] = evv_.std(axis=0)
c4["eww"] = eww_.std(axis=0)
c4["ebb"] = ebb_.std(axis=0)
c4["euv"] = euv_.std(axis=0)
c4["euw"] = euw_.std(axis=0)
c4["evw"] = evw_.std(axis=0)
c4["eub"] = eub_.std(axis=0)
c4["evb"] = evb_.std(axis=0)

# %% [markdown]
# Error on gradients.

# %%
finite_diff_err = 0.06  # Assume 6 percent...

SNR_dudz = np.load("../data/SNR_dudz.npy")
SNR_dvdz = np.load("../data/SNR_dvdz.npy")
SNR_nstrain = np.load("../data/SNR_nstrain.npy")
SNR_sstrain = np.load("../data/SNR_sstrain.npy")

ones = np.ones_like(c4["euu"])

c4["edudz"] = ones * np.sqrt(c4["dudz"].var(axis=0) / SNR_dudz)
c4["edvdz"] = ones * np.sqrt(c4["dvdz"].var(axis=0) / SNR_dvdz)
c4["enstrain"] = esum(
    ones * np.sqrt(c4["nstrain"].var(axis=0) / SNR_nstrain),
    finite_diff_err * c4["nstrain"],
)
c4["esstrain"] = esum(
    ones * np.sqrt(c4["sstrain"].var(axis=0) / SNR_sstrain),
    finite_diff_err * c4["sstrain"],
)

# %% [markdown]
# Error propagation.

# %%
euumvv = 0.5 * esum(c4["euu"], c4["evv"])
c4["enstress"] = euumvv.copy()
enorm = emult(
    -0.5 * (c4["Puu_int"] - c4["Pvv_int"]), c4["nstrain"], euumvv, c4["enstrain"]
)
eshear = emult(c4["Cuv_int"], c4["sstrain"], c4["euv"], c4["esstrain"])
c4["errF_horiz_norm"] = enorm.copy()
c4["errF_horiz_shear"] = eshear.copy()
c4["errF_horiz"] = esum(enorm, eshear)

euumvv = 0.5 * esum(c4["euu"], c4["evv"])
c4["enstress_cov"] = euumvv.copy()
enorm = emult(-0.5 * (c4["couu"] - c4["covv"]), c4["nstrain"], euumvv, c4["enstrain"])
eshear = emult(c4["couv"], c4["sstrain"], c4["euv"], c4["esstrain"])
c4["errF_horiz_norm_cov"] = enorm.copy()
c4["errF_horiz_shear_cov"] = eshear.copy()
c4["errF_horiz_cov"] = esum(enorm, eshear)

euwmvb = esum(c4["euw"], np.abs(cc["f"] / c4["N"] ** 2) * c4["evb"])
evwpub = esum(c4["evw"], np.abs(cc["f"] / c4["N"] ** 2) * c4["eub"])
c4["evstressu"] = euwmvb
c4["evstressv"] = evwpub
edu = emult(
    -(c4["Cuwg_int"] - cc["f"] * c4["Cvb_int"] / c4["N"] ** 2),
    c4["dudz"],
    euwmvb,
    c4["edudz"],
)
edv = emult(
    -(c4["Cvwg_int"] + cc["f"] * c4["Cub_int"] / c4["N"] ** 2),
    c4["dvdz"],
    evwpub,
    c4["edvdz"],
)
c4["errEPu"] = edu.copy()
c4["errEPv"] = edv.copy()
c4["errF_vert"] = esum(edu, edv)

c4["errEPu_alt"] = emult(-c4["Tuwg_int"], c4["dudz"], c4["euw"], c4["edudz"])
c4["errEPv_alt"] = emult(-c4["Tvwg_int"], c4["dvdz"], c4["evw"], c4["edvdz"])
c4["errF_vert_alt"] = esum(c4["errEPu_alt"], c4["errEPv_alt"])

edu = emult(
    -(c4["couw"] - cc["f"] * c4["covb"] / c4["N"] ** 2), c4["dudz"], euwmvb, c4["edudz"]
)
edv = emult(
    -(c4["covw"] + cc["f"] * c4["coub"] / c4["N"] ** 2), c4["dvdz"], evwpub, c4["edvdz"]
)
c4["errEPu_cov"] = edu.copy()
c4["errEPv_cov"] = edv.copy()
c4["errF_vert_cov"] = esum(edu, edv)

c4["errF_total"] = esum(c4["errF_vert"], c4["errF_horiz"])
c4["errF_total_cov"] = esum(c4["errF_vert_cov"], c4["errF_horiz_cov"])

# %% [markdown]
# Save the interpolated data.

# %% ########################## SAVE CORRECTED FILES ##########################
io.savemat(os.path.join(data_out, "C_alt.mat"), c4)
io.savemat(os.path.join(data_out, "C_altw.mat"), c4w)

# %% [markdown]
# <a id="ADCP"></a>

# %% [markdown]
# # ADCP Processing

# %% ########################## PROCESS ADCP DATA #############################
print("ADCP PROCESSING")
tf = np.array([16.0, 2.0])  # band pass filter cut off hours
tc_hrs = 40.0  # Low pass cut off (hours)
dt = 0.5  # Data sample period hr

print("Loading ADCP data from file.")
file = os.path.expanduser(os.path.join(data_in, "ladcp_data.mat"))
adcp = utils.loadmat(file)["ladcp2"]

print("Removing all NaN rows.")
varl = ["u", "v", "z"]
for var in varl:  # Get rid of the all nan row.
    adcp[var] = adcp.pop(var)[:-1, :]

print("Calculating vertical shear.")
z = adcp["z"]
dudz = np.diff(adcp["u"], axis=0) / np.diff(z, axis=0)
dvdz = np.diff(adcp["v"], axis=0) / np.diff(z, axis=0)
nans = np.isnan(dudz) | np.isnan(dvdz)
dudz[nans] = np.nan
dvdz[nans] = np.nan

adcp["zm"] = utils.mid(z, axis=0)
adcp["dudz"] = dudz
adcp["dvdz"] = dvdz

# Low pass filter data.
print("Low pass filtering at {:1.0f} hrs.".format(tc_hrs))
varl = ["u", "v", "dudz", "dvdz"]
for var in varl:
    data = adcp[var]
    nans = np.isnan(data)
    adcp[var + "_m"] = np.nanmean(data, axis=0)
    datalo = utils.butter_filter(
        utils.interp_nans(adcp["dates"], data, axis=1), 1 / tc_hrs, 1 / dt, btype="low"
    )

    # Then put nans back...
    if nans.any():
        datalo[nans] = np.nan

    namelo = var + "_lo"
    adcp[namelo] = datalo
    namehi = var + "_hi"
    adcp[namehi] = adcp[var] - adcp[namelo]

# Band pass filter the data.
print("Band pass filtering between {:1.0f} and {:1.0f} hrs.".format(*tf))
varl = ["u", "v", "dudz", "dvdz"]
for var in varl:
    data = adcp[var]
    nans = np.isnan(data)
    databp = utils.butter_filter(
        utils.interp_nans(adcp["dates"], data, axis=1), 1 / tf, 1 / dt, btype="band"
    )

    # Then put nans back...
    if nans.any():
        databp[nans] = np.nan

    namebp = var + "_bp"
    adcp[namebp] = databp

io.savemat(os.path.join(data_out, "ADCP.mat"), adcp)

# %% [markdown]
# <a id="VMP"></a>

# %% [markdown]
# ## VMP data

# %%
print("VMP PROCESSING")
vmp = utils.loadmat(os.path.join(data_in, "jc054_vmp_cleaned.mat"))["d"]

box = np.array([[-58.0, -58.0, -57.7, -57.7], [-56.15, -55.9, -55.9, -56.15]]).T
p = path.Path(box)
in_box = p.contains_points(np.vstack((vmp["startlon"], vmp["startlat"])).T)
idxs = np.argwhere(in_box).squeeze()
Np = len(idxs)

print("Isolate profiles in match around mooring.")
for var in vmp:
    ndim = np.ndim(vmp[var])
    if ndim == 2:
        vmp[var] = vmp[var][:, idxs]
    if ndim == 1 and vmp[var].size == 36:
        vmp[var] = vmp[var][idxs]

print("Rename variables.")
vmp["P"] = vmp.pop("press")
vmp["T"] = vmp.pop("temp")
vmp["S"] = vmp.pop("salin")

print("Deal with profiles where P[0] != 1.")
P_ = np.arange(1.0, 10000.0)
i0o = np.zeros((Np), dtype=int)
i1o = np.zeros((Np), dtype=int)
i0n = np.zeros((Np), dtype=int)
i1n = np.zeros((Np), dtype=int)
pmax = 0.0
for i in range(Np):
    nans = np.isnan(vmp["eps"][:, i])
    i0o[i] = i0 = np.where(~nans)[0][0]
    i1o[i] = i1 = np.where(~nans)[0][-1]
    P0 = vmp["P"][i0, i]
    P1 = vmp["P"][i1, i]
    i0n[i] = np.searchsorted(P_, P0)
    i1n[i] = np.searchsorted(P_, P1)
    pmax = max(P1, pmax)

P = np.tile(np.arange(1.0, pmax + 2)[:, np.newaxis], (1, len(idxs)))
eps = np.full_like(P, np.nan)
chi = np.full_like(P, np.nan)
T = np.full_like(P, np.nan)
S = np.full_like(P, np.nan)

for i in range(Np):
    eps[i0n[i] : i1n[i] + 1, i] = vmp["eps"][i0o[i] : i1o[i] + 1, i]
    chi[i0n[i] : i1n[i] + 1, i] = vmp["chi"][i0o[i] : i1o[i] + 1, i]
    T[i0n[i] : i1n[i] + 1, i] = vmp["T"][i0o[i] : i1o[i] + 1, i]
    S[i0n[i] : i1n[i] + 1, i] = vmp["S"][i0o[i] : i1o[i] + 1, i]

vmp["P"] = P
vmp["eps"] = eps
vmp["chi"] = chi
vmp["T"] = T
vmp["S"] = S
vmp["z"] = gsw.z_from_p(vmp["P"], vmp["startlat"])

print("Calculate neutral density.")
# Compute potential temperature using the 1983 UNESCO EOS.
vmp["PT0"] = seawater.ptmp(vmp["S"], vmp["T"], vmp["P"])
# Flatten variables for analysis.
lons = np.ones_like(P) * vmp["startlon"]
lats = np.ones_like(P) * vmp["startlat"]
S_ = vmp["S"].flatten()
T_ = vmp["PT0"].flatten()
P_ = vmp["P"].flatten()
LO_ = lons.flatten()
LA_ = lats.flatten()
gamman = gamma_GP_from_SP_pt(S_, T_, P_, LO_, LA_)
vmp["gamman"] = np.reshape(gamman, vmp["P"].shape) + 1000.0

io.savemat(os.path.join(data_out, "VMP.mat"), vmp)
