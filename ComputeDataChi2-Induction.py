#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import numpy as np
import matplotlib.pyplot as plt
import uproot
import matplotlib as mpl
import math

from scipy.optimize import curve_fit
from scipy import interpolate, special
from scipy.stats import norm as normal

import cProfile

import inspect

import os


# In[56]:


# MPL config
mpl.rc('font', size=14)

dosave = True
savedir = "/home/grayputnam/Work/Winter2023/Mar18FitSignal/test/"
# override
if "SAVEDIR" in os.environ:
    savedir = os.environ["SAVEDIR"]

# Clear figures in main loop
CLEARPLT = False


# In[41]:


# Configuration

# Plane
IPLANE = 0
if "IPLANE" in os.environ:
    IPLANE = int(os.environ["IPLANE"])

# Parameters for measurement resolution
if IPLANE == 2:
    SIGMA_A = 0.31
    SIGMA_B = 0.43
elif IPLANE == 1:
    SIGMA_A = 0.67
    SIGMA_B = 0.59
elif IPLANE == 0:
    SIGMA_A = 0.74
    SIGMA_B = 0.65

# Central index of electron path response
if IPLANE == 2:
    CENTER_IND = 625 # not banded
    # CENTER_IND = 610
elif IPLANE == 1:
    CENTER_IND = 608 # not banded
    # CENTER_IND = 595 # banded
elif IPLANE == 0:
    CENTER_IND = 591 # not banded
    # CENTER_IND = 578 # banded
    
if IPLANE == 0:
    # What range of time to fit the response
    FIT_PERIOD = 10
    # What range of time to plot the response
    PLT_PERIOD = 15
if IPLANE == 1:
    # What range of time to fit the response
    FIT_PERIOD = 7
    # What range of time to plot the response
    PLT_PERIOD = 15
if IPLANE == 2:
    # What range of time to fit the response
    FIT_PERIOD = 7
    # What range of time to plot the response
    PLT_PERIOD = 15

# Filename for WC configuration containing electron paths
WCpaths_filename_nom = "./garfield-icarus-fnal-rev1.json"
#Sergey's WC paths
# WCpaths_filename_nom = "./icarus_fnal_band.json"

WCpaths_filename_noY = "./icarus_fnal_noY.json"

# Time spacing in file
time = np.linspace(0,100,1000, endpoint=False)

# Filename and histogram with electronics response
fER = "elecResp_ICARUS.root"
fERh = uproot.open(fER)["ER"]

# Filename with simulated signal response
SR_filename = "./mc/WFresults_Plane%i_WC.root" % IPLANE

angle_min = 20
angle_max = 80
angle_increment = 2
all_angles = np.array(list(range(angle_min, angle_max, angle_increment))) # 20-80, inclusive

# Whether to include Guassian broadening in fit
FITGAUS = False

# Include skew?
FITSKEW = False

# Plot various things
PLTPATHS = False # electron paths
PLTFR = False # field response
PLTSR = False # signal response
PLTFIT = False # fit

#Time increment
dt = time[1] - time[0]


# In[42]:


# FIT_ANGLES = np.array([30, 36, 40, 46, 50, 54, 60, 64])
FIT_ANGLES = np.array(all_angles[3:-7])
FIT_ANGLES


# In[43]:


# Helper functions
def convolve(f1, f2):
    '''
    Return the simple convolution of the two arrays using FFT+mult+invFFT method.
    '''
    # fftconvolve adds an unwanted time shift
    #from scipy.signal import fftconvolve
    #return fftconvolve(field, elect, "same")
    s1 = np.fft.fft(f1)
    s2 = np.fft.fft(f2)
    sig = np.fft.ifft(s1*s2)

    return np.real(sig)

def gaus(t, sigma):
    return np.exp(-t**2/(2*sigma**2))

def skewgaus(t, sigma, skew):
    skew = -1/skew
    return gaus(t, sigma)*(1 + special.erf(skew*t/np.sqrt(2)))

def agaus(t, sigmalo, sigmahi):
    return np.exp(-t**2/(2*sigmalo**2))*(t < 0) + np.exp(-t**2/(2*sigmahi**2))*(t >= 0)

def norm_v(v):
    if IPLANE == 2:
        return v.max()
    else:
        return np.abs(v.min())

def norm(v, vnorm=None):
    if vnorm is None:
        vnorm = v
    return v / norm_v(vnorm)

def areanorm(v):
    return v / np.sum(v)
    
def center(t, v):
    if IPLANE == 0: # Ind-0 -- center on down-peak
        return t - t[np.argmin(v)]
    elif IPLANE == 2: # Collection -- center on peak
        return t - t[np.argmax(v)]
    else: # Ind-1 -- center on zero-cross
        center_ind = np.argmin(np.abs(v[np.argmax(v):np.argmin(v)])) + np.argmax(v)
        return t - t[center_ind]


# In[44]:


#Old electronics response lookup table
ER_val = fERh.values()
ER_time = fERh.axis().centers()
ER = interpolate.interp1d(ER_time, ER_val, kind="linear", bounds_error=False, fill_value=0)

#Parametrized electronics response
er_width = 1.3
def electronics_response(t,width):
    mu = 0
    amp = 10.
    y = (t>=mu)*amp*(1-np.exp(-0.5*(0.9*(t-mu)/width)**2))*np.exp(-0.5*(0.5*(t-mu)/width)**2)
    return y

plt.clf()

plt.plot(time, electronics_response(time,er_width) / electronics_response(time,er_width).max())
#plt.plot(ER_time, ER_val / ER_val.max())
plt.xlabel("Time [$\\mu$s]")
plt.ylabel("Amplitude")
plt.xlim([0, 10])
plt.title("Electronics Response")
plt.tight_layout()

if dosave: plt.savefig(savedir + "elecResp.pdf")


# In[45]:


def RC_filter(t, tau):
    return (t/tau-2)*(1/tau)*np.exp(-t/tau)*(t>=0)


# In[46]:


plt.clf()

ER_plt = electronics_response(time,er_width)
plt.plot(time, ER_plt / ER_plt.max(), label="Bessel")

ER_plt_conv = -convolve(ER_plt, RC_filter(time, 1)) + ER_plt
plt.plot(time, ER_plt_conv / np.abs(ER_plt_conv).max(), label="Bessel $\\circledast$ RC-RC")
plt.xlim([0, 15])
plt.legend()


# In[47]:


dat = json.load(open(WCpaths_filename_nom))

driftV = dat["FieldResponse"]["speed"]*1e3 # mm/us

thispaths = dat["FieldResponse"]["planes"][IPLANE]["PlaneResponse"]["paths"]

pitchpos_f = [path["PathResponse"]["pitchpos"] for path in thispaths] # mm

paths_f = [np.array(path["PathResponse"]["current"]["array"]["elements"])
             for path in thispaths]

pitchpos = []
paths_nom = []

for i, (ppos, path) in enumerate(zip(pitchpos_f, paths_f)):
    pitchpos.append(ppos)
    paths_nom.append(path)
    if -ppos not in pitchpos_f:
        pitchpos.append(-ppos)
        paths_nom.append(path)
    # handle boundary between (half-)wires
    else:
        paths_nom[-1] = tuple([paths_nom[-1], paths_f[pitchpos_f.index(-ppos)]])
        
pitchpos, paths_nom = zip(*sorted(zip(pitchpos, paths_nom), key=lambda pair: pair[0]))
pitchpos = np.array(pitchpos)

pitchpos_interp = np.linspace(pitchpos.min(), pitchpos.max(), 2101) + 0.015


# In[48]:


np.argwhere(np.abs(pitchpos) == 0.9)[0]


# In[49]:


dat = json.load(open(WCpaths_filename_noY))

driftV = dat["FieldResponse"]["speed"]*1e3 # mm/us

thispaths = dat["FieldResponse"]["planes"][IPLANE]["PlaneResponse"]["paths"]

pitchpos_f = [path["PathResponse"]["pitchpos"] for path in thispaths] # mm

paths_f = [np.array(path["PathResponse"]["current"]["array"]["elements"])
             for path in thispaths]

pitchpos = []
paths_noY = []

for i, (ppos, path) in enumerate(zip(pitchpos_f, paths_f)):
    pitchpos.append(ppos)
    paths_noY.append(path)
    if -ppos not in pitchpos_f:
        pitchpos.append(-ppos)
        paths_noY.append(path)
    # handle boundary between (half-)wires
    else:
        paths_noY[-1] = tuple([paths_noY[-1], paths_f[pitchpos_f.index(-ppos)]])
        
pitchpos, paths_noY = zip(*sorted(zip(pitchpos, paths_noY), key=lambda pair: pair[0]))
pitchpos = np.array(pitchpos)

paths_noY = list(paths_noY)

# The 1.2mm path still "collects" on collection for some reason -- overwrite this
paths_noY_09mm = paths_noY[np.argwhere(np.abs(pitchpos) == 0.9)[0][0]]
for i_12 in np.argwhere(np.abs(pitchpos) == 1.2):
    paths_noY[i_12[0]] = paths_noY_09mm

pitchpos_interp = np.linspace(pitchpos.min(), pitchpos.max(), 2101) + 0.015


# In[50]:


plt.clf()

for p, path in zip(pitchpos, paths_nom):
    if isinstance(path, tuple):
        plt.plot(time, -path[0])
    else:
        plt.plot(time, -path)

plt.title("Wire-Cell Electron Path Responses")
plt.xlabel("Time [$\\mu$s]")
plt.ylabel("Current")

plt.xlim([57 - 2*(2-IPLANE), 67 - 2*(2-IPLANE)])
plt.tight_layout()

# plt.axvline([time[CENTER_IND]], color="r")

if dosave: plt.savefig(savedir + "pathResp.pdf")


# In[51]:


plt.clf()

for p, path in zip(pitchpos, paths_noY):
    if isinstance(path, tuple):
        plt.plot(time, -path[0])
    else:
        plt.plot(time, -path)
    
plt.title("Wire-Cell Electron Path Responses")
plt.xlabel("Time [$\\mu$s]")
plt.ylabel("Current")

plt.xlim([58 - 2*(2-IPLANE), 66 - 2*(2-IPLANE)])
plt.tight_layout()

if dosave: plt.savefig(savedir + "pathResp_noY.pdf")


# In[52]:


if IPLANE == 2:
    txt_x = 0.025
else:
    txt_x = 0.55


# In[53]:


# Pick plane, load in data, etc

# Filename with measured signal response
file_path = "../data/run_"
file_path = "./data/far_TrackBased_official/"

data_files = []
run_list = ["8749","9133"]
tpc_list = ["EE","EW","WE","WW"]
for run in run_list:
    for tpc in tpc_list:
        data_files.append(uproot.open(file_path+run+"/WFresults_Plane"+str(IPLANE)+"_"+tpc+".root"))
        
#Create arrays over angle bins to store angle info, data filenames, data products
angle_min = 20
angle_max = 80
angle_increment = 2

nall_angles = len(all_angles)
nfiles = len(data_files)
ndata = 401
data_hists_allfiles = [[[0.0 for k in range(ndata)] for j in range(nall_angles)] for i in range(nfiles)]
data_hists    = [[0.0 for k in range(ndata)] for j in range(nall_angles)]
data_err_vecs = [[0.0 for k in range(ndata)] for j in range(nall_angles)]

data_times = []
data_when_list = []
data_fit_list = []

for j,angle in enumerate(all_angles):
    thlo = angle
    thhi = thlo + angle_increment
    
    angle_data_hists = []
    angle_err_hists = []
    for i, uhf in enumerate(data_files):
        data = uhf["AnodeRecoHist1D_%ito%i" % (thlo, thhi)]
        data_err = uhf["AnodeTrackUncertHist2D_%ito%i" % (thlo, thhi)].to_numpy()[0][5]
        angle_data_hists.append(data.values())
        angle_err_hists.append(data_err)
        data_hists[j] += data.values()
        data_err_vecs[j] = np.sqrt(np.square(data_err_vecs[j])+np.square(data_err))
        if i == 0:
            data_times.append(center(data.axis().centers(), data.values()))
            # data_times.append(data.axis().centers())
            data_when_list.append(np.abs(data_times[-1]) < PLT_PERIOD)
            data_fit_list.append(np.abs(data_times[-1]) < FIT_PERIOD)

    # NOTE: use std-dev to get uncertainty -- ignore intrinsic error
    data_hists[j][:] = np.average(angle_data_hists, axis=0)
    data_err_vecs[j][:] = np.std(angle_data_hists, axis=0) / len(data_files)


# In[54]:


#Helper function to compute chi2
def calc_chi2(pred, meas, err):
    return np.sum(((meas - pred)/err)**2)

#Convert from prepared time (containing angle info) to actual time
def convert_time(t0):
    angle_bin = np.floor(t0/(2*PLT_PERIOD)).astype(int)
    angle_bin_set = to_set(angle_bin)
    angle_bin_index = np.zeros(len(angle_bin),dtype=int)
    for i in range(len(angle_bin_set)):
        angle_bin_index += i*(angle_bin==angle_bin_set[i]).astype(int)
    t = (t0 % (2*PLT_PERIOD)) - PLT_PERIOD
    angle = (all_angles[angle_bin_set]+all_angles[angle_bin_set+1])/2
    
    return (t,angle_bin_set,angle_bin_index,angle)

def contains(array,x):
    return np.any(array == x)

#returns the set of elements in the input array - no duplicates
def to_set(array):
    return np.unique(array)

#Takes as input the scaled times for each path, the WC paths and the interpolated path positions and returns a list of interpolated paths
def interpolate_paths(paths, pitchpos_interp):
    paths_interp = []

    for j, p in enumerate(pitchpos_interp):
        i_pitchpos = int((p - pitchpos[0]+1e-6) / (pitchpos[1] - pitchpos[0]))
        if i_pitchpos == len(pitchpos) - 1:
            path = paths[i_pitchpos]
        else:
            F1 = paths[i_pitchpos]
            F2 = paths[i_pitchpos+1]

            # handle boundary between (half-)wires
            if isinstance(F1, tuple):
                if p > pitchpos[i_pitchpos]:
                    F1 = F1[0]
                else:
                    F1 = F1[1]
            if isinstance(F2, tuple):
                if p > pitchpos[i_pitchpos+1]:
                    F2 = F2[0]
                else:
                    F2 = F2[1]

            interp = (pitchpos[i_pitchpos+1] - p)/(pitchpos[1] - pitchpos[0])
            path = F1*interp + F2*(1-interp)

        paths_interp.append(path)
    return paths_interp

def shifted_paths_sum(angle, paths_interp, pitchpos_interp):
    thxw = angle*np.pi/180
    shift = (np.outer(np.tan(thxw),pitchpos_interp/driftV/dt)).astype(int)
    summed_paths_angle = []
    for i in range(len(angle)):
        summed_paths_angle.append(np.zeros(paths_interp[0].shape))
        
        for j in range(len(paths_interp)):
            s = shift[i][j]
            if s < 0:
                summed_paths_angle[-1][:s] += -paths_interp[j][-s:]
            elif s > 0:
                summed_paths_angle[-1][s:] += -paths_interp[j][:-s]
            else:
                summed_paths_angle[-1] += -paths_interp[j]
                
    return summed_paths_angle

#Finds the index where a strictly increasing array first goes above 0
def find_zero(array):
    return np.argmax(array >= 0)

#Scale the array of input times t by values sl and sr for the left and right sides of 0, respectively
def scale_time(t,sl,sr):
    t0_index = find_zero(t)
    scaled_time = np.zeros(len(t))
    scaled_time[:t0_index] = t[:t0_index]*sl
    scaled_time[t0_index:] = t[t0_index:]*sr
    return scaled_time

#Weights each path based on left and right parameters wl, wr
def weight_path(path, path_time, wl, wr):
    center_index = find_zero(path_time)
    path_left  = path*wl
    path_right = path*wr
    return np.concatenate((path_left[:center_index], path_right[center_index:]))

#Computes a scaled path at given offset by evaluating an input path at times path_time
def scale_path(path, path_time, scale_left, scale_right, wl, wr):
    scaled_time = scale_time(path_time, scale_left, scale_right)
    if isinstance(path, tuple):
        weighted_path_left  = weight_path(path[0],path_time,wl,wr)
        weighted_path_right = weight_path(path[1],path_time,wl,wr)
        pathl_interp = interpolate.interp1d(path_time, weighted_path_left, kind="linear", bounds_error=False, fill_value=0)
        pathr_interp = interpolate.interp1d(path_time, weighted_path_right, kind="linear", bounds_error=False, fill_value=0)
        scaled_path = [pathl_interp(scaled_time), pathr_interp(scaled_time)]
        scaled_path = tuple(scaled_path)

    else:
        weighted_path = weight_path(path,path_time,wl,wr)
        path_interp = interpolate.interp1d(path_time, weighted_path, kind="linear", bounds_error=False, fill_value=0)
        scaled_path = path_interp(scaled_time)  
        
    return scaled_path

#Sum the field responses over drift offset to get the overall field response
def compute_field_response(shifted_paths, angles):
    field_response_angles = []
    for i in range(nangles):
        field_response_angles.append(-sum(shifted_paths[i]))
    return field_response_angles

# TODO FIX -- this doesn't really work
def compute_path_center_ind(path):
    if isinstance(path, tuple):
        v = -path[0]
    else:
        v = -path
        
    if IPLANE == 2:
        center_ind = np.argmax(v)
    #elif IPLANE == 0:
    #    center_ind = np.argmin(v)
    else: # Ind-1 -- center on zero-cross
        center_ind = np.argmax(v[575:] <= 0) + 575
    return center_ind

def compute_path_center_time(path, time):
    return time - time[CENTER_IND]


# In[55]:


centers = []
for p in paths_nom[100:110]:
    centers.append(compute_path_center_ind(p))
centers


# In[19]:


def mc_waveform(t0, er_width=1.3, rcrc_width=0, sl0=1, sr0=1, sl1=0, sr1=0, sl2=0, sr2=0,
                weight_ratio=0, nom_frac=1, sigma_a=SIGMA_A, sigma_b=SIGMA_B, mu0=[0],
                      ANGLES=None, normalize=True):
    if ANGLES is None:
        ANGLES = MC_ANGLES
    
    nangle = len(ANGLES)
    if len(mu0) == 1:
        mu0 = mu0*nangle
    else: 
        assert(len(mu0) == nangle)
        
    t0_angle = np.split(t0, nangle)
    
    #Compuate scaled times and interpolate to get scaled paths, all for each angle
    offset_param = 1-np.exp(-np.absolute(pitchpos)/1.5)  #3 mm between wires
    # offset_param = (np.abs(pitchpos)%3)/3
    scaling_left  = sl1*offset_param + sl2*offset_param**2
    scaling_right = sr1*offset_param + sr2*offset_param**2
    weights_left  = (1 - weight_ratio)
    weights_right = 1. / weights_left
    
    scaled_paths = []
    for i, (p_nom, p_noY) in enumerate(zip(paths_nom, paths_noY)):
        if isinstance(p_nom, tuple):
            p = (p_nom[0]*nom_frac + p_noY[0]*(1-nom_frac), 
                 p_nom[1]*nom_frac + p_noY[1]*(1-nom_frac))
        else:
            p = p_nom*nom_frac + p_noY*(1-nom_frac)
        
        sp = scale_path(p, compute_path_center_time(p, time),
                        sl0 + scaling_left[i], sr0 + scaling_right[i],
                        weights_left, weights_right)
        scaled_paths.append(sp)
            
    # interpolate the paths with a finer spacing (0.03mm)
    # avoid edge effects by spacing in between the discontinuities in the paths (every 1.5mm)
    paths_interp = interpolate_paths(scaled_paths, pitchpos_interp)
        
    # Compute the interpolated field response at each track angle
    field_response_angles = shifted_paths_sum(ANGLES, paths_interp, pitchpos_interp)
        
    #Convolve with electronics response and Gaussian smearing
    gaus_sigma = np.sqrt(sigma_a**2 + (np.tan(ANGLES*np.pi/180)*sigma_b)**2)
    
    # Build the electronics response
    ER = electronics_response(time, er_width)
    if rcrc_width > 0:
        ER = -convolve(ER, RC_filter(time, rcrc_width)) + ER
    if normalize:
        ER = ER / np.abs(ER).max()
    
    mc_val_interp_list = []
    vals = []
    for i in range(nangle):
        SR = convolve(field_response_angles[i], ER)
        if gaus_sigma[i] > 1e-4:
            SR_gaus = convolve(SR, gaus(time, gaus_sigma[i]))
        else:
            SR_gaus = SR
        if normalize:
            SR_gaus = norm(SR_gaus)
        centered_time = center(time, SR_gaus)
        SR_interp = interpolate.interp1d(centered_time + mu0[i], SR_gaus, kind="linear", bounds_error=False, fill_value=0)
        vals.append(SR_interp(t0_angle[i]))

    return np.concatenate(vals)


# In[20]:


FIT_ANGLES = np.array([26, 44, 62])

FIT_ANGLES = np.array(all_angles[3:-7])


# In[21]:


#Create arrays for time,data in fit and plot regions
time_fitregion_list = []
time_plotregion_list = []

data_fitregion_list = []
data_plotregion_list = []
data_fitregion_norm_list = []
data_plotregion_norm_list = []

data_err_fitregion_list = []
data_err_plotregion_list = []
data_err_fitregion_norm_list = []
data_err_plotregion_norm_list = []

#Loop over angles
nangles = len(FIT_ANGLES)
for i_angle in range(nangles):
    angle = FIT_ANGLES[i_angle]
    i_angle_h = np.where(all_angles == angle)[0][0]
    thlo = angle
    thhi = thlo + angle_increment
    
    time_fitregion_list.append(data_times[i_angle_h][data_fit_list[i_angle_h]])
    time_plotregion_list.append(data_times[i_angle_h][data_when_list[i_angle_h]])

    data_fitregion_list.append(data_hists[i_angle_h][data_fit_list[i_angle_h]])
    data_plotregion_list.append(data_hists[i_angle_h][data_when_list[i_angle_h]])
    data_fitregion_norm_list.append(norm(data_fitregion_list[-1]))
    data_plotregion_norm_list.append(norm(data_plotregion_list[-1]))

    data_err_fitregion_list.append(data_err_vecs[i_angle_h][data_fit_list[i_angle_h]])
    data_err_plotregion_list.append(data_err_vecs[i_angle_h][data_when_list[i_angle_h]])
    data_err_fitregion_norm_list.append(norm(data_err_fitregion_list[-1], data_fitregion_list[-1]))
    data_err_plotregion_norm_list.append(norm(data_err_plotregion_list[-1], data_plotregion_list[-1]))

time_fitregion           = np.concatenate(time_fitregion_list)
time_plotregion          = np.concatenate(time_plotregion_list)

data_fitregion           = np.concatenate(data_fitregion_list)
data_plotregion          = np.concatenate(data_plotregion_list)
data_fitregion_norm      = np.concatenate(data_fitregion_norm_list)
data_plotregion_norm     = np.concatenate(data_plotregion_norm_list)

data_err_fitregion       = np.concatenate(data_err_fitregion_list)
data_err_plotregion      = np.concatenate(data_err_plotregion_list)
data_err_fitregion_norm  = np.concatenate(data_err_fitregion_norm_list)
data_err_plotregion_norm = np.concatenate(data_err_plotregion_norm_list)

# Fit

# everything you could fit for -- using the mc_waveform spec
all_fit_params = inspect.getfullargspec(mc_waveform).args[1:-2]

# all defaults and bounds
default_params = dict(zip(all_fit_params, inspect.getfullargspec(mc_waveform).defaults))
default_params["rcrc_width"] = 1
# setup bounds manually
bounds = {
    "er_width": (0.5, 1.5),
    "rcrc_width": (0, 2),
    "sl0": (0.05, 5),
    "sr0": (0.05, 5),
    "sl1": (-0.2, 2),
    "sr1": (-0.2, 2),
    "sl2": (-0.2, 2),
    "sr2": (-0.2, 2),
    "weight_ratio": (-0.9, 0.9),
    "nom_frac": (0, 1.5),
    "sigma_a" : (SIGMA_A / 2, SIGMA_A * 2),
    "sigma_b" : (SIGMA_B / 2, SIGMA_B * 2),
    "mu0": (-2, 2)
}

assert(default_params.keys() == bounds.keys())

nmu = nangles
# nmu = 1

# Option to fit for each waveform center individually
for i in range(nmu):
    name = "mu" + str(i)
    default_params[name] = 0
    bounds[name] = (-2, 2)
    if i > 0:
        all_fit_params.append(name)
        
MC_ANGLES = FIT_ANGLES
MC_ANGLES


# In[22]:


debugging = True

#Default prediction for comparison
do_fit = False

pred_default_plotregion = mc_waveform(time_plotregion)
pred_default_plotregion_norm_list = [norm(v) for v in np.split(pred_default_plotregion, nangles)]

pred_default_plotregion_weird = mc_waveform(time_plotregion, rcrc_width=1)
pred_default_plotregion_norm_weird_list = [norm(v) for v in np.split(pred_default_plotregion_weird, nangles)]

plt.close("all")

for i,a in enumerate(MC_ANGLES):
    thlo = a
    thhi = thlo + angle_increment
    
    plt.figure(i)
    plt.plot(time_plotregion_list[i], pred_default_plotregion_norm_list[i])
    plt.plot(time_plotregion_list[i], pred_default_plotregion_norm_weird_list[i])
    plt.text(txt_x, 0.8, "$%i^\\circ < \\theta_{xw} < %i^\\circ$" % (thlo, thhi),
             fontsize=16,transform=plt.gca().transAxes)
    
    plt.errorbar(time_plotregion_list[i], data_plotregion_norm_list[i], data_err_plotregion_norm_list[i],
             linestyle="none", marker=".", color="black", label="Data")


# In[ ]:





# In[23]:


# Define the parameters we will fit for here

tofit = [
    #"er_width",
    "rcrc_width",
    "sl0",
    "sr0",
    "sl1",
    "sr1",
    "weight_ratio",
    #"nom_frac",
] 

if "TOFIT" in os.environ:
    tofit = os.environ["TOFIT"].split(",")

tofit += ["mu%i" % i for i in range(nmu)]

nparams = len(tofit) - nmu

def mc_waveform_fit(t0, *params):
    assert(len(params) == len(tofit))
    args = {}
    for k, p in zip(tofit, params):
        if k == "mu0":
            args[k] = [p]
        elif k.startswith("mu"):
            args["mu0"].append(p)
        else:
            args[k] = p
            
    return mc_waveform(t0, **args)
            
p0 = [default_params[k] for k in tofit]
bound_lo, bound_hi = list(zip(*[bounds[k] for k in tofit]))

MC_ANGLES = FIT_ANGLES


# In[24]:


#%timeit mc_waveform(time_fitregion)


# In[25]:


#cProfile.run('mc_waveform(time_fitregion)', sort="time")


# In[26]:


debugging = False

print('fitting')
popt, perr = curve_fit(mc_waveform_fit, time_fitregion, data_fitregion_norm, 
                       p0=p0, sigma=data_err_fitregion_norm, absolute_sigma=True, bounds=(bound_lo, bound_hi))

for ip in range(len(p0)):
    print(tofit[ip], p0[ip], bound_lo[ip], bound_hi[ip], popt[ip])

#Create arrays for mc in fit and plot regions
print('pred')
pred_fitregion  = mc_waveform_fit(time_fitregion, *popt)
pred_plotregion = mc_waveform_fit(time_plotregion, *popt)

pred_fitregion_norm_list = [norm(v) for v in np.split(pred_fitregion, nangles)]
pred_plotregion_norm_list = [norm(v) for v in np.split(pred_plotregion, nangles)]

chi2 = calc_chi2(np.concatenate(pred_fitregion_norm_list), data_fitregion_norm, data_err_fitregion_norm)
ndf = len(np.concatenate(pred_fitregion_norm_list))

print('fit chi2/ndf =     '+str(chi2)+' / '+str(ndf))


# In[27]:


debugging = True

#Default prediction for comparison
do_fit = False
pred_default_fitregion  = mc_waveform(time_fitregion, mu0=list(popt[-nmu:]))
pred_default_plotregion = mc_waveform(time_plotregion, mu0=list(popt[-nmu:]))

pred_default_fitregion_norm_list = [norm(v) for v in np.split(pred_default_fitregion, nangles)]
pred_default_plotregion_norm_list = [norm(v) for v in np.split(pred_default_plotregion, nangles)]

chi2_default = calc_chi2(np.concatenate(pred_default_fitregion_norm_list), data_fitregion_norm, data_err_fitregion_norm)
print('default chi2/ndf = '+str(chi2_default)+' / '+str(ndf))

#Ratio of Default/Fit MC
pred_ratio_plotregion_norm = np.concatenate(pred_default_plotregion_norm_list)/np.concatenate(pred_plotregion_norm_list)
pred_ratio_plotregion_norm_list = np.split(pred_ratio_plotregion_norm, nangles)


# In[28]:


plt.close("all")

#Draw subplots
for i,a in enumerate(FIT_ANGLES):
    thlo = a
    thhi = thlo + angle_increment
    plt.figure(i+1)
    
    fig,(ax1,ax2) = plt.subplots(num=i+1,figsize=(6.4, 5.6),nrows=2, sharex=True, gridspec_kw={'height_ratios': [3, 1]})

    ax1.errorbar(time_plotregion_list[i], data_plotregion_norm_list[i], data_err_plotregion_norm_list[i],
                 linestyle="none", marker=".", color="black", label="Data")
    ax1.plot(time_plotregion_list[i], pred_default_plotregion_norm_list[i], 
             label="Nominal Response")
    ax1.plot(time_plotregion_list[i], pred_plotregion_norm_list[i], 
             label="Fit Response",color='orange')
    ax1.legend(ncol=2 ,loc='upper center', bbox_to_anchor=(0.5, 1.35))
    ax1.axhline(0, color="r")
    
#     ax2.errorbar(time_plotregion_list[i], data_plotregion_norm_list[i]/pred_plotregion_norm_list[i],
#             abs(data_err_plotregion_norm_list[i]/pred_plotregion_norm_list[i]),linestyle="none", marker=".", color="black")
#     ax2.set_ylim([0.7, 1.3])
#     ax2.axhline(1, color='orange')
    
    ax2.errorbar(time_plotregion_list[i], data_plotregion_norm_list[i]-pred_default_plotregion_norm_list[i],
            abs(data_err_plotregion_norm_list[i]),linestyle="none", marker=".")
    
    ax2.errorbar(time_plotregion_list[i], data_plotregion_norm_list[i]-pred_plotregion_norm_list[i],
            abs(data_err_plotregion_norm_list[i]),linestyle="none", marker=".")

    ax2.set_ylim([-0.05, 0.05])
    ax2.axhline(0, color='black')

    
    #ax2.plot(time_plotregion_list[i], pred_ratio_plotregion_norm_list[i], label="Default/Fit MC Ratio")
    
    #plt.xlim([-5, 5])
    plt.subplots_adjust(hspace=0)
    ax1.text(txt_x, 0.85, "$%i^\\circ < \\theta_{xw} < %i^\\circ$" % (thlo, thhi),fontsize=16,transform=ax1.transAxes)
    
    if IPLANE == 2: 
        txt_xy = (0.675, 0.6)
    else:
        txt_xy = (0.05, 0.05)
    #ax1.text(*txt_xy, "$\\chi^2_{nom}/n:$\n\t$%.0f /(%i - %i)$\n$\\chi^2_{fit}/n:$\n\t$%.0f /(%i - %i)$" % (chi2_default, ndf, nmu, chi2, ndf, len(p0)),
    #        fontsize=10, transform=ax1.transAxes)
    
    ax2.set_xlabel("Time [$\\mu$s]")
    ax1.set_ylabel("Amp. Normalized")
    ax2.set_ylabel("Data - Resp.", fontsize=12)
    
    plt.tight_layout()
    if dosave:
        plt.savefig(savedir + "signalfit_th%ito%i.pdf" % (thlo, thhi))


# In[29]:


fit_param_args = dict(zip(tofit[:nparams], popt[:nparams]))


# In[30]:


# try out plotting some angles to see how the fit compares to the nominal
MC_ANGLES = np.array(list(range(0, 90, 5)))
plt_time = np.linspace(-40, 40, 800)
plt_times = np.tile(plt_time, len(MC_ANGLES))

nom_wvf = mc_waveform(plt_times)
nom_wvfs = np.split(nom_wvf, len(MC_ANGLES))

fit_wvf = mc_waveform(plt_times, **fit_param_args)
fit_wvfs = np.split(fit_wvf, len(MC_ANGLES))
plt.close("all")

for i, th in enumerate(MC_ANGLES):
    plt.figure(i)
    plt.plot(plt_time, norm(nom_wvfs[i]), label="Nominal")
    plt.plot(plt_time, norm(fit_wvfs[i]), label="Fit")
    plt.title("Plane %i, $\\theta_{xw} = %i^\\circ$ Signal Response" % (IPLANE, th))
    plt.legend()
    if th < 75:
        plt.xlim([-20, 20])
        
    plt.xlabel("Time [$\\mu$s]")
    plt.tight_layout()
    if dosave: plt.savefig(savedir + "fitvnom_signals_th%i.pdf" %(int(th)))


# In[31]:


plt.clf()

ER_plt = electronics_response(time, er_width)
plt.plot(time, ER_plt / ER_plt.max(), label="Nominal")

fit_er_width = fit_param_args.get("er_width", 1.3)

ER_fit = electronics_response(time, fit_er_width)
if fit_param_args.get("rcrc_width", 0.) > 0:
    ER_fit = -convolve(ER_fit, RC_filter(time, fit_param_args["rcrc_width"])) + ER_fit
ER_fit = ER_fit / np.abs(ER_fit).max()

plt.plot(time, ER_fit / ER_fit.max(), label="Fit")

plt.xlim([0, 15])
plt.legend()
plt.title("Electronics Response")
plt.xlabel("Time [$\\mu$s]")
plt.tight_layout()

if dosave: plt.savefig(savedir + "fit_elecResp.pdf")


# In[32]:


sl1 = fit_param_args["sl1"]
sr1 = fit_param_args["sr1"]
sl0 = fit_param_args["sl0"]
sr0 = fit_param_args["sr0"]
weight_ratio = fit_param_args["weight_ratio"]
nom_frac = 1

#Compuate scaled times and interpolate to get scaled paths, all for each angle
offset_param = 1-np.exp(-np.absolute(pitchpos)/1.5)  #3 mm between wires
scaling_left  = sl1*offset_param
scaling_right = sr1*offset_param
weights_left  = (1 - weight_ratio)
weights_right = 1. / weights_left

scaled_paths = []
for i, (p_nom, p_noY) in enumerate(zip(paths_nom, paths_noY)):
    if isinstance(p_nom, tuple):
        p = (p_nom[0]*nom_frac + p_noY[0]*(1-nom_frac), 
             p_nom[1]*nom_frac + p_noY[1]*(1-nom_frac))
    else:
        p = p_nom*nom_frac + p_noY*(1-nom_frac)

    sp = scale_path(p, compute_path_center_time(p, time),
                    sl0 + scaling_left[i], sr0 + scaling_right[i],
                    weights_left, weights_right)
    scaled_paths.append(sp)


# In[33]:


plt.clf()

for p, path in zip(pitchpos, scaled_paths):
    if isinstance(path, tuple):
        plt.plot(time, -path[0])
    else:
        plt.plot(time, -path)
    
plt.title("Fit Electron Path Responses")
plt.xlabel("Time [$\\mu$s]")
plt.ylabel("Current")

plt.xlim([57 - 2*(2-IPLANE), 67 - 2*(2-IPLANE)])
plt.tight_layout()

# plt.axvline([time[CENTER_IND]], color="r")

if dosave: plt.savefig(savedir + "fit_pathResp.pdf")


# In[34]:


for p, path in zip(pitchpos, scaled_paths):
    if p > 0:
        othr_path = scaled_paths[list(pitchpos).index(-p)]
        if isinstance(path, tuple):
            path = path[0]
            othr_path = othr_path[1]

        assert((path == othr_path).all())


# In[35]:


CENTRAL_PATHS = [100, 101, 102, 103, 104, 105]


# In[36]:


plt.clf()

for ip in CENTRAL_PATHS:
    p = pitchpos[ip]
    path = scaled_paths[ip]
    
    if isinstance(path, tuple):
        path = path[0]

    plt.plot(time, -path)
    
plt.title("Fit Electron Path Responses")
plt.xlabel("Time [$\\mu$s]")
plt.ylabel("Current")

plt.xlim([57 - 2*(2-IPLANE), 67 - 2*(2-IPLANE)])
plt.tight_layout()


# In[37]:


# Compute the normalization for the scaled paths
norm_nom = norm_v(mc_waveform(plt_time, ANGLES=np.array([0]), normalize=False, er_width=fit_param_args.get("er_width", 1.3)))
norm_fit = norm_v(mc_waveform(plt_time, ANGLES=np.array([0]), normalize=False, **fit_param_args))


# In[38]:


# Save paths if configured
scaled_paths_conv = []

for p in scaled_paths:
    if isinstance(p, tuple):
        p = p[0]
        
    if fit_param_args.get("rcrc_width", 0.) > 0:
        f = RC_filter(time, fit_param_args["rcrc_width"])
        p = (-convolve(p, f) + p)
        
    # normalize
    p = p * norm_nom / norm_fit
        
    scaled_paths_conv.append(p)
    
if dosave:
    for p, path in zip(pitchpos, scaled_paths_conv):
        with open(savedir + ("paths_plane%i_pitch%.1f" % (IPLANE, p)).replace(".","_") + ".txt", "w") as f:
            for e in path:
                f.write(str(e) + "\n")


# In[39]:


for ip in CENTRAL_PATHS:
    p = pitchpos[ip]
    path = scaled_paths_conv[ip]
    
    if isinstance(path, tuple):
        path = path[0]
        
    path_nom = paths_nom[ip]
    
    if isinstance(path_nom, tuple):
        path_nom = path_nom[0]

    print(p, np.sum(-path), np.sum(-path_nom))


# In[40]:


# And parameters
if dosave:
    with open(savedir + "popt.txt", "w") as f:
        for ip in range(len(p0)):
            print(tofit[ip], p0[ip], bound_lo[ip], bound_hi[ip], popt[ip], file=f)

