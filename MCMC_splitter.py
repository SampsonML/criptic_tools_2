"""
Fitting Script for diffusion coefficents CR study
Matt Sampson 2022 -- updated for split output
"""
import argparse
import numpy as np
from numpy import diff
import astropy.units as u
import astropy.constants as const
from glob import glob
import os.path as osp
import matplotlib.pyplot as plt
import cmasher as cmr
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import matplotlib as mpl
from matplotlib import rc_context
import argparse
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.colors as colors
from scipy.optimize import minimize
from scipy.stats import levy_stable
from scipy.stats import invgamma
from scipy import stats
from scipy.stats import norm
from scipy.stats import rv_histogram
###################################
# MCMC things
#import scipy.stats as st, levy
from levy import levy
import emcee
import corner
from multiprocessing import Pool, cpu_count
###################################
############################################
#### Astro Plot Aesthetics Pre-Amble
############################################
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.top'] = True
mpl.rcParams['xtick.bottom'] = True
mpl.rcParams['ytick.left'] = True
mpl.rcParams['ytick.right'] = True
mpl.rcParams['xtick.minor.visible'] = True
mpl.rcParams['ytick.minor.visible'] = True
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.Set1.colors)
#Direct input 
#plt.rcParams['text.latex.preamble']=[r"\usepackage{lmodern}"]
#Options
params =   {'text.usetex' : True,
            'font.size' : 11,
            'font.family' : 'lmodern'
            }
plt.rcParams.update(params) 

### Setting new default colour cycles
# Set the default color cycle custom colours (Synthwave inspired)
green_m = (110/256, 235/256, 52/256)
purp_m = (210/356, 113/256, 235/256)
blue_m = (50/256, 138/256, 129/256)
dblue_m = (0.1, 0.2, 0.5)
salmon_m = (240/256, 102/256, 43/256)
dgreen_m = (64/256, 128/256, 4/256)
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=[dblue_m, green_m, purp_m,  dgreen_m, blue_m]) 
# Dark background
#plt.style.use('dark_background')
##############################################
# Parse arguements
parser = argparse.ArgumentParser(
    description="Analysis script")
parser.add_argument("-d", "--dir",
                    help="directory containing run to analyze",
                    default=".", type=str)
parser.add_argument("-m", "--Mach",
                    help="Amount of chunk files",
                    default="1", type=float)
parser.add_argument("-a", "--Alfven",
                    help="Amount of chunk files",
                    default="1", type=float)
parser.add_argument("-s", "--c_s",
                    help="Amount of chunk files",
                    default="1", type=float)
parser.add_argument("-i", "--chi",
                    help="Amount of chunk files",
                    default="1", type=float)
parser.add_argument("-r", "--dens",
                    help="Amount of chunk files",
                    default="1", type=float)                  
parser.add_argument("-n", "--name",
                    help="name to save",
                    default="test", type=str)

args = parser.parse_args()

# ------------------ #
# Initialise Problem
# ------------------ #
displacement = [0] ; displacement_par = [0]
displacement_perp_x = [0] ; displacement_perp_y = [0]
time_vals = [0]
# 1 /2 Domain Size
L = 3.09e19
# Getting MHD specific parameters
Mach = args.Mach
Alfven = args.Alfven
chi = 1 * 10**(-args.chi)
c_s = args.c_s
rho_0 = 2e-21 # This value for all sims
B =  c_s * Mach * (1 / Alfven) * np.sqrt(4 * np.pi * rho_0)
# Defining Speeds
V_alfven = B / (np.sqrt(4 * np.pi * rho_0))
Vstr = (1/np.sqrt(chi)) * V_alfven
# Defining timescales
t_cross = 2 * L / Vstr
t_turb = (2 * L) / (c_s * Mach)
t_turnover = t_turb / 2
t_alfven = (2 * L) / V_alfven

# ------------ #
# Read in file #
# ------------ #
filename = args.dir + '.txt'
Data_Use = np.loadtxt(filename,delimiter=',')

# ---------------------------------------------------------------------------- #
# Function to return the dimensionless y coordinate given a position x, time t,
# index alpha, and drift velocty u; this is equation (11) of the notes
def ycoord(x, t, alpha, u):
    return (x - u*t) / t**(1./alpha)


# Function to return the Green's function on an infinite domain; this is
# equation (10) of the notes
def Ginf(x, t, sigma, alpha, u):
    return levy(ycoord(x, t, alpha, u), alpha, 0.0, sigma=sigma)/t**(1/alpha)


# Function to return the non-normalised Green's function on a periodic domain;
# this is equation (16) of the notes; here eps is the tolerance on epsilon^{(N)}
# in the notes, and nmax is a maximum allowed value of N, set for safety
def Gperiod(x, t, sigma, alpha, u, period, eps=5.0e-3, nmax=500):
    
    # Compute the Green's function for the image of the source that is not shifted
    # relative to the domain
    G = np.zeros(x.shape)
    G = Ginf(x, t, sigma, alpha, u)
    
    # Now successively add the contribution from shifted images of the source
    n = 1
    while n <= nmax:
        
        # Save last estimate
        Glast = np.copy(G)
        
        # Add contribution from shifted images of source
        G += Ginf(x + period*n, t, sigma, alpha, u)
        G += Ginf(x - period*n, t, sigma, alpha, u)
        
        # Check for convergence
        err = 1.0 - Glast / G
        if np.amax(err) < eps: break
            
        # Next iteration
        n += 1

    # Zero out points outside domain
    G[x < -period/2] = 0.0
    G[x > period/2] = 0.0
            
    # Return
    return G

# Define the log likelihood function; this is just the sum of the log's of the Green's functions
def logL(pars, x, t, period=2):
    
    # Extract parameters
    sigma = pars[0]
    alpha = pars[1]
    u = pars[2]
    
    # Evaluate Green's function
    gf = Gperiod(x, t, sigma, alpha, u, period=period) 
    
    # Return log likelihood
    return np.sum(np.log(gf))

###################################
### Defining prior distribution
def log_prior(pars):
    sigma = pars[0]
    alpha = pars[1]
    u = pars[2]
    if 0.75 < alpha <= 1.99 and 0.0 < sigma < 50 and 0 <= u < 10000:
        return 0.0
    return -np.inf

###################################
### Define log probability
def log_probability(pars, x, t, period):
    lp = log_prior(pars)
    if not np.isfinite(lp):
        return -np.inf
    return lp + logL(pars, x, t, period)

#######################################################
'''
   Beginning of MCMC routine, here the function 
   log_lik_drift is the likelyhood function.
   Searching over 3 parameters, alpha (from Levy),
   scale/kappa (from Levy) and drift/gamma which
   is to account for the drift in parallel diffusion.
'''
#######################################################
if __name__ == '__main__':
    
    ###################################
    ### Perpendicular x
    ###################################
    ''' Non dimensionalise everything'''
    # L = driving scale
    pos = Data_Use[:,0] / L                      # Divide by Driving scale
    t = Data_Use[:,3] / ( L / (c_s * Mach) )    # Divide by turbulent time
    perp = Data_Use[:,2] / L 
    ####################################
    print('')
    print('=============================')
    print('         MCMC trials         ')
    print(f'    Using {len(pos)} data points')
    print('=============================')
    print('')
    ####################################
    # Set initial drift guess as mean 
    n_cores = 48 # Core num
    num_walkers = n_cores * 1
    num_of_steps = 200
    burn_num = 50
    alpha = 1.4
    sigma = 1.5
    drift = 0 
    params = np.array([sigma, alpha,  drift])
    init_guess = params
    position = init_guess +  2e-3 * np.random.randn(num_walkers, 3) # Add random noise to guess
    nwalkers, ndim = position.shape
    period = 2
    
    ###################################
    # Multiprocessing    
    with Pool(n_cores) as pool:
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, log_probability, args=(pos, t, period), pool=pool)
        sampler.run_mcmc(position,
                        num_of_steps,
                        progress=True)
    
    ###################################
    
    ### Visualisations
    fig, axes = plt.subplots(3, figsize=(10, 7), sharex=True)
    samples = sampler.get_chain()
    labels = [r"$\kappa$",r"$\alpha$", r"$u$"]
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "midnightblue", alpha=0.4)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)

    
    axes[-1].set_xlabel("step number")
    name = 'drift_bands_x_' + str(Mach) + '_' + str(Alfven) + '_' + str(args.chi) + '_perp' + '.pdf'
    plt.savefig(name)
    plt.close()
    
    flat_samples = sampler.get_chain(discard=burn_num, thin=15, flat=True)
    #flat_samples[:,0] = flat_samples[:,0]**(flat_samples[:,1])

    ################################################
    ### Saving results as txt file
    ################################################
    Filename = 'walkers_perp_x_' + args.dir + '.txt'
    np.savetxt(Filename, flat_samples, delimiter=',')
    ################################################
    
    ###################################
    ### Corner plot
    labels          = [r"$\kappa$",r"$\alpha$", r"$u$"]
    #labels          = [r"$\kappa$", r"$\gamma$"]
    # initialise figure
    fig             = corner.corner(flat_samples,
                        labels=labels,
                        quantiles=[0.16, 0.5, 0.84],
                        show_titles=True,
                        title_kwargs={"fontsize": 17},
                        color='midnightblue',
                        label_kwargs={"fontsize": 26},
                        truth_color='red',
                        truths=[np.median(flat_samples[:,0]),np.median(flat_samples[:,1]),np.median(flat_samples[:,2])])
    # for control of labelsize of x,y-ticks:
    for ax in fig.get_axes():
      #ax.tick_params(axis='both', which='major', labelsize=14)
      #ax.tick_params(axis='both', which='minor', labelsize=12)    
      ax.tick_params(axis='both', labelsize=15)
    #fig.suptitle(r'$\mathcal{M} = 2 \ \ \ \mathcal{M}_{A0} \approx 2 \ \ \ \chi = 1 \times10^{-4}$',
    #fontsize = 24,y=0.98)
    name = 'fix_converge_drift_corner_x_' + str(Mach) + '_' + str(Alfven) + '_' + str(args.chi) + '_perp' + '.pdf'
    plt.savefig(name)
    plt.close()
    
    ############################################################
    ### Save data and 1 sigma levels
    ############################################################
    alpha_perp  = np.median(flat_samples[:,1])
    a_perp_lo = alpha_perp - np.quantile(flat_samples[:,1],0.16) 
    a_perp_hi = np.quantile(flat_samples[:,1],0.84) - alpha_perp
    scale_perp = np.median(flat_samples[:,0])
    s_perp_lo = scale_perp - np.quantile(flat_samples[:,0],0.16) 
    s_perp_hi = np.quantile(flat_samples[:,0],0.84) - scale_perp
    drift_perp = np.median(flat_samples[:,2])
    d_perp_lo = drift_perp - np.quantile(flat_samples[:,2],0.16) 
    d_perp_hi = np.quantile(flat_samples[:,2],0.84) - drift_perp

###################################
    ### Perpendicular y
    ###################################
    ''' Non dimensionalise everything'''
    # L = driving scale
    pos = Data_Use[:,1] / L                       # Divide by Driving scale
    t = Data_Use[:,3] / ( L / (c_s * Mach) )    # Divide by turbulent time
    ####################################
    print('')
    print('=============================')
    print('         MCMC trials         ')
    print(f'    Using {len(pos)} data points')
    print('=============================')
    print('')
    ####################################
    # Set initial drift guess as mean 
    alpha = 1.4
    sigma = 1.5
    drift = 0 
    params = np.array([sigma, alpha,  drift])
    init_guess = params
    position = init_guess +  2e-3 * np.random.randn(num_walkers, 3) # Add random noise to guess
    nwalkers, ndim = position.shape
    period = 2
    
    ###################################
    # Multiprocessing    
    with Pool(n_cores) as pool:
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, log_probability, args=(pos, t, period), pool=pool)
        sampler.run_mcmc(position,
                        num_of_steps,
                        progress=True)
    
    ###################################
    
    ### Visualisations
    fig, axes = plt.subplots(3, figsize=(10, 7), sharex=True)
    samples = sampler.get_chain()
    labels = [r"$c$",r"$\alpha$", r"$u$"]
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "midnightblue", alpha=0.4)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)

    
    axes[-1].set_xlabel("step number")
    name = 'drift_bands_y_' + str(Mach) + '_' + str(Alfven) + '_' + str(args.chi) + '_perp' + '.pdf'
    plt.savefig(name)
    plt.close()
    
    flat_samples = sampler.get_chain(discard=burn_num, thin=15, flat=True)
    #flat_samples[:,0] = flat_samples[:,0]**(flat_samples[:,1])

    ################################################
    ### Saving results as txt file
    ################################################
    Filename = 'walkers_perp_y_' + args.dir + '.txt'
    np.savetxt(Filename, flat_samples, delimiter=',')
    ################################################
    
    ###################################
    ### Corner plot
    labels          = [r"$\kappa$",r"$\alpha$", r"$u$"]
    #labels          = [r"$\kappa$", r"$\gamma$"]
    # initialise figure
    fig             = corner.corner(flat_samples,
                        labels=labels,
                        quantiles=[0.16, 0.5, 0.84],
                        show_titles=True,
                        title_kwargs={"fontsize": 17},
                        color='midnightblue',
                        label_kwargs={"fontsize": 26},
                        truth_color='red',
                        truths=[np.median(flat_samples[:,0]),np.median(flat_samples[:,1]),np.median(flat_samples[:,2])])
    #fig.suptitle(r'$\mathcal{M} = 2 \ \ \ \mathcal{M}_{A0} \approx 2 \ \ \ \chi = 1 \times10^{-4}$',
    #fontsize = 24,y=0.98)
# for control of labelsize of x,y-ticks:
    for ax in fig.get_axes():
      #ax.tick_params(axis='both', which='major', labelsize=14)
      #ax.tick_params(axis='both', which='minor', labelsize=12)    
      ax.tick_params(axis='both', labelsize=15)
    name = 'fix_converge_drift_corner_' + str(Mach) + '_' + str(Alfven) + '_' + str(args.chi) + '_perp' + '.pdf'
    plt.savefig(name)
    plt.close()
    
    ############################################################
    ### Save data and 1 sigma levels
    ############################################################
    alpha_perp_y  = np.median(flat_samples[:,1])
    a_perp_lo_y = alpha_perp_y - np.quantile(flat_samples[:,1],0.16) 
    a_perp_hi_y = np.quantile(flat_samples[:,1],0.84) - alpha_perp
    scale_perp_y = np.median(flat_samples[:,0])
    s_perp_lo_y = scale_perp_y - np.quantile(flat_samples[:,0],0.16) 
    s_perp_hi_y = np.quantile(flat_samples[:,0],0.84) - scale_perp
    drift_perp_y = np.median(flat_samples[:,2])
    d_perp_lo_y = drift_perp_y - np.quantile(flat_samples[:,2],0.16) 
    d_perp_hi_y = np.quantile(flat_samples[:,2],0.84) - drift_perp
    
    ###################################
    ### Parallel
    ###################################
    # Non dimensionalise everything
    # L = driving scale
    pos = Data_Use[:,2] / L                     # Divide by Driving scale
    t = Data_Use[:,3] / ( L / (c_s * Mach) )    # Divide by turbulent time
    ###################################
    
    ###################################
    # Set initial guess 
    alpha = 1.4
    sigma = 1.5
    drift = 10 
    params = np.array([sigma, alpha,  drift])
    init_guess = params
    position = init_guess + 2e-1 * np.random.randn(num_walkers, 3) # Add random noise to guess
    nwalkers, ndim = position.shape
    ###################################
    # Multiprocessing    
    with Pool(n_cores) as pool:
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, log_probability, args=(pos, t, period), pool=pool)
        sampler.run_mcmc(position,
                        num_of_steps,
                        progress=True)
    
    ###################################
    ### Visualisations
    fig, axes = plt.subplots(3, figsize=(10, 7), sharex=True)
    samples = sampler.get_chain()
    labels = [r"$c$",r"$\alpha$", r"$u$"]
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "midnightblue", alpha=0.4)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)

    
    axes[-1].set_xlabel("step number")
    name = 'drift_bands_' + str(Mach) + '_' + str(Alfven) + '_' + str(args.chi) + '_par' + '.pdf'
    plt.savefig(name)
    plt.close()
    flat_samples = sampler.get_chain(discard=burn_num, thin=15, flat=True)
    #flat_samples[:,0] = flat_samples[:,0]**(flat_samples[:,1])

    ################################################
    ### Saving results as txt file
    ################################################
    Filename = 'walkers_par_' + args.dir + '.txt'
    np.savetxt(Filename, flat_samples, delimiter=',')
    ################################################
    
    ###################################
    ### Corner plot
    labels          = [r"$\kappa$",r"$\alpha$", r"$u$"]
    # initialise figure
    fig             = corner.corner(flat_samples,
                        labels=labels,
                        quantiles=[0.16, 0.5, 0.84],
                        show_titles=True,
                        title_kwargs={"fontsize": 17},
                        color='midnightblue',
                        label_kwargs={"fontsize": 26},
                        truth_color='red',
                        truths=[np.median(flat_samples[:,0]), np.median(flat_samples[:,1]), np.median(flat_samples[:,2])])
    #fig.suptitle(r'$\mathcal{M} = 2 \ \ \ \mathcal{M}_{A0} \approx 2 \ \ \ \chi = 1 \times10^{-4}$',
    #fontsize = 24,y=0.98)
    # for control of labelsize of x,y-ticks:
    for ax in fig.get_axes():
      #ax.tick_params(axis='both', which='major', labelsize=14)
      #ax.tick_params(axis='both', which='minor', labelsize=12)    
      ax.tick_params(axis='both', labelsize=15)
    name = 'fix_converge_drift_corner_' + str(Mach) + '_' + str(Alfven) + '_' + str(args.chi) + '_par' + '.pdf'
    plt.savefig(name)
    plt.close()
    
    ############################################################
    ### Save data and 1 sigma levels
    ############################################################
    alpha_par  = np.median(flat_samples[:,1])
    a_par_lo = alpha_par - np.quantile(flat_samples[:,1],0.16) 
    a_par_hi = np.quantile(flat_samples[:,1],0.84) - alpha_par
    scale_par = np.median(flat_samples[:,0])
    s_par_lo = scale_par - np.quantile(flat_samples[:,0],0.16) 
    s_par_hi = np.quantile(flat_samples[:,0],0.84) - scale_par
    drift_par = np.median(flat_samples[:,2])
    d_par_lo = drift_par - np.quantile(flat_samples[:,2],0.16) 
    d_par_hi = np.quantile(flat_samples[:,2],0.84) - drift_par
    ################################################
    ### Plotting fitted Levy distribution
    ################################################
    '''
    #Need to sum analytical fits over "finite range" bounds
    #Need to check the normalisation,
    #int dx (analytic) == \int dx (Hist)
    '''


    ### Get hist data
    fig = plt.figure(figsize=(18.0,8.0), dpi = 150)
    #plt.style.use('dark_background')
    print('')
    print('========== Parameters ===========')
    print(f'Alpha perp x: {alpha_perp}')
    print(f'Scale perp x: {scale_perp}')
    print(f'Drift perp x: {drift_perp}')
    print(f'Alpha perp y: {alpha_perp_y}')
    print(f'Scale perp y: {scale_perp_y}')
    print(f'Drift perp y: {drift_perp_y}')
    print(f'Alpha par: {alpha_par}')
    print(f'Scale par: {scale_par}')
    print(f'Drift par: {drift_par}')
    print(f'================================')
    print('')
    ####################################


    #################################################
    ### Saving outputs
    ### Making results array
    ############################################################
    '''
    # Results arrary, format is: 
    # Mach || Alfven Mach || Ion Fraction || D_par || D_perp || Error_par || Error_perp || Alpha_par || Alpha_perp
    '''
    APar = alpha_par
    APerp = alpha_perp
    ErrorAPar_L = a_par_lo
    ErrorAPar_H = a_par_hi
    ErrorAPerp_L = a_perp_lo
    ErrorAPerp_H = a_perp_hi
    KPar = scale_perp 
    KPerp = scale_par 
    ErrorKPar_L = s_par_lo
    ErrorKPar_H = s_par_hi
    ErrorKPerp_L = s_perp_lo
    ErrorKPerp_H = s_perp_hi
    IonFrac = 1 * 10**(-args.chi)
    drift_perp = drift_perp
    drift_par = drift_par
    Error_DPar_L = d_par_lo
    Error_DPar_H = d_par_hi
    Error_DPerp_L = d_perp_lo
    Error_DPerp_H = d_perp_hi
    # For y valas
    APerp_y = alpha_perp_y
    ErrorAPerp_L_y = a_perp_lo_y
    ErrorAPerp_H = a_perp_hi_y
    KPar_y = scale_perp_y 
    ErrorKPerp_L_y = s_perp_lo_y
    ErrorKPerp_H_y = s_perp_hi_y
    drift_perp_y = drift_perp_y
    Error_DPerp_L_y = d_perp_lo_y
    Error_DPerp_H_y = d_perp_hi_y

    results = np.array([args.Mach, args.Alfven, IonFrac, KPar , KPerp ,
    ErrorKPar_L , ErrorKPar_H, ErrorKPerp_L, ErrorKPerp_H,
    APar, APerp,
    ErrorAPar_L , ErrorAPar_H, ErrorAPerp_L, ErrorAPerp_H,
    drift_perp, drift_par, Error_DPar_L, Error_DPar_H,
    Error_DPerp_L, Error_DPerp_H,
    APerp_y,ErrorAPerp_L_y,ErrorAPerp_H,
    KPar_y,ErrorKPerp_L_y,ErrorKPerp_H_y,
    drift_perp_y,Error_DPerp_L_y,Error_DPerp_H_y])

    ################################################
    ### Saving results as txt file
    ################################################
    Filename = 'fitted_params_' + args.dir + '.txt'
    np.savetxt(Filename, results, delimiter=',')
    ################################################
   
