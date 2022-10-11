"""
Fitting Script for diffusion coefficents CR study
Matt Sampson 2022 -- updated
"""
import argparse
import numpy as np
from numpy import diff
import astropy.units as u
import astropy.constants as const
from glob import glob
import os.path as osp
from cripticpy import readchk
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
parser.add_argument("--testonly",
                    help="report test results only, do not make plots",
                    default=False, action="store_true")
parser.add_argument("-c", "--chk",
                    help="number of checkpoint file to examine (default "
                    "is last checkpoint file in directory)",
                    default=-1, type=int)
parser.add_argument("-d", "--dir",
                    help="directory containing run to analyze",
                    default=".", type=str)
parser.add_argument("-f", "--filenum",
                    help="Amount of chunk files",
                    default="1", type=int)
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

# Extract parameters we need from the input file
fp = open(osp.join(args.dir, 'criptic.in'), 'r')
chkname = 'criptic_'
for line in fp:
    l = line.split('#')[0]    # Strip comments
    s = l.split()
    if len(s) == 0: continue
    if s[0] == 'output.chkname':
        chkname = s[1]
    elif s[0] == 'cr.kPar0':
        kPar0 = float(s[1]) * u.cm**2 / u.s
    elif s[0] == 'cr.kParIdx':
        kParIdx = float(s[1])
    elif s[0] == 'prob.L':
        Lsrc = np.array([ float(s1) for s1 in s[1:]]) * u.erg/u.s
    elif s[0] == 'prob.T':
        Tsrc = np.array([ float(s1) for s1 in s[1:]]) * u.GeV
    elif s[0] == 'prob.r':
        rsrc = float(s[1]) * u.cm
fp.close()
###################################
# Initialise Problem
##################################
chunk_age       = np.zeros(args.filenum)
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

#############################################################
### Beginning data read in
#############################################################
by_num = 5 # Added to speed up small scale local PC testing set to 1 for GADI
if (by_num > args.filenum):
    by_num = 1
    print(f'Chunk gap revert to {1}')
total_load_ins = int(args.filenum / by_num)

for k in range(total_load_ins):  
    #########################################################
    # Figure out which checkpoint file to examine
    args.chk = 1 + k * by_num
    if args.chk >= 0:
        chknum = "{:05d}".format(args.chk)
        chknum = chknum + '.hdf5'
    else:
        chkfiles = glob(osp.join(args.dir,
                                 chkname+"[0-9][0-9][0-9][0-9][0-9].hdf5"))
        chknums = [ int(c[-9:-4]) for c in chkfiles ]
        chknum = "{:05d}".format(np.amax(chknums))
        chknum = chknum + '.hdf5'
    ########################################################
    # Check if file exists
    Path = osp.join(args.dir,chkname+chknum) 
    #print(Path)
    file_exist = osp.exists(Path)
    if (file_exist == False):
        print(f'File {Path} dosnt exist')
        break
    ########################################################
    # Read the checkpoint
    data = readchk(osp.join(args.dir,chkname+chknum),units=False, ke=True, ndot=False,
            sort=True, meta=False)
    packets = data['packets']
    t = data['t']
    
    ########################################################
    ### Extract useful particle information
    ########################################################
    # Get Particle Positions
    x = packets['x']
    x_pos =  x[:,0]
    y_pos =  x[:,1]
    z_pos =  x[:,2]

    # Get source
    source = data['sources']
    part_source = source['x']
    source_ID = packets['source']
    source_x, source_y, source_z = zip(*part_source)
    # Get Particle Age
    particle_age =  (t - packets['tinj']) 
    particle_age = np.asarray(particle_age)
    x_pos = np.asarray(x_pos)
    y_pos = np.asarray(y_pos)
    z_pos = np.asarray(z_pos)
    ##########################################################
    ### Adjust for Periodicity and Source Location
    ##########################################################
    
    for i in range(len(x_pos)):
        # The bounds adjustment
        if (np.abs(x_pos[i] - source_x[source_ID[i]]) >= L):
            x_pos[i] = x_pos[i] - np.sign(x_pos[i]) * 2 * L
        if (np.abs(y_pos[i] - source_y[source_ID[i]]) >= L):
            y_pos[i] = y_pos[i] - np.sign(y_pos[i]) * 2 * L
        if (np.abs(z_pos[i] - source_z[source_ID[i]]) >= L):
            z_pos[i] = z_pos[i] - np.sign(z_pos[i]) * 2 * L
        # The source adjustment to center all CRs
        x_pos[i] = x_pos[i] - source_x[source_ID[i]]
        y_pos[i] = y_pos[i] - source_y[source_ID[i]]
        z_pos[i] = z_pos[i] - source_z[source_ID[i]] 
    
    ########################################
    ### Displacement All Data
    ########################################
    displacement_perp_y = np.append(displacement_perp_y, y_pos) 
    displacement_perp_x = np.append(displacement_perp_x, x_pos) 
    displacement_par = np.append(displacement_par, z_pos)
    time_vals = np.append(time_vals, particle_age)

########################################
### Age Selections
########################################
Data_Combine = np.column_stack((displacement_perp_x, displacement_perp_y,displacement_par,time_vals))
# Subset data for speed
Data_Use = Data_Combine[Data_Combine[:,3] > 1e2]
Data_Use = Data_Use[Data_Use[:,3] < 5e13] #3e13 old val
print(f'Total data points {len(Data_Use[:,1])}')


################################################
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
    pos = Data_Use[:,0] / L                       # Divide by Driving scale
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
    num_of_steps = 700
    burn_num = 300
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
    ### Perp Ewald Sum
    """
    tol = 0.01
    L_n =  2 
    finite_range = 10
    beta = 0             # Skew
    mu = 0           
    # Location
    ########################
    ### Perp x
    pos = Data_Use[:,0] / L 
    Gp = Gperiod(pos,t, scale_perp, alpha_perp, drift_perp, period=2)
    sort_index = np.argsort(pos)
    pos = pos[sort_index]
    Gp = Gp[sort_index]

    ########################
    ### Perp y
    pos_y = Data_Use[:,1] / L 
    Gp_y = Gperiod(pos_y,t, scale_perp_y, alpha_perp_y, drift_perp_y, period=2)
    sort_index_y = np.argsort(pos_y)
    pos_y = pos_y[sort_index_y]
    Gp_y = Gp_y[sort_index_y]

    ################################################
    ###############################################################
    ### Parallel ###
    pos_par = Data_Use[:,2] / L 
    Gpar = Gperiod(pos_par,t, scale_par, alpha_par, drift_par, period=2)
    sort_index2 = np.argsort(pos_par)
    pos_par = pos_par[sort_index2]
    Gpar = Gpar[sort_index2]

    #############################################################
    lab_a_perp = r'$\alpha = ' + str(round(alpha_perp,3)) + r'^{' + str(round(a_perp_hi,3)) + r'}_{' + str(round(a_perp_lo,3)) + r'}$' 
    lab_s_perp = r'$\kappa = ' + str(round(scale_perp,3)) + r'^{' + str(round(s_perp_hi,3)) + r'}_{' + str(round(s_perp_lo,3)) + r'}$' 
    lab_d_perp = r'$u = ' + str(round(drift_perp,3)) + r'^{' + str(round(d_perp_hi,3)) + r'}_{' + str(round(d_perp_lo,3)) + r'}$' 

    lab_a_perp_y = r'$\alpha = ' + str(round(alpha_perp_y,3)) + r'^{' + str(round(a_perp_hi_y,3)) + r'}_{' + str(round(a_perp_lo_y,3)) + r'}$' 
    lab_s_perp_y = r'$\kappa = ' + str(round(scale_perp_y,3)) + r'^{' + str(round(s_perp_hi_y,3)) + r'}_{' + str(round(s_perp_lo_y,3)) + r'}$' 
    lab_d_perp_y = r'$u = ' + str(round(drift_perp_y,3)) + r'^{' + str(round(d_perp_hi_y,3)) + r'}_{' + str(round(d_perp_lo_y,3)) + r'}$' 

    
    lab_a_par = r'$\alpha = ' + str(round(alpha_par,3)) + r'^{' + str(round(a_par_hi,3)) + r'}_{' + str(round(a_par_lo,3)) + r'}$' 
    lab_s_par = r'$\kappa = ' + str(round(scale_par,3)) + r'^{' + str(round(s_par_hi,3)) + r'}_{' + str(round(s_par_lo,3)) + r'}$' 
    lab_d_par = r'$u = ' + str(round(drift_par,3)) + r'^{' + str(round(d_par_hi,3)) + r'}_{' + str(round(d_par_lo,3)) + r'}$' 
    
    bin_numL = 23
    from scipy.stats import binned_statistic
    sx, edges, _ = binned_statistic(pos,Gp,
      statistic='mean', bins=bin_numL)
    for i in range(len(sx)):
        if (np.isfinite(sx[i]) == False):
            sx[i] = 0
    sx = np.append(sx,sx[bin_numL-1])
    edges = edges + 0.5 * (edges[2] - edges[1])
    sx = sx / np.sum(sx * (edges[2] - edges[1]))

    s1, edges1, _ = binned_statistic(pos_par,Gpar,
      statistic='mean', bins=bin_numL)
    for i in range(len(s1)):
        if (np.isfinite(s1[i]) == False):
            s1[i] = 0
    s1 = np.append(s1,s1[bin_numL-1])
    edges1 = edges1 + 0.5 * (edges1[2] - edges1[1])
    s1 = s1 / np.sum(s1 * (edges1[2] - edges1[1]))

    sy, edgesy, _ = binned_statistic(pos_y,Gp_y,
      statistic='mean', bins=bin_numL)
    for i in range(len(sy)):
        if (np.isfinite(sy[i]) == False):
            sy[i] = 0
    sy = np.append(sy,sy[bin_numL-1])
    edgesy = edgesy + 0.5 * (edgesy[2] - edgesy[1])
    sy = sy / np.sum(sy * (edgesy[2] - edgesy[1]))


    ### Colors
    bin_num = 33
    fig = plt.figure(figsize=(22,6.5), dpi = 200)
    Hist_col = 'midnightblue'
    Levy_col = 'red'
    purp_m = 'red'
    error_col = Hist_col

    #############################
    ### Perp x
    #############################
    plt.subplot(1,3,1)
    #countsE, binsE = np.histogram(pos[0:len(pos) -1], bins = bin_num)
    counts, bins, _ = plt.hist(pos, bins = bin_num,histtype='step', 
    edgecolor = Hist_col, label = r'Simulation', linewidth=2, zorder = 1, density = True)
    bin_centers = 0.5*(bins[1:] + bins[:-1])
    plt.errorbar( bin_centers, counts, yerr = (counts**0.5) / np.sum(counts), 
    marker = '.',drawstyle = 'steps-mid', c = error_col, fmt='none',capsize=4)
    #plt.hist(edges[:-1], edges,weights=sx, histtype='step',
    #edgecolor = Levy_col, label=f'Levy', linewidth=2, zorder = 2)
    ### Normalise
    #sx = sx * (np.sum(sx*(edges[2] - edges[1])) / np.sum(counts * (bins[2] - bins[1])))
    print(np.sum(sx*(edges[2] - edges[1])))
    print(np.sum(counts*(bins[2] - bins[1])))
    plt.plot(edges,sx,label='Model', color = 'red',linewidth=2,zorder=2)
    #import scipy.stats as stats
    #density = stats.gaussian_kde(edges * sx)
    #plt.plot(edges[:-1],density(edges[:-1]),label='Levy', color = 'red',linewidth=2,zorder=2)
    plt.legend(frameon=False,fontsize=16, loc = 'upper right')
    plt.ylabel(r'PDF', fontsize = 26)
    plt.title(r'Perpendicular $\mathcal{M}_{A0} \approx$ ' + str(Alfven),fontsize = 28)
    # Add labels
    xmin, xmax, ymin, ymax = plt.axis()
    x_text = 0.9*xmin
    y_text_1 = 0.9*ymax ; y_text_2 = 0.8*ymax ; y_text_3 = 0.7*ymax ; y_text_4 = 0.6*ymax
    plt.text(x_text, y_text_1, lab_a_perp, fontsize = 22)
    plt.text(x_text, y_text_2, lab_s_perp, fontsize = 22)
    plt.text(x_text, y_text_3, lab_d_perp, fontsize = 22)
    plt.xlabel(r'$\Delta x$', fontsize = 26)
    #plt.text(x_text, y_text_1, chi_lab, fontsize = 22)
    '''
    plt.subplot(2,3,4)
    counts, bins, _ = plt.hist(pos, bins = bin_num,histtype='step', 
    edgecolor = Hist_col, label = r'CRIPTIC', linewidth=2, zorder = 1, density = True)
    bin_centers = 0.5*(bins[1:] + bins[:-1])
    plt.errorbar( bin_centers, counts, yerr = counts**0.5, 
    marker = '.',drawstyle = 'steps-mid', c = error_col, fmt='none',capsize=4)
    plt.hist(edges[:-1], edges,weights=s, histtype='step',
    edgecolor = Levy_col, label=f'Levy', linewidth=2, zorder = 2)
    plt.legend(frameon=False,fontsize=16, loc = 'upper right')
    plt.xlabel(r'$\Delta x$', fontsize = 26)
    plt.ylabel(r'log(PDF)', fontsize = 26)
    plt.yscale('log')
    '''

    #############################
    ### Perp y
    #############################
    plt.subplot(1,3,2)
    #countsE, binsE = np.histogram(pos[0:len(pos) -1], bins = bin_num)
    counts, bins, _ = plt.hist(pos_y, bins = bin_num,histtype='step', 
    edgecolor = Hist_col, label = r'Simulation', linewidth=2, zorder = 1, density = True)
    bin_centers = 0.5*(bins[1:] + bins[:-1])
    plt.errorbar( bin_centers, counts, yerr = (counts**0.5) / np.sum(counts), 
    marker = '.',drawstyle = 'steps-mid', c = error_col, fmt='none',capsize=4)
    #plt.hist(edgesy[:-1], edgesy,weights=sy, histtype='step',
    #edgecolor = Levy_col, label=f'Levy', linewidth=2, zorder = 2)
    plt.plot(edgesy,sy,label='Model', color = 'red',linewidth=2,zorder=2)
    plt.legend(frameon=False,fontsize=16, loc = 'upper right')
    plt.ylabel(r'PDF', fontsize = 26)
    plt.title(r'Perpendicular $\mathcal{M}_{A0} \approx$ ' + str(Alfven),fontsize = 28)
    # Add labels
    xmin, xmax, ymin, ymax = plt.axis()
    x_text = 0.9*xmin
    y_text_1 = 0.9*ymax ; y_text_2 = 0.8*ymax ; y_text_3 = 0.7*ymax ; y_text_4 = 0.6*ymax
    plt.text(x_text, y_text_1, lab_a_perp_y, fontsize = 22)
    plt.text(x_text, y_text_2, lab_s_perp_y, fontsize = 22)
    plt.text(x_text, y_text_3, lab_d_perp_y, fontsize = 22)
    plt.xlabel(r'$\Delta y$', fontsize = 26)    
    #plt.text(x_text, y_text_1, chi_lab, fontsize = 22)
    '''
    plt.subplot(2,3,5)
    counts, bins, _ = plt.hist(pos_y, bins = bin_num,histtype='step', 
    edgecolor = Hist_col, label = r'CRIPTIC', linewidth=2, zorder = 1, density = True)
    bin_centers = 0.5*(bins[1:] + bins[:-1])
    plt.errorbar( bin_centers, counts, yerr = counts**0.5, 
    marker = '.',drawstyle = 'steps-mid', c = error_col, fmt='none',capsize=4)
    plt.hist(edgesy[:-1], edgesy,weights=sy, histtype='step',
    edgecolor = Levy_col, label=f'Levy', linewidth=2, zorder = 2)
    plt.legend(frameon=False,fontsize=16, loc = 'upper right')
    plt.xlabel(r'$\Delta y$', fontsize = 26)
    plt.ylabel(r'log(PDF)', fontsize = 26)
    plt.yscale('log')
    '''
    plt.subplot(1,3,3)
    counts, bins, _ = plt.hist(pos_par, bins = bin_num,histtype='step', 
    edgecolor = Hist_col, label = r'Simulation', linewidth=2, zorder = 1, density = True)
    bin_centers = 0.5*(bins[1:] + bins[:-1])
    plt.errorbar( bin_centers, counts, yerr = (counts**0.5) / np.sum(counts), 
    marker = '.',drawstyle = 'steps-mid', c = error_col, fmt='none',capsize=4)  
    #plt.hist(edges1[:-1], edges1,weights=s1, histtype='step',
    #edgecolor = Levy_col, label=f'Levy', linewidth=2, zorder = 2)
    plt.plot(edges1,s1,label='Model', color = 'red',linewidth=2,zorder=2)
    plt.title(r'Parallel $\mathcal{M}_{A0} \approx$ ' + str(Alfven),fontsize = 28)
    plt.legend(frameon=False,fontsize=16, loc = 'upper right')
    # Add labels
    xmin, xmax, ymin, ymax = plt.axis()
    x_text = 0.9*xmin
    y_text_1 = 0.9*ymax ; y_text_2 = 0.8*ymax ; y_text_3 = 0.7*ymax
    plt.text(x_text, y_text_1, lab_a_par, fontsize = 24)
    plt.text(x_text, y_text_2, lab_s_par, fontsize = 24)
    plt.text(x_text, y_text_3, lab_d_par, fontsize = 24)
    plt.xlabel(r'$\Delta z$', fontsize = 26)
    '''
    plt.subplot(2,3,6)
    counts, bins, _ = plt.hist(pos_par, bins = bin_num,histtype='step', 
    edgecolor = Hist_col, label = r'CRIPTIC', linewidth=2, zorder = 1, density = True)
    bin_centers = 0.5*(bins[1:] + bins[:-1])
    plt.errorbar( bin_centers, counts, yerr = counts**0.5, 
    marker = '.',drawstyle = 'steps-mid', c = error_col, fmt='none',capsize=4)
    plt.hist(edges1[:-1], edges1,weights=s1, histtype='step',
    edgecolor = Levy_col, label=f'Levy', linewidth=2, zorder = 2)
    plt.legend(frameon=False,fontsize=16, loc = 'upper right')
    plt.xlabel(r'$\Delta z$', fontsize = 26)
    plt.yscale('log')
    '''
    fig.tight_layout()
    plt.savefig('zzz_drift_' +  str(Mach) + '_' + str(Alfven) + '_' + str(args.chi)  + "_fitting.pdf",)
    plt.close()
    """

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
    Filename = 'parameters_' + args.dir + '.txt'
    np.savetxt(Filename, results, delimiter=',')
    ################################################
   
