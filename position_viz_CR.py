#-------------------------------------#
#     CR position visualisation       #
#     Matt Sampson  Sep. 2022         #
#-------------------------------------#

#-----------------#
# package imports #
#-----------------#
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
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sys
import time 

#---------------------#
# Making progress bar #
#---------------------#
from tqdm import tqdm
from time import sleep


#--------------#
#  arg parser  #
#--------------#
parser = argparse.ArgumentParser(
    description="structure functions")
parser.add_argument("-c", "--chk",
                    help="number of checkpoint to start on)",
                    default=-1, type=int)
parser.add_argument("-d", "--dir",
                    help="directory containing run to analyze",
                    default=".", type=str)
parser.add_argument("-f", "--filenum",
                    help="number of chekcpoint to end on",
                    default="1", type=int)
parser.add_argument("-m", "--Mach",
                    help="Mach number",
                    default="1", type=float)
parser.add_argument("-a", "--Alfven",
                    help="Alfven number",
                    default="1", type=float)
parser.add_argument("-i", "--chi",
                    help="ion fraction",
                    default="1", type=float)
parser.add_argument("-s", "--c_s",
                    help="Amount of chunk files",
                    default="1", type=float)

# grab the arguements
args = parser.parse_args()

#------------------------------------------------#
# Extract parameters we need from the input file #
#------------------------------------------------#
fp = open(osp.join(args.dir, 'criptic.in'), 'r')
chkname = 'criptic_'
for line in fp:
    l = line.split('#')[0]
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

#------------------------------------------#
# determine how many files to iterate over #
#------------------------------------------#
dt_gap = 1
total_load_ins = int((args.filenum - args.chk) / dt_gap)

#------------------#
# Define variables #
#------------------#
L = 3.09e19
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


#--------------------#
# Animation function #
#--------------------#

def animate(k):
        #---------------------------------------------#
        # Figure out which checkpoint file to examine #
        #---------------------------------------------#
        chk_number =  args.chk + k * dt_gap
        if chk_number >= 0:
            chknum = "{:05d}".format(chk_number)
            chknum = chknum + '.hdf5'
        else:
            chkfiles = glob(osp.join(args.dir,
                                    chkname+"[0-9][0-9][0-9][0-9][0-9].hdf5"))
            chknums = [ int(c[-9:-4]) for c in chkfiles ]
            chknum = "{:05d}".format(np.amax(chknums))
            chknum = chknum + '.hdf5'


        #----------------------#
        # Check if file exists #
        #----------------------#
        Path = osp.join(args.dir,chkname+chknum)
        file_exist = osp.exists(Path)
        if (file_exist == False):
            print(f'File {Path} dosnt exist')

        #---------------------#
        # Read the checkpoint #
        #---------------------#
        data = readchk(osp.join(args.dir,chkname+chknum),units=False, ke=False, ndot=False,
                sort=True, meta=False)
        packets = data['packets']

        #------------------------#
        # Get Particle Positions #
        #------------------------#
        x = packets['x']
        x_pos =  x[:,0]
        y_pos =  x[:,1]
        z_pos =  x[:,2]

        t = data['t'] / t_turnover
        # Get source
        source = data['sources']
        part_source = source['x']
        source_ID = packets['source']
        source_x, source_y, source_z = zip(*part_source)
        # Get Particle Age
        particle_age =  (t - (packets['tinj'] / t_turnover))
        particle_age = np.asarray(particle_age)
        x_pos = np.asarray(x_pos)
        y_pos = np.asarray(y_pos)
        z_pos = np.asarray(z_pos)

        #-----------------------------------------------#
        #   Adjust for Periodicity and Source Location  #
        #-----------------------------------------------#
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

        x_pos = x_pos / ( L )
        y_pos = y_pos / ( L )
        z_pos = z_pos / ( L )
        
        
        #print(f'x - max: {np.max(x_pos)}  min: {np.min(x_pos)}')
        #print(f'y - max: {np.max(y_pos)}  min: {np.min(y_pos)}')
        #print(f'z - max: {np.max(z_pos)}  min: {np.min(z_pos)}')
    

        #----------------------------------#
        #      plotting CR positions       #
        #----------------------------------#
        plt.clf()
        plt.style.use("dark_background")

        col1 = 'cornflowerblue'
        m_size = 1
        # plot x
        plt.subplot(3, 1, 1)
        plt.scatter(particle_age, x_pos, alpha = 0.35,zorder = 1,c = col1, edgecolor = col1, s = m_size)
        plt.ylabel(r'$\Delta_x (\ell_0)$', fontsize = 28)
        plt.ylim([-1,1])
        plt.xlim([0,1/2])


        plt.subplot(3, 1, 2)
        plt.scatter(particle_age, y_pos, alpha = 0.35,zorder = 1,c = col1, edgecolor = col1, s = m_size)
        plt.ylabel(r'$\Delta_y (\ell_0)$', fontsize = 28)
        plt.ylim([-1,1])
        plt.xlim([0,1/2])

    
        plt.subplot(3, 1, 3)
        plt.scatter(particle_age, z_pos, alpha = 0.35,zorder = 1,c = col1, edgecolor = col1, s = m_size)
        plt.ylabel(r'$\Delta_z (\ell_0)$', fontsize = 28)
        plt.xlabel(r'time ($\tau$)', fontsize = 28)
        plt.ylim([-1,1])
        plt.xlim([0,1/2])

    
# run the animation
import mplcyberpunk
fig = plt.figure(figsize=(14, 10))
ani = FuncAnimation(fig, animate, frames=total_load_ins, interval=60, repeat=False)
f = "space_time_" + str(args.dir) +  ".gif" 
ani.save(f, writer='pillow')

