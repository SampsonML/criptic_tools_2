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

print(f'Mag strength is {B} Gauss')
print(f'Alfven speed is {V_alfven} cm/s')
print(f'Streaming speed is {Vstr} cm/s')


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

        x_pos = x_pos / ( L )
        y_pos = y_pos / ( L )
        z_pos = z_pos / ( L )

        #----------------------------------#
        #      plotting CR positions       #
        #----------------------------------#
        plt.clf()
        plt.style.use("dark_background")

        col1 = particle_age
        map_c = cmr.ember_r
        map_c = cmr.gothic_r
        m_size = 1.5
        label_alf = r"$\mathcal{M}_{A0} = $" + str(args.Alfven)
        lab_time = 'Sim time: ' + str(round(t,2)) + r'$\tau$'
        # plot x
        plt.subplot(1, 2, 1)
        im = plt.scatter(x_pos, y_pos, alpha = 0.75,zorder = 1,
                    c = col1, edgecolor = 'none', s = m_size, cmap = map_c,
                    vmin = 0, vmax = 1.5)
        plt.xlabel(r'$L_{\perp}$', fontsize = 28)
        plt.ylabel(r'$L_{\perp}$', fontsize = 28)
        plt.text(-0.9,0.8,r'$\mathbf{B}_0 \odot$',fontsize = 28)
        plt.text(-0.9,0.65,label_alf,fontsize = 24)
        plt.text(-0.9,0.50,lab_time,fontsize = 21)
        plt.ylim([-1,1])
        plt.xlim([-1,1])


        plt.subplot(1, 2, 2)
        im = plt.scatter(x_pos, z_pos, alpha = 0.75,zorder = 1,
                    c = col1, edgecolor = 'none', s = m_size, cmap = map_c,
                    vmin = 0, vmax = 1.5)
        plt.xlabel(r'$L_{\perp}$', fontsize = 28)
        plt.ylabel(r'$L_{\parallel}$', fontsize = 28)
        plt.text(-0.9,0.8,r'$\mathbf{B}_0 \uparrow$',fontsize = 28)
        plt.ylim([-1,1])
        plt.xlim([-1,1])
        
        #fig.subplots_adjust(right=0.75)
        cbar_ax = fig.add_axes([0.92, 0.112, 0.02, 0.77])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label(r'CR packet age ($\tau$)', rotation=270, fontsize = 23,labelpad= 35)
        
        #plt.tight_layout()

    
# run the animation
fig = plt.figure(figsize=(15, 7), dpi = 150)
ani = FuncAnimation(fig, animate, frames=total_load_ins, interval=40, repeat=False)
import matplotlib.animation as animation
f = "space_space_" + str(args.dir) +  ".mp4" 
writervideo = animation.FFMpegWriter()
ani.save(f, writer='ffmpeg')


