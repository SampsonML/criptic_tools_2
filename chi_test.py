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
chi = 1 
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

#print(f'Mag strength is {B} Gauss')
#print(f'Alfven speed is {V_alfven} cm/s')
#print(f'Streaming speed is {Vstr} cm/s')
#print(f'Alfven speed is {(L * V_alfven) / t_turnover} L / tau')
#print(f'Streaming speed is {(L * Vstr) /t_turnover } L / tau')

slice_nums = args.filenum - args.chk
n_src = 81

#--------------------#
# Animation function #
#--------------------#

for k in range(2):
        #---------------------------------------------#
        # Figure out which checkpoint file to examine #
        #---------------------------------------------#
        chk_number =  args.chk + k * slice_nums
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
        print(f'Reading in {Path}')

        #---------------------#
        # Read the checkpoint #
        #---------------------#
        data = readchk(osp.join(args.dir,chkname+chknum),units=True, ke=True, ndot=False,
                sort=True, meta=False)
        packets = data['packets']
           
        #------------------------#
        # Get Particle Positions #
        #------------------------#
        x = packets['x']
        x_pos =  x[:,0]
        y_pos =  x[:,1]
        z_pos =  x[:,2]

        #t = data['t'] / t_turnover
        t = data['t']

        # Get source
        source = data['sources']
        part_source = source['x']
        source_x, source_y, source_z = zip(*part_source)
        # Get Particle Age
        
        source_ID = np.asarray(packets['source'])
        #particle_age =  np.asarray( (t - (packets['tinj'] / t_turnover)) )
        particle_age =  np.asarray( (t - (packets['tinj'] )) )
        grammage = np.asarray( packets['T'] ) ## change if wanting units
        x_pos = np.asarray(x_pos)
        y_pos = np.asarray(y_pos)
        z_pos = np.asarray(z_pos)
        
        
        
        if (k > 0):
            source_ID = source_ID[0:length]
            particle_age =  particle_age[0:length]
            x_pos =  x_pos[0:length]
            y_pos =  y_pos[0:length]
            z_pos =  z_pos[0:length]
            x_pos = np.asarray(x_pos)
            y_pos = np.asarray(y_pos)
            z_pos = np.asarray(z_pos)
            grammage = grammage[0:length]
            #print(f'Initial packet count: {length}')
            #print(f'Final packet count: {len(x_pos)}')
        
        
        # now get permuation to sort by source ID
        p = np.argsort(source_ID)

        # update the sorted data
        x_pos = x_pos[p]
        y_pos = y_pos[p]
        z_pos = z_pos[p]
        particle_age = particle_age[p]
        source_ID = source_ID[p]
        grammage = grammage[p]
        
        # now grab the start and end points of the chi batches
        chi_batch_0_end = np.argmin(np.abs( source_ID - (n_src +1) )) -1
        chi_batch_1_end = np.argmin(np.abs( source_ID - (2*n_src +1) )) -1
        chi_batch_2_end = np.argmin(np.abs( source_ID - (3*n_src +1) )) -1
        chi_batch_3_end = np.argmin(np.abs( source_ID - (4*n_src +1) )) -1
        chi_batch_4_end = np.argmin(np.abs( source_ID - (5*n_src +1) )) -1
        chi_batch_5_end = len(x_pos) -1
        
        #---------------------------#
        # now sort into chi batches #
        #---------------------------#
        #- chi 1
        particle_age_c0 = np.asarray(particle_age[0:chi_batch_0_end])
        x_pos_c0 = np.asarray(x_pos[0:chi_batch_0_end])
        y_pos_c0 = np.asarray(y_pos[0:chi_batch_0_end])
        z_pos_c0 = np.asarray(z_pos[0:chi_batch_0_end])
        grammage_c0 = np.asarray(grammage[0:chi_batch_0_end])
        
        #- chi 0.1
        particle_age_c1 = np.asarray(particle_age[chi_batch_0_end:chi_batch_1_end])
        x_pos_c1 = np.asarray(x_pos[chi_batch_0_end:chi_batch_1_end])
        y_pos_c1 = np.asarray(y_pos[chi_batch_0_end:chi_batch_1_end])
        z_pos_c1 = np.asarray(z_pos[chi_batch_0_end:chi_batch_1_end])
        grammage_c1 = np.asarray(grammage[chi_batch_0_end:chi_batch_1_end])
        
        #- chi 0.01
        particle_age_c2 = np.asarray(particle_age[chi_batch_1_end:chi_batch_2_end])
        x_pos_c2 = np.asarray(x_pos[chi_batch_1_end:chi_batch_2_end])
        y_pos_c2 = np.asarray(y_pos[chi_batch_1_end:chi_batch_2_end])
        z_pos_c2 = np.asarray(z_pos[chi_batch_1_end:chi_batch_2_end])
        grammage_c2 = np.asarray(grammage[chi_batch_1_end:chi_batch_2_end])
        
        #- chi 0.001
        particle_age_c3 = np.asarray(particle_age[chi_batch_2_end:chi_batch_3_end])
        x_pos_c3 = np.asarray(x_pos[chi_batch_2_end:chi_batch_3_end])
        y_pos_c3 = np.asarray(y_pos[chi_batch_2_end:chi_batch_3_end])
        z_pos_c3 = np.asarray(z_pos[chi_batch_2_end:chi_batch_3_end])
        grammage_c3 = np.asarray(grammage[chi_batch_2_end:chi_batch_3_end])
        
        #- chi 1e-4
        particle_age_c4 = np.asarray(particle_age[chi_batch_3_end:chi_batch_4_end])
        x_pos_c4 = np.asarray(x_pos[chi_batch_3_end:chi_batch_4_end])
        y_pos_c4 = np.asarray(y_pos[chi_batch_3_end:chi_batch_4_end])
        z_pos_c4 = np.asarray(z_pos[chi_batch_3_end:chi_batch_4_end])
        grammage_c4 = np.asarray(grammage[chi_batch_3_end:chi_batch_4_end])
        
        #- chi 1e-5
        particle_age_c5 = np.asarray(particle_age[chi_batch_4_end:chi_batch_5_end])
        x_pos_c5 = np.asarray(x_pos[chi_batch_4_end:chi_batch_5_end])
        y_pos_c5 = np.asarray(y_pos[chi_batch_4_end:chi_batch_5_end])
        z_pos_c5 = np.asarray(z_pos[chi_batch_4_end:chi_batch_5_end])
        grammage_c5 = np.asarray(grammage[chi_batch_4_end:chi_batch_5_end])

        # now get average velocities normalised by streaming speed
        if (k == 0):
            init_sources = source_ID
            
            length = len(x_pos)
            #chi 1
            age_c0_init = particle_age_c0 
            x_c0_init   = x_pos_c0 
            y_c0_init   = y_pos_c0 
            z_c0_init   = z_pos_c0 
            grammage_c0_init = grammage_c0

            #chi 0.1
            age_c1_init = particle_age_c1 
            x_c1_init   = x_pos_c1 
            y_c1_init   = y_pos_c1 
            z_c1_init   = z_pos_c1 
            grammage_c1_init = grammage_c1
            
            #chi 0.01
            age_c2_init = particle_age_c2
            x_c2_init   = x_pos_c2 
            y_c2_init   = y_pos_c2 
            z_c2_init   = z_pos_c2 
            grammage_c2_init = grammage_c2
            
            #chi 0.001
            age_c3_init = particle_age_c3 
            x_c3_init   = x_pos_c3 
            y_c3_init   = y_pos_c3 
            z_c3_init   = z_pos_c3 
            grammage_c3_init = grammage_c3
            
            #chi 1e-4
            age_c4_init = particle_age_c4 
            x_c4_init   = x_pos_c4 
            y_c4_init   = y_pos_c4 
            z_c4_init   = z_pos_c4 
            grammage_c4_init = grammage_c4
            
            #chi 1e-5
            age_c5_init = particle_age_c5 
            x_c5_init   = x_pos_c5 
            y_c5_init   = y_pos_c5 
            z_c5_init   = z_pos_c5 
            grammage_c5_init = grammage_c5
            

# ------------------------------------------ #
# Now calculate effective streaming velocity #
# ------------------------------------------ #
grammage_change_c0 = np.mean( grammage_c0 )
grammage_change_c1 = np.mean( grammage_c1 )
grammage_change_c2 = np.mean( grammage_c2 )
grammage_change_c3 = np.mean( grammage_c3 )
grammage_change_c4 = np.mean( grammage_c4 )
grammage_change_c5 = np.mean( grammage_c5 )

vel_c0 = np.round(  (np.mean(z_pos_c0) - np.mean(z_c0_init)) / (np.mean(particle_age_c0) - np.mean(age_c0_init)), 1 )
vel_c1 = np.round(  (np.mean(z_pos_c1) - np.mean(z_c1_init)) / (np.mean(particle_age_c1) - np.mean(age_c1_init)), 1 )
vel_c2 = np.round(  (np.mean(z_pos_c2) - np.mean(z_c2_init)) / (np.mean(particle_age_c2) - np.mean(age_c2_init)), 1 )
vel_c3 = np.round(  (np.mean(z_pos_c3) - np.mean(z_c3_init)) / (np.mean(particle_age_c3) - np.mean(age_c3_init)), 1 )
vel_c4 = np.round(  (np.mean(z_pos_c4) - np.mean(z_c4_init)) / (np.mean(particle_age_c4) - np.mean(age_c4_init)), 1 )
vel_c5 = np.round(  (np.mean(z_pos_c5) - np.mean(z_c5_init)) / (np.mean(particle_age_c5) - np.mean(age_c5_init)), 1 )

speed_factor1 = (1 / ( (0.1)**(1/2) ) )
speed_factor2 = (1 / ( (0.01)**(1/2) ) )
speed_factor3 = (1 / ( (0.001)**(1/2) ) )
speed_factor4 = (1 / ( (0.0001)**(1/2) ) )
speed_factor5 = (1 / ( (0.00001)**(1/2) ) )


print('')
print('--------------------------------------------------')
print(f'Streaming velocity fully ionised is {Vstr} cm/s')
print('--------------------------------------------------')
print(f'chi 1    -- <velocity> = {407704.5} cm/s')
print(f'chi 0.1  -- <velocity> = {vel_c1} cm/s')
print(f'chi 0.01 -- <velocity> = {vel_c2} cm/s')
print(f'chi 1e-3 -- <velocity> = {vel_c3} cm/s')
print(f'chi 1e-4 -- <velocity> = {vel_c4} cm/s')
print(f'chi 1e-5 -- <velocity> = {vel_c5} cm/s')

print('')
print('----------------------------------------------------')
print(f'CR packet velocity normalised by ionic alfven speed')
print('----------------------------------------------------')
print(f'chi 1    -- <velocity> / vstr = {1.019}')
print(f'chi 0.1  -- <velocity> / vstr = {np.round( vel_c1/ (speed_factor1 * Vstr), 3)}')
print(f'chi 0.01 -- <velocity> / vstr = {np.round( vel_c2/ (speed_factor2 * Vstr), 3)}')
print(f'chi 1e-3 -- <velocity> / vstr = {np.round( vel_c3/ (speed_factor3 * Vstr), 3)}')
print(f'chi 1e-4 -- <velocity> / vstr = {np.round( vel_c4/ (speed_factor4 * Vstr), 3)}')
print(f'chi 1e-5 -- <velocity> / vstr = {np.round( vel_c5/ (speed_factor5 * Vstr), 3)}')
print('----------------------------------------------------')


