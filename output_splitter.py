#-------------------------------------#
#       Chi output splitter           #
#     Matt Sampson  Oct. 2022         #
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
import sys
import time 

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

# -------------------------- #
# Initialise storage vectors #
# -------------------------- #
displacement_c0 = [0] ; displacement_par_c0 = [0]
displacement_perp_x_c0 = [0] ; displacement_perp_y_c0 = [0]
time_vals_c0 = [0]

displacement_c1 = [0] ; displacement_par_c1 = [0]
displacement_perp_x_c1 = [0] ; displacement_perp_y_c1 = [0]
time_vals_c1 = [0]

displacement_c2 = [0] ; displacement_par_c2 = [0]
displacement_perp_x_c2 = [0] ; displacement_perp_y_c2 = [0]
time_vals_c2 = [0]

displacement_c3 = [0] ; displacement_par_c3 = [0]
displacement_perp_x_c3 = [0] ; displacement_perp_y_c3 = [0]
time_vals_c3 = [0]

displacement_c4 = [0] ; displacement_par_c4 = [0]
displacement_perp_x_c4 = [0] ; displacement_perp_y_c4 = [0]
time_vals_c4 = [0]

displacement_c5 = [0] ; displacement_par_c5 = [0]
displacement_perp_x_c5 = [0] ; displacement_perp_y_c5 = [0]
time_vals_c5 = [0]

slice_nums = args.filenum - args.chk
n_src = 81

#--------------------#
# Animation function #
#--------------------#

for k in range(args.filenum):
        #---------------------------------------------#
        # Figure out which checkpoint file to examine #
        #---------------------------------------------#
        chk_number =  args.chk + k 
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

        #t = data['t'] / t_turnover
        t = data['t']

        # Get source
        source = data['sources']
        part_source = source['x']
        source_x, source_y, source_z = zip(*part_source)
        source_x = np.asarray(source_x)
        source_y = np.asarray(source_y)
        source_z = np.asarray(source_z)
        # Get Particle Age
        
        source_ID = np.asarray(packets['source'])
        #particle_age =  np.asarray( (t - (packets['tinj'] / t_turnover)) )
        particle_age =  np.asarray( (t - (packets['tinj'] )) )
        x_pos = np.asarray(x_pos)
        y_pos = np.asarray(y_pos)
        z_pos = np.asarray(z_pos)
        
        # ------------------------------------------ #
        # Adjust for Periodicity and Source Location #
        # ------------------------------------------ #
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
        
        # now get permuation to sort by source ID
        p = np.argsort(source_ID)

        # update the sorted data
        x_pos = x_pos[p]
        y_pos = y_pos[p]
        z_pos = z_pos[p]
        particle_age = particle_age[p]
        source_ID = source_ID[p]
        
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
        # ------------------------------------------------------------------ #
        displacement_perp_y_c0 = np.append(displacement_perp_y_c0, y_pos_c0) 
        displacement_perp_x_c0 = np.append(displacement_perp_x_c0, x_pos_c0) 
        displacement_par_c0 = np.append(displacement_par_c0, z_pos_c0)
        time_vals_c0 = np.append(time_vals_c0, particle_age_c0)
        # ------------------------------------------------------------------ #
        
        #- chi 0.1
        particle_age_c1 = np.asarray(particle_age[chi_batch_0_end:chi_batch_1_end])
        x_pos_c1 = np.asarray(x_pos[chi_batch_0_end:chi_batch_1_end])
        y_pos_c1 = np.asarray(y_pos[chi_batch_0_end:chi_batch_1_end])
        z_pos_c1 = np.asarray(z_pos[chi_batch_0_end:chi_batch_1_end])
        # ------------------------------------------------------------------ #
        displacement_perp_y_c1 = np.append(displacement_perp_y_c1, y_pos_c1) 
        displacement_perp_x_c1 = np.append(displacement_perp_x_c1, x_pos_c1) 
        displacement_par_c1 = np.append(displacement_par_c1, z_pos_c1)
        time_vals_c1 = np.append(time_vals_c1, particle_age_c1)
        # ------------------------------------------------------------------ #
        
        #- chi 0.01
        particle_age_c2 = np.asarray(particle_age[chi_batch_1_end:chi_batch_2_end])
        x_pos_c2 = np.asarray(x_pos[chi_batch_1_end:chi_batch_2_end])
        y_pos_c2 = np.asarray(y_pos[chi_batch_1_end:chi_batch_2_end])
        z_pos_c2 = np.asarray(z_pos[chi_batch_1_end:chi_batch_2_end])
        # ------------------------------------------------------------------ #
        displacement_perp_y_c2 = np.append(displacement_perp_y_c2, y_pos_c2) 
        displacement_perp_x_c2 = np.append(displacement_perp_x_c2, x_pos_c2) 
        displacement_par_c2 = np.append(displacement_par_c2, z_pos_c2)
        time_vals_c2 = np.append(time_vals_c2, particle_age_c2)
        # ------------------------------------------------------------------ #
        
        #- chi 0.001
        particle_age_c3 = np.asarray(particle_age[chi_batch_2_end:chi_batch_3_end])
        x_pos_c3 = np.asarray(x_pos[chi_batch_2_end:chi_batch_3_end])
        y_pos_c3 = np.asarray(y_pos[chi_batch_2_end:chi_batch_3_end])
        z_pos_c3 = np.asarray(z_pos[chi_batch_2_end:chi_batch_3_end])
        # ------------------------------------------------------------------ #
        displacement_perp_y_c3 = np.append(displacement_perp_y_c3, y_pos_c3) 
        displacement_perp_x_c3 = np.append(displacement_perp_x_c3, x_pos_c3) 
        displacement_par_c3 = np.append(displacement_par_c3, z_pos_c3)
        time_vals_c3 = np.append(time_vals_c3, particle_age_c3)
        # ------------------------------------------------------------------ #
        
        #- chi 1e-4
        particle_age_c4 = np.asarray(particle_age[chi_batch_3_end:chi_batch_4_end])
        x_pos_c4 = np.asarray(x_pos[chi_batch_3_end:chi_batch_4_end])
        y_pos_c4 = np.asarray(y_pos[chi_batch_3_end:chi_batch_4_end])
        z_pos_c4 = np.asarray(z_pos[chi_batch_3_end:chi_batch_4_end])
        # ------------------------------------------------------------------ #
        displacement_perp_y_c4 = np.append(displacement_perp_y_c4, y_pos_c4) 
        displacement_perp_x_c4 = np.append(displacement_perp_x_c4, x_pos_c4) 
        displacement_par_c4 = np.append(displacement_par_c4, z_pos_c4)
        time_vals_c4 = np.append(time_vals_c4, particle_age_c4)
        # ------------------------------------------------------------------ #
        
        #- chi 1e-5
        particle_age_c5 = np.asarray(particle_age[chi_batch_4_end:chi_batch_5_end])
        x_pos_c5 = np.asarray(x_pos[chi_batch_4_end:chi_batch_5_end])
        y_pos_c5 = np.asarray(y_pos[chi_batch_4_end:chi_batch_5_end])
        z_pos_c5 = np.asarray(z_pos[chi_batch_4_end:chi_batch_5_end])
        # ------------------------------------------------------------------ #
        displacement_perp_y_c5 = np.append(displacement_perp_y_c5, y_pos_c5) 
        displacement_perp_x_c5 = np.append(displacement_perp_x_c5, x_pos_c5) 
        displacement_par_c5 = np.append(displacement_par_c5, z_pos_c5)
        time_vals_c5 = np.append(time_vals_c5, particle_age_c5)
        # ------------------------------------------------------------------ #
        
        
# ----------------------------- #
# Now add to concatinated array #
# ----------------------------- #

# -------------- #
# Age Selections #
# -------------- #
age_max = t_turnover # define cutoff age since we don't delete packets

# ---------------------------------------------------------------#
Data_Combine_c0 = np.column_stack((displacement_perp_x_c0, 
                                displacement_perp_y_c0,
                                displacement_par_c0,time_vals_c0))
# Subset data for speed
Data_Use_c0 = Data_Combine_c0[Data_Combine_c0[:,3] > 1e2]
Data_Use_c0 = Data_Use_c0[Data_Use_c0[:,3] < age_max] #3e13 old val
# ---------------------------------------------------------------#

# ---------------------------------------------------------------#
Data_Combine_c1 = np.column_stack((displacement_perp_x_c1, 
                                displacement_perp_y_c1,
                                displacement_par_c1,time_vals_c1))
# Subset data for speed
Data_Use_c1 = Data_Combine_c1[Data_Combine_c1[:,3] > 1e2]
Data_Use_c1 = Data_Use_c1[Data_Use_c1[:,3] < age_max] #3e13 old val
# ---------------------------------------------------------------#

# ---------------------------------------------------------------#
Data_Combine_c2 = np.column_stack((displacement_perp_x_c2, 
                                displacement_perp_y_c2,
                                displacement_par_c2,time_vals_c2))
# Subset data for speed
Data_Use_c2 = Data_Combine_c2[Data_Combine_c2[:,3] > 1e2]
Data_Use_c2 = Data_Use_c2[Data_Use_c2[:,3] < age_max] #3e13 old val
# ---------------------------------------------------------------#

# ---------------------------------------------------------------#
Data_Combine_c3 = np.column_stack((displacement_perp_x_c3, 
                                displacement_perp_y_c3,
                                displacement_par_c3,time_vals_c3))
# Subset data for speed
Data_Use_c3 = Data_Combine_c3[Data_Combine_c3[:,3] > 1e2]
Data_Use_c3 = Data_Use_c3[Data_Use_c3[:,3] < age_max] #3e13 old val
# ---------------------------------------------------------------#

# ---------------------------------------------------------------#
Data_Combine_c4 = np.column_stack((displacement_perp_x_c4, 
                                displacement_perp_y_c4,
                                displacement_par_c4,time_vals_c4))
# Subset data for speed
Data_Use_c4 = Data_Combine_c4[Data_Combine_c4[:,3] > 1e2]
Data_Use_c4 = Data_Use_c4[Data_Use_c4[:,3] < age_max] #3e13 old val
# ---------------------------------------------------------------#

# ---------------------------------------------------------------#
Data_Combine_c5 = np.column_stack((displacement_perp_x_c5, 
                                displacement_perp_y_c5,
                                displacement_par_c5,time_vals_c5))
# Subset data for speed
Data_Use_c5 = Data_Combine_c5[Data_Combine_c5[:,3] > 1e2]
Data_Use_c5 = Data_Use_c5[Data_Use_c5[:,3] < age_max] #3e13 old val
# ---------------------------------------------------------------#

# ---------------------------- #
# Save the files per chi batch #
# ---------------------------- #
print('')
print('-----------------------------------')
Filename = 'chi_0_' + args.dir + '.txt'
print(f'Writing {Filename}')
np.savetxt(Filename, Data_Use_c0, delimiter=',')

Filename = 'chi_1_' + args.dir + '.txt'
print(f'Writing {Filename}')
np.savetxt(Filename, Data_Use_c1, delimiter=',')

Filename = 'chi_2_' + args.dir + '.txt'
print(f'Writing {Filename}')
np.savetxt(Filename, Data_Use_c2, delimiter=',')

Filename = 'chi_3_' + args.dir + '.txt'
print(f'Writing {Filename}')
np.savetxt(Filename, Data_Use_c3, delimiter=',')

Filename = 'chi_4_' + args.dir + '.txt'
print(f'Writing {Filename}')
np.savetxt(Filename, Data_Use_c4, delimiter=',')

Filename = 'chi_5_' + args.dir + '.txt'
print(f'Writing {Filename}')
np.savetxt(Filename, Data_Use_c5, delimiter=',')
print('-----------------------------------')
# --------------------------------------------- #


