#!/bin/env python                                                                                                                                                             
# Author: Brett Savoie (brettsavoie@gmail.com)

import sys,argparse,os,datetime,fnmatch,os,re,glob,math

# Add TAFFY Lib to path
sys.path.append('/'.join(os.path.abspath(__file__).split('/')[:-1])+'/Lib')
from adjacency import *
from file_parsers import xyz_parse
from id_types import *

import random
import numpy as np
import time
from shutil import move,copyfile

def main(argv):

    parser = argparse.ArgumentParser(description='Reads in a geometry and FF.db file (usually outputted from an intramolecular mode parse) '+\
                                                 'and writes inputs for a LAMMPS job to generate configurations for the parsing VDW parameters.')


    #required (positional) arguments                                                                                                  
    parser.add_argument('coord_files', help = 'A quoted list of the *.xyz files to be included in the sim, or a folder holding a previous md cycles. If a folder is supplied then the FF parameters and last configuration '+\
                                              'are used to start the new job along with the most recently fit VDW parameters.')

    #optional arguments    
    parser.add_argument('-FF',dest='FF_files', default="taffi_gen.db",
                         help = 'A quoted list of the force-field files. Formatting of the force-field files is as produced by the FF_gen.py/FF_extract.py programs)')

    parser.add_argument('-N', dest='N_mol', default="25",
                        help = 'Controls the number of molecules put in the simulation cell. The program expects a list of integers. If less integers '+\
                               'are supplied than the number of coord files, then the list is automatically expanded using the last supplied (or default) '+\
                               'element. A single integer rather than a list is also accepted. (default: 25, i.e. 25 of each molecule supplied are placed in the simulation cell)')

    parser.add_argument('-T', dest='T_equil', default=400,
                        help = 'Controls the temperature of the MD simulation. (in Kelvin; default: 400)')

    parser.add_argument('-t', dest='t_equil', default=1E6,
                        help = 'Controls the length of the MD equilibration segment. (default: 1E6)')

    parser.add_argument('-T_A', dest='T_anneal', default=400,
                        help = 'Controls the temperature of the MD simulation. (in Kelvin; default: 400)')

    parser.add_argument('-t_A', dest='t_anneal', default=1E6,
                        help = 'Controls the length of the MD equilibration segment. (default: 1E6)')

    parser.add_argument('-t_ext', dest='t_ext', default=1E6,
                        help = 'Controls the length of the MD extension job script. (default: 1E6)')

    parser.add_argument('-P', dest='pressure', default="1",
                        help = 'Controls the pressure during the MD simulation. (in ATM; default: 1)')

    parser.add_argument('-d', dest='density', default=0,
                        help = 'Sets the density for the MD simulation. When this option is set, an NVT simulation is performed instead '
                               'of an NPT simulation, and the value supplied for pressure (-P) is disregarded. '+\
                               'This option is incompatible with the -d_N option, only one can be supplied at a time (default: 0, NPT simulation; units of g/cubic cm)')

    parser.add_argument('-d_N', dest='N_density', default=0,
                        help = 'Sets the number density for the MD simulation. When this option is set, an NVT simulation is performed instead '
                               'of an NPT simulation, and the value supplied for pressure (-P) is disregarded. '+\
                               'This option is incompatible with the -d option, only one can be supplied at a time (default: 0, NPT simulation; units of atoms/cubic Angstrom)')

    parser.add_argument('-fix', dest='fixes', default="",
                        help = 'Determines if bonds and/or angles are contstrained in the simulation. "bonds" and "angles" are values options that can be supplied as a space-delimited string. By default the rattle '+\
                               'algorithm is used for constraining the modes. (default: no constrained modes)')

    parser.add_argument('-f', dest='frequency', default=1000,
                        help = 'Controls the sampling frequency during the MD simulation. (default: 1000)')

    parser.add_argument('-q', dest='q_list', default="none",
                        help = 'Controls the total charge on the molecule. By default, the rounded integer charge is used for each molecule (round). The program expects a list of integers (or none, or round for the default). '+\
                               'If less integers are supplied than the number of coord files, then the list is automatically expanded using the last supplied (or default) element. (default: None)') 

    parser.add_argument('-o', dest='outputname', default='',
                        help = 'Sets the output filename prefix (default: None)')

    parser.add_argument('-ts', dest='timestep', default=1.0,
                        help = 'Sets the timestep for the md simulation (default: 1.0, units of fs)')

    parser.add_argument('-gens', dest='gens', default=2,
                        help = 'When atomtypes are being automatically determined (i.e. when the supplied *.xyz files do not have atomtype information) this variable controls the bond depth used to define unique atomtypes. (default: 2)')

    parser.add_argument('-onefourscale_coul', dest='onefourscale_coul', default=0.0,
                        help = 'Sets scaling for 1-4 Electrostatic interactions (default: 0.0)')

    parser.add_argument('-onefourscale_lj', dest='onefourscale_lj', default=0.0,
                        help = 'Sets scaling for 1-4 VDW interactions (default: 0.0)')

    parser.add_argument('-eps_scale', dest='eps_scale', default=1.0,
                        help = 'Sets scaling for the default UFF eps parameters. To increase configurational sampling, it can be advantageous '+\
                               'to reduce the stiffness of the VDW potentials. (default: 1.0; full value)')

    parser.add_argument('-sigma_scale', dest='sigma_scale', default=1.0,
                        help = 'Sets scaling for the default UFF sigma parameters. To increase configurational sampling, it can be advantageous '+\
                               'to reduce the LJ minima of the VDW potentials. (default: 1.0; full value)')

    parser.add_argument('-mixing_rule', dest='mixing_rule', default="wh",
                        help = 'Defines the mixing rule to be used for missing LJ parameters. Waldman-Hagler (wh) and Lorentz-Berthelot (lb) are valid options. (default: "wh")')

    parser.add_argument('-pair_styles', dest='pair_styles', default='lj/cut/coul/long 14.0 14.0',
                        help = 'Supplies the information for the pair style setting(s). Supplied as a string and should be formatted for comply '+\
                               'with the LAMMPS "pair_style hybrid" command (default: "lj/cut/coul/long 14.0 14.0")')

    parser.add_argument('--tail', dest='tail_opt', default=False, action='store_const', const=True,
                        help = 'Use tail correction to compensate for LJ cutoff. (default: off, i.e., the shifted potential is used)')

    parser.add_argument('--velocities', dest='velocity_flag', default=False, action='store_const', const=True,
                        help = 'When present, the simulation will print the velocities to the dump file (default: False)')

    parser.add_argument('--molecule', dest='molecule_flag', default=False, action='store_const', const=True,
                        help = 'When present, the simulation will print the molids to the dump file (default: False)')

    parser.add_argument('--remove_multi', dest='remove_multi', default=False, action='store_const', const=True,
                        help = 'When present, the simulation will remove bimodal angles from the simulation (default: False)')

    parser.add_argument('--force_read', dest='force_read_opt', default=False, action='store_const', const=True,
                        help = 'When present, the simulation will use the force-field parameters that are discovered, even if the type does not match the expectations of the lewis structure parse. '+\
                               'In particular, sometimes the user wants to use a harmonic dihedral type for an ostensibly flexible dihedral. If only the harmonic type is supplied, the script will '+\
                               'print a missing mode warning and exit. When this flag is supplied the script will proceed to use the dihedral that is discovered in the force-field file. (default: off)')

    # Make parse inputs
    args=parser.parse_args(argv)

    # Make fake input config
    option = dict(args.__dict__.items())
    
    fun(option,argv)


def fun(config,argv):


    option = miss2default(config)
    
    coord_files = option['coord_files']
    FF_files = option['FF_files']
    N_mol = option['N_mol']
    T_equil = option['T_equil']
    t_equil = option['t_equil']
    T_anneal = option['T_anneal']
    t_anneal = option['t_anneal']
    t_ext =  option['t_ext']
    pressure = option['pressure']
    density = option['density']
    N_density = option['N_density']
    fixes = option['fixes']
    frequency = option['frequency']
    q_list = option['q_list']
    outputname = option['outputname']
    timestep = option['timestep']
    gens = option['gens']
    onefourscale_coul = option['onefourscale_coul']
    onefourscale_lj =  option['onefourscale_lj']
    eps_scale = option['eps_scale']
    sigma_scale = option['sigma_scale']
    force_UFF = option['force_UFF']
    mixing_rule = option['mixing_rule']
    pair_styles = option['pair_styles']
    tail_opt = option['tail_opt']
    improper_flag = option['improper_flag']
    velocity_flag = option['velocity_flag']
    molecule_flag = option['molecule_flag']
    remove_multi = option['remove_multi']
    force_read_opt = option['force_read_opt']

    # Converting each list input argument. 
    coord_files = coord_files.split()
    FF_files = FF_files.split()
    N_mol = [ int(i) for i in N_mol.split() ]
    q_list = [ str(i) for i in q_list.split() ]

    # Check if a directory was supplied to the coord_files list.
    bools = [ os.path.isdir(i) for i in coord_files ]
    if True in bools: 
        if len([ True for i in bools if i == True ]) > 1:
            print("ERROR: multiple directories were supplied to the script. Only one directory can\nbe supplied if the user intends to start the job from a previous run. Exiting...")
            quit()        
        coord_files = [coord_files[bools.index(True)]]
        coord_files = glob.glob("{}/*.xyz".format(os.path.join(os.getcwd(), coord_files[0])), recursive=True)
        read_flag = 1
    else:
        read_flag = 0

    # Extend N_mol and q_list to match the length of coord_file
    while len(N_mol) < len(coord_files): N_mol = N_mol + [N_mol[-1]] 
    while len(q_list) < len(coord_files): q_list = q_list + [q_list[-1]] 

    # Parse input
    T_equil      = float(T_equil)
    T_anneal     = float(T_anneal)
    t_equil      = int(float(t_equil))
    t_anneal     = int(float(t_anneal))
    t_ext        = int(float(t_ext))
    if "gpa" in pressure.lower():
        pressure = float(pressure.lower().strip("gpa"))*9869.23169314269
    elif "atm" in pressure.lower():
        pressure = float(pressure.lower().strip("atm"))
    else:
        pressure     = float(pressure)
    frequency    = int(float(frequency))
    eps_scale    = float(eps_scale)
    sigma_scale  = float(sigma_scale)
    density      = float(density)
    N_density    = float(N_density)
    gens         = int(gens)
    timestep     = float(timestep)
    mixing_rule = str(mixing_rule).lower()
    
    if density != 0 and N_density != 0: print("ERROR: Setting both the -d and -d_N options is inconsistent (both specify density). Please revise. Exiting..."); quit()
    if mixing_rule not in ["wh","lb"]:
        print("ERROR: '{}' is not a valid argument for -mixing_rule. Valid options are 'wh' and 'lb'. Exiting...".format(mixing_rule))
        quit()
    if timestep < 0: print("ERROR: -ts must be set to a positive value. Exiting..."); quit()

    # Parse output filename 
    if outputname != '':

        # Handle the special case where the script is run from without the output directory
        if outputname == ".":
            print("ERROR: the output folder must differ from the current folder.")
            quit()

        # Else, directly assign the output directory based on the -o argument
        else:
            Filename = outputname        

    # If the output folder is not specified then the program defaults to the root of the first supplied *xyz file
    else:
        Filename = coord_files[0].split('.')[0]
    
    # Check that the input is an .xyz file. 
    if read_flag == 0:
        for i in coord_files:
            if i.split('.')[-1] != 'xyz':
                print("ERROR: Check to ensure that the input coordinate file(s) are in .xyz format.")
                quit()
            elif os.path.isfile(i) != True:
                print("ERROR: Specified *.xyz file ({}) does not exit. Please check the path. Exiting...".format(i))
                quit()

    # Check that the supplied FF files exist
    for i in FF_files:
        if os.path.isfile(i) != True:
            print("ERROR: Specified FF_file ({}) does not exist. Please check the path. Exiting...".format(i))
            quit()

    # If the output directory already exists, exit to avoid overwriting data    
    if os.path.isdir(Filename):        
        print("ERROR: Specified folder ({}) already exist. Please try another name. Exiting...".format(Filename))
        quit()

    # Else, create a bare directory
    else:
        os.makedirs(Filename)
        sys.stdout = Logger(Filename)
        print("PROGRAM CALL: python gen_md_for_sampling.py {}\n".format(' '.join([ i for i in argv])))

    # Catenate the FF files and copy them to the output folder
    FF_all = Filename+"/"+Filename.split('/')[-1]+'.db'
    with open(FF_all,'w') as f:
        for i in FF_files:
            with open(i,'r') as ff_file:
                for j in ff_file:
                    f.write(j)
            f.write("\n")

    # Grab data on each molecule being added to the MD simulation.
    Data = get_data(FF_all,coord_files,N_mol,q_list,gens,Improper_flag=improper_flag,force_read_opt=force_read_opt,remove_multi=remove_multi)

    # Pack the simulation cell
    Elements,Atom_types,Geometry,Bonds,Bond_types,Angles,Angle_types,Dihedrals,Dihedral_types,Impropers,Improper_types,Charges,Molecule,Molecule_names,Adj_mat,Sim_Box,b_min = \
    Pack_Box(Data,Box_offset=0,Density=density,N_Density=N_density)

    # Generate VDW parameters    
    VDW_params = initialize_VDW(sorted(set(Atom_types)),sigma_scale=sigma_scale,eps_scale=eps_scale,VDW_FF=Data[coord_files[0]]["VDW_params"],\
                                Force_UFF=force_UFF,mixing_rule=mixing_rule)    

    # Generate Simulation Dictionaries
    # The bond, angle, and diehdral parameters for each molecule are combined into one dictionary
    Bond_params = {}; Angle_params = {}; Dihedral_params = {}; Improper_params = {}; Masses = {}
    for i in list(Data.keys()):
        for j in list(Data[i]["Bond_params"].keys()): Bond_params[j] = Data[i]["Bond_params"][j]
        for j in list(Data[i]["Angle_params"].keys()): Angle_params[j] = Data[i]["Angle_params"][j]
        for j in list(Data[i]["Dihedral_params"].keys()): Dihedral_params[j] = Data[i]["Dihedral_params"][j]
        for j in list(Data[i]["Improper_params"].keys()): Improper_params[j] = Data[i]["Improper_params"][j]
        for j in list(Data[i]["Masses"].keys()): Masses[j] = Data[i]["Masses"][j]

    # Write the lammps datafile (the function returns a dictionary, Atom_type_dict, holding the mapping between atom_types and the lammps id numbers; this mapping is needed for setting fixes)
    print("Writing LAMMPS datafile ({})...".format(Filename+'.data'))
    print("Writing LAMMPS settings file ({})...".format(Filename+'.in.settings'))
    Atom_type_dict,fixed_modes = Write_data(Filename,Atom_types,Sim_Box,Elements,Geometry,Bonds,Bond_types,Bond_params,Angles,Angle_types,Angle_params,\
                                            Dihedrals,Dihedral_types,Dihedral_params,Impropers,Improper_types,Improper_params,Charges,VDW_params,Masses,Molecule,Improper_flag = improper_flag)

    # Gather the different dihedral_styles
    Dihedral_styles = set([ Dihedral_params[i][0] for i in Dihedral_types ])

    # Write the lammps input files
    print("Writing LAMMPS input file ({})...".format(Filename.split('/')[-1]+'.in.init'))
    Write_input(Filename,T_equil,t_equil,T_anneal,t_anneal,t_ext,pressure,frequency,onefourscale_coul,onefourscale_lj,pair_styles,\
                Dihedral_styles,b_min,fixes,fixes,Bond_types,Angle_types,tail_opt=tail_opt,timestep=timestep,fixed_modes=fixed_modes,improper_flag=improper_flag,\
                velocity_opt=velocity_flag,mol_opt=molecule_flag,mixing_rule=mixing_rule)

    # Write map file to more easily post-process the trajectory
    print("Writing mapfile ({})...".format(Filename.split('/')[-1]+'.map'))
    write_map(Filename,Elements,Atom_types,Charges,Masses,Adj_mat,np.zeros([len(Elements)]),N_mol)

    # Writing molecule map file
    print("Writing molfile ({})...".format(Filename.split('/')[-1]+'.mol.txt'))
    write_molecule(Filename,Molecule_names)

    # Remove concatenated FF file from run folder
    os.remove(FF_all)

    # Print banner
    print("\n{}\n* {:^173s} *\n{}\n".format("*"*177,"Success! Have a Nice Day!","*"*177))

# Description: A wrapper for the write commands for generating the lammps input file
def Write_input(Filename,equil_temp,equil_time,anneal_temp,anneal_time,extend_time,pressure,frequency,Onefourscale_coul,Onefourscale_lj,Pair_styles,\
                Dihedral_styles,b_min,fixes,ba_fixes,Bond_types,Angle_types,tail_opt=False,timestep=1.0,skip_resize=False,fixed_modes={'bonds':[], 'angles':[]},improper_flag=False,\
                velocity_opt=False,mol_opt=False,mixing_rule='wh'):
    
    # Initialize rigid bond/angle fix commands
    # NOTE: the ba_fixes arguments override the specific bond and angle fixes.
    ba_fix_cmd = ''

    # Case: all bonds and angles are treated as fixed
    if 'bonds' in ba_fixes  and 'angles' in ba_fixes:
        ba_fix_cmd="fix rigid all rattle 0.0001 20 ${coords_freq} "+"b {} a {}\n".format(" ".join([ str(count_i+1) for count_i,i in enumerate(set(Bond_types))])," ".join([ str(count_i+1) for count_i,i in enumerate(set(Angle_types))]))

    # Case: all bonds and only a subset of angles are treated as fixed
    elif 'bonds' in ba_fixes:
        bond_fixes  = " ".join([ str(count_i+1) for count_i,i in enumerate(set(Bond_types))])
        angle_fixes = " ".join([ str(i) for i in fixed_modes["angles"] ])
        if len(angle_fixes) == 0:
            ba_fix_cmd="fix rigid all rattle 0.0001 20 ${coords_freq} "+"b {}\n".format(bond_fixes)
        else:
            ba_fix_cmd="fix rigid all rattle 0.0001 20 ${coords_freq} "+"b {} a {}\n".format(bond_fixes,angle_fixes)

    # Case: all angles and only a subset of bonds are treated as fixed
    elif 'angles' in ba_fixes:
        bond_fixes  = " ".join([ str(i) for i in fixed_modes["bonds"] ])
        angle_fixes = " ".join([ str(count_i+1) for count_i,i in enumerate(set(Angle_types))])
        if len(bond_fixes) == 0:
            ba_fix_cmd="fix rigid all rattle 0.0001 20 ${coords_freq} "+"a {}\n".format(angle_fixes)
        else:
            ba_fix_cmd="fix rigid all rattle 0.0001 20 ${coords_freq} "+"b {} a {}\n".format(bond_fixes,angle_fixes)

    # Case: only a subset of bonds/angles are treated as fixed
    else:
        bond_fixes  = " ".join([ str(i) for i in fixed_modes["bonds"] ])
        angle_fixes = " ".join([ str(i) for i in fixed_modes["angles"] ])
        if len(bond_fixes) != 0 and len(angle_fixes) != 0:
            ba_fix_cmd="fix rigid all rattle 0.0001 20 ${coords_freq} "+"b {} a {}\n".format(bond_fixes,angle_fixes)            
        elif len(bond_fixes) != 0:
            ba_fix_cmd="fix rigid all rattle 0.0001 20 ${coords_freq} "+"b {}\n".format(bond_fixes)
        elif len(angle_fixes) != 0:
            ba_fix_cmd="fix rigid all rattle 0.0001 20 ${coords_freq} "+"a {}\n".format(angle_fixes)

    # Initialize velocity_string and molecule_string
    if velocity_opt is True:
        velocity_string = " vx vy vz"
    else:
        velocity_string = ""
    if mol_opt is True:
        molecule_string = " mol"
    else:
        molecule_string = ""

    # Specify mixing rule used
    mixing_name = {"lb":"Lorentz-Berthelot", "wh":"Waldman-Hagler"}

    with open(Filename+'/'+Filename.split('/')[-1]+'.in.init','w') as f:
        f.write("# lammps input file for polymer simulation with dilute ions\n\n"+\
                "# VARIABLES\n"+\
                "variable        data_name       index   {}\n".format(Filename.split('/')[-1]+'.data')+\
                "variable        settings_name   index   {}\n".format(Filename.split('/')[-1]+'.in.settings')+\
                "variable        log_name        index   {}\n".format(Filename.split('/')[-1]+'.log')+\
                "variable        nSteps_ramp     index   {} # number of data steps for the ramped anneal\n".format(int(round((anneal_time/2)/timestep)))+\
                "variable        nSteps_equil    index   {} # number of data steps for the ramped anneal\n".format(int(round(equil_time/timestep)))+\
                "variable        avg_freq        index   1000\n".format(int(round(frequency/timestep)))+\
                "variable        coords_freq     index   1000\n".format(int(round(frequency/timestep)))+\
                "variable        thermo_freq     index   1000\n".format(int(round(frequency/timestep)))+\
                "variable        dump4avg        index   100\n"+\
                "variable        vseed           index   {: <6d}\n".format(int(random.random()*100000))+\
                "variable        ANNEAL_TEMP     index   {} # Temperature during the initial anneal\n".format(anneal_temp)+\
                "variable        FINAL_TEMP      index   {} # Temperature ramped to during the final anneal\n".format(equil_temp)+\
                "variable        pressure        index   {} # Pressure during the simulations\n\n".format(pressure)+\

                "# Change the name of the log output #\n"+\
                "log ${log_name}\n\n"+\
                
                "#===========================================================\n"+\
                "# GENERAL PROCEDURES\n"+\
                "#===========================================================\n"+\
                "units		real	# g/mol, angstroms, fs, kcal/mol, K, atm, charge*angstrom\n"+\
                "dimension	3	    # 3 dimensional simulation\n"+\
                "newton		on	    # use Newton's 3rd law\n"+\
                "boundary	p p p	# periodic boundary conditions \n"+\
                "atom_style	full    # molecular + charge\n\n"+\

                "#===========================================================\n"+\
                "# FORCE FIELD DEFINITION\n"+\
                "#===========================================================\n"+\
                'special_bonds  lj   0.0 0.0 {}  coul 0.0 0.0 {}     # NO     1-4 LJ/COUL interactions\n'.format(Onefourscale_lj,Onefourscale_coul)+\
                'pair_style     hybrid {}      # outer_LJ outer_Coul (cutoff values, see LAMMPS Doc)\n'.format(Pair_styles)+\
                'kspace_style   pppm 0.0001                            # long-range electrostatics sum method\n'+\
                'bond_style     harmonic                               # parameters needed: k_bond, r0\n'+\
                'angle_style    harmonic                               # parameters needed: k_theta, theta0\n')
        if len(Dihedral_styles) >= 1:
            f.write('dihedral_style hybrid {}                            # parameters needed: V1, V2, V3, V4\n'.format(' '.join(Dihedral_styles)))            
        if improper_flag:
            f.write('improper_style harmonic                             # parameters needed: k_psi, psi0\n')
        if tail_opt is True:
            f.write('pair_modify    tail yes mix arithmetic                # using {} mixing rules\n\n'.format(mixing_name[mixing_rule]))
        else:
            f.write('pair_modify    shift yes mix arithmetic               # using {} mixing rules\n\n'.format(mixing_name[mixing_rule]))

        f.write("#===========================================================\n"+\
                "# SETUP SIMULATIONS\n"+\
                "#===========================================================\n\n"+\
                "# READ IN COEFFICIENTS/COORDINATES/TOPOLOGY\n"+\
                'read_data ${data_name}\n'+\
                'include ${settings_name}\n\n'+\

                "# SET RUN PARAMETERS\n"+\
                "timestep	{}		# fs\n".format(timestep)+\
                "run_style	verlet 		# Velocity-Verlet integrator\n"+\
                "neigh_modify every 1 delay 0 check no one 10000 # More relaxed rebuild criteria can be used\n\n")
        
        if fixes != []:
            f.write("{}".format(fixes))

        f.write("#===========================================================\n"+\
                "# RUN CONSTRAINED RELAXATION\n"+\
                "#===========================================================\n\n"+

                "# SET OUTPUTS\n"+\
                "thermo_style    custom step temp vol density etotal pe ebond eangle edihed ecoul elong evdwl enthalpy press\n"+\
                "thermo_modify   format float %20.10f\n"+\
                "thermo ${thermo_freq}\n\n"+\

                "# DECLARE RELEVANT OUTPUT VARIABLES\n"+\
                "variable        my_step equal   step\n"+\
                "variable        my_temp equal   temp\n"+\
                "variable        my_rho  equal   density\n"+\
                "variable        my_pe   equal   pe\n"+\
                "variable        my_ebon equal   ebond\n"+\
                "variable        my_eang equal   eangle\n"+\
                "variable        my_edih equal   edihed\n"+\
                "variable        my_evdw equal   evdwl\n"+\
                "variable        my_eel  equal   (ecoul+elong)\n"+\
                "variable        my_ent  equal   enthalpy\n"+\
                "variable        my_P    equal   press\n"+\
                "variable        my_vol  equal   vol\n\n"+\

                "fix  averages all ave/time ${dump4avg} $(v_avg_freq/v_dump4avg) ${avg_freq} v_my_temp v_my_rho v_my_vol v_my_pe v_my_edih v_my_evdw v_my_eel v_my_ent v_my_P file thermo.avg\n\n"+\

                "# INITIALIZE VELOCITIES AND CREATE THE CONSTRAINED RELAXATION FIX\n"+\
                "velocity        all create ${ANNEAL_TEMP} ${vseed} mom yes rot yes     # DRAW VELOCITIES\n\n"+\

                "fix relax all nve/limit 0.01\n")

        # Write rattle constraints
        if len(ba_fix_cmd) != 0:
            f.write(ba_fix_cmd)
        f.write("run             10000\n")
        if len(ba_fix_cmd) != 0:
            f.write("unfix rigid\n")
        f.write("unfix relax\n\n")

        # Write the box resize command for constant density simulations
        if b_min is not None and skip_resize is False:
            f.write("#===========================================================\n"+\
                    "# RESHAPE BOX FOR CONSTANT DENSITY NVT \n"+\
                    "#===========================================================\n"+\
                    "fix dynamics all nvt temp ${ANNEAL_TEMP} ${ANNEAL_TEMP} 100.0\n"+\
                    "velocity all create ${ANNEAL_TEMP} ${vseed} mom yes rot yes\n"+\
                    "fix mom all momentum 100 linear 1 1 1 angular\n\n"+\
                    
                    "fix reshape  all deform 1 x final -{} {} y final -{} {} z final -{} {} units box\n".format(b_min,b_min,b_min,b_min,b_min,b_min))

            # Write rattle constraints
            if len(ba_fix_cmd) != 0:
                f.write(ba_fix_cmd)
            f.write("run             10000\n")
            if len(ba_fix_cmd) != 0:
                f.write("unfix rigid\n")

            
            f.write("unfix dynamics\n"+\
                    "unfix reshape\n"+\
                    "unfix mom\n\n")                    
                    
        f.write("#===========================================================\n"+\
                "# RUN RAMPED ANNEAL\n"+\
                "#===========================================================\n\n"+\
                
                "# REINITIALIZE THE VELOCITIES AND CREATE THE ANNEALING FIX\n"+\
                "velocity        all create ${ANNEAL_TEMP} ${vseed} mom yes rot yes     # DRAW VELOCITIES\n\n")
        f.write("fix mom all momentum 1000 linear 1 1 1 angular # Zero out system linear and angular momentum every ps \n")
        if b_min is None:
            f.write("fix anneal all npt temp ${ANNEAL_TEMP} ${FINAL_TEMP} 100.0 iso ${pressure} ${pressure} 1000.0 # NPT, nose-hoover 100 fs T relaxation\n\n")
        else:
            f.write("fix anneal all nvt temp ${ANNEAL_TEMP} ${FINAL_TEMP} 100.0 # NVT, nose-hoover 100 fs T relaxation\n\n")

        f.write("# CREATE COORDINATE DUMPS FOR ANNEAL\n"+\
                "dump anneal all custom ${coords_freq} anneal.lammpstrj id type x y z" + velocity_string + molecule_string + "\n"+\
                "dump_modify anneal sort id format float %20.10g\n\n"+\

                "# RUN RAMPED ANNEAL\n")

        # Write rattle constraints
        if len(ba_fix_cmd) != 0:
            f.write(ba_fix_cmd)
        f.write("run ${nSteps_ramp}\n")
        if len(ba_fix_cmd) != 0:
            f.write("unfix rigid\n")
        f.write("unfix anneal\n\n"+\

                "# RUN EQUILIBRATION PHASE AT FINAL TEMP\n")
        if b_min is None:
            f.write("fix anneal all npt temp ${FINAL_TEMP} ${FINAL_TEMP} 100.0 iso ${pressure} ${pressure} 1000.0 # NPT, nose-hoover 100 fs T relaxation\n")
        else:
            f.write("fix anneal all nvt temp ${FINAL_TEMP} ${FINAL_TEMP} 100.0 # NVT, nose-hoover 100 fs T relaxation\n")

        # Write rattle constraints
        if len(ba_fix_cmd) != 0:
            f.write(ba_fix_cmd)
        f.write("run ${nSteps_ramp}\n")
        if len(ba_fix_cmd) != 0:
            f.write("unfix rigid\n")
        f.write("unfix anneal\n"+\
                "undump anneal\n\n")

        # Restore more conservative neighbor list rebuilds
        f.write('# RETURN TO LESS FREQUENT NEIGHBOR BUILDS\n'+\
                "neigh_modify every 1 delay 10 check yes one 10000 # More relaxed rebuild criteria can be used\n\n")

        if equil_time > 0:
            f.write("#===========================================================\n"+\
                    "# RUN EQUILIBRIUM SIM\n"+\
                    "#===========================================================\n\n"+

                    "# UPDATE RUN PARAMETERS AND CREATE FIX\n")
            if b_min is None:
                f.write("fix equil all npt temp ${FINAL_TEMP} ${FINAL_TEMP} 100.0 iso ${pressure} ${pressure} 1000.0 # NPT, nose-hoover 100 fs T relaxation\n\n")
            else:
                f.write("fix equil all nvt temp ${FINAL_TEMP} ${FINAL_TEMP} 100.0 # NVT, nose-hoover 100 fs T relaxation\n\n")

            f.write("# CREATE COORDINATE DUMPS FOR EQUILIBRIUM\n"+\
                    "dump equil all custom ${coords_freq} equil.lammpstrj id type x y z" + velocity_string + molecule_string + "\n"+\
                    "dump_modify equil sort id format float %20.10g\n\n"+\

                    "# RUN EQUIL\n")

            # Write rattle constraints
            if len(ba_fix_cmd) != 0:
                f.write(ba_fix_cmd)
            f.write("run		${nSteps_equil}\n")
            if len(ba_fix_cmd) != 0:
                f.write("unfix rigid\n")
            f.write("unfix equil\n"+\
                    "undump equil\n\n")
                        
        f.write("# WRITE RESTART FILES, CLEANUP, AND EXIT\n"+\
                "write_restart   {}\n".format(Filename.split('/')[-1]+'.end.restart')+\
                "write_data      {} pair ii\n".format(Filename.split('/')[-1]+'.end.data')+\
                "unfix		   averages\n")
    
    # Write header of trajectory extension file
    with open(Filename+'/extend.in.init','w') as f:
        f.write("# lammps input file for insertion trajectory\n\n"+\
                "# VARIABLES\n"+\
                "log {}\n".format('extend.log'))

        f.write("variable    data_name       index   {}\n".format(Filename.split('/')[-1]+'.end.data')+\
                "variable    restart_name    index   extend.end.restart\n"+\
                "variable    settings_name   index   {}\n".format(Filename.split('/')[-1]+'.in.settings')+\
                "variable    n_equil         index   {}	  # number of data steps equilibration segment\n".format(int(round(extend_time/timestep)))+\
                "variable    coords_freq     index   {} # the frequency that coordinates are prints\n".format(int(round(frequency/timestep)))+\
                "variable    thermo_freq     index   {} # the frequency that instantaneous thermodynamic variables are printed to output.\n".format(int(round(frequency/timestep)))+\
                "variable    avg_freq        index   {} # the frequency that the thermodynamics averages are printed (to file)\n".format(int(round(frequency/timestep)))+\
                "variable    avg_spacing     index   1    # the spacing of snapshots used for average values calculations (printed to file)\n"+\
                "variable    Temp            index   {}\n".format(equil_temp)+\
                "variable    pressure        index   {} # Pressure during the simulations\n".format(pressure)+\
                "variable    run             index   0\n\n")

        f.write("#===========================================================\n"+\
                "# GENERAL PROCEDURES\n"+\
                "#===========================================================\n"+\
                "units		real	# g/mol, angstroms, fs, kcal/mol, K, atm, charge*angstrom\n"+\
                "dimension	3	# 3 dimensional simulation\n"+\
                "newton		on	# use Newton's 3rd law\n"+\
                "boundary	p p p	# periodic boundary conditions \n"+\
                "atom_style	full    # molecular + charge\n\n"+\

                "#===========================================================\n"+\
                "# FORCE FIELD DEFINITION\n"+\
                "#===========================================================\n"+\
                'special_bonds  lj   0.0 0.0 {}  coul 0.0 0.0 {}     # NO     1-4 LJ/COUL interactions\n'.format(Onefourscale_lj,Onefourscale_coul)+\
                'pair_style     hybrid {} # outer_LJ outer_Coul (cutoff values, see LAMMPS Doc)\n'.format(Pair_styles)+\
                'kspace_style   pppm 0.0001          # long-range electrostatics sum method\n'+\
                'bond_style     harmonic             # parameters needed: k_bond, r0\n'+\
                'angle_style    harmonic             # parameters needed: k_theta, theta0\n')
        if len(Dihedral_styles) >= 1:
            f.write('dihedral_style hybrid {}            # parameters needed: V1, V2, V3, V4\n'.format(' '.join(Dihedral_styles)))            
        if improper_flag:
            f.write('improper_style harmonic             # parameters needed: k_psi, psi0\n')
        if tail_opt is True:
            f.write('pair_modify    tail yes mix arithmetic       # using Lorenz-Berthelot mixing rules\n\n')
        else:
            f.write('pair_modify    shift yes mix arithmetic       # using Lorenz-Berthelot mixing rules\n\n')

        f.write("#===========================================================\n"+\
                "# SETUP SIMULATIONS\n"+\
                "#===========================================================\n\n"+\
                "# READ IN COEFFICIENTS/COORDINATES/TOPOLOGY\n"+\
                'if "${run} == 0" then &\n'+\
                '   "read_data ${data_name}" &\n'+\
                'else &\n'+\
                '   "read_restart ${restart_name}"\n'+\
                'include ${settings_name}\n\n')

        f.write("# SET RUN PARAMETERS\n"+\
                "timestep	{}		# fs\n".format(timestep)+\
                "run_style	verlet 		# Velocity-Verlet integrator\n"+\
                "neigh_modify every 1 delay 10 check yes one 10000\n\n"+\

                "# SET OUTPUTS\n"+\
                "thermo_style    custom step temp vol density etotal pe ebond eangle edihed ecoul elong evdwl enthalpy press\n"+\
                "thermo_modify   format float %20.10f\n"+\
                "thermo ${thermo_freq}\n\n"+\

                "# Declare relevant output variables and create averages fix\n"+\
                "variable        my_step equal   step\n"+\
                "variable        my_temp equal   temp\n"+\
                "variable        my_rho  equal   density\n"+\
                "variable        my_pe   equal   pe\n"+\
                "variable        my_ebon equal   ebond\n"+\
                "variable        my_eang equal   eangle\n"+\
                "variable        my_edih equal   edihed\n"+\
                "variable        my_evdw equal   evdwl\n"+\
                "variable        my_eel  equal   (ecoul+elong)\n"+\
                "variable        my_ent  equal   enthalpy\n"+\
                "variable        my_P    equal   press\n"+\
                "variable        my_vol  equal   vol\n"+\
                "fix averages all ave/time ${avg_spacing} $(v_thermo_freq/v_avg_spacing) ${thermo_freq} v_my_temp v_my_rho v_my_vol v_my_pe v_my_edih v_my_evdw v_my_eel v_my_ent v_my_P file ${run}.thermo.avg\n\n"+\

                "#===========================================================\n"+\
                "# RUN EXTENSION\n"+\
                "#===========================================================\n\n"+\

                "# Set npt/nvt fix for the runs\n"+\
                "fix mom all momentum 1000 linear 1 1 1 angular # Zero out system linear and angular momentum every ps \n")

        # No dilatometry option
        if b_min is None:
            f.write('fix tmp all npt temp ${Temp} ${Temp} 100.0 iso ${pressure} ${pressure} 1000.0 # NPT, nose-hoover 100 fs T relaxation\n\n')
        else:
            f.write('fix tmp all nvt temp ${Temp} ${Temp} 100.0 # NVT, nose-hoover 100 fs T relaxation\n\n')


        f.write('# Create coordinate dump\n')
        f.write('dump equil all custom ${coords_freq} ${run}.sys.lammpstrj id type x y z' + velocity_string + molecule_string + "\n"+\
                'dump_modify equil sort id format float %20.10g\n')

        # Write rattle constraints
        if len(ba_fix_cmd) != 0:
            f.write(ba_fix_cmd)

        # Write run command
        f.write('\n# Run equilibration segment\n'+\
                'run		${n_equil}\n\n')
        
        # Write unfix command
        if len(ba_fix_cmd) != 0:
            f.write("unfix rigid\n\n")

        f.write('# Write restart files, cleanup, and exit\n'+\
                'write_restart   extend.end.restart\n'+\
                'write_data      extend.end.data pair ii\n\n'+\

                '# Reset dump and increment\n'+\
                'undump equil\n\n'+\

                '# Update run number\n'+\
                'variable sub equal (v_run+1)\n'+\
                'shell sed -i /variable.*run.*index/s/${run}/${sub}/g extend.in.init\n\n')

        f.close()

    return

# Description: A wrapper for the commands to generate a cubic box and array of molecules for the lammps run
def Pack_Box(Data,Box_offset=0,Density=0.0,N_Density=0.0):
    
    print("\nPacking simulation cell...")
    # Center each molecule at the origin
    for i in list(Data.keys()):
        Data[i]["Geometry"] -= (np.mean(Data[i]["Geometry"][:,0]),np.mean(Data[i]["Geometry"][:,1]),np.mean(Data[i]["Geometry"][:,2]))

    # Define box circumscribing the molecule
    for i in list(Data.keys()):
        Data[i]["Mol_Box"] = \
        ( np.min(Data[i]["Geometry"][:,0]),np.max(Data[i]["Geometry"][:,0]),np.min(Data[i]["Geometry"][:,1]),np.max(Data[i]["Geometry"][:,1]),np.min(Data[i]["Geometry"][:,2]),np.max(Data[i]["Geometry"][:,2]) )

    # Find the largest step_size
    # Use the geometric norm of the circumscribing cube as the step size for tiling
    Step_size=0.0
    for i in list(Data.keys()):
        Current_step = ( (Data[i]["Mol_Box"][1]-Data[i]["Mol_Box"][0])**2 +\
                         (Data[i]["Mol_Box"][3]-Data[i]["Mol_Box"][2])**2 +\
                         (Data[i]["Mol_Box"][5]-Data[i]["Mol_Box"][4])**2 )**(0.5)+3  # +3 is just to be safe in the case of perfect alignment
        if Step_size < Current_step:
            Step_size = Current_step


    # Find the smallest N^3 cubic lattice that is greater than N_tot
    N_tot = np.sum([ Data[i]["N_mol"] for i in list(Data.keys())]) # Define the total number of molecules to be placed
    N_lat = 1
    while(N_lat**3 < N_tot):
        N_lat = N_lat + 1

    # Find molecular centers
    Centers = np.zeros([N_tot,3])
    count = 0
    for i in range(N_lat):

        if count == N_tot:
            break

        for j in range(N_lat):

            if count == N_tot:
                    break

            for k in range(N_lat):

                if count == N_tot:
                    break
                
                Centers[i*N_lat**2 + j*N_lat + k] = np.array([Step_size*i,Step_size*j,Step_size*k])
                count = count + 1    

    # Randomize the order of the centers so that molecules are randomly placed
    np.random.shuffle(Centers[:])
    
    # Find the box extrema. The same value is used for x y and z so that a cubic box is obtained (Note, the extra "Step_size" builds in padding even without Box_offset)
    low = np.min([np.min(Centers[:,0])-Box_offset-Step_size,np.min(Centers[:,1])-Box_offset-Step_size,np.min(Centers[:,2])-Box_offset-Step_size])
    high = np.max([np.max(Centers[:,0])+Box_offset+Step_size,np.max(Centers[:,1])+Box_offset+Step_size,np.max(Centers[:,2])+Box_offset+Step_size])

    # Center the box at the origin
    disp = -1*np.mean([low,high])
    low += disp
    high += disp
    for count_i,i in enumerate(Centers):
        Centers[count_i] = i + disp

    # Define sim box
    Sim_Box = np.array([low,high,low,high,low,high])

    # Intialize lists for iterating over molecules and keeping track of how many have been placed
    keys = list(Data.keys())             # list of unique molecule keys
    placed_num = [0]*len(keys)     # list for keeping track of how many of each molecule have been placed
    atom_index = 0                 # an index for keeping track of how many atoms have been placed
    mol_index = 0                  # an index for keeping track of how many molecules have been placed

    # Initialize various lists to hold the elements, atomtype, molid labels etc.
    # Create a list of molecule ids for each atom        
    Geometry_sim       = np.zeros([np.sum([ Data[i]["N_mol"]*len(Data[i]["Geometry"]) for i in list(Data.keys()) ]),3])
    Adj_mat_sim        = np.zeros([np.sum([ Data[i]["N_mol"]*len(Data[i]["Geometry"]) for i in list(Data.keys()) ]),np.sum([ Data[i]["N_mol"]*len(Data[i]["Geometry"]) for i in list(Data.keys()) ])])
    Molecule_sim       = []
    Molecule_files     = []
    Elements_sim       = [] 
    Atom_types_sim     = []
    Bonds_sim          = []
    Bond_types_sim     = []
    Angles_sim         = []
    Angle_types_sim    = []
    Dihedrals_sim      = []
    Dihedral_types_sim = []
    Impropers_sim      = []
    Improper_types_sim = []
    Charges_sim        = []
    Masses_sim         = []

    # Place molecules in the simulation box and extend simulation lists 
    while (np.sum(placed_num) < N_tot):

        # Place the molecules in round-robin fashion to promote mixing
        # Each molecule key is iterated over, if all molecules of this type have been
        # placed then the molecule type is skipped
        for count_k,k in enumerate(keys):
            
            # If all the current molecules types have been placed continue
            # else: increment counter
            if placed_num[count_k] >= Data[k]["N_mol"]:
                continue
            else:
                placed_num[count_k]+=1
            print("placing atoms {}:{} with molecule {}".format(atom_index,atom_index+len(Data[k]["Geometry"]),k))

            # perform x rotations
            angle = random.random()*360
            for count_j,j in enumerate(Data[k]["Geometry"]):
                Data[k]["Geometry"][count_j,:] = axis_rot(j,np.array([1.0,0.0,0.0]),np.array([0.0,0.0,0.0]),angle,mode='angle')

            # perform y rotations
            angle = random.random()*360
            for count_j,j in enumerate(Data[k]["Geometry"]):
                Data[k]["Geometry"][count_j,:] = axis_rot(j,np.array([0.0,1.0,0.0]),np.array([0.0,0.0,0.0]),angle,mode='angle')

            # perform z rotations
            angle = random.random()*360
            for count_j,j in enumerate(Data[k]["Geometry"]):
                Data[k]["Geometry"][count_j,:] = axis_rot(j,np.array([0.0,0.0,1.0]),np.array([0.0,0.0,0.0]),angle,mode='angle')

            # Move the current molecule to its box, append to Geometry_sim, return to the molecule to the origin
            Data[k]["Geometry"] += Centers[mol_index]
            Geometry_sim[atom_index:(atom_index+len(Data[k]["Geometry"])),:] = Data[k]["Geometry"]
            Data[k]["Geometry"] -= Centers[mol_index]

            # Extend various lists (total elements lists, atomtypes lists, etc)
            # Note: the lammps input expects bonds,angles,dihedrals, etc to be defined in terms of atom
            #       id so, the atom_index is employed to keep track of how many atoms have been placed.
            Adj_mat_sim[atom_index:(atom_index+len(Data[k]["Geometry"])),atom_index:(atom_index+len(Data[k]["Geometry"]))] = Data[k]["Adj_mat"]
            Molecule_sim       = Molecule_sim + [mol_index]*len(Data[k]["Elements"])
            Molecule_files     = Molecule_files + [k]
            Elements_sim       = Elements_sim + Data[k]["Elements"]
            Atom_types_sim     = Atom_types_sim + Data[k]["Atom_types"]
            Bonds_sim          = Bonds_sim + [ (j[0]+atom_index,j[1]+atom_index) for j in Data[k]["Bonds"] ]
            Bond_types_sim     = Bond_types_sim + Data[k]["Bond_types"]
            Angles_sim         = Angles_sim + [ (j[0]+atom_index,j[1]+atom_index,j[2]+atom_index) for j in Data[k]["Angles"] ]
            Angle_types_sim    = Angle_types_sim + Data[k]["Angle_types"]
            Dihedrals_sim      = Dihedrals_sim + [ (j[0]+atom_index,j[1]+atom_index,j[2]+atom_index,j[3]+atom_index) for j in Data[k]["Dihedrals"] ]
            Dihedral_types_sim = Dihedral_types_sim + Data[k]["Dihedral_types"]
            Impropers_sim      = Impropers_sim + [ (j[0]+atom_index,j[1]+atom_index,j[2]+atom_index,j[3]+atom_index) for j in Data[k]["Impropers"] ]
            Improper_types_sim = Improper_types_sim + Data[k]["Improper_types"]
            Charges_sim        = Charges_sim + Data[k]["Charges"]
            Masses_sim         = Masses_sim + [ Data[k]["Masses"][j] for j in Data[k]["Atom_types"] ] 

            # Increment atom_index based on the number of atoms in the current geometry
            atom_index += len(Data[k]["Geometry"])
            mol_index += 1
            
    # If a mass density is set, then the coordinates of the molecules and simulation box are rescaled to match the requested density.
    b_min = None
    if Density != 0.0:
        print("\nRescaling box size and coordinates to match requested mass density ({} g/cc):\n".format(Density))
        A = 6.0221413e23
        mass_in_box = np.sum(Masses_sim) / A                                 # sum atomic masses then convert to g
        box_vol = ((Sim_Box[1]-Sim_Box[0])*10.0**(-8))**3                 # Standard density units are per cm^3 in lammps
        current_density = mass_in_box/box_vol                             # save current density to variable
        rescale_factor = (current_density/Density)**(1./3.)               # calculate rescale_factor based on the density ratio. cubed root is owing to the 3-dimensionality of the sim box.
        b_min = ( Sim_Box[1] - Sim_Box[0] ) * rescale_factor / 2.0
        print("\tmass_in_box:      {:< 12.6f} g".format(mass_in_box))
        print("\tbox_vol:          {:< 12.6f} cm^3".format(box_vol))
        print("\tcurrent_density:  {:< 12.6f} g/cm^3".format(current_density))
        print("\trescale_factor:   {:< 12.6f} ".format(rescale_factor))
        print("\tfinal_box_length: {:< 12.6f} angstroms".format(b_min*2.0))

        print("\nFixing the mass density by rescaling the box during equilibration by {}\n".format(rescale_factor))

    # If a number density is set, then the coordinates of the molecules and simulation box are rescaled to match the requested density.
    if N_Density != 0.0:
        print("\nRescaling box size and coordinates to match requested number density ({} atoms/A^3):\n".format(N_Density))
        A = 6.0221413e23
        atoms_in_box = len(Elements_sim)                                  # sum atomic masses then convert to g
        box_vol = (Sim_Box[1]-Sim_Box[0])**3                              # number density is specified in 1/A^3
        current_N_density = atoms_in_box/box_vol                          # save current density to variable
        rescale_factor = (current_N_density/N_Density)**(1./3.)           # calculate rescale_factor based on the density ratio. cubed root is owing to the 3-dimensionality of the sim box.
        b_min = ( Sim_Box[1] - Sim_Box[0] ) * rescale_factor / 2.0
        print("\tatoms_in_box:      {:< 16d} atoms".format(atoms_in_box))
        print("\tbox_vol:           {:< 16.6f} A^3".format(box_vol))
        print("\tcurrent_N_density: {:< 16.6f} atoms/A^3".format(current_N_density))
        print("\trescale_factor:    {:< 16.6f} ".format(rescale_factor))
        print("\tfinal_box_length:  {:< 12.6f} angstroms".format(b_min*2.0))

        print("\nFixing the number density by rescaling the box during equilibration by {}\n".format(rescale_factor))        

    print("The simulation cell has {} molecules and dimensions of {:<3.2f} x {:<3.2f} x {:<3.2f}".format(N_tot,Sim_Box[1]-Sim_Box[0],Sim_Box[3]-Sim_Box[2],Sim_Box[5]-Sim_Box[4]))
    if Density != 0.0 or N_Density != 0.0:
        print("To reach target density simulation box will be rescaled to dimensions of {:<3.2f} x {:<3.2f} x {:<3.2f} during equilibration".format(2.0*b_min,2.0*b_min,2.0*b_min))
    return Elements_sim,Atom_types_sim,Geometry_sim,Bonds_sim,Bond_types_sim,Angles_sim,Angle_types_sim,Dihedrals_sim,Dihedral_types_sim,Impropers_sim,Improper_types_sim,Charges_sim,Molecule_sim,Molecule_files,Adj_mat_sim,Sim_Box,b_min

# Wrapper function for writing the lammps data file which is read by the .in.init file during run initialization
def Write_data(Filename,Atom_types,Sim_Box,Elements,Geometry,Bonds,Bond_types,Bond_params,Angles,Angle_types,Angle_params,Dihedrals,Dihedral_types,Dihedral_params,\
               Impropers,Improper_types,Improper_params,Charges,VDW_params,Masses,Molecule,Improper_flag=False):

    # Write an xyz for easy viewing
    with open(Filename+'/'+Filename.split('/')[-1]+'.xyz','w') as f:
        f.write('{}\n\n'.format(len(Geometry)))
        for count_i,i in enumerate(Geometry):
            f.write('{:20s} {:< 20.6f} {:< 20.6f} {:< 20.6f}\n'.format(Elements[count_i],i[0],i[1],i[2]))

    # Create type dictionaries (needed to convert each atom,bond,angle, and dihedral type to consecutive numbers as per LAMMPS convention)
    # Note: LAMMPS orders atomtypes, bonds, angles, dihedrals, etc as integer types. Each of the following dictionaries holds the mapping between
    #       the true type (held in the various types lists) and the lammps type_id, obtained by enumerated iteration over the respective set(types).
    Atom_type_dict = {}
    for count_i,i in enumerate(sorted(set(Atom_types))):
        for j in Atom_types:
            if i == j:
                Atom_type_dict[i]=count_i+1
            if i in list(Atom_type_dict.keys()):
                break
    Bond_type_dict = {}
    for count_i,i in enumerate(sorted(set(Bond_types))):
        for j in Bond_types:
            if i == j:
                Bond_type_dict[i]=count_i+1
            if i in list(Bond_type_dict.keys()):
                break
    Angle_type_dict = {}
    for count_i,i in enumerate(sorted(set(Angle_types))):
        for j in Angle_types:
            if i == j:
                Angle_type_dict[i]=count_i+1
            if i in list(Angle_type_dict.keys()):
                break
    Dihedral_type_dict = {}
    for count_i,i in enumerate(sorted(set(Dihedral_types))):
        for j in Dihedral_types:
            if i == j:
                Dihedral_type_dict[i]=count_i+1
            if i in list(Dihedral_type_dict.keys()):
                break
    Improper_type_dict = {}
    for count_i,i in enumerate(sorted(set(Improper_types))):
        for j in Improper_types:
            if i == j:
                Improper_type_dict[i]=count_i+1
            if i in list(Improper_type_dict.keys()):
                break

    # Write the data file
    with open(Filename+'/'+Filename.split('/')[-1]+'.data','w') as f:
        
        # Write system properties
        f.write("LAMMPS data file via vdw_self_gen.py, on {}\n\n".format(datetime.datetime.now()))

        f.write("{} atoms\n".format(len(Elements)))
        f.write("{} atom types\n".format(len(set(Atom_types))))
        if len(Bonds) > 0:
            f.write("{} bonds\n".format(len(Bonds)))
            f.write("{} bond types\n".format(len(set(Bond_types))))
        if len(Angles) > 0:
            f.write("{} angles\n".format(len(Angles)))
            f.write("{} angle types\n".format(len(set(Angle_types))))
        if len(Dihedrals) > 0:
            f.write("{} dihedrals\n".format(len(Dihedrals)))
            f.write("{} dihedral types\n".format(len(set(Dihedral_types))))
        if Improper_flag and len(Impropers) > 0:
            f.write("{} impropers\n".format(len(Impropers)))
            f.write("{} improper types\n".format(len(set(Improper_types))))
        f.write("\n")

        # Write box dimensions
        f.write("{:< 20.16f} {:< 20.16f} xlo xhi\n".format(Sim_Box[0],Sim_Box[1]))
        f.write("{:< 20.16f} {:< 20.16f} ylo yhi\n".format(Sim_Box[2],Sim_Box[3]))
        f.write("{:< 20.16f} {:< 20.16f} zlo zhi\n\n".format(Sim_Box[4],Sim_Box[5]))

        # Write Masses
        f.write("Masses\n\n")
        for count_i,i in enumerate(sorted(set(Atom_types))):
            for j in set(Atom_types):
                if Atom_type_dict[j] == count_i+1:
                    f.write("{} {:< 8.6f}\n".format(count_i+1,Masses[str(j)])) # count_i+1 bc of LAMMPS 1-indexing
        f.write("\n")

        # Write Atoms
        f.write("Atoms\n\n")
        for count_i,i in enumerate(Atom_types):
            f.write("{:<8d} {:< 4d} {:< 4d} {:< 20.16f} {:< 20.16f} {:< 20.16f} {:< 20.16f} {:d} {:d} {:d}\n"\
            .format(count_i+1,Molecule[count_i],Atom_type_dict[i],Charges[count_i],Geometry[count_i,0],Geometry[count_i,1],Geometry[count_i,2],0,0,0))

        # Write Bonds
        if len(Bonds) > 0:
            f.write("\nBonds\n\n")
            for count_i,i in enumerate(Bonds):
                f.write("{:<8d} {:<8d} {:<8d} {:<8d}\n".format(count_i+1,Bond_type_dict[Bond_types[count_i]],i[0]+1,i[1]+1))

        # Write Angles
        if len(Angles) > 0:
            f.write("\nAngles\n\n")
            for count_i,i in enumerate(Angles):
                f.write("{:<8d} {:<8d} {:<8d} {:<8d} {:<8d}\n".format(count_i+1,Angle_type_dict[Angle_types[count_i]],i[0]+1,i[1]+1,i[2]+1))

        # Write Dihedrals
        if len(Dihedrals) > 0: 
            f.write("\nDihedrals\n\n")
            for count_i,i in enumerate(Dihedrals):
                f.write("{:<8d} {:<8d} {:<8d} {:<8d} {:<8d} {:<8d}\n".format(count_i+1,Dihedral_type_dict[Dihedral_types[count_i]],i[0]+1,i[1]+1,i[2]+1,i[3]+1))

        # Write Impropers
        if Improper_flag and len(Impropers) > 0: 
            f.write("\nImpropers\n\n")
            for count_i,i in enumerate(Impropers):
                f.write("{:<8d} {:<8d} {:<8d} {:<8d} {:<8d} {:<8d}\n".format(count_i+1,Improper_type_dict[Improper_types[count_i]],i[0]+1,i[1]+1,i[2]+1,i[3]+1))

    # Write the settings file
    fixed_modes = {'bonds':[], 'angles':[]}
    with open(Filename+'/'+Filename.split('/')[-1]+'.in.settings','w') as f:

        # Write non-bonded interactions (the complicated form of this loop is owed
        # to desire to form a nicely sorted list in terms of the lammps atom types
        # Note: Atom_type_dict was initialize according to sorted(set(Atom_types) 
        #       so iterating over this list (twice) orders the pairs, too.
        f.write("     {}\n".format("# Non-bonded interactions (pair-wise)"))
        for count_i,i in enumerate(sorted(set(Atom_types))):     
            for count_j,j in enumerate(sorted(set(Atom_types))): 

                # Skip duplicates
                if count_j < count_i:
                    continue

                # Conform to LAMMPS i <= j formatting
                if Atom_type_dict[i] <= Atom_type_dict[j]:
                    f.write("     {:20s} {:<10d} {:<10d} ".format("pair_coeff",Atom_type_dict[i],Atom_type_dict[j]))
                else:
                    f.write("     {:20s} {:<10d} {:<10d} ".format("pair_coeff",Atom_type_dict[j],Atom_type_dict[i]))

                # Determine key (ordered by initialize_VDW function such that i > j)
                if i > j:
                    key = (i,j)
                else:
                    key = (j,i)

                # Write the parameters
                for k in VDW_params[key]:
                    if type(k) is str:
                        f.write("{:20s} ".format(k))
                    if type(k) is float:
                        f.write("{:< 20.6f} ".format(k))
                f.write("\n")                        

        # Write stretching interactions
        # Note: Bond_type_dict was initialized by looping over sorted(set(Bond_types)), so 
        #       iterating over this list resulted in ordered parameters in the in.settings file
        f.write("\n     {}\n".format("# Stretching interactions"))
        for i in sorted(set(Bond_types)):
            f.write("     {:20s} {:<10d} ".format("bond_coeff",Bond_type_dict[i]))
            for j in Bond_params[i]:
                if j == "fixed":
                    continue
                if type(j) is str:
                    f.write("{:20s} ".format(j))
                if type(j) is float:
                    f.write("{:< 20.6f} ".format(j))
            f.write("\n")

            # populate fixed_modes
            if Bond_params[i][0] == "fixed":
                fixed_modes["bonds"] += [Bond_type_dict[i]]

        # Write bending interactions
        # Note: Angle_type_dict was initialized by looping over sorted(set(Angle_types)), so 
        #       iterating over this list resulted in ordered parameters in the in.settings file
        f.write("\n     {}\n".format("# Bending interactions"))
        for i in sorted(set(Angle_types)):
            f.write("     {:20s} {:<10d} ".format("angle_coeff",Angle_type_dict[i]))
            for j in Angle_params[i]:
                if j == "fixed":
                    continue
                if type(j) is str:
                    f.write("{:20s} ".format(j))
                if type(j) is float:
                    f.write("{:< 20.6f} ".format(j))
            f.write("\n")

            # populate fixed_modes
            if Angle_params[i][0] == "fixed":
                fixed_modes["angles"] += [Angle_type_dict[i]]

        # Write dihedral interactions
        # Note: Dihedral_type_dict was initialized by looping over sorted(set(Dihedral_types)), so 
        #       iterating over this list resulted in ordered parameters in the in.settings file
        f.write("\n     {}\n".format("# Dihedral interactions"))
        for i in sorted(set(Dihedral_types)):
            f.write("     {:20s} {:<10d} ".format("dihedral_coeff",Dihedral_type_dict[i]))
            for j in Dihedral_params[i]:
                if type(j) is str:
                    f.write("{:20s} ".format(j))
                if type(j) is float:
                    f.write("{:< 20.6f} ".format(j))
                if type(j) is int:
                    f.write("{:< 20d} ".format(j))
            f.write("\n")

        # Write improper interactions
        # Note: Improper_type_dict was initialized by looping over sorted(set(Improper_types)), so 
        #       iterating over this list resulted in ordered parameters in the in.settings file
        if Improper_flag:
            f.write("\n     {}\n".format("# Improper interactions"))
            for i in sorted(set(Improper_types)):
                f.write("     {:20s} {:<10d} ".format("improper_coeff",Improper_type_dict[i]))
                for j in Improper_params[i][1:]:
                    if type(j) is str:
                        f.write("{:20s} ".format(j))
                    if type(j) is float:
                        f.write("{:< 20.6f} ".format(j))
                    if type(j) is int:
                        f.write("{:< 20d} ".format(j))
                f.write("\n")

    return Atom_type_dict,fixed_modes


# A wrapper for the commands to parse the bonds, angles, and dihedrals from the adjacency matrix.
# Returns:   list of atomtypes, bond_types, bond instances, angle_types, angle instances, dihedral_types,
#            diehdral instances, charges, and VDW parameters.
def Find_parameters(Adj_mat,Bond_mats,Geometry,Atom_types,FF_db="FF_file",Improper_flag = False, force_read=False,remove_multi=False):

    # List comprehension to determine bonds from a loop over the adjacency matrix. Iterates over rows (i) and individual elements
    # ( elements A[count_i,count_j] = j ) and stores the bond if the element is "1". The count_i < count_j condition avoids
    # redudant bonds (e.g., (i,j) vs (j,i) ). By convention only the i < j definition is stored.
    print("Parsing bonds...")
    Bonds          = [ (count_i,count_j) for count_i,i in enumerate(Adj_mat) for count_j,j in enumerate(i) if j == 1 ]
    Bond_types     = [ (Atom_types[i[0]],Atom_types[i[1]]) for i in Bonds ]

    # List comprehension to determine angles from a loop over the bonds. Note, since there are two bonds in every angle, there will be
    # redundant angles stored (e.g., (i,j,k) vs (k,j,i) ). By convention only the i < k definition is stored.
    print("Parsing angles...")
    Angles          = [ (count_j,i[0],i[1]) for i in Bonds for count_j,j in enumerate(Adj_mat[i[0]]) if j == 1 and count_j not in i ]
    Angle_types     = [ (Atom_types[i[0]],Atom_types[i[1]],Atom_types[i[2]]) for i in Angles ]

    # List comprehension to determine dihedrals from a loop over the angles. Note, since there are two angles in every dihedral, there will be
    # redundant dihedrals stored (e.g., (i,j,k,m) vs (m,k,j,i) ). By convention only the i < m definition is stored.
    print("Parsing dihedrals...")
    Dihedrals      = [ (count_j,i[0],i[1],i[2]) for i in Angles for count_j,j in enumerate(Adj_mat[i[0]]) if j == 1 and count_j not in i ]
    Dihedral_types = [ (Atom_types[i[0]],Atom_types[i[1]],Atom_types[i[2]],Atom_types[i[3]]) for i in Dihedrals ]

    # List comprehension to determine dihedrals from a loop over the angles. Note, since there are two angles in every dihedral, there will be
    # redundant dihedrals stored (e.g., (i,j,k,m) vs (m,k,j,i) ). By convention only the i < m definition is stored.
    if Improper_flag:    print("Parsing impropers...")
    Impropers      = [ (i[1],i[0],i[2],count_j) for i in Angles for count_j,j in enumerate(Adj_mat[i[1]]) if j == 1 and count_j not in i ]
    Improper_types = [ (Atom_types[i[0]],Atom_types[i[1]],Atom_types[i[2]],Atom_types[i[3]]) for i in Impropers ]

    # Canonicalize the modes
    for i in range(len(Bonds)):
        Bond_types[i],Bonds[i] = canon_bond(Bond_types[i],ind=Bonds[i])
    for i in range(len(Angles)):
        Angle_types[i],Angles[i] = canon_angle(Angle_types[i],ind=Angles[i])
    for i in range(len(Dihedrals)):
        Dihedral_types[i],Dihedrals[i] = canon_dihedral(Dihedral_types[i],ind=Dihedrals[i])
    for i in range(len(Impropers)):
        Improper_types[i],Impropers[i] = canon_improper(Improper_types[i],ind=Impropers[i])        

    # Remove redundancies    
    if len(Bonds) > 0: Bonds,Bond_types = list(map(list, list(zip(*[ (i,Bond_types[count_i]) for count_i,i in enumerate(Bonds) if count_i == [ count_j for count_j,j in enumerate(Bonds) if j == i or j[::-1] == i ][0]  ]))))
    if len(Angles) > 0: Angles,Angle_types = list(map(list, list(zip(*[ (i,Angle_types[count_i]) for count_i,i in enumerate(Angles) if count_i == [ count_j for count_j,j in enumerate(Angles) if j == i or j[::-1] == i ][0]  ]))))
    if len(Dihedrals) > 0: Dihedrals,Dihedral_types = list(map(list, list(zip(*[ (i,Dihedral_types[count_i]) for count_i,i in enumerate(Dihedrals) if count_i == [ count_j for count_j,j in enumerate(Dihedrals) if j == i or j[::-1] == i ][0]  ]))))
    if len(Impropers) > 0: Impropers,Improper_types = list(map(list, list(zip(*[ (i,Improper_types[count_i]) for count_i,i in enumerate(Impropers) if count_i == [ count_j for count_j,j in enumerate(Impropers) if j[0] == i[0] and len(set(i[1:]).intersection(set(j[1:]))) ][0] ]))))

    # Remove bimodal angles (e.g., if the same angle is present in
    # more than one configuration, then only the most abundant is kept)
    if remove_multi is True:
        for i in set(Angle_types):

            # Calculate most populated bin
            bins = [ 20.0*_ for _ in np.arange(9) ]
            hist = np.zeros(len(bins))
            angles = []
            for count_j,j in enumerate(Angle_types):
                if j == i:
                    v1 = Geometry[Angles[count_j][0]] - Geometry[Angles[count_j][1]]
                    v2 = Geometry[Angles[count_j][2]] - Geometry[Angles[count_j][1]]
                    v1 = v1/np.linalg.norm(v1)
                    v2 = v2/np.linalg.norm(v2)
                    angles += [ math.acos(np.dot(v1,v2)) * 180/np.pi ]
                    hist[int(angles[-1]/20.0 - 1E-6)] += 1 # put linear angle to the last bin SEO 05/03/21        
                else:
                    angles += [0.0]
            angle = bins[np.where(hist == np.max(hist))[0][0]]

            # Determine the angles to be removed
            del_inds = []
            for count_j,j in enumerate(Angle_types):                
                if j == i and np.abs(angles[count_j] - angle) > 20.0:
                    del_inds += [count_j]
            Angle_types = [ j for count_j,j in enumerate(Angle_types) if count_j not in del_inds ]
            Angles = [ j for count_j,j in enumerate(Angles) if count_j not in del_inds ]

    # Add the opls/harmonic type as a fifth element in the Dihedral_types tuples
    for count_i,i in enumerate(Dihedrals):
        if 2 in [ j[i[1],i[2]] for j in Bond_mats ]:
            Dihedral_types[count_i] = tuple(list(Dihedral_types[count_i]) + ["harmonic"])        
        # elif ( "R" in Dihedral_types[count_i][1] or "E" in Dihedral_types[count_i][1] or "Z" in Dihedral_types[count_i][1] ) and \
        #      ( "R" in Dihedral_types[count_i][2] or "E" in Dihedral_types[count_i][2] or "Z" in Dihedral_types[count_i][2] ):
        #     Dihedral_types[count_i] = tuple(list(Dihedral_types[count_i]) + ["harmonic"])        
        else:
            Dihedral_types[count_i] = tuple(list(Dihedral_types[count_i]) + ["opls"])                        

    ##############################################################
    # Read in parameters: Here the stretching, bending, dihedral #
    # and non-bonding interaction parameters are read in from    #
    # parameters file. Mass and charge data is also included.    #
    # The program looks for a simple match for the first entry   #
    # in each line with one of the bond or angle types.          #
    # INPUT: param_file, BOND_TYPES_LIST, ANGLE_TYPES_LIST       #
    #        DIHEDRAL_TYPES_LIST, ELEMENTS                       #
    # OUTPUT: CHARGES, MASSES, BOND_PARAMS, ANGLE_PARAMS,        #
    #         DIHERAL_PARAMS, PW_PARAMS                          #
    ##############################################################

    # Initialize dictionaries

    # Read in masses and charges
    Masses = {}
    with open(FF_db,'r') as f:
        content=f.readlines()
        
    for lines in content:
        fields=lines.split()

        # Skip empty lines
        if len(fields) == 0:
            continue

        if fields[0].lower() == 'atom':
            Masses[fields[1]] = float(fields[3]) 

    # Read in bond parameters
    Bond_params = {}
    with open(FF_db,'r') as f:
        content=f.readlines()

    for lines in content:
        fields=lines.split()

        # Skip empty lines
        if len(fields) == 0:
            continue

        if fields[0] == 'bond':
            if fields[3] == "harmonic":
                Bond_params[(fields[1],fields[2])] = [float(fields[4]),float(fields[5])]
            elif fields[3] == "fixed":
                Bond_params[(fields[1],fields[2])] = ["fixed",0.0,float(fields[4])]
            else:
                print("ERROR: only harmonic bond definitions are currently supported by gen_md_for_sampling.py. Exiting...")
                quit()

    # Read in angle parameters
    Angle_params = {}
    with open(FF_db,'r') as f:
        content=f.readlines()

    for lines in content:
        fields=lines.split()

        # Skip empty lines
        if len(fields) == 0:
            continue

        if fields[0].lower() == 'angle':
            if fields[4] == "harmonic":
                Angle_params[(fields[1],fields[2],fields[3])] = [float(fields[5]),float(fields[6])]
            elif fields[4] == "fixed":
                Angle_params[(fields[1],fields[2],fields[3])] = ["fixed",0.0,float(fields[5])]
            else:
                print("ERROR: only harmonic angle definitions are currently supported by gen_md_for_sampling.py. Exiting...")
                quit()

    # Read in dihedral parameters
    Dihedral_params = {}
    with open(FF_db,'r') as f:
        content=f.readlines()

    for lines in content:
        fields=lines.split()

        # Skip empty lines
        if len(fields) == 0:
            continue

        if fields[0].lower() in ['dihedral','torsion']:            
            if fields[5] == "opls":
                Dihedral_params[(fields[1],fields[2],fields[3],fields[4],fields[5])] = [fields[5]] + [ float(i) for i in fields[6:10] ]
            elif fields[5] == "harmonic":
                Dihedral_params[(fields[1],fields[2],fields[3],fields[4],fields[5])] = [fields[5]] + [ float(fields[6]),int(float(fields[7])),int(float(fields[8])) ]
            else:
                print("ERROR: Only opls and harmonic dihedral types are currently supported by gen_md_for_sampling.py. Exiting...")
                quit()
        
    # Read in improper parameters
    Improper_params = {}
    with open(FF_db,'r') as f:
        content=f.readlines()

    for lines in content:
        fields=lines.split()

        # Skip empty lines
        if len(fields) == 0:
            continue

        if fields[0].lower() in ['improper']:
            if fields[5] == "harmonic":
                Improper_params[(fields[1],fields[2],fields[3],fields[4])] = [fields[5]] + [ float(fields[6]),float(fields[7])]
            else:
                print("ERROR: Only opls type dihedral definitions are currently supported by gen_md_for_vdw.py. Exiting...")
                quit()
                
    # Search for charges based on atom type
    with open(FF_db,'r') as f:
        content=f.readlines()

    Charges = np.zeros(len(Atom_types))
    for i in range(len(Atom_types)):
        for lines in content:
            fields=lines.split()

            # Skip empty lines
            if len(fields) == 0:
                continue
                    
            if fields[0].lower() in ['charge'] and Atom_types[i] == fields[1]:
                Charges[i] = float(fields[2])

    # Search for VDW parameters
    VDW_params = {}
    with open(FF_db,'r') as f:
        for lines in f:
            fields = lines.split()

            # Skip empty lines
            if len(fields) == 0:
                continue
                    
            if fields[0].lower() in ['vdw']:

                # Only two parameters are required for lj types
                if fields[3] == "lj":
                    if fields[1] > fields[2]:
                        VDW_params[(fields[1],fields[2])] = [fields[3],float(fields[4]),float(fields[5])]
                    else:
                        VDW_params[(fields[2],fields[1])] = [fields[3],float(fields[4]),float(fields[5])]
                elif fields[3] == "buck":
                    if fields[1] > fields[2]:
                        VDW_params[(fields[1],fields[2])] = [fields[3],float(fields[4]),float(fields[5]),float(fields[6])]
                    else:
                        VDW_params[(fields[2],fields[1])] = [fields[3],float(fields[4]),float(fields[5]),float(fields[6])]

    # Check for missing parameters
    Missing_masses = [ i for i in Atom_types if str(i) not in list(Masses.keys()) ] 
    Missing_charges = [ count_i for count_i,i in enumerate(Charges) if i == -100.0 ]; Missing_charges = [ Atom_types[i] for i in Missing_charges ]
    Missing_bonds = [ i for i in Bond_types if (i[0],i[1]) not in list(Bond_params.keys()) ]
    Missing_angles = [ i for i in Angle_types if (i[0],i[1],i[2]) not in list(Angle_params.keys()) ]
    Missing_dihedrals = [ i for i in Dihedral_types if (i[0],i[1],i[2],i[3],i[4]) not in list(Dihedral_params.keys()) ]
    Missing_impropers = []
    if Improper_flag is True: Missing_impropers = [ i for i in Improper_types if (i[0],i[1],i[2],i[3]) not in list(Improper_params.keys()) ]

    # When the force_read option is set to True, the script will attempt to replace the missing dihedral types with whatever (matching)
    # parameters are in the force-field file. For example, if an opls type is expected but a harmonic type is supplied, the harmonic type
    # will be used IF THE FLAG IS TRUE, otherwise the script will print an error and exit.
    if force_read:

        # Assemble lists of types and keys present in the force field dictionary
        bt,bk = ( [ tuple(i[:2]) for i in list(Bond_params.keys()) ],     [ i for i in list(Bond_params.keys()) ] )
        at,ak = ( [ tuple(i[:3]) for i in list(Angle_params.keys()) ],    [ i for i in list(Angle_params.keys()) ] )
        dt,dk = ( [ tuple(i[:4]) for i in list(Dihedral_params.keys()) ], [ i for i in list(Dihedral_params.keys()) ] )

        # Replace missing dihedral types with whatever matching type is present in the supplied force-field
        for i in set(Missing_dihedrals):
            if i[:4] in dt:
                new_type = dk[dt.index(i[:4])]
                print("using dihedral type {} in place of {}...".format(new_type,i))
                r_inds = [ count_j for count_j,j in enumerate(Dihedral_types) if j == i ]
                for j in r_inds:
                    Dihedral_types[j] = new_type
                Missing_dihedrals = [ j for j in Missing_dihedrals if j != i ]


    # Print diagnostics on missing parameters and quit if the prerequisites are missing.
    if ( len(Missing_masses) + len(Missing_charges) + len(Missing_bonds) + len(Missing_angles) + len(Missing_dihedrals) + len(Missing_impropers) ) > 0:
        print("\nUh Oh! There are missing FF parameters...\n")

        if Missing_masses:
            print("Missing masses for the following atom types: {}".format([ i for i in set(Missing_masses) ]))
        if Missing_charges:
            print("Missing charges for the following atom types: {}".format([ i for i in set(Missing_charges) ]))
        if Missing_bonds:
            print("Missing bond parameters for the following bond types: {}".format([ i for i in set(Missing_bonds) ]))
        if Missing_angles:
            print("Missing angle parameters for the following angle types: {}".format([ i for i in set(Missing_angles) ]))
        if Missing_dihedrals:
            print("Missing dihedral parameters for the following dihedral types: {}".format([ i for i in set(Missing_dihedrals) ]))
        if Improper_flag and Missing_impropers:
            print("Missing improper parameters for the following improper types: {}".format([ i for i in set(Missing_impropers) ]))
        
        print("\nEnsure the specification of the missing parameters. Exiting...")
        quit()

    return list(Bonds),list(Bond_types),Bond_params,list(Angles),list(Angle_types),Angle_params,list(Dihedrals),list(Dihedral_types),Dihedral_params,list(Impropers),list(Improper_types),Improper_params,Charges.tolist(),Masses,VDW_params

# Description: finds the number of disconnected subnetworks in the 
#              adjacency matrix, which corresponds to the number of 
#              separate molecules.
#
# Inputs:      adj_mat: numpy array holding a 1 in the indices of bonded
#                        atom types. 
#
# Returns:     mol_count: scalar, the number of molecules in the adj_mat
def mol_count(adj_mat):
    
    # Initialize list of atoms assigned to molecules and a counter for molecules
    placed_idx = []    
    mol_count = 0

    # Continue the search until all the atoms have been assigned to molecules
    while len(placed_idx)<len(adj_mat):

        # Use sequential elements of the adj_mat as seeds for the spanning network search
        for count_i,i in enumerate(adj_mat):

            # Only proceed with search if the current atom hasn't been placed in a molecule
            if count_i not in placed_idx:

                # Increment mol_count for every new seed and add the seed to the list of placed atoms
                mol_count += 1               
                placed_idx += [count_i]
                
                # Find connections
                idx = [ count_j for count_j,j in enumerate(i) if j==1 and count_j not in placed_idx ]
                
                # Continue until no new atoms are found
                while len(idx) > 0:
                    current = idx.pop(0)
                    if current not in placed_idx:
                        placed_idx += [current]
                        idx += [ count_k for count_k,k in enumerate(adj_mat[current]) if k == 1 and count_k not in placed_idx ]
    return mol_count

# Wrapper function for the write commands for creating the *.map file
def write_map(Filename,Elements,Atom_types,Charges,Masses,Adj_mat,Structure,N_mol):

    # Open file for writing and write header (first two lines of the map file are header)    
    with open(Filename+'/'+Filename.split('/')[-1]+'.map','w') as f:
        f.write('{} {}\n {:<50} {:<10} {:<10} {:<14}  {:<13} {}\n'.format(len(Atom_types),np.sum(N_mol),'Atom_type','Element','Structure','Mass','Charge','Adj_mat'))
        for count_i,i in enumerate(Atom_types):
            adj_mat_entry = (' ').join([ str(_) for _ in np.where(Adj_mat[count_i,:] == 1)[0] ])
            f.write(' {:<50} {:<10} {:< 9d} {:<14.6f} {:< 14.8f} {}\n'.format(i,Elements[count_i],int(Structure[count_i]),Masses[str(i)],Charges[count_i],adj_mat_entry))
        f.close()

# Wrapper function for the write commands for creating the *.map file
def write_molecule(Filename,Molecules):

    # Open file for writing and write header (first two lines of the map file are header)    
    with open(Filename+'/'+Filename.split('/')[-1]+'.mol.txt','w') as f:
        f.write('{:<30s} {:<30s}\n'.format("Molecule_id","Molecule_Name"))
        for count_i,i in enumerate(Molecules):
            f.write("{:<30} {:<30}\n".format(count_i,i))

        # Write mol clusters
        f.write('\nLists for each molecule:\n')
        for i in sorted(set(Molecules)):
            f.write('{}: '.format(i))
            for count_j,j in enumerate(Molecules):
                if i == j:
                    f.write(' {}'.format(count_j))
            f.write('\n')


# Description: Initialize VDW_dict based on UFF parameters for the initial guess of the fit.
def initialize_VDW(atomtypes,sigma_scale=1.0,eps_scale=1.0,VDW_FF={},Force_UFF=0,mixing_rule='lb'):

    # Initialize UFF parameters (tuple corresponds to eps,sigma pairs for each element)
    # Taken from UFF (Rappe et al. JACS 1992)
    # Note: LJ parameters in the table are specificed in the eps,r_min form rather than eps,sigma
    #       the conversion between r_min and sigma is sigma = r_min/2^(1/6)
    # Note: Units for sigma = angstroms and eps = kcal/mol 
    UFF_dict = { 1:(0.044,2.5711337005530193),  2:(0.056,2.1043027722474816),  3:(0.025,2.183592758161972),  4:(0.085,2.4455169812952313),\
                 5:(0.180,3.6375394661670053),  6:(0.105,3.4308509635584463),  7:(0.069,3.260689308393642),  8:(0.060,3.1181455134911875),\
                 9:(0.050,2.996983287824101),  10:(0.042,2.88918454292912),   11:(0.030,2.657550876212632), 12:(0.111,2.6914050275019648),\
                13:(0.505,4.008153332913386),  14:(0.402,3.82640999441276),   15:(0.305,3.694556984127987), 16:(0.274,3.594776327696269),\
                17:(0.227,3.5163772404999194), 18:(0.185,3.4459962417668324), 19:(0.035,3.396105913550973), 20:(0.238,3.0281647429590133),\
                21:(0.019,2.935511276272418),  22:(0.017,2.828603430095577),  23:(0.016,2.800985569833227), 24:(0.015,2.6931868249382456),\
                25:(0.013,2.6379511044135446), 26:(0.013,2.5942970672246677), 27:(0.014,2.558661118499054), 28:(0.015,2.5248069672097215),\
                29:(0.005,3.113691019900486),  30:(0.124,2.4615531582217574), 31:(0.415,3.904809081609107), 32:(0.379,3.813046513640652),\
                33:(0.309,3.7685015777336357), 34:(0.291,3.746229109780127),  35:(0.251,3.731974730289881), 36:(0.220,3.689211591819145),\
                37:(0.040,3.6651573264293558), 38:(0.235,3.2437622327489755), 39:(0.072,2.980056212179435), 40:(0.069,2.78316759547042),\
                41:(0.059,2.819694442914174),  42:(0.056,2.7190228877643157), 43:(0.048,2.670914356984738), 44:(0.056,2.6397329018498255),\
                45:(0.053,2.6094423454330538), 46:(0.048,2.5827153838888437), 47:(0.036,2.804549164705788), 48:(0.228,2.537279549263686),\
                49:(0.599,3.976080979060334),  50:(0.567,3.9128271700723705), 51:(0.449,3.937772334180300), 52:(0.398,3.982317270087316),\
                53:(0.339,4.009044231631527),  54:(0.332,3.923517954690054),  55:(0.045,4.024189509839913), 56:(0.364,3.2989979532736764),\
                72:(0.072,2.798312873678806),  73:(0.081,2.8241489365048755), 74:(0.067,2.734168165972701), 75:(0.066,2.631714813386562),\
                76:(0.037,2.7796040005978586), 77:(0.073,2.5301523595185635), 78:(0.080,2.453535069758495), 79:(0.039,2.9337294788361374),\
                80:(0.385,2.4098810325696176), 81:(0.680,3.872736727756055),  82:(0.663,3.828191791849038), 83:(0.518,3.893227398273283),\
                84:(0.325,4.195242063722858),  85:(0.284,4.231768911166611),  86:(0.248,4.245132391938716) }

    # NEW: Initialize VDW_dict first guess based on element types and Lorentz-Berthelot mixing rules
    # Order of operations: (1) If the parameters are in the supplied FF database then they are used as is
    # (2) If the self-terms are in the supplied FF datebase then they are used to generate the mixed
    # interactions (3) UFF parameters are used. 
    VDW_dict = {}
    origin = {}
    VDW_styles = []
    for count_i,i in enumerate(atomtypes):
        for count_j,j in enumerate(atomtypes):
            if count_i < count_j:
                continue

            # Check for parameters in the database
            if (i,j) in VDW_FF and Force_UFF != 1:

                # Determine appropriate lammps style
                if VDW_FF[(i,j)][0] == "lj":
                    VDW_type = "lj/cut/coul/long"
                elif VDW_FF[(i,j)][0] == "buck":
                    VDW_type = "buck/coul/long"
                else:
                    print("ERROR in initialize_VDW: only lj and buck pair types are supported. Exiting...")
                    quit()

                # Assign style
                if i > j: 
                    VDW_dict[(i,j)] = [VDW_type] + VDW_FF[(i,j)][1:]
                else:
                    VDW_dict[(j,i)] = [VDW_type] + VDW_FF[(i,j)][1:]
                origin[(i,j)] = origin[(j,i)] = "read"

            # Check for reverse combination
            elif (j,i) in VDW_FF and Force_UFF != 1:

                # Determine appropriate lammps style
                if VDW_FF[(j,i)][0] == "lj":
                    VDW_type = "lj/cut/coul/long"
                elif VDW_FF[(j,i)][0] == "buck":
                    VDW_type = "buck/coul/long"
                else:
                    print("ERROR in initialize_VDW: only lj and buck pair types are supported. Exiting...")
                    quit()

                # Assign style
                if i > j: 
                    VDW_dict[(i,j)] = [VDW_type] + VDW_FF[(j,i)][1:]
                else:
                    VDW_dict[(j,i)] = [VDW_type] + VDW_FF[(j,i)][1:]
                origin[(i,j)] = origin[(j,i)] = "read"

            # Check if the database has the self-terms necessary for applying mixing rules
            elif (i,i) in VDW_FF and (j,j) in VDW_FF and Force_UFF != 1 and mixing_rule == 'lb':

                # Check compatibility with mixing rules
                if VDW_FF[(i,i)][0] != "lj" or VDW_FF[(j,j)][0] != "lj":
                    print("ERROR in initialize_VDW: only lj styles support mixing rules. Exiting...")
                    quit()

                # Apply mixing rules and assign
                VDW_type = "lj/cut/coul/long"
                eps    = (VDW_FF[(i,i)][1]*VDW_FF[(j,j)][1])**(0.5) * eps_scale
                sigma  = (VDW_FF[(i,i)][2]+VDW_FF[(j,j)][2])/2.0 * sigma_scale
                if i > j: 
                    VDW_dict[(i,j)] = [VDW_type,eps,sigma]
                else:
                    VDW_dict[(j,i)] = [VDW_type,eps,sigma]
                origin[(i,j)] = origin[(j,i)] = "lb"

            # Check if the database has the self-terms necessary for applying mixing rules
            elif (i,i) in VDW_FF and (j,j) in VDW_FF and Force_UFF != 1 and mixing_rule == 'wh':

                # Check compatibility with mixing rules
                if VDW_FF[(i,i)][0] != "lj" or VDW_FF[(j,j)][0] != "lj":
                    print("ERROR in initialize_VDW: only lj styles support mixing rules. Exiting...")
                    quit()

                # Apply mixing rules and assign
                VDW_type = "lj/cut/coul/long"
                sigma  = ((VDW_FF[(i,i)][2]**(6.0)+VDW_FF[(j,j)][2]**(6.0))/2.0)**(1.0/6.0)
                eps    = (VDW_FF[(i,i)][1]*VDW_FF[(i,i)][2]**(6.0) * VDW_FF[(j,j)][1]*VDW_FF[(j,j)][2]**(6.0) )**(0.5) / sigma**(6.0)
                if i > j: 
                    VDW_dict[(i,j)] = [VDW_type,eps,sigma]
                else:
                    VDW_dict[(j,i)] = [VDW_type,eps,sigma]
                origin[(i,j)] = origin[(j,i)] = "wh"

            # Last resort: Use UFF parameters. Does not happen for the benchmark set SEO 05/03/2021
            else:
                VDW_type = "lj/cut/coul/long"
                type_1 = int(i.split('[')[1].split(']')[0])
                type_2 = int(j.split('[')[1].split(']')[0])
                eps    = (UFF_dict[type_1][0]*UFF_dict[type_2][0])**(0.5) * eps_scale
                sigma  = (UFF_dict[type_1][1]+UFF_dict[type_2][1])/2.0 * sigma_scale
                if i > j:
                    VDW_dict[(i,j)] = [VDW_type,eps,sigma]
                else:
                    VDW_dict[(j,i)] = [VDW_type,eps,sigma]
                origin[(i,j)] = origin[(j,i)] = "UFF"
                
            # Collect a list of the LAMMPS styles used in the simulation
            VDW_styles += [VDW_type]

    # Print summary
    print("\n{}".format("*"*177))
    print("* {:^173s} *".format("Initializing VDW parameters for the simulation (those with * were read from the FF file(s))"))
    print("*{}*".format("-"*175))
    print("* {:<50s} {:<50s} {:<20s}  {:<18s} {:<18s} {:<8s}    *".format("Type","Type","VDW_type","eps (kcal/mol)","sigma (angstroms)","origin"))
    print("{}".format("*"*177))
    for j in list(VDW_dict.keys()):
        print("  {:<50s} {:<50s} {:<20s} {:< 18.4f} {:< 18.4f}  {:<18s}".format(j[0],j[1],VDW_dict[j][0],VDW_dict[j][1],VDW_dict[j][2],origin[j]))
    print("")


    return VDW_dict


# Description:   A wrapper for the commands to create a data dictionary holding the FF, mode, and geometry information for each molecule
# Inputs:  Filename:
# Returns: Data: a dictionary keyed to the list of coord_files
#                a keyed entry is itself a dictionary holding the geometry, 
#                elements, charges, bonds, bond_types, angles, angle_types,
#                dihedrals, dihedral_types and masses.
#                subdictionary keys (e.g., Data[*]["Geometry"])
#                Geometry: array of the molecule's geometry
#                Elements: list of the molecule's elements (indexed to the rows of "Geometry")
#                Atom_types: list of the molecule's atom_types (indexed to the rows of "Geometry")
#                Charges: list of the molecule's charges (indexed to the rows of "Geometry")
#                Bonds: a list of the molecule's bonds, each element being a tuple of the atom indices that are bonded
#                Angles: a list of the molecule's angles, each element being a tuple of the atom indices that form an angle mode
#                Dihedrals: a list of the molecule's dihedrals, each element being a tuple of the atom indices that form a dihedral
#                Bond_types: a list of the molecule's bond_types, indexed to Bonds
#                Angle_types: a list of the molecule's angle_types, indexed to Angles
#                Dihedral_types: a list of the molecule's dihedral_types, indexed to Dihedrals
#                Impropers: a list of the molecule's improper dihedrals by atom tindex
#                Improper_types: a list of the molecule's improper_types, indexed to Impropers
#
def get_data(FF_all,coord_files,N_mol,q_list,gens,Improper_flag=False,force_read_opt=False,remove_multi=False):

    # Initialize dictionary to hold the FF, mode, and geometry information about each molecule
    Data = {}

    # Iterate over all molecules being simulated and collect their FF, mode, and geometry information.
    for count_i,i in enumerate(coord_files):

        # Initialize dictionary for this geometry and set number of molecules to place in the simulated system
        Data[i] = {}      
        Data[i]["N_mol"] = N_mol[count_i]

        # Extract Element list and Coord list from the file
        Data[i]["Elements"],Data[i]["Geometry"] = xyz_parse(i)

        # Generate adjacency table
        Data[i]["Adj_mat"] = Table_generator(Data[i]["Elements"],Data[i]["Geometry"])

        # Find atom types
        print("Determining atom types for {} based on a {}-bond deep search...".format(i,gens))
        Data[i]["Atom_types"] = id_types(Data[i]["Elements"],Data[i]["Adj_mat"],gens) 

        # Determine bonding matrix for the compound
        if q_list[count_i] == "none":
            q_tmp = 0.0
        else:
            q_tmp = q_list[count_i]
        #Data[i]["Bond_mats"] = find_lewis(Data[i]["Atom_types"], Data[i]["Adj_mat"], q_tot=int(q_tmp),b_mat_only=True,verbose=False)        
        Data[i]["Bond_mats"] = find_lewis(Data[i]["Elements"], Data[i]["Adj_mat"], q_tot=int(q_tmp),b_mat_only=True,verbose=False) # find_lewis v.062520 

        # Check the number of molecules
        mol_in_in = mol_count(Data[i]["Adj_mat"])
        if mol_in_in > 1:
            print("ERROR: {} molecules were discovered in geometry {}. Check the geometry of the input file. Exiting...".format(mol_in_in,i))
            quit()

        # Generate list of bonds angles and dihedrals    
        print("\n{}\n* {:^163s} *\n{}\n".format("*"*167,"Parsing Modes and FF Information for Molecule {}".format(i),"*"*167))
        Data[i]["Bonds"],Data[i]["Bond_types"],Data[i]["Bond_params"],Data[i]["Angles"],Data[i]["Angle_types"],Data[i]["Angle_params"],\
        Data[i]["Dihedrals"],Data[i]["Dihedral_types"],Data[i]["Dihedral_params"],Data[i]["Impropers"],Data[i]["Improper_types"],Data[i]["Improper_params"],Data[i]["Charges"],Data[i]["Masses"],Data[i]["VDW_params"] =\
            Find_parameters(Data[i]["Adj_mat"],Data[i]["Bond_mats"],Data[i]["Geometry"],Data[i]["Atom_types"],FF_db=FF_all,Improper_flag = Improper_flag, force_read=force_read_opt,remove_multi=remove_multi)

        # Print System characteristics
        print("\n{}\n* {:^163s} *\n{}\n".format("*"*167,"Mode Summary for Molecule {}".format(i),"*"*167))
        print("\nAtom_types ({}):\n".format(len(set(Data[i]["Atom_types"]))))
        for j in sorted(set(Data[i]["Atom_types"])):
            print("\t{}".format(j))
        print("\nBond types ({}):\n".format(len(set(Data[i]["Bond_types"]))))
        for j in sorted(set(Data[i]["Bond_types"])):
            print("\t{}".format(j))
        print("\nAngle types ({}):\n".format(len(set(Data[i]["Angle_types"]))))
        for j in sorted(set(Data[i]["Angle_types"])):
            print("\t{}".format(j))
        print("\nDihedral types ({}):\n".format(len(set(Data[i]["Dihedral_types"]))))
        for j in sorted(set(Data[i]["Dihedral_types"])):
            print("\t{}".format(j))
        if Improper_flag:
            print("\nImproper types ({}):\n".format(len(set(Data[i]["Improper_types"]))))
            for j in sorted(set(Data[i]["Improper_types"])):
                print("\t{}".format(j))
        
        # Subtract off residual
        print("\n{:40s} {}".format("Residual Charge:",np.sum(Data[i]["Charges"])))
        if q_list[count_i] == "round":
            q_tot = int(round(np.sum(Data[i]["Charges"])))
        elif q_list[count_i] != "none":
            q_tot = int(q_list[count_i])

        # Avoid subtracting off the residual if q_list is none
        if q_list[count_i] != "none":
            correction = (float(q_tot)-np.sum(Data[i]["Charges"]))/float(len(Data[i]["Atom_types"]))
            print("{:40s} {}".format("Charge added to each atom:",correction))
            for j in range(len(Data[i]["Atom_types"])):
                Data[i]["Charges"][j] += correction

        print("{:40s} {}".format("Final Total Charge:",np.sum(Data[i]["Charges"])))
        print("\n{}".format("*"*167))
        print("* {:^163s} *".format("System Characteristics"))
        print("*{}*".format("-"*165))
        print("* {:<87s} {:<24s} {:<24s} {:<25s} *".format("Type","Element","Mass","Charge"))
        print("{}".format("*"*167))
        for j in range(len(Data[i]["Atom_types"])):
            print(" {:<88s} {:<23s} {:< 24.6f} {:< 24.6f}".format(Data[i]["Atom_types"][j],Data[i]["Elements"][j],Data[i]["Masses"][str(Data[i]["Atom_types"][j])],Data[i]["Charges"][j]))

    return Data


# Generate dictionaries for mapping between taffy atomtypes and lammps atomtypes
def taffy2lammps(mapfile,datafile):

    # Check existence conditions for the input files
    if os.path.isfile(mapfile) == False:
        print("ERROR in taffy2lammps: there is no mapfile named {}. Exiting...".format(mapfile))
        quit()
    elif os.path.isfile(datafile) == False:
        print("ERROR in taffy2lammps: there is no datafile named {}. Exiting...".format(datafile))
        quit()

    # Get taffy types
    with open(mapfile,'r') as f:
        for lc,lines in enumerate(f):
            if lc == 0:
                N_atoms = int(lines.split()[0])
                taffytypes = ["X"]*N_atoms
                lammpstypes = ["X"]*N_atoms
            if lc > 1:
                taffytypes[lc-2] = lines.split()[0]
            if lc == N_atoms+1:
                break
    
    # Get LAMMPS types
    flag = 0
    count = 0
    with open(datafile,'r') as f:
        for lc,lines in enumerate(f):
            fields = lines.split()
            if flag == 0 and len(fields) > 0 and fields[0] == "Atoms":
                flag = 1
                continue
            if flag > 0: 
                if len(fields) == 0:
                    flag += 1            
                    if flag == 3:
                        break
                else:
                    lammpstypes[int(fields[0])-1] = fields[2]
                    count += 1

    # Return dictionaries for mapping taffy onto lammps types and the reverse
    return { i:lammpstypes[taffytypes.index(i)] for i in set(taffytypes) }, { j:taffytypes[lammpstypes.index(j)] for j in set(lammpstypes) } 

def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

class Logger(object):
    def __init__(self,folder):
        self.terminal = sys.stdout
        self.log = open(folder+"/gen_md_for_sampling.log", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  
    def flush(self):
        pass

    def flush(self):
        pass

def miss2default(config):

 
    # Create default dict
    options = ['coord_files','FF_files','N_mol','T_equil','t_equil','T_anneal','t_anneal',\
               't_ext','pressure','density','N_density','fixes','frequency','q_list','outputname','timestep','gens','onefourscale_coul'\
               ,'onefourscale_lj','eps_scale','sigma_scale','charge_scale','force_UFF','mixing_rule','pair_styles','tail_opt','improper_flag'\
               ,'velocity_flag','molecule_flag','remove_multi','force_read_opt']

    defaults = ["", "", "25", 400, 1E6, 400, 1E6, 1E6, "1", 0, 0, "", 1000, "none", '', 1.0, 2, 0.0, 0.0, 1.0, 1.0, 1.0, 0, "wh",\
                'lj/cut/coul/long 14.0 14.0', True, False, False, False, False, False]  
    

    N_position = int (len(options) - len(defaults))

    default = {}
    for count_i,i in enumerate(defaults):
      default[options[N_position + count_i]] = i


    # Combine config
    option = {}
    for key in config:
      option[key] = config[key]

    missing = [ i in option for i in options]
    
    
    # set missing option to default
    for count_i,i in enumerate(missing):
      if i is False:
         option[options[count_i]] = default[options[count_i]]

    return option

if __name__ == "__main__":
   main(sys.argv[1:])
