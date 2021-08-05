#!/bin/env python
# Author: Brett Savoie (brettsavoie@gmail.com)

import os,sys
from subprocess import Popen,PIPE
import numpy as np 
from copy import deepcopy
from file_parsers import xyz_parse
from adjacency import Table_generator,axis_rot
from write_functions import mol_write

def main(argv):

    # Extract Element list and Coord list from the file
    elements,geo = xyz_parse(argv[0])
    adj_mat = Table_generator(elements,geo)
    geo = transify(geo,adj_mat,opt_terminals=True,opt_final=True,elements=elements)

    # Save the straightened geometry    
    with open("straightened.xyz",'w') as f:
        f.write("{}\n\n".format(len(elements)))
        for count_i,i in enumerate(elements):
            f.write("{} {}\n".format(i," ".join([ "{:< 20.8f}".format(j) for j in geo[count_i] ])))

# This function takes a geometry and adjacency matrix then returns an "all-trans" aligned geometric conformer                
def transify(geo,adj_mat,start=-1,end=-1,elements=None,opt_terminals=False,opt_final=False):

    # Return the geometry if it is empty
    if len(geo) == 0: 
        return geo

    # Calculate the pair-wise graphical separations between all atoms
    seps = graph_seps(adj_mat)

    # Find the most graphically separated pair of atoms. 
    if start == -1 and end == -1:
        max_ind = np.where(seps == seps.max())
        start,end =  max_ind[0][0],max_ind[1][0]

    # Find the most graphically separated atom from the start atom
    elif end == -1:
        end = np.argmax(seps[start])        

    print("start (end): {} ({})".format(start,end))
                
    # Find the shortest pathway between these points
    pathway = Dijkstra(adj_mat,start,end)

    # If the molecule doesn't have any dihedrals then return
    if len(pathway) < 4:
        return geo
    print(pathway)
    # Initialize the list of terminal atoms
    terminals = set([ count_i for count_i,i in enumerate(adj_mat) if np.sum(i) == 1 ])

    # Work through the backbone dihedrals and straighten them out
    for i in range(1,len(pathway)-2):

        # Collect the atoms that are connected to the 2 atom of the dihedral
        group_1 = return_connected(adj_mat,start=pathway[i],avoid=[pathway[i+1]])   
        group_2 = return_connected(adj_mat,start=pathway[i+1],avoid=[pathway[i]])

        # Skip if the two groups are equal
        if group_1 == group_2: continue

        # Calculate the rotation vector and angle
        rot_vec = geo[pathway[i+1]] - geo[pathway[i]]
        stationary = geo[pathway[i]]
        theta = np.pi - dihedral_calc(geo[pathway[i-1:i+3]])                
        
        # Perform the rotation
        for j in group_2:
            geo[j] = axis_rot(geo[j],rot_vec,stationary,theta,mode="rad")

        # Check for overlaps
        theta = np.pi - dihedral_calc(geo[pathway[i-1:i+3]])

    # Identify the branch points
    branches  = [ count_i for count_i,i in enumerate(adj_mat) if len([ j for count_j,j in enumerate(i) if j == 1 and count_j not in terminals]) > 2 ]
    avoid_list = terminals

    # Loop over the branch points and correct their dihedrals
    for count_i,i in enumerate(branches):
        
        branching_cons = [ count_j for count_j,j in enumerate(adj_mat[i]) if j == 1 and count_j not in pathway ]

        for c in branching_cons:

            # Find the connections to this branch point that are terminal atoms
            conn_terminals = [ j for j in branching_cons if j != c ]

            # If a terminal connection exists then it is used to orient the branch by prepending it to the branch index list
            if len(conn_terminals) > 0:
                branch_ind = [conn_terminals[0]] + list(return_connected(adj_mat,start=i,avoid=[ count_j for count_j,j in enumerate(adj_mat[i]) if j == 1 and count_j in pathway ]+conn_terminals))
                geo[branch_ind[1:]] = transify(geo[branch_ind],adj_mat[branch_ind,:][:,branch_ind],start=0)[1:]

            # If no terminal connections exists then the first dihedral of the branch is not adjusted. 
            else:
                branch_ind = list(return_connected(adj_mat,start=i,avoid=[ count_j for count_j,j in enumerate(adj_mat[i]) if j == 1 and count_j in pathway ]+conn_terminals))
                geo[branch_ind] = transify(geo[branch_ind],adj_mat[branch_ind,:][:,branch_ind],start=next( count_j for count_j,j in enumerate(branch_ind) if j == i))
    
    # If optimize structure is set to True then the transified structure is relaxed using obminimize
    if opt_final == True:
        opt_geo(geo,adj_mat,elements,ff='uff')

    # If optimize terminals is set to true then a conformer search is performed over the terminal groups
    if opt_terminals == True:
        geo = opt_terminal_centers(geo,adj_mat,elements,ff='uff')

    return geo

# Description: This function calls obminimize (open babel geometry optimizer function) to optimize the current geometry
#
# Inputs:      geo:      Nx3 array of atomic coordinates
#              adj_mat:  NxN array of connections
#              elements: N list of element labels
#              ff:       force-field specification passed to obminimize (uff, gaff)
#
# Returns:     geo:      Nx3 array of optimized atomic coordinates
# 
def opt_geo(geo,adj_mat,elements,ff='uff',steps=100):

    # Write a temporary molfile for obminimize to use
    tmp_filename = '.tmp.mol'
    count = 0
    while os.path.isfile(tmp_filename):
        count += 1
        if count == 10:
            print( "ERROR in opt_geo: could not find a suitable filename for the tmp geometry. Exiting...")
            return geo
        else:
            tmp_filename = ".tmp" + tmp_filename            

    # Use the mol_write function imported from the write_functions.py 
    # to write the current geometry and topology to file
    mol_write(tmp_filename,elements,geo,adj_mat,append_opt=False)
    
    # Popen(stdout=PIPE).communicate() returns stdout,stderr as strings
    substring = 'obabel {} -O result.xyz --sd --minimize --steps {} --ff {}'.format(tmp_filename,steps,ff)
    output = Popen(substring, shell=True, stdin=PIPE, stdout=PIPE, stderr=PIPE,bufsize=-1).communicate()[0]
    element,geo = xyz_parse("result.xyz")
    # Remove the tmp file that was read by obminimize
    try:
        os.remove(tmp_filename)
    except:
        pass
    return geo

# Description: This function instantiates all conformers involving adjustments of terminal centers (non-terminal atoms with
#              only one non-terminal connection, such as a methyl carbon). The function performs repeated calls to obminimize
#              (open babel geometry optimizer function) to minimize each conformer and returns the lowest energy geometry.
#
# Inputs:      geo:      Nx3 array of atomic coordinates
#              adj_mat:  NxN array of connections
#              elements: N list of element labels
#              ff:       force-field specification passed to obminimize (uff, gaff)
#
# Returns:     geo:      Nx3 array of optimized atomic coordinates
# 
def opt_terminal_centers(geo,adj_mat,elements,ff='uff'):

    # Generate lists of rotation centers, the terminal and non-terminal atoms connected to each center and the rotation vectors for each center
    centers = terminal_centers(adj_mat)
    terminal_cons = [ [ count_j for count_j,j in enumerate(adj_mat[i]) if j == 1 and np.sum(adj_mat[count_j]) == 1 ] for i in centers ]
    nonterminal_cons = [ next( count_j for count_j,j in enumerate(adj_mat[i]) if j == 1 and np.sum(adj_mat[count_j]) != 1 ) for i in centers ]
    rot_vecs = [ geo[nonterminal_cons[count_i]]-geo[i] for count_i,i in enumerate(centers) ]

    # Print a warning if there are a large number of centers
    if len(centers) > 5:
        print( "WARNING in opt_terminal_centers: Iterating through all terminal center conformers. There are {} centers, so this might take a while.".format(len(centers))) 

    # Collect the rotation operations
    ops = []    
    for i in terminal_cons:

        if len(i) == 1:
            ops += [[ 0.0 ]]
        elif len(i) == 2 and False in [elements[i[0]] == elements[j] for j in i ]:
            ops += [[ j*90.0 for j in range(4) ]]
        elif len(i) == 2:
            ops += [[ j*90.0 for j in range(2) ]]
        elif len(i) == 3 and False in [elements[i[0]] == elements[j] for j in i ]:
            ops += [[ j*60.0 for j in range(6) ]]
        elif len(i) == 3:
            ops += [[ j*60.0 for j in range(2) ]]

    # Loop over all combinations of rotation operations and retain the lowest energy conformer
    E_min   = None
    geo_min = None    
    min_num = None
    for count_i,i in enumerate(combos(ops)):
        new_geo = deepcopy(geo)
        for count_j,j in enumerate(i):            
            for k in terminal_cons[count_j]:
                new_geo[k] = axis_rot(new_geo[k],rot_vecs[count_j],v2=new_geo[centers[count_j]],theta=j)

        # Optimize the current geometry
        new_geo = opt_geo(new_geo,adj_mat,elements,ff='uff')        

        # Generate a temporary filename for opt_terminal_centers to use
        tmp_filename = '.tmp.mol'
        count = 0
        while os.path.isfile(tmp_filename):
            count += 1
            if count == 10:
                print( "ERROR in opt_terminal_centers: could not find a suitable filename for the tmp geometry. Exiting...")
                return geo
            else:
                tmp_filename = ".tmp" + tmp_filename            

        # Calculate the energy of the optimized conformer
        # use os.devnull to suppress stderr from the obminimize call
        mol_write(tmp_filename,elements,new_geo,adj_mat,append_opt=True)

        # Popen(stdout=PIPE).communicate() returns stdout,stderr as strings
        output = Popen('obenergy -ff {} {}'.format(ff,tmp_filename).split(), stdin=PIPE, stdout=PIPE, stderr=PIPE,bufsize=-1).communicate()[0]
        output = str(output)
        
        # Grab the energy from the output
        Energy = next( float(j.split()[3]) for j in output.split(r'\n') if len(j.split()) > 4 and j.split()[0] == "TOTAL" and j.split()[1] == "ENERGY" and j.split()[2] == "=" )
        if E_min is None or Energy < E_min:
            E_min = Energy
            geo_min = deepcopy(new_geo)
            min_num = count_i
        os.remove(tmp_filename)

    return geo_min

# Description: Generates all unique combinations (independent of order) of objects in a list of lists (x)
#
# Inputs:        x: a list of lists
#        
# Returns:       an iterable list of objects that yields a unique combination of objects (one from each sublist in x)
#                from the x.
def combos(x):
    if len(x) == 0:
        yield []
    elif hasattr(x[0], '__iter__'):
        for i in x[0]:
            for j in combos(x[1:]):
                yield [i]+j
    else:
        for i in x:
            yield [i]

# Calculates the dihedral angle for a quadruplet of atoms
def dihedral_calc(xyz):
    
    # Calculate the 2-1 vector           
    v1 = (xyz[1]-xyz[0]) 
                                                             
    # Calculate the 3-2 vector           
    v2 = (xyz[2]-xyz[1]) 
                                                             
    # Calculate the 4-3 bond vector      
    v3 = (xyz[3]-xyz[2]) 

    # Calculate dihedral 
    angle = np.arctan2( np.dot(v1,np.cross(v2,v3))*(np.dot(v2,v2))**(0.5) , np.dot(np.cross(v1,v2),np.cross(v2,v3)) )
    
    return angle


# Returns a matrix of graphical separations for all nodes in a graph defined by the inputted adjacency matrix 
def graph_seps(adj_mat_0):

    # Create a new name for the object holding A**(N), initialized with A**(1)
    adj_mat = deepcopy(adj_mat_0)
    
    # Initialize an np.array to hold the graphical separations with 
    # -1 for all unassigned elements and 0 for the diagonal.
    seps = np.ones([len(adj_mat),len(adj_mat)])*-1
    np.fill_diagonal(seps,0)

    # Perform searches out to len(adj_mat) bonds 
    for i in np.arange(len(adj_mat)):        

        # All perform assignments to unassigned elements (seps==-1) and all perform an assignment if the value in the adj_mat is > 0        
        seps[np.where((seps==-1)&(adj_mat>0))] = i+1

        # set the larger than 1 values to 1 to ensure numerical stability 
        adj_mat[np.where(adj_mat>1)] = 1
        
        # Break once all of the elements have been assigned
        if -1 not in seps: break

        # Take the inner product of the A**(i+1) with A**(1)
        adj_mat = np.dot(adj_mat,adj_mat_0)

    return seps


# Returns the list of terminal centers, where a terminal center is an atom with only one 
# bond to a non-terminal atom.
def terminal_centers(adj_mat):    
    return [ count_i for count_i,i in enumerate(adj_mat) if sum(i) > 1 and len([ j for count_j,j in enumerate(i) if j == 1 and sum(adj_mat[count_j]) > 1 ]) == 1 ]


# This is an implementation of Dijkstra algorithm for finding the backbone 
def Dijkstra(Adj_mat,start=0,end=-1):
    
    # Default to the last node in Adj_mat if end is unassigned or less than 0
    if end < 0:
        end = len(Adj_mat)+end

    # Remove terminal sites (sites with only a single length 2 self walk). 
    # Continue until all terminal structure has been removed from the topology.
    Adj_trimmed = np.copy(Adj_mat)

    # Initialize Distances, Previous, and Visited lists    
    Distances = np.array([100000]*(len(Adj_mat))) # Shortest distance to origin
    Distances[start] = 0 # Sets the separation of the initial node
    Previous = np.array([-1]*len(Adj_mat)) # Previous site on the shortest distance to origin
    Visited = [0]*len(Adj_mat) # Visited sites

    # Initialize current site (i) and neighbors list
    i = start # current site
    neighbors = []

    # Iterate through sites until the terminal site is identified. 
    while( 0 in Visited):

        # If the current site is the terminal site the algorithm is finished
        if i == end: break

        # Add new neighbors to the list and remove current site from unvisited
        neighbors = [ count_j for count_j,j in enumerate(Adj_trimmed[i]) if j == 1 ]
        Visited[i] = 1

        # Iterate over neighbors and update shortest paths
        for j in neighbors:

            # Update distances for current generation of connections
            if Distances[i] + Adj_trimmed[j,i] < Distances[j]:
                Distances[j] = Distances[i] + Adj_trimmed[j,i]
                Previous[j] = i

        # Find new site based on the minimum separation
        tmp = min([ j for count_j,j in enumerate(Distances) if Visited[count_j] == 0 ])
        i = [ count_j for count_j,j in enumerate(Distances) if j == tmp and Visited[count_j] == 0 ][0]

    # Find shortest path by iterating backwards starting with the end site.
    Shortest_path = [end]
    i=end
    while( i != start):
        Shortest_path = Shortest_path + [Previous[i]]    
        i = Previous[i]

    # Reverse order of the list to go from start to finish
    Shortest_path = Shortest_path[::-1]
    return Shortest_path


# Returns the set of connected nodes to the start node, while avoiding any connections through nodes in the avoid list. 
def return_connected(adj_mat,start=0,avoid=[]):

    # Initialize the avoid list and encountered nodes with the starting index
    avoid = set(avoid+[start])
    new_0 = [start] # most recently encountered nodes
    new_1 = set([start]) # all encountered nodes

    # keep looping until no new nodes are encountered
    while len(new_0) > 0:        

        # reinitialize new_0 with new connections
        new_0 = [ count_j for i in new_0 for count_j,j in enumerate(adj_mat[i]) if j == 1 and count_j not in avoid ]

        # update new_1 set and avoid list with newly encountered nodes
        new_1.update(new_0)
        avoid.update(new_0)

    # return the set of encountered nodes
    return new_1

if __name__ == "__main__": 
    main(sys.argv[1:])
