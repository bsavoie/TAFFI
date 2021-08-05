
# Author: Brett Savoie (brettsavoie@gmail.com)

import sys
from file_parsers import xyz_parse
from adjacency import Table_generator
import numpy as np

def main(argv):

    # Extract Element list and Coord list from the file
    elements,geo = xyz_parse(argv[0])
    adj_mat = Table_generator(elements,geo)    
    atom_types = id_types(elements,adj_mat,gens=2)

    # Print out molecular diagnostics
    print( "idx {:20s} {:20s}".format("Element","atomtype"))
    for count_i,i in enumerate(atom_types):
        print( "{:<2d}: {:20s} {:40s}".format(count_i,elements[count_i],i))
    print( "\nadj_mat:")
    for i in adj_mat:
        print(i)
    return

# identifies the taffi atom types from an adjacency matrix/list (A) and element identify. 
def id_types(elements,A,gens=2,avoid=[],geo=None):

    # On first call initialize dictionaries
    if not hasattr(id_types, "mass_dict"):

        # Initialize mass_dict 
        # (used for identifying the dihedral among a coincident set that will be explicitly scanned)
        id_types.mass_dict = {'H':1.00794,'He':4.002602,'Li':6.941,'Be':9.012182,'B':10.811,'C':12.011,'N':14.00674,'O':15.9994,'F':18.9984032,'Ne':20.1797,\
                             'Na':22.989768,'Mg':24.3050,'Al':26.981539,'Si':28.0855,'P':30.973762,'S':32.066,'Cl':35.4527,'Ar':39.948,\
                              'K':39.0983,'Ca':40.078,'Sc':44.955910,'Ti':47.867,'V':50.9415,'Cr':51.9961,'Mn':54.938049,'Fe':55.845,'Co':58.933200,'Ni':58.6934,'Cu':63.546,'Zn':65.39,\
                             'Ga':69.723,'Ge':72.61,'As':74.92159,'Se':78.96,'Br':79.904,'Kr':83.80,\
                             'Rb':85.4678,'Sr':87.62,'Y':88.90585,'Zr':91.224,'Nb':92.90638,'Mo':95.94,'Tc':98.0,'Ru':101.07,'Rh':102.90550,'Pd':106.42,'Ag':107.8682,'Cd':112.411,\
                             'In':114.818,'Sn':118.710,'Sb':121.760,'Te':127.60,'I':126.90447,'Xe':131.29,\
                             'Cs':132.90545,'Ba':137.327,'La':138.9055,'Hf':178.49,'Ta':180.9479,'W':183.84,'Re':186.207,'Os':190.23,'Ir':192.217,'Pt':195.078,'Au':196.96655,'Hg':200.59,\
                             'Tl':204.3833,'Pb':207.2,'Bi':208.98038,'Po':209.0,'At':210.0,'Rn':222.0}

    # Assemble prerequisite masses and Loop over the inidices that need to be id'ed
    masses = [ id_types.mass_dict[i] for i in elements ]
    atom_types = [ "["+taffi_type(i,elements,A,masses,gens)+"]" for i in range(len(elements)) ]

    # Add ring atom designation for atom types that belong are intrinsic to rings 
    # (depdends on the value of gens)
    for count_i,i in enumerate(atom_types):
        if ring_atom(A,count_i,ring_size=(gens+2)) == True:
            atom_types[count_i] = "R" + atom_types[count_i]            

    return atom_types

# adjacency matrix based algorithm for identifying the taffi atom type
def taffi_type(ind,elements,adj_mat,masses,gens=2,avoid=[]):

    # On first call initialize dictionaries
    if not hasattr(taffi_type, "periodic"):

        # Initialize periodic table
        taffi_type.periodic = { "h": 1,  "he": 2,\
                               "li":3,  "be":4,                                                                                                      "b":5,    "c":6,    "n":7,    "o":8,    "f":9,    "ne":10,\
                               "na":11, "mg":12,                                                                                                     "al":13,  "si":14,  "p":15,   "s":16,   "cl":17,  "ar":18,\
                                "k":19,  "ca":20,  "sc":21,  "ti":22,  "v":23,  "cr":24,  "mn":25,  "fe":26,  "co":27,  "ni":28,  "cu":29,  "zn":30,  "ga":31,  "ge":32,  "as":33,  "se":34,  "br":35,  "kr":36,\
                               "rb":37, "sr":38,  "y":39,   "zr":40,  "nb":41, "mo":42,  "tc":43,  "ru":44,  "rh":45,  "pd":46,  "ag":47,  "cd":48,  "in":49,  "sn":50,  "sb":51,  "te":52,  "i":53,   "xe":54,\
                               "cs":55, "ba":56,            "hf":72,  "ta":73, "w":74,   "re":75,  "os":76,  "ir":77,  "pt":78,  "au":79,  "hg":80,  "tl":81,  "pb":82,  "bi":83,  "po":84,  "at":85,  "rn":86}

    # Find connections, avoid is used to avoid backtracking
    cons = [ count_i for count_i,i in enumerate(adj_mat[ind]) if i == 1 and count_i not in avoid ]

    # Sort the connections based on the hash function 
    if len(cons) > 0:
        cons = list(zip(*sorted([ (atom_hash(i,adj_mat,masses,gens=gens-1),i) for i in cons ])[::-1]))[1]

    # Calculate the subbranches
    # NOTE: recursive call with the avoid list results 
    if gens == 0:
        subs = []
    else:
        subs = [ taffi_type(i,elements,adj_mat,masses,gens=gens-1,avoid=[ind]) for i in cons ]

    return "{}".format(taffi_type.periodic[elements[ind].lower()]) + "".join([ "["+i+"]" for i in subs ])


def ring_atom(adj_mat,idx,start=None,ring_size=10,counter=0,avoid=[]):

    # Consistency/Termination checks
    if ring_size < 3:
        print( "ERROR in ring_atom: ring_size variable must be set to an integer greater than 2!")
    if counter == ring_size:
        return False

    # Automatically assign start to the supplied idx value. For recursive calls this is set manually
    if start is None:
        start = idx
    
    # Loop over connections and recursively search for idx
    cons = [ count_i for count_i,i in enumerate(adj_mat[idx]) if i == 1 and count_i not in avoid ]
    if len(cons) == 0:
        return False
    elif start in cons:
        return True
    else:
        for i in cons:
            if ring_atom(adj_mat,i,start=start,ring_size=ring_size,counter=counter+1,avoid=[idx]) == True:
                return True
        return False

# hashing function for canonicalizing geometries based on their adjacency matrices and elements
def atom_hash(ind,A,M,alpha=100.0,beta=0.1,gens=10):    
    if gens <= 0:
        return rec_sum(ind,A,M,beta,gens=0)
    else:
        return alpha * sum(A[ind]) + rec_sum(ind,A,M,beta,gens)

# recursive function for summing up the masses at each generation of connections. 
def rec_sum(ind,A,M,beta,gens,avoid_list=[]):
    if gens != 0:
        tmp = M[ind]*beta
        new = [ count_j for count_j,j in enumerate(A[ind]) if j == 1 and count_j not in avoid_list ]
        if len(new) > 0:
            for i in new:
                tmp += rec_sum(i,A,M,beta*0.1,gens-1,avoid_list=avoid_list+[ind])
            return tmp
        else:
            return tmp
    else:
        return M[ind]*beta

if __name__ == "__main__": 
    main(sys.argv[1:])
    
