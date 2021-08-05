import os,sys,argparse,glob

def main(argv):

    parser = argparse.ArgumentParser(description='assign groups to the xyz files.')

    parser.add_argument('-folder', dest = 'folder', default='.')

    parser.add_argument('-xyz', dest = 'xyz', default=None)



    args = parser.parse_args(argv)

    if args.xyz != None:
        xyzs = [ args.xyz ]

    else:
        xyzs = [ i.split('/')[-1][:-4] for i in glob.glob("{}/*.xyz".format(args.folder)) ]

    print("\nTOTAL OF {} MOLECULES".format(len(xyzs)))
    groups = ["thiol","ketone","aldehyde","alkane","alkene","alkyne","carbacid","sulfide","alcohol","ester","halide","amine","ether","cyanide","sulphone","nitrile","amide"]

    mol_dict = {}
    for i in groups:
        mol_dict[i] = []
    
    for i in xyzs:
        if "thiol" in i:
            mol_dict["thiol"].append(i)
        if "one" in i:
            mol_dict["ketone"].append(i)
        if "al" in i:
            mol_dict["aldehyde"].append(i)
        if "acid" in i:
            mol_dict["carbacid"].append(i)
        if "sulf" in i:
            mol_dict["sulfide"].append(i)
        if "ol" in i and "thiol" not in i:
            mol_dict["alcohol"].append(i)
        if "ate" in i:
            mol_dict["ester"].append(i)
        if "fluoro" in i or "chloro" in i or "bromo" in i or "iodo" in i:
            mol_dict["halide"].append(i)
        if "amine" in i:
            mol_dict["amine"].append(i)
        if "oxy" in i:
            mol_dict["ether"].append(i)
        if "nitrile" in i:
            mol_dict["nitrile"].append(i)
        if "amide" in i:
            mol_dict["amide"].append(i)
    
    mol_dict["alkane"].append("2-methylpropane")
    mol_dict["alkene"].append("(E)-hex-2-ene")

    for i in groups:
        #print("#"*30 + "\n{}: {}".format(i, mol_dict[i]))
        print("\nGROUP {}:\n {}".format(i, ', '.join([mol for mol in mol_dict[i]])))
    
    for i in xyzs:
        in_flag = False
        for g in groups:
            if i in mol_dict[g]:
                in_flag = True
        if in_flag == False:
            print("\n{} not in any group".format(i))

    return mol_dict

if __name__ == "__main__":
    main(sys.argv[1:])
