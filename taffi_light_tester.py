import os, sys, glob, subprocess, shutil


# Parse xyz files of the benchmark molecules
run_dir = os.getcwd()
xyzs = glob.glob("{}/xyzs/*.xyz".format(os.getcwd()), recursive=True)
FF_file = os.path.join(run_dir, "taffi_gen.db")
outputfolder = os.path.join(run_dir, "test_runs")

# Types of generated files that are tested
files = ["data", "in.settings", "in.init", "map", "mol.txt"]

# Generate output folder
if os.path.isdir(outputfolder) is False:
    os.mkdir(outputfolder)

# Run tests
print("\n### RUNNING TESTS ON {} BENCHMARK MOLECULES".format(len(xyzs)))
for idx, xyz in enumerate(xyzs):

    name = xyz.split("/")[-1].split(".")[0]

    # Only run if the folder doesn't already exist
    if os.path.isdir("{}/{}".format(outputfolder, name)) is False:
            
        # Reproduce runs from the paper (see script for additional options)
        substring = 'python gen_md_for_sampling.py {} -FF {} -N 500 -T 298 -T_A 10 -t_A 2E6 -t 2E6 -t_ext 2E6  --tail -q 0 -o {}/{}'

        substring = substring.format(xyz, FF_file, outputfolder, name)

        # Run gen_md_for_sampling
        output    = subprocess.Popen(substring.split(),stdout=subprocess.PIPE,stderr=subprocess.PIPE,encoding='utf-8').communicate()[0].strip("\r\n")

        # Check generated files
        for inputfile in files:
            if os.path.isfile(os.path.join(run_dir, outputfolder, "{}/{}.{}".format(name,name,inputfile))) is False:
                print("{}/{}/{}.{} was not generated".format(outputfolder,name,name,inputfile))
                quit()
        if os.path.isfile(os.path.join(run_dir, outputfolder, "{}/extend.in.init".format(name))) is False:
            print("{}/{}/extend.in.init was not generated".format(outputfolder,name))
            quit()

        # Test transify
        substring = 'python {}/Lib/transify.py {} -o {}/{}/straightened.xyz'.format(run_dir,xyz, outputfolder, name)
        output    = subprocess.Popen(substring.split(),stdout=subprocess.PIPE,stderr=subprocess.PIPE,encoding='utf-8').communicate()[0].strip("\r\n")
        if os.path.isfile(os.path.join(run_dir,"straightened.xyz")) is False:
            print("transify function for {} have failed.".format(xyz))
            quit()
        else: shutil.move("{}/{}".format(run_dir,"straightened.xyz"),"{}/{}/{}".format(outputfolder,name,"straightened.xyz"))
        print("  {:>2.0f}/{} SUCCESS: {}".format(idx+1, len(xyzs), name))

    # Print message to the user if the job creation was skipped because the input folder was already present. 
    else:
        print("The folder {}/{} already exists, skipping this job.".format(outputfolder, name))
