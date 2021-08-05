
# TAFFI-gen

Topology Automated Force-Field Interactions (TAFFI) is a framework for parameterizing transferable force-fields to quantum chemistry data. Here we are distributing force-field parameters for TAFFI-gen, a fixed charge, harmonic force field fit to wB97X-D3/def2-TZVP data with coverage for small organic molecules. The current distribution provides the TAFFI-gen force-field parameters used for the simulations reported [here](https://doi.org/10.26434/chemrxiv.14527299.v1). 

# What's Included

The included scripts are provided as stand-alone python programs that only provide atomtyping and force-field assignment capability, not the full TAFFI functionality required for extending the force-field or parameterizing alternative functional forms.

* `taffi_lite_tester.py`: this is a driver script that will generate lammps input files for the benchmarked taffi-gen structures. This script calls the others and should be the first thing that the beginning user consults and tries to run.

* `gen_md_for_sampling.py`: this python script performs taffi-gen atom typing, force-field parameter assignment, and lammps input file generation.

* `Lib`: this folder contains some TAFFI library functions used by gen_md_for_sampling.py

* `taffi_gen.db`: this is a text file that contains the taffi-gen force-field parameters required to simulate the benchmarked structures.

* `xyzs`: this folder contains the `*.xyz` files for each benchmark molecule.

* `OKKJLVBELUTLKV-UHFFFAOYSA-N`: this folder contains quantum chemistry training data for the methanol model compound. This is included here as an example of the training data for interested readers of the TAFFI-gen reference. A full description of model compound and training data generation are supplied there.

### Dependencies

* Python 3
* numpy
* openbabel 2.3.2 and above

### Usage

* To create lammps input files using your xyz, run
```
python gen_md_for_sampling.py example.xyz
```
where example.xyz is a user-supplied structure, or one of the structures included in the "xyzs" folder. The output of this script will be the inputes for a lammps job. Optional arguments to the gen_md_for_sampling.py script can be reviewed by running it with the -h option. Note that the supplied `taffi_gen.db` file only has guarranteed coverage for the benchmarked model compounds. If the user supplies a compound with missing parameters, the script will print out the missing paramters for the user.

* To create input files based on the included benchmark `.xyz` files inside of the `xyz` folder run:
```
python taffi_lite_tester.py
```
Running this script will react a folder called `test_runs` that contains the input files for each job. This folder already exists in the repository for the user to check the expected behavior. If running from scratch, the existing `test_runs` folder should be renamed before running the script. 

## Citation

Please cite the following if you use TAFFI-gen

Bumjoon Seo, Zih-Yu Lin, Qiyuan Zhao, Michael A. Webb, Brett M. Savoie. *Topology Automated Force-Field Interactions
(TAFFI): A Framework for Developing Transferable Force-Fields*. ChemRxiv. https://doi.org/10.26434/chemrxiv.14527299.v1
