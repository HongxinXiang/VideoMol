# Docking Guidance

Here is a step-by-step tutorial on docking. Through this tutorial, you will easily learn how to complete molecular docking. Of course, the docking results of VideoMol are also generated through the following steps.



## 1. Environment and system requirements

- Ubuntu 18.04

- Windows 10

- Dock 6.10

- Chimera：For protein and ligand structure preparation

- PyMOL

Please note that if your Ubuntu has a UI interface, you do not need a Windows computer. Because some steps can only be performed in the UI interface.



### 1.1 Install Dock6.10

The installation link of Dock is: https://dock.compbio.ucsf.edu/DOCK_6/index.htm. 

Note: You need to fill out an application to download Dock.

```bash
cd ~/softwares/Dock6.10
# Compile from source
tar -zxvf dock.6.10_source.tar.gz
cd dock6/install/
./configure gnu  # Non-parallel version. Please note that the parallel version requires MPI support. Since our server does not have it installed, we do not use parallel
make all
make dockclean
make utils
# Test whether the installation is successful
cd test
make test
make check
# Add to environment variables
vim ~/.bashrc
export PATH=~/softwares/Dock6.10/dock6/bin:$PATH
source ~/.bashrc
```



Summary of the errors we encountered when `make all` and solutions are follows:

```text
Problem 1: zlib.h: No such file or directory
Solution: sudo apt-get install libz-dev

Problem 2: Makefile:21: recipe for target 'is_LEX_available' failed
Solution: Install flex: sudo apt-get install flex

Problem 3: make[2]: gfortran: Command not found
Solution: Install gfortran: sudo apt-get install gfortran

Problem 4: make[2]: yacc: Command not found
Solution: sudo apt-get install byacc flex

Problem 5: /usr/bin/python: bad interpreter: No such file or directory
Solution: The system does not have a default /usr/bin/python file, create a soft link: sudo ln -s /usr/bin/python3.6 /usr/bin/python
```



### 1.2 Install UCSF Chimera

UCSF Chimera is an interactive visualization software mainly used for the analysis of protein three-dimensional structure.

Please note that this software requires an interface. Since my Linux server is interfaceless, I installed Chimera on a `Windows computer`. Go to https://www.cgl.ucsf.edu/chimera/download.html and download the installation package:

```bash
cd /home/xianghongxin/softwares/Dock6.9
chmod +x chimera-1.17.3-linux_x86_64.bin
./chimera-1.17.3-linux_x86_64.bin
```



### 1.3 Install PyMOL

Please reference: https://www.pymol.org/



## 2. Preparation of protein and ligand structures

### 2.1 Preparation of protein structures

Because we are doing molecular docking, we need information about the combination of protein and ligand to discover the pocket location.

First, download the pdb file of protein and ligand binding from PDBank (https://www.pdbus.org/search). Use Chimera software to open the downloaded pdb file. Here I take 4ivs.pdb as an example.

Enter the Chimera software, click `File->Open->select a pdb file`, and the content in the middle is the structure of the pdb file I opened.

**Prepare the receptor:**

To remove a ligand:

- Select ligand: `Select->Structure->ligand`
- Delete ligand: `Actions->Atom/Bonds->delete`

Here, you need to use the Dock Prep tool to complete the preparation of the receptor:

`Tools->Structure Editing->Dock prep->Always OK (default parameters are fine)`

Here, we save the processed structure as `4ivs_rec_charged.mol2`.

At this point, the receptor has not been processed yet, and the protein structure without hydrogenation needs to be processed. This is necessary to generate the molecular surface.

First remove the hydrogen:

- Select hydrogen: `Select->Chemistry->element->H`
- Delete selected hydrogen: `Action->Atoms/Bonds->delete`

Then save the pdb file:

- `File->Save PDB...`

we save the file as: `4ivs_rec_noH.pdb`

At this point, the receptor is ready, and the next step is to prepare the ligand.



### 2.2 Prepare the ligand

Reopen the 4ivs.pdb file.

Remove all atoms in the protein except the ligand:

- `Select->structure->ligand`

- `Select->invert (selected models)`

- `Action->Atoms/Bonds->delete`

Then, add hydrogen: 

- `Tools->Structure Editing->AddH->OK`

Add charge:

- `Tools->Structure Editing->Add Charge->OK`

Then, we save the processed structure to `4ivs_lig_charged.mol2` by `File->Save Mol2`.



## 3. Generate a sphere

### 3.1 Generating the molecular surface of the receptor

Reopen Chimera:

- Read file: `File -> Open -> rec_noH.pdb`
- Create surface: `Actions -> Surface -> Show`
- Save DMS file: `Tools -> Structure Editing -> Write DMS`



### 3.2 Generate a sphere around the receptor (must be done on a Linux server)

Spheres are computed over the entire surface, yielding approximately one sphere per surface point. This dense representation is then filtered to retain only the largest spheres associated with each surface atom. The filtered set is then clustered using a single linkage algorithm. Each resulting cluster represents an outgrowth in the target. The sphgen input file must be named INSPH and contain the following information:

```bash
4ivs_rec.dms	# molecular surface file (no default)
R	            # sphere outside of surface (R) or inside surface (L) (no default)
X	            # specifies subset of surface points to be used (X=all points) (no default)
0.0	            # prevents generation of large spheres with close surface contacts (default=0.0)
4.0	            # maximum sphere radius in Angstroms (default=5.0)
1.4	            # minimum sphere radius in Angstroms (no default)
4ivs_rec.sph	        # clustered spheres file (no default)
```

The next step requires the use of Dock 6.10 software. Since the Windows version is not supported, we will perform the following steps on Linux:

- command: `sphgen`

After entering the command, we will get two files: `4ivs_rec.sph` and `OUTSPH`.



### 3.3 Select a subset of generated Spheres as your binding site (generate docking pocket)

Selecting a sphere within a certain radius of the desired position: If the active site is known, then a sphere can be selected within the radius of the set of atoms that describe that site. To do this, we use the sphere_selector program, which is distributed as an add-on to DOCK. The command of sphere_selector is:

- `sphere_selector 4ivs_rec.sph 4ivs_lig_charged.mol2 10.0`

Here we use the command to select all spheres within 10.0 angstroms of the root mean square deviation (RMSD) of each atom in the ligand crystal structure. The output file is always named `selected_spheres.sph`.



## 4. Generate Grid

### 4.1 Building a box around the active site

We first create a `box.in` with the following content:

```text
Y
5
selected_spheres.sph
1
4ivs_rec_box.pdb
```

Enter the command to generate box: `showbox < box.in`



### 4.2 Generate Grid

The grid-based energy scoring functionality is used here. The energy scoring component of DOCK is a force field score. The force field score is an approximation of the molecular mechanical interaction energy, consisting of van der Waals and electrostatic components.

To generate the grid itself, the program grid distributed as an add-on to DOCK is used. Using the boxes generated in the previous step, the program grid precomputes the contact and electrostatic potentials at active positions at a specified grid spacing. To run the grid, a file called grid.in must be manually generated interactively by answering questions to create a text file.

So here, we run the following command:

- `grid -i 4ivs_grid.in`

Next, start generating Grid by: 

- `grid -i 4ivs_grid.in -o 4ivs_grid.out `

Below are the specific values we fill in:

```bash
compute_grids [no] (yes, no): yes
grid_spacing [0.3] (float): 0.3
output_molecule [no] (yes, no): no
contact_score [no] (yes, no): no
energy_score [no] (yes, no): yes
energy_cutoff_distance [10] (float): 10
atom_model [u] (u, a): a
attractive_exponent [6] (int): 6
repulsive_exponent [12] (int): 12
distance_dielectric [yes] (yes, no): yes
dielectric_factor[4.0] (float): 4
allow_non_integral_charges: yes
bump_filter [no, yes]: yes
bump_overlay [0.75]: 0.75
receptor_file: 4ivs_rec_charged.mol2
box_file: 4ivs_rec_box.pdb
vdw_definition_file: /home/xianghongxin/softwares/Dock6.10/dock6/parameters/vdw_AMBER_parm99.defn
```

Three files will be generated: `4ivs_grid.bmp, 4ivs_grid.in, 4ivs_grid.nrg`



## 5. Molecular docking using Dock 6.10 [semi-flexible]

Create the file dock.in (parameters explained here: http://dock.compbio.ucsf.edu/DOCK_6/tutorials/ligand_sampling_dock/ligand_sampling_dock.html)

```bash
conformer_search_type                                        flex
write_fragment_libraries                                     no
user_specified_anchor                                        no
limit_max_anchors                                            no
min_anchor_size                                              5
pruning_use_clustering                                       yes
pruning_max_orients                                          1000
pruning_clustering_cutoff                                    100
pruning_conformer_score_cutoff                               100.0
pruning_conformer_score_scaling_factor                       1.0
use_clash_overlap                                            no
write_growth_tree                                            no
use_internal_energy                                          yes
internal_energy_rep_exp                                      12
internal_energy_cutoff                                       100.0
ligand_atom_file                                             4ivs_lig_charged.mol2
limit_max_ligands                                            no
skip_molecule                                                no
read_mol_solvation                                           no
calculate_rmsd                                               no
use_database_filter                                          no
orient_ligand                                                yes
automated_matching                                           yes
receptor_site_file                                           selected_spheres.sph
max_orientations                                             1000
critical_points                                              no
chemical_matching                                            no
use_ligand_spheres                                           no
bump_filter                                                  yes
bump_grid_prefix                                             grid
max_bumps_anchor                                             2
max_bumps_growth                                             2
score_molecules                                              yes
contact_score_primary                                        no
grid_score_primary                                           yes
grid_score_rep_rad_scale                                     1
grid_score_vdw_scale                                         1
grid_score_es_scale                                          1
grid_score_grid_prefix                                       grid
minimize_ligand                                              yes
minimize_anchor                                              yes
minimize_flexible_growth                                     yes
use_advanced_simplex_parameters                              no
minimize_flexible_growth_ramp                                no
simplex_max_cycles                                           1
simplex_score_converge                                       0.1
simplex_cycle_converge                                       1.0
simplex_trans_step                                           1.0
simplex_rot_step                                             0.1
simplex_tors_step                                            10.0
simplex_anchor_max_iterations                                500
simplex_grow_max_iterations                                  250
simplex_grow_tors_premin_iterations                          0
simplex_random_seed                                          0
simplex_restraint_min                                        no
atom_model                                                   all
vdw_defn_file                                                /home/xianghongxin/softwares/Dock6.10/dock6/parameters/vdw_AMBER_parm99.defn
flex_defn_file                                               /home/xianghongxin/softwares/Dock6.10/dock6/parameters/flex.defn
flex_drive_file                                              /home/xianghongxin/softwares/Dock6.10/dock6/parameters/flex_drive.tbl
ligand_outfile_prefix                                        output
write_orientations                                           no
num_scored_conformers                                        1
rank_ligands                                                 no
```

Some files have incorrect default settings (so it is recommended not to add the 4ivs_ prefix): `mv 4ivs_grid.bmp grid.bmp; mv 4ivs_grid.nrg grid.nrg`

Start docking: 

- `dock6 -i dock.in -o dock.out`

The output file dock.out contains the docking results:

```bash
...
Grid_Score: -52.468266
...
```



# References

We would like to thank the following tutorials for providing us with very useful assistance:

DOCK6.9 Learning (IV): https://blog.csdn.net/Wanderers111/article/details/103828295
DOCK6.9 Learning (V): https://blog.csdn.net/Wanderers111/article/details/104012626?spm=1001.2014.3001.5502
DOCK6.9 Learning (VI): https://blog.csdn.net/Wanderers111/article/details/103968992?spm=1001.2014.3001.5502
DOCK Tutorial-1: https://cloud.tencent.com/developer/article/1785373
DOCK-2-Preparation of protein receptors and ligands: https://cloud.tencent.com/developer/article/1785374
DOCK-3-Generate Spheres: https://cloud.tencent.com/developer/article/1785377
DOCK-4-Generate Grid: https://cloud.tencent.com/developer/article/1785376
DOCK-5-Docking: https://cloud.tencent.com/developer/article/1785375
"Molecular Docking Tutorial" Protein/Nucleic Acid/Peptide-Small Molecule Docking (DOCK 6.9): https://baijiahao.baidu.com/s?id=1674141651219145929&wfr=spider&for=pc
Talk about the mol2 file that records the chemical system structure: http://sobereva.com/655