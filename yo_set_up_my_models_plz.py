import numpy as np 
import os
import glob
import argparse
import sys

def main():
	dir_name = sys.argv[1]

	BT_fid = 0.5
	stellarfrac_fid = 0.01e0
	gasfrac_fid = 0.1
	halo_fbar_fid=0.16


	Npart_disk_fid = 100000
	Npart_bulge_fid = 100000
	Npart_gas_fid = 100000
	Npart_ghalo_fid = 1000000

		
	diskfrac_input_fid, diskfrac_fid, bulgefrac_fid, mstar_fid = \
			calc_model_params(stellarfrac_fid, gasfrac_fid, BT_fid)

	stellarfrac_list = 	[0.01, 	0.01, 	0.01, 	0.01, 	0.01, 	0.01, 	0.01, 	0.01,	0.01,	0.01,	0.01,	0.01, 	0.05]
	gasfrac_list = 		[0.1, 	0.1, 	0.1, 	0.1, 	0.1, 	0.1, 	0.1, 	0.1, 	0.1, 	0.1, 	0.05, 	0.6, 	0.1] 
	BT_list = 			[0, 	0, 		0, 		0, 		0.1, 	0.25, 	0.50, 	0.1, 	0.25, 	0.50, 	0, 		0, 		0]
	fbulge_list =		[0, 	0, 		0, 		0, 		0.05, 	0.05, 	0.05, 	0.2, 	0.2, 	0.2, 	0, 		0, 		0]
	halo_fbar_list = 	[0, 	0.08, 	0.16, 	0.32, 	0.16, 	0.16, 	0.16, 	0.16, 	0.16, 	0.16,	0.16,	0.16,	0.16]
	

	for ii in range(len(BT_list)):
		print('model number', ii)
		BT = BT_list[ii]
		gasfrac = gasfrac_list[ii]
		fbulge = fbulge_list[ii]
		stellarfrac = stellarfrac_list[ii]
		halo_fbar = halo_fbar_list[ii]

		diskfrac_input, diskfrac, bulgefrac, mstar = calc_model_params(stellarfrac, gasfrac, BT)

		Npart_disk, Npart_bulge, Npart_gas, Npart_ghalo = \
		calc_Nparticles(BT ,BT_fid,\
						 gasfrac, gasfrac_fid, \
						 stellarfrac, stellarfrac_fid,\
						 halo_fbar, halo_fbar_fid,\
				Npart_disk_fid, Npart_bulge_fid, Npart_disk_fid, Npart_ghalo_fid)



		print('Input disk fraction:',diskfrac_input)
		print('Disk reduction factor:',mstar)
		print('Disk fraction:', diskfrac)
		print('Bulge fraction:', bulgefrac)
		print('B/T:', bulgefrac/(bulgefrac+diskfrac))
		print('Ndisk:',Npart_disk)
		print('Nbulge:',Npart_bulge)
		print('Ngas:',Npart_gas)
		print('')


		print('Disk particle mass fiducial:',diskfrac_fid/Npart_disk_fid)
		print('Bulge particle mass fiducial:',bulgefrac_fid/Npart_bulge_fid)
		print('Gas particle mass fiducial:',gasfrac_fid*stellarfrac_fid/Npart_gas_fid)
		print('')

		if Npart_disk!=0:
			print('Disk particle mass:',diskfrac/Npart_disk)
		if Npart_bulge !=0:
			print('Bulge particle mass:',bulgefrac/Npart_bulge)
		if Npart_gas!=0:
			print('Gas particle mass:',gasfrac*stellarfrac/Npart_gas)
		print('')

		model_dir = create_dir(dir_name, stellarfrac, halo_fbar, BT, fbulge, gasfrac)
	

		write_GalIC_files(model_dir, diskfrac_input, bulgefrac, fbulge, Npart_disk, Npart_bulge)

		write_buildgas_files(model_dir, diskfrac_input, mstar, bulgefrac, fbulge, Npart_gas, halo_fbar, Npart_ghalo)

		write_Gadget_files(model_dir)


def create_dir(dir_name, stellarfrac, halo_fbar, BT, fbulge, gasfrac):
	model_dir = '/home/awatts/Adam_PhD/models_fitting/GadgetSimulations/simulations/'\
			'{dir}/fstar{fstar}/BT{BT}/GF{GF}_fB{fB}_fhalo{fhalo}'\
					.format(dir = dir_name,fstar=int(100*stellarfrac), fhalo = int(100*halo_fbar), \
						BT=int(BT*100),fB=int(fbulge*100),GF=int(gasfrac*100))
	if glob.glob(model_dir) == []:
		os.mkdir(model_dir)
	if glob.glob('{}/snaps/'.format(model_dir)) == []:
		os.mkdir('{}/snaps/'.format(model_dir))
	if glob.glob('{}/model_setup/'.format(model_dir)) == []:
		os.mkdir('{}/model_setup/'.format(model_dir))
	if glob.glob('{}/data/'.format(model_dir)) == []:
		os.mkdir('{}/data/'.format(model_dir))
	if glob.glob('{}/figures/'.format(model_dir)) == []:
		os.mkdir('{}/figures/'.format(model_dir))
	return model_dir

def calc_model_params(stellarfrac, gasfrac, BT):
	diskfrac = (1 - BT)*stellarfrac
	bulgefrac = BT*stellarfrac
	mstar = (1.e0-BT)/ (1.e0-BT + gasfrac)
	diskfrac_input = diskfrac / mstar
	return diskfrac_input, diskfrac, bulgefrac, mstar

def calc_Nparticles(BT , BT_fid, \
	gasfrac, gasfrac_fid, \
	stellarfrac, stellarfrac_fid, \
	halo_fbar, halo_fbar_fid,
	Npart_disk_fid, Npart_bulge_fid, Npart_gas_fid, Npart_ghalo_fid):
	Npart_disk = Npart_disk_fid * (1-BT)/(1-BT_fid) * stellarfrac/stellarfrac_fid
	Npart_bulge = Npart_bulge_fid * BT/BT_fid * stellarfrac/stellarfrac_fid
	Npart_gas = Npart_gas_fid * (gasfrac/gasfrac_fid) * stellarfrac/stellarfrac_fid
	Npart_ghalo = Npart_ghalo_fid * (halo_fbar / halo_fbar_fid)
	return int(Npart_disk), int(Npart_bulge), int(Npart_gas), int(Npart_ghalo)

def write_GalIC_files(model_dir, mdisk, mbulge, fbulge, Npart_disk, Npart_bulge):
	filename = '{model_dir}/Nbody.param'.format(model_dir=model_dir)
	f = open(filename, 'w')
	f.write(
	'%------   File and path names, as well as output file format\n'\
	'\n'\
	'OutputDir       ./model_setup\n'\
	'\n'\
	'OutputFile      snap    % Base filename of generated sequence of files\n'\
	'SnapFormat      3       %  File format selection\n'\
	'\n'\
	'\n'\
	'%------   Basic structural parameters of model\n'\
	'\n'\
	'CC             10.0       %	halo concentration\n'\
	'V200          200.0       %	circular velocity v_200 (in km/sec)\n'\
	'LAMBDA         0.035      %	spin parameter     \n'\
	'MD             {mdisk:.4f}      %	disk mass fraction   \n'\
	'MB             {mbulge:.4f}        % bulge mass fraction     \n'\
	'MBH            0.0        % black hole mass fraction. If zero, no black\n'\
	'                          %	hole is generated, otherwise one at the centre\n'\
	'                          %	is added.\n'\
	' \n'\
	'JD             {mdisk:.4f}      %	disk spin fraction, typically chosen equal to MD\n'\
	'\n'\
	'DiskHeight     0.2        % thickness of stellar disk in units of radial scale length \n'\
	'BulgeSize      {fbulge:.3f}        % bulge scale length in units of halo scale length \n'\
	'\n'\
	'HaloStretch    1.0        %	should be one for a spherical halo, smaller than one corresponds to prolate distortion, otherwise oblate\n'\
	'BulgeStretch   1.0        %	should be one for a spherical bulge, smaller than one corresponds to prolate distortion, otherwise oblate\n'\
	'\n'\
	'%------   Particle numbers in target model\n'\
	'\n'\
	'N_HALO         100000          %	desired number of particles in dark halo \n'\
	'N_DISK         {Npart_disk}           %	desired number of collisionless particles in disk \n'\
	'N_BULGE        {Npart_bulge}                % number of bulge particles \n'\
	'\n'\
	'\n'\
	'\n'\
	'%------   Selection of symmetry constraints of velocity structure\n'\
	'\n'\
	'TypeOfHaloVelocityStructure    2      %  0 = spherically symmetric, isotropic\n'\
	'                                      %  1 = spherically symmetric, anisotropic (with beta parameter specified)\n'\
	'                                      %  2 = axisymmetric, f(E, Lz), with specified net rotation\n'\
	'                                      %  3 = axisymmetric, f(E, Lz, I_3), with <vz^2>/<vR^2> specified and net rotation specified\n'\
	'                                      \n'\
	'TypeOfDiskVelocityStructure    2      %  0 = spherically symmetric, isotropic\n'\
	'                                      %  1 = spherically symmetric, anisotropic (with beta parameter specified)\n'\
	'                                      %  2 = axisymmetric, f(E, Lz), with specified net rotation\n'\
	'                                      %  3 = axisymmetric, f(E, Lz, I_3), with <vz^2>/<vR^2> specified and net rotation specified\n'\
	'                                      \n'\
	'TypeOfBulgeVelocityStructure   0      %  0 = spherically symmetric, isotropic\n'\
	'                                      %  1 = spherically symmetric, anisotropic (with beta parameter specified)\n'\
	'                                      %  2 = axisymmetric, f(E, Lz), with specified net rotation\n'\
	'                                      %  3 = axisymmetric, f(E, Lz, I_3), with <vz^2>/<vR^2> specified and net rotation specified\n'\
	'	                                      \n'\
	'\n'\
	'HaloBetaParameter              0    %  only relevant for TypeOfHaloVelocityStructure=1\n'\
	'BulgeBetaParameter             0    %  only relevant for TypeOfBulgeVelocityStructure=1\n'\
	'\n'\
	'\n'\
	'HaloDispersionRoverZratio      4.0   %  only relevant for TypeOfHaloVelocityStructure=3\n'\
	'DiskDispersionRoverZratio      4.0   %  only relevant for TypeOfDiskVelocityStructure=3\n'\
	'BulgeDispersionRoverZratio     4.0   %  only relevant for TypeOfBulgeVelocityStructure=3\n'\
	'\n'\
	'\n'\
	'HaloStreamingVelocityParameter     0.0    %	gives the azimuthal streaming velocity in the TypeOf*VelocityStructure=2/3 cases (\'k parameter\')\n'\
	'DiskStreamingVelocityParameter     1.0    %	gives the azimuthal streaming velocity in the TypeOf*VelocityStructure=2/3 cases (\'k parameter\')\n'\
	'BulgeStreamingVelocityParameter    0.0    %	gives the azimuthal streaming velocity in the TypeOf*VelocityStructure=2/3 cases (\'k parameter\')\n'\
	'\n'\
	'\n'\
	'%------   Orbit integration accuracy\n'\
	'\n'\
	'TorbitFac                          10.0  %	regulates the integration time of orbits\n'\
	'                                         % (this is of the order of the typical number of orbits per particle)\n'\
	'TimeStepFactorOrbit                0.01\n'\
	'TimeStepFactorCellCross            0.25\n'\
	'\n'\
	'\n'\
	'%------   Iterative optimization parameters\n'\
	'\n'\
	'FractionToOptimizeIndependendly    0.001\n'\
	'IndepenentOptimizationsPerStep     100\n'\
	'StepsBetweenDump                   10\n'\
	'MaximumNumberOfSteps               100\n'\
	'\n'\
	'MinParticlesPerBinForDispersionMeasurement  100\n'\
	'MinParticlesPerBinForDensityMeasurement     50 \n'\
	'\n'\
	'\n'\
	'%------   Grid dimension and extenstion/resolution\n'\
	'\n'\
	'DG_MaxLevel    7\n'\
	'EG_MaxLevel    7\n'\
	'FG_Nbin        256                  % number of bins for the acceleration grid in the R- and z-directions \n'\
	'\n'\
	'\n'\
	'OutermostBinEnclosedMassFraction  0.999   %	regulates the fraction of mass of the Hernquist \n'\
	'                                          %	halo that must be inside the grid (determines grid extension)\n'\
	'\n'\
	'InnermostBinEnclosedMassFraction  0.0000001 %	regulates the fraction of mass enclosed by the innermost \n'\
	'				            % bin (regulates size of innermost grid cells)\n'\
	'\n'\
	'\n'\
	'\n'\
	'MaxVelInUnitsVesc                 0.9999  % maximum allowed velocity in units of the local escape velocity\n'\
	'\n'\
	'\n'\
	'%------   Construction of target density field\n'\
	'\n'\
	'SampleDensityFieldForTargetResponse 1               %	if set to 1, the code will randomly sample points to construct the density field\n'\
	'SampleParticleCount                 10000000       % number of points sampled for target density field \n'\
	'\n'\
	'\n'\
	'%------   Construction of force field\n'\
	'\n'\
	'SampleForceNhalo                    0               % number of points to use to for computing force field with a tree \n'\
	'SampleForceNdisk                    10000000\n'\
	'SampleForceNbulge                   0\n'\
	'\n'\
	'Softening                           0.05\n'\
	'\n'\
	'\n'\
	'%------   Accuracy settings of tree code used in construction of force field\n'\
	'\n'\
	'TypeOfOpeningCriterion    1\n'\
	'ErrTolTheta               0.4\n'\
	'ErrTolForceAcc            0.0025\n'\
	'\n'\
	'%------   Domain decomposition parameters used in parallel tree code\n'\
	'\n'\
	'MultipleDomains 4\n'\
	'TopNodeFactor   4\n'\
	'\n'\
	'\n'\
	'%------   Parallel I/O paramaters, only affects writing of galaxy files\n'\
	'\n'\
	'NumFilesPerSnapshot       1\n'\
	'NumFilesWrittenInParallel 1\n'\
	'\n'\
	'\n'\
	'%------   Memory allocation parameters\n'\
	'\n'\
	'MaxMemSize                2300.0    %	in MB\n'\
	'BufferSize                100.0\n'\
	'BufferSizeGravity         100.0\n'\
	'\n'\
	'\n'\
	'%------   Specification of internal system of units\n'\
	'\n'\
	'UnitLength_in_cm         3.085678e21        %  1.0 kpc\n'\
	'UnitMass_in_g            1.989e43           %  1.0e10 solar masses\n'\
	'UnitVelocity_in_cm_per_s 1e5                %  1 km/sec\n'\
	'GravityConstantInternal  0'\
	.format(mdisk = mdisk, mbulge = mbulge, Npart_disk = Npart_disk, 
		Npart_bulge= Npart_bulge, fbulge = fbulge))
	f.close()

	slurm_file = '{model_dir}/build_Nbody.slurm'.format(model_dir = model_dir)
	f = open(slurm_file, 'w')
	f.write(
	'#!/bin/bash\n'\
	'#SBATCH --nodes=1\n'\
	'#SBATCH --ntasks-per-node=24\n'\
	'#SBATCH --time=100:00:00\n'\
	'#SBATCH --output=Nbody.out\n'\
	'#SBATCH --error=Nbody.err\n'\
	'#SBATCH --mail-type=END,FAIL\n'\
	'#SBATCH --mail-user=adam.watts@icrar.org\n'\
	'#SBATCH --mem=10GB\n'\
	'\n'\
	'# export OMP_NUM_THREADS=1\n'\
	'\n'\
	'module load gsl/2.3\n'\
	# 'module load openmpi\n'\
	'export LD_LIBRARY_PATH=/home/awatts/local/lib\n'\
	'\n'\
	'mpirun -n 24 \\\n'\
	'	/home/awatts/Adam_PhD/models_fitting/GadgetSimulations/simulations/GalIC \\\n'\
	'	./Nbody.param \n'\
	'\n')#\
	# 'cp ./model_setup/snap_010.hdf5 ./Nbody_ICs.hdf5')
	f.close()

def write_buildgas_files(model_dir, mdisk, mstar, mbulge, fbulge, Npart_gas, halo_fbar, Npart_ghalo):
	
	filename = '{}/gas.param'.format(model_dir)
	f = open(filename, 'w')
	f.write(
	'&HALO \n'\
	'm200=?m200\n'\
	'c200=10\n'\
	'/\n'\
	'\n'\
	'&DISC\n'\
	'mdisc={mdisk:.4f}\n'\
	'fdisc=?fdisk\n'\
	'mstar={mstar:.4f}\n'\
	'fhole=0.005\n'\
	'/\n'\
	'\n'\
	'&BULGE\n'\
	'mbulge={mbulge:.4f}\n'\
	'fbulge={fbulge:.3f}\n'\
	'/\n'\
	'\n'\
	'&GAS_DISC\n'\
	'temp=5.d4\n'\
	'ndisc={Npart_gas}\n'\
	'/\n'\
	'\n'\
	'&GAS_HALO\n'\
	'fbaryon={halo_fbar}\n'\
	'nhalo={Npart_ghalo}\n'\
	'lambda_b=0.0\n'\
	'v_radial=0.0\n'\
	'/\n'\
	'\n'\
	'&BLACK_HOLE\n'\
	'mbh=0.0e-2\n'\
	'/\n'\
	'\n'\
	'&NBODY\n'\
	'ncomp=1\n'\
	'npart_comp(1)=100000\n'\
	'npart_comp(2)=100000\n'\
	'npart_comp(3)=0\n'\
	'fmass_comp(1)=1\n'\
	'fmass_comp(2)=0.035\n'\
	'fmass_comp(3)=0.0\n'\
	'form_comp(1)=\'hernquist\'\n'\
	'form_comp(2)=\'hernquist\'\n'\
	'form_comp(3)=\'hernquist\'\n'\
	'rscale_comp(1)=1.0\n'\
	'rscale_comp(2)=0.2\n'\
	'rscale_comp(3)=0.2\n'\
	'rtrunc_comp(1)=1.5\n'\
	'rtrunc_comp(2)=1.5\n'\
	'rtrunc_comp(3)=0.5\n'\
	'istaper=.false.\n'\
	'/\n'\
	'\n'\
	'&OUTPUT\n'\
	'disc_file=\'./model_setup/gas_disc\'\n'\
	'halo_file=\'./model_setup/gas_halo\'\n'\
	'nbody_file=\'nbody_halo\'\n'\
	'snap_format=3\n'\
	'/\n'\
	'\n'\
	'&MISC\n'\
	'ispoisson=.true.\n'\
	'/\n'.format(mdisk = mdisk, 
		mstar = mstar, mbulge = mbulge, Npart_gas = Npart_gas, Npart_ghalo = Npart_ghalo, 
		halo_fbar=halo_fbar, fbulge = fbulge))
	f.close()

	slurm_file = '{model_dir}/build_gas.slurm'.format(model_dir = model_dir)
	f = open(slurm_file, 'w')
	f.write(
	'#!/bin/bash\n'\
	'#SBATCH --nodes=1\n'\
	'#SBATCH --ntasks-per-node=1\n'\
	'#SBATCH --time=12:00:00\n'\
	'#SBATCH --output=gas.out\n'\
	'#SBATCH --error=gas.err\n'\
	'#SBATCH --mail-type=END,FAIL\n'\
	'#SBATCH --mail-user=adam.watts@icrar.org\n'\
	'#SBATCH --mem=4GB\n'\
	'\n'\
	'export LD_LIBRARY_PATH=/home/awatts/local/lib\n'\
	'\n'\
	'python /home/awatts/Adam_PhD/models_fitting/GadgetSimulations/simulations/get_structparams.py $PWD\n'\
	'\n'\
	'/home/awatts/Adam_PhD/models_fitting/GadgetSimulations/simulations/mk_gas_disc.exe ./gas.param\n'\
	)
	if halo_fbar != 0:
		f.write(
		'/home/awatts/Adam_PhD/models_fitting/GadgetSimulations/simulations/mk_gas_halo.exe ./gas.param\n'\
		'\n'\
		)
	f.write(
	'/home/awatts/Adam_PhD/models_fitting/GadgetSimulations/simulations/merge_comps.exe \\\n'\
	'-in_hdf5 -out_hdf5 \\\n'\
	'-in ./model_setup/gas_disc.gdt.hdf5 \\\n'\
	)
	if halo_fbar != 0:
		f.write(
		'-in ./model_setup/gas_halo.gdt.hdf5 \\\n'\
		)
	f.write(
	'-in ./model_setup/snap_010.hdf5 \\\n'\
 	'-out model_ICs.hdf5 \\\n'\
	'-reduce_stellar_mass 3 -reduce_stellar_mass_fraction {mstar:.4f} \\\n'\
	'-taper_profile -taper_mass 0.99\n'\
	'\n'\
	.format(mstar = mstar))

def write_Gadget_files(model_dir):
	param_file = '{}/run_gal.param'.format(model_dir)
	f = open(param_file, 'w')
	f.write(
	'%  Relevant files\n'\
	'\n'\
	'InitCondFile    ./model_ICs\n'\
	'OutputDir       ./snaps/\n'\
	'\n'\
	'EnergyFile        energy.txt\n'\
	'InfoFile          info.txt\n'\
	'TimingsFile       timings.txt\n'\
	'CpuFile           cpu.txt\n'\
	'\n'\
	'RestartFile       restart\n'\
	'SnapshotFileBase  snapshot\n'\
	'\n'\
	'OutputListFilename    Output/output_list.txt\n'\
	'\n'\
	'% CPU-time limit\n'\
	'\n'\
	'TimeLimitCPU      720000       \n'\
	'ResubmitOn        0\n'\
	'ResubmitCommand   xyz\n'\
	'\n'\
	'% Code options\n'\
	'\n'\
	'ICFormat                 3\n'\
	'ComovingIntegrationOn    0\n'\
	'TypeOfTimestepCriterion  0\n'\
	'OutputListOn             0\n'\
	'PeriodicBoundariesOn     0\n'\
	'CoolingOn                1\n'\
	'StarformationOn          1\n'\
	'\n'\
	'%  Caracteristics of run\n'\
	'\n'\
	'TimeBegin	      0.0\n'\
	'TimeMax	              0.5\n'\
	'\n'\
	'Omega0	              0\n'\
	'OmegaLambda           0\n'\
	'OmegaBaryon           0\n'\
	'HubbleParam         1.0\n'\
	'BoxSize               0\n'\
	'\n'\
	'% Output frequency\n'\
	'\n'\
	'TimeBetSnapshot        0.005\n'\
	'TimeOfFirstSnapshot    0.0\n'\
	'\n'\
	'CpuTimeBetRestartFile     600.0    ; here in seconds\n'\
	'TimeBetStatistics         0.001\n'\
	'\n'\
	'% Accuracy of time integration\n'\
	'\n'\
	'ErrTolIntAccuracy      0.01    %  used for TypeOfTimestepCritrion==0\n'\
	'CourantFac             0.4      %  for SPH\n'\
	'MaxSizeTimestep        0.005   % was 0.05\n'\
	'MinSizeTimestep        0.0\n'\
	'\n'\
	'TimeStepFac	       4.\n'\
	'\n'\
	'% Tree algorithm, force accuracy, domain update frequency\n'\
	'\n'\
	'ErrTolTheta            0.5\n'\
	'TypeOfOpeningCriterion 1\n'\
	'ErrTolForceAcc         0.03\n'\
	'\n'\
	'%  Further parameters of SPH\n'\
	'\n'\
	'DesNumNgb              200 \n'\
	'ArtBulkViscConst       1.e0\n'\
	'ArtBulkDissConst       1.e0\n'\
	'InitGasTemp            0    %  1.e4       %  always ignored if set to 0 \n'\
	'MinGasTemp             1.e2  \n'\
	'%  EquilibriumTemp        1.e4\n'\
	'%  MeanWeight             1.0\n'\
	'\n'\
	'% Memory allocation\n'\
	'\n'\
	'PartAllocFactor        4.0\n'\
	'%TreeAllocFactor       1.5\n'\
	'\n'\
	'% System of units\n'\
	'\n'\
	'UnitLength_in_cm         3.086e21       % 1 kpc\n'\
	'UnitMass_in_g            1.989e43       %  1e10 solar masses\n'\
	'UnitVelocity_in_cm_per_s 1.e5         %20733965.\n'\
	'\n'\
	'GravityConstantInternal  0\n'\
	'\n'\
	'% Softening lengths\n'\
	'\n'\
	'MinGasHsmlFractional     1.00 % minimum softening in terms of the gravitational \n'\
	'                                %  softening length\n'\
	'\n'\
	'SofteningGas       0.001\n'\
	'SofteningHalo      0.01 % times 2.8 gives 80 pc\n'\
	'SofteningDisk      0.001\n'\
	'SofteningBulge     0.001 %  used as photon/virtual particles\n'\
	'SofteningStars     0.001\n'\
	'SofteningBndry     0.001\n'\
	'\n'\
	'SofteningGasMaxPhys       0.001\n'\
	'SofteningHaloMaxPhys      0.01\n'\
	'SofteningDiskMaxPhys      0.001\n'\
	'SofteningBulgeMaxPhys     0.001 %  sets maximum search radius for photons\n'\
	'SofteningStarsMaxPhys     0.001\n'\
	'SofteningBndryMaxPhys     0.001\n'\
	'\n'\
	'% New Parameters\n'\
	'SnapFormat            3\n'\
	'TreeDomainUpdateFrequency    0.0\n'\
	'MaxRMSDisplacementFac  0.5\n'\
	'MaxNumNgbDeviation  1.\n'\
	'NumFilesPerSnapshot       1\n'\
	'NumFilesWrittenInParallel 1\n'\
	'BufferSize              100\n'\
	'\n'\
	'% Cooling Data\n'\
	'AssumedMetallicity        1.0e-1\n'\
	'TabulatedCoolingRatesFile  /home/awatts/Adam_PhD/models_fitting/GadgetSimulations/simulations/cooling_rates.txt\n'\
	'%  GrackleOn                 0\n'\
	'%  GrackleVerboseOn          0\n'\
	'%  GrackleChemistry          0\n'\
	'%  GrackleRadiativeCoolingOn 0\n'\
	'%  GrackleMetalCoolingOn     0\n'\
	'%  GrackleUVBackgroundOn     0\n'\
	'%  GrackleDataFile       /Users/cpower/Codes/grackle/input/CloudyData_noUVB.h5\n'\
	'\n'\
	'\n'\
	'% Star Formation \n'\
	'StellarNgbFactor          1\n'\
	'StellarFeedbackFactor     1\n'\
	'StellarMassMinimum        0.1\n'\
	'StellarMassMaximum        100.0\n'\
	'SlopeIMF                  2.35\n'\
	'\n'\
	'%  IMFType                   0\n'\
	'%  MetalsFeedbackFactor      1.0\n'\
	'\n'\
	'MaxTempThreshold          1.e4\n'\
	'SF_ThresholdNumberDensity 5.0e2\n'\
	'\n'\
	'%  Black Holes \n'\
	'%  BlackHoleAccretionRadius   1.0\n'\
	'%  BlackHoleAccretionFactor   1.0\n'\
	'%  BlackHoleFeedbackFactor    1.0\n'\
	'%  BlackHoleEddingtonFactor   1.0\n'\
	'%  BlackHoleNgbFactor         1.0')
	f.close()

	slurm_file = '{model_dir}/sub_model{BT}_{model}.slurm'.format(model_dir = model_dir,\
				BT = model_dir.split('/')[-2], model=model_dir.split('/')[-1])
	f = open(slurm_file, 'w')
	f.write(
	'#!/bin/bash\n'\
	'#SBATCH --nodes=1\n'\
	'#SBATCH --ntasks-per-node=24\n'\
	'#SBATCH --time=200:00:00\n'\
	'#SBATCH --output=galaxy.out\n'\
	'#SBATCH --error=galaxy.err\n'\
	'#SBATCH --mail-type=END,FAIL\n'\
	'#SBATCH --mail-user=adam.watts@icrar.org\n'\
	'#SBATCH --mem=15GB\n'\
	'\n'\
	'# export OMP_NUM_THREADS=1\n'\
	'\n'\
	'export LD_LIBRARY_PATH=/home/cpower/Codes/lib/\n'\
	'module load gsl/2.3\n'\
	'module load openmpi\n'\
	'module load hdf5\n'\
	'\n'\
	'mpirun -n 24 \\\n'\
	'	/home/awatts/Adam_PhD/models_fitting/GadgetSimulations/simulations/main_test \\\n'\
	'	./run_gal.param')
	f.close()
	
if __name__ == '__main__':
	main()



