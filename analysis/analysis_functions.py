import numpy as np 
from astropy.table import Table
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FormatStrFormatter
from matplotlib.colorbar import Colorbar
import h5py
import glob
from astropy.convolution import convolve, Gaussian2DKernel
from scipy.special import erf
from scipy.optimize import curve_fit
from mpi4py import MPI
import sys
import Find_H2_LGS
import galcalc_ARHS


def compare_pressures_spatial():
	basedir = sys.argv[1]
	theta = 90.				#inclination
	phi = 0.				#position angle
	view_theta = theta*np.pi/180.e0 
	view_phi = phi*np.pi/180.e0  

	unitmass = 1.e10

	snaplist = [0]

	for ii in range(len(snaplist)):
		tt = snaplist[ii]
		print(rank,tt)

		filename = '{dir}snaps/snapshot_{tt}.hdf5'.format(dir=basedir, tt=str(tt).zfill(3))
		file = h5py.File(filename,'r')
		head = file['Header']
		DM = file['PartType1']
		disk = file['PartType2']
		gas = file['PartType0']
		# newstars = file['PartType5']
		parttypes = list(file.keys())
		# exit()
		DM_coordinates, DM_masses, DM_velocities = particle_info(DM, unitmass)
		gas_coordinates, gas_masses, gas_velocities, gas_neutral_masses, \
			gas_densities, gas_internal_energy = particle_info(gas, unitmass, gas=True)
		disk_coordinates, disk_masses, disk_velocities = particle_info(disk, unitmass)

		if 'PartType3' in parttypes:
			bulge = file['PartType3']
			bulge_coordinates, bulge_masses ,bulge_velocities = particle_info(bulge, unitmass)

			stars_coordinates = np.append(disk_coordinates, bulge_coordinates, axis=0)
			stars_masses = np.append(disk_masses, bulge_masses)
			stars_velocities = np.append(disk_velocities, bulge_velocities)
		else:
			stars_coordinates, stars_masses, stars_velocities = particle_info(disk, unitmass)
	
		if 'PartType5' in parttypes:
			newstars = file['PartType5']
			newstars_coordinates, newstars_masses, newstars_velocities = particle_info(newstars, unitmass)
		
			stars_coordinates = np.append(stars_coordinates, newstars_coordinates, axis=0)
			stars_masses = np.append(stars_masses, newstars_masses)
			stars_velocities = np.append(stars_velocities, newstars_velocities)


		print('DM mass[1.e12]',np.nansum(DM_masses)/1.e12)
		print('stellar mass[1.e10]',np.nansum(stars_masses)/1.e10)
		print('Total gas mass[1.e10]', np.nansum(gas_masses)/1.e10)
		print('Stellar fraction',np.nansum(stars_masses)/np.nansum(DM_masses))
		print('Total Gas fraction',np.nansum(gas_masses)/np.nansum(stars_masses))
				
		COM_DM = calc_COM(DM_coordinates,DM_masses,5.e3)

		DM_coordinates -= COM_DM
		stars_coordinates -= COM_DM
		disk_coordinates -= COM_DM
		gas_coordinates -= COM_DM
		if 'PartType3' in parttypes:
			bulge_coordinates -= COM_DM

		gas_coords_edgeon = calc_coords_obs(gas_coordinates, 0, 90)
		gas_coords_faceon = calc_coords_obs(gas_coordinates, 0, 0)
		gas_vel_edgeon = calc_vel_obs(gas_velocities, 0 , 90)

		gas_Pextk_MP = calc_Pextk_midpressure(stars_coordinates, stars_masses, gas_coordinates, gas_neutral_masses)
		
		gas_Pextk_particle = calc_Pextk_densityenergy(gas_densities, gas_internal_energy)

		compare_pressures_spatial(filename,gas_coords_edgeon, gas_Pextk_MP, gas_Pextk_particle, gas_neutral_masses/gas_masses)

def compare_scaleheights():
	basedir = sys.argv[1]
	theta = 90.				#inclination
	phi = 0.				#position angle
	view_theta = theta*np.pi/180.e0 
	view_phi = phi*np.pi/180.e0  

	unitmass = 1.e10

	# create_gif(basedir)
	# exit()
	# snaplist = [10,50,80,70,80,90,100]
	snaplist = np.arange(0,102,2)
	scaleheights_disk = []
	scaleheights_all = []
	scaleheights_percentile = []
	for ii in range(len(snaplist)):
		tt = snaplist[ii]
		print(rank,tt)


		filename = '{dir}snaps/snapshot_{tt}.hdf5'.format(dir=basedir, tt=str(tt).zfill(3))
		file = h5py.File(filename,'r')
		head = file['Header']
		DM = file['PartType1']
		disk = file['PartType2']
		gas = file['PartType0']
		# newstars = file['PartType5']
		parttypes = list(file.keys())
		# exit()
		DM_coordinates, DM_masses, DM_velocities = particle_info(DM, unitmass)
		gas_coordinates, gas_masses, gas_velocities, gas_neutral_masses, \
			gas_densities, gas_internal_energy = particle_info(gas, unitmass, gas=True)
		disk_coordinates, disk_masses, disk_velocities = particle_info(disk, unitmass)

		if 'PartType3' in parttypes:
			bulge = file['PartType3']
			bulge_coordinates, bulge_masses ,bulge_velocities = particle_info(bulge, unitmass)

			stars_coordinates = np.append(disk_coordinates, bulge_coordinates, axis=0)
			stars_masses = np.append(disk_masses, bulge_masses)
			stars_velocities = np.append(disk_velocities, bulge_velocities)
		else:
			stars_coordinates, stars_masses, stars_velocities = particle_info(disk, unitmass)
	
		if 'PartType5' in parttypes:
			newstars = file['PartType5']
			newstars_coordinates, newstars_masses, newstars_velocities = particle_info(newstars, unitmass)
		
			stars_coordinates = np.append(stars_coordinates, newstars_coordinates, axis=0)
			stars_masses = np.append(stars_masses, newstars_masses)
			stars_velocities = np.append(stars_velocities, newstars_velocities)


		print('DM mass[1.e12]',np.nansum(DM_masses)/1.e12)
		print('stellar mass[1.e10]',np.nansum(stars_masses)/1.e10)
		print('Total gas mass[1.e10]', np.nansum(gas_masses)/1.e10)
		print('Stellar fraction',np.nansum(stars_masses)/np.nansum(DM_masses))
		print('Total Gas fraction',np.nansum(gas_masses)/np.nansum(stars_masses))
				
		COM_DM = calc_COM(DM_coordinates,DM_masses,5.e3)

		DM_coordinates -= COM_DM
		stars_coordinates -= COM_DM
		disk_coordinates -= COM_DM
		gas_coordinates -= COM_DM
		# if 'PartType3' in parttypes:
			# bulge_coordinates -= COM_DM

		hstar = hstar_from_radfit(disk_coordinates, disk_masses, 40)
		scaleheights_disk.extend([hstar])
		hstar = hstar_from_radfit(stars_coordinates, stars_masses, 40)
		scaleheights_all.extend([hstar])
		hstar = radial_scaleheight(stars_coordinates, stars_masses, 40)
		scaleheights_percentile.extend([hstar])


	plt.scatter(snaplist,scaleheights_disk, label = 'disk particles')
	plt.scatter(snaplist,scaleheights_all, label = 'all particles')
	plt.scatter(snaplist,scaleheights_percentile, label = 'all particles: percentile')
	plt.xlabel('Snapshot')
	plt.ylabel('stellar scaleheight')
	plt.legend()
	plt.show()
	exit()

def analyse_datacube():

	comm = MPI.COMM_WORLD
	rank = comm.Get_rank()
	nproc = comm.Get_size()
	basedir = sys.argv[1]
	theta = 90.				#inclination
	phi = 0.				#position angle
	view_theta = theta*np.pi/180.e0 
	view_phi = phi*np.pi/180.e0  

	unitmass = 1.e10

	# create_gif(basedir)
	# exit()
	for tt in [0,10,20,30,40,50,60,70,80,90,100]:#range(rank,101,nproc):

		spacebins, velbins, datacube, mjy_conv = read_datacube(tt, basedir)
		plot_datacube(tt, basedir, spacebins, velbins, datacube, mjy_conv)

def create_datacube():

	comm = MPI.COMM_WORLD
	rank = comm.Get_rank()
	nproc = comm.Get_size()
	basedir = sys.argv[1]
	theta = 90.				#inclination
	phi = 0.				#position angle
	view_theta = theta*np.pi/180.e0 
	view_phi = phi*np.pi/180.e0  

	unitmass = 1.e10

	# create_gif(basedir)
	# exit()
	# snaplist = [10,50,80,70,80,90,100]
	snaplist = np.arange(0,102,2)
	for ii in range(rank,len(snaplist),nproc):
		tt = snaplist[ii]
		print(rank,tt)

		filename = '{dir}snaps/snapshot_{tt}.hdf5'.format(dir=basedir, tt=str(tt).zfill(3))
		file = h5py.File(filename,'r')
		head = file['Header']
		DM = file['PartType1']
		disk = file['PartType2']
		gas = file['PartType0']
		# newstars = file['PartType5']
		parttypes = list(file.keys())
		# exit()
		DM_coordinates, DM_masses, DM_velocities = particle_info(DM, unitmass)
		gas_coordinates, gas_masses, gas_velocities, gas_neutral_masses, \
			gas_densities, gas_internal_energy = particle_info(gas, unitmass, gas=True)
		disk_coordinates, disk_masses, disk_velocities = particle_info(disk, unitmass)

		if 'PartType3' in parttypes:
			bulge = file['PartType3']
			bulge_coordinates, bulge_masses ,bulge_velocities = particle_info(bulge, unitmass)

			stars_coordinates = np.append(disk_coordinates, bulge_coordinates, axis=0)
			stars_masses = np.append(disk_masses, bulge_masses)
			stars_velocities = np.append(disk_velocities, bulge_velocities)
		else:
			stars_coordinates, stars_masses, stars_velocities = particle_info(disk, unitmass)
	
		if 'PartType5' in parttypes:
			newstars = file['PartType5']
			newstars_coordinates, newstars_masses, newstars_velocities = particle_info(newstars, unitmass)
		
			stars_coordinates = np.append(stars_coordinates, newstars_coordinates, axis=0)
			stars_masses = np.append(stars_masses, newstars_masses)
			stars_velocities = np.append(stars_velocities, newstars_velocities)


		print('DM mass[1.e12]',np.nansum(DM_masses)/1.e12)
		print('stellar mass[1.e10]',np.nansum(stars_masses)/1.e10)
		print('Total gas mass[1.e10]', np.nansum(gas_masses)/1.e10)
		print('Stellar fraction',np.nansum(stars_masses)/np.nansum(DM_masses))
		print('Total Gas fraction',np.nansum(gas_masses)/np.nansum(stars_masses))
				
		COM_DM = calc_COM(DM_coordinates,DM_masses,5.e3)

		DM_coordinates -= COM_DM
		stars_coordinates -= COM_DM
		disk_coordinates -= COM_DM
		gas_coordinates -= COM_DM
		if 'PartType3' in parttypes:
			bulge_coordinates -= COM_DM

		gas_coords_edgeon = calc_coords_obs(gas_coordinates, 0, 90)
		gas_coords_faceon = calc_coords_obs(gas_coordinates, 0, 0)
		gas_vel_edgeon = calc_vel_obs(gas_velocities, 0 , 90)

		gas_Pextk = calc_Pextk_midpressure(stars_coordinates, stars_masses, gas_coordinates, gas_neutral_masses)
		gas_Rmol = calc_Rmol(gas_Pextk)
		
		# gas_Pextk = calc_Pextk_densityenergy(gas_densities, gas_internal_energy)
		# Rmol_part = calc_Rmol(gas_Pextk)


		HI_masses = gas_neutral_masses / (1.e0 + Rmol_part)
		H2_masses = gas_neutral_masses - HI_masses

		create_datacube(tt, filename, 45, gas_coordinates, gas_velocities, HI_masses)
		create_datacube(tt, filename, 90, gas_coordinates, gas_velocities, HI_masses)

def measure_controlled_run():

	comm = MPI.COMM_WORLD
	rank = comm.Get_rank()
	nproc = comm.Get_size()
	basedir = sys.argv[1]
	theta = 90.				#inclination
	phi = 0.				#position angle
	view_theta = theta*np.pi/180.e0 
	view_phi = phi*np.pi/180.e0  

	unitmass = 1.e10

	# create_gif(basedir)
	# exit()
	# snaplist = [10,50,80,70,80,90,100]
	snaplist = np.arange(0,101)
	snaplist = [16]

	for ii in range(rank,len(snaplist),nproc):
		tt = snaplist[ii]
		print(rank,tt)

		filename = '{dir}snaps/snapshot_{tt}.hdf5'.format(dir=basedir, tt=str(tt).zfill(3))
		file = h5py.File(filename,'r')
		parttypes = list(file.keys())
		head = file['Header']
		DM = file['PartType1']
		disk = file['PartType2']
		gas = file['PartType0']

		[DM_coordinates, DM_masses, DM_velocities] = \
			particle_info(DM, unitmass, ['Coordinates','Masses','Velocities'])
		[gas_coordinates, gas_masses, gas_velocities, gas_densities,gas_neutral_fraction, \
			gas_internal_energy] = particle_info(gas, unitmass, ['Coordinates','Masses',\
				'Velocities','Density','NeutralHydrogenAbundance','InternalEnergy'])
		gas_neutral_masses = gas_masses*gas_neutral_fraction

		[disk_coordinates, disk_masses, disk_velocities] = \
			particle_info(disk, unitmass, ['Coordinates','Masses','Velocities'])

		if 'PartType3' in parttypes:
			bulge = file['PartType3']
			[bulge_coordinates, bulge_masses, bulge_velocities] = \
			particle_info(bulge, unitmass, ['Coordinates','Masses','Velocities'])

			stars_coordinates = np.append(disk_coordinates, bulge_coordinates, axis=0)
			stars_masses = np.append(disk_masses, bulge_masses)
			stars_velocities = np.append(disk_velocities, bulge_velocities)
		else:
			stars_coordinates, stars_masses, stars_velocities = \
					particle_info(disk, unitmass, ['Coordinates','Masses','Velocities'])
	
		if 'PartType5' in parttypes:
			newstars = file['PartType5']

			[newstars_coordinates, newstars_masses, newstars_velocities] = \
			particle_info(newstars, unitmass, ['Coordinates','Masses','Velocities'])

			stars_coordinates = np.append(stars_coordinates, newstars_coordinates, axis=0)
			stars_masses = np.append(stars_masses, newstars_masses)
			stars_velocities = np.append(stars_velocities, newstars_velocities)

		# print('DM mass[1.e12]',np.nansum(DM_masses)/1.e12)
		# print('stellar mass[1.e10]',np.nansum(stars_masses)/1.e10)
		# print('Total gas mass[1.e10]', np.nansum(gas_masses)/1.e10)
		# print('Stellar fraction',np.nansum(stars_masses)/np.nansum(DM_masses))
		# print('Total Gas fraction',np.nansum(gas_masses)/np.nansum(stars_masses))
				
		COM_DM = calc_COM(DM_coordinates,DM_masses,5.e3)

		DM_coordinates -= COM_DM
		stars_coordinates -= COM_DM
		disk_coordinates -= COM_DM
		gas_coordinates -= COM_DM
		if 'PartType3' in parttypes:
			bulge_coordinates -= COM_DM

		gas_temps = gas_internal_energy * 1.e5*1.e5 * 1.67e-24 / ((5./3. - 1) *1.381e-16) 	
		print(gas_temps)


		offender = np.where(np.array(gas['ParticleIDs']) == 82454)[0]
		print(gas_coordinates[offender])
		print(gas_velocities[offender])
		print(gas_densities[offender])
		print(gas_internal_energy[offender])
		print(gas_temps[offender])

		print(np.nanmax(gas_densities),np.nanmin(gas_densities))
		print(np.nanmax(gas_internal_energy),np.nanmin(gas_internal_energy))
		print(np.nanmax(gas_temps),np.nanmin(gas_temps))

		plt.scatter(gas_coordinates[:,0],gas_coordinates[:,1],s=0.05)
		plt.scatter(gas_coordinates[offender,0],gas_coordinates[offender,1],color='Red',s=2)
		plt.show()


		# radial_scaleheight(stars_coordinates, stars_masses, 40)

		# gas_coords_edgeon = calc_coords_obs(gas_coordinates, 0, 90)
		# gas_coords_faceon = calc_coords_obs(gas_coordinates, 0, 0)
		# gas_vel_edgeon = calc_vel_obs(gas_velocities, 0 , 90)

		# gas_Pextk = calc_Pextk_midpressure(stars_coordinates, stars_masses, gas_coordinates, gas_neutral_masses)
		# gas_Rmol = calc_Rmol(gas_Pextk)
		
		gas_Pextk = calc_Pextk_densityenergy(gas_densities, gas_internal_energy)
		Rmol_part = calc_Rmol(gas_Pextk)
		HI_masses = gas_neutral_masses / (1.e0 + Rmol_part)
		H2_masses = gas_neutral_masses - HI_masses 

		# spectrum_spatial_contribution(gas_coordinates, gas_velocities, HI_masses)

		# map_asymmetry_viewangle(tt, gas_coordinates, gas_velocities, HI_masses, save = basedir)
		file = '/media/data/simulations/parameterspace_models/iso_fbar0.01_BT0_FB0_GF10/data/snaps013_Afr_viewgrid.dat'
		data = np.loadtxt(file)
		# map_asymmetry_viewangle(tt, gas_coordinates, gas_velocities, HI_masses,save=True)
		map_asymmetry_viewangle(data, -1,-1,-1,save='view')

		# print(np.nanmax(HI_masses),np.nanmin(HI_masses))
		# print('total HI mass',np.nansum(HI_masses),'fraction=',np.nansum(HI_masses)/np.nansum(stars_masses))
		# print('total H2 mass',np.nansum(H2_masses),'fraction=',np.nansum(H2_masses)/np.nansum(stars_masses))

		# plot_spatial_radial_spectrum(0, Rvir, gas_coordinates, gas_velocities, HI_masses, H2_masses)

		# np.savetxt('{dir}/data/snap{tt}_vel.txt'.format(dir=basedir,tt=str(tt).zfill(3)),vel)
		# np.savetxt('{dir}/data/snap{tt}_rad_points.txt'.format(dir=basedir,tt=str(tt).zfill(3)),rad_points)
		# np.savetxt('{dir}/data/snap{tt}_spec.txt'.format(dir=basedir,tt=str(tt).zfill(3)),spectrum)
		# np.savetxt('{dir}/data/snap{tt}_spec_all.txt'.format(dir=basedir,tt=str(tt).zfill(3)),spectrum_all)
		# np.savetxt('{dir}/data/snap{tt}_sigma_all.txt'.format(dir=basedir,tt=str(tt).zfill(3)),sigma_all)
		# np.savetxt('{dir}/data/snap{tt}_sigma.txt'.format(dir=basedir,tt=str(tt).zfill(3)),sigma_HI)
		# np.savetxt('{dir}/data/snap{tt}_RC_all.txt'.format(dir=basedir,tt=str(tt).zfill(3)),RC_all)
		# np.savetxt('{dir}/data/snap{tt}_RC_HI.txt'.format(dir=basedir,tt=str(tt).zfill(3)),RC_HI)
		# np.savetxt('{dir}/data/snap{tt}_spacebins.txt'.format(dir=basedir,tt=str(tt).zfill(3)),spacebins)
		# np.savetxt('{dir}/data/snap{tt}_H2faceon_mom0.txt'.format(dir=basedir,tt=str(tt).zfill(3)),H2faceon_mom0)
		# np.savetxt('{dir}/data/snap{tt}_H2edgeon_mom0.txt'.format(dir=basedir,tt=str(tt).zfill(3)),H2edgeon_mom0)
		# np.savetxt('{dir}/data/snap{tt}_HIfaceon_mom0.txt'.format(dir=basedir,tt=str(tt).zfill(3)),HIfaceon_mom0)
		# np.savetxt('{dir}/data/snap{tt}_HIedgeon_mom0.txt'.format(dir=basedir,tt=str(tt).zfill(3)),HIedgeon_mom0)

		# vel = np.loadtxt('{dir}/data/snap{tt}_vel.txt'.format(dir=basedir,tt=str(tt).zfill(3)))
		# rad_points = np.loadtxt('{dir}/data/snap{tt}_rad_points.txt'.format(dir=basedir,tt=str(tt).zfill(3)))
		# spectrum = np.loadtxt('{dir}/data/snap{tt}_spec.txt'.format(dir=basedir,tt=str(tt).zfill(3)))
		# spectrum_all = np.loadtxt('{dir}/data/snap{tt}_spec_all.txt'.format(dir=basedir,tt=str(tt).zfill(3)))
		# sigma_all = np.loadtxt('{dir}/data/snap{tt}_sigma_all.txt'.format(dir=basedir,tt=str(tt).zfill(3)))
		# sigma_HI = np.loadtxt('{dir}/data/snap{tt}_sigma.txt'.format(dir=basedir,tt=str(tt).zfill(3)))
		# RC_all = np.loadtxt('{dir}/data/snap{tt}_RC_all.txt'.format(dir=basedir,tt=str(tt).zfill(3)))
		# RC_HI = np.loadtxt('{dir}/data/snap{tt}_RC_HI.txt'.format(dir=basedir,tt=str(tt).zfill(3)))
		# spacebins = np.loadtxt('{dir}/data/snap{tt}_spacebins.txt'.format(dir=basedir,tt=str(tt).zfill(3)))
		# H2faceon_mom0 = np.loadtxt('{dir}/data/snap{tt}_H2faceon_mom0.txt'.format(dir=basedir,tt=str(tt).zfill(3)))
		# H2edgeon_mom0 = np.loadtxt('{dir}/data/snap{tt}_H2edgeon_mom0.txt'.format(dir=basedir,tt=str(tt).zfill(3)))
		# HIfaceon_mom0 = np.loadtxt('{dir}/data/snap{tt}_HIfaceon_mom0.txt'.format(dir=basedir,tt=str(tt).zfill(3)))
		# HIedgeon_mom0 = np.loadtxt('{dir}/data/snap{tt}_HIedgeon_mom0.txt'.format(dir=basedir,tt=str(tt).zfill(3)))

def plot_controlled_run():

	comm = MPI.COMM_WORLD
	rank = comm.Get_rank()
	nproc = comm.Get_size()
	basedir = sys.argv[1]
	theta = 90.				#inclination
	phi = 0.				#position angle
	view_theta = theta*np.pi/180.e0 
	view_phi = phi*np.pi/180.e0  

	unitmass = 1.e10

	
	snaplist = np.arange(0,102,2)
	for ii in range(rank,len(snaplist),nproc):
		tt = snaplist[ii]
	
		vel = np.loadtxt('{dir}/data/snap{tt}_vel.txt'.format(dir=basedir,tt=str(tt).zfill(3)))
		rad_points = np.loadtxt('{dir}/data/snap{tt}_rad_points.txt'.format(dir=basedir,tt=str(tt).zfill(3)))
		spectrum = np.loadtxt('{dir}/data/snap{tt}_spec.txt'.format(dir=basedir,tt=str(tt).zfill(3)))
		spectrum_all = np.loadtxt('{dir}/data/snap{tt}_spec_all.txt'.format(dir=basedir,tt=str(tt).zfill(3)))
		sigma_all = np.loadtxt('{dir}/data/snap{tt}_sigma_all.txt'.format(dir=basedir,tt=str(tt).zfill(3)))
		sigma_HI = np.loadtxt('{dir}/data/snap{tt}_sigma.txt'.format(dir=basedir,tt=str(tt).zfill(3)))
		RC_all = np.loadtxt('{dir}/data/snap{tt}_RC_all.txt'.format(dir=basedir,tt=str(tt).zfill(3)))
		RC_HI = np.loadtxt('{dir}/data/snap{tt}_RC_HI.txt'.format(dir=basedir,tt=str(tt).zfill(3)))
		spacebins = np.loadtxt('{dir}/data/snap{tt}_spacebins.txt'.format(dir=basedir,tt=str(tt).zfill(3)))
		H2faceon_mom0 = np.loadtxt('{dir}/data/snap{tt}_H2faceon_mom0.txt'.format(dir=basedir,tt=str(tt).zfill(3)))
		H2edgeon_mom0 = np.loadtxt('{dir}/data/snap{tt}_H2edgeon_mom0.txt'.format(dir=basedir,tt=str(tt).zfill(3)))
		HIfaceon_mom0 = np.loadtxt('{dir}/data/snap{tt}_HIfaceon_mom0.txt'.format(dir=basedir,tt=str(tt).zfill(3)))
		HIedgeon_mom0 = np.loadtxt('{dir}/data/snap{tt}_HIedgeon_mom0.txt'.format(dir=basedir,tt=str(tt).zfill(3)))


		PeaklocL, PeaklocR = locate_peaks(spectrum)
		widths = locate_width(spectrum, [spectrum[PeaklocL],spectrum[PeaklocR]], 0.2)
		Sint, Afr = areal_asymmetry(spectrum, widths, np.abs(np.diff(vel)[0]))

		plot_spatial_radial_spectrum(tt,rad_points,sigma_HI,sigma_all,RC_HI,RC_all,vel,spectrum,\
								spectrum_all,widths, spacebins,HIfaceon_mom0,HIedgeon_mom0,\
								H2faceon_mom0,H2edgeon_mom0,save=basedir)

def asymmetry_time():

	dir1 = '/home/awatts/Adam_PhD/models_fitting/GadgetSimulations/simulations/parameterspace_models/iso_fbar0.01_BT0_FB0_GF10/'
	dir2 = '/home/awatts/Adam_PhD/models_fitting/GadgetSimulations/simulations/parameterspace_models/iso_fbar0.01_BT50_FB20_GF10/'
	dir3 = '/home/awatts/Adam_PhD/models_fitting/GadgetSimulations/simulations/parameterspace_models/iso_fbar0.01_BT50_FB5_GF10/'

	snaplist = np.arange(0,102,2)
	dirs = [dir1,dir2,dir3]
	Afr_vals = np.zeros([len(snaplist), 3])
	for ii in range(len(dirs)):
		basedir = dirs[ii]
		for jj in range(len(snaplist)):
			tt = snaplist[jj]
	

			vel = np.loadtxt('{dir}/data/snap{tt}_vel.txt'.format(dir=basedir,tt=str(tt).zfill(3)))
			spectrum = np.loadtxt('{dir}/data/snap{tt}_spec.txt'.format(dir=basedir,tt=str(tt).zfill(3)))

			PeaklocL, PeaklocR = locate_peaks(spectrum)
			widths = locate_width(spectrum, [spectrum[PeaklocL],spectrum[PeaklocR]], 0.2)
			Sint, Afr = areal_asymmetry(spectrum, widths, np.abs(np.diff(vel)[0]))

			Afr_vals[jj,ii] = Afr

	time = 5*snaplist
	plt.plot(time,Afr_vals[:,0], label = 'B/T = 0')
	plt.plot(time,Afr_vals[:,1], label = 'B/T = 50, extended')
	plt.plot(time,Afr_vals[:,2], label = 'B/T = 50, compact')
	plt.xlabel('Time [Myr]')
	plt.ylabel('Asymmetry Measure $A_{{fr}}$')
	plt.legend()
	plt.show()

	# plt.plot(vel, spectrum)
	# plt.plot([vel[int(widths[0])],vel[int(widths[0])]],[0,40])
	# plt.plot([vel[int(widths[1])],vel[int(widths[1])]],[0,40])
	# plt.show()
	# print(Afr)

def hydro_run():
	theta = 90.				#inclination
	phi = 0.				#position angle
	view_theta = theta*np.pi/180.e0 
	view_phi = phi*np.pi/180.e0  

	unitmass = 1.e10
	Rvir = 0.2697

	COD = np.array([13.3031, 34.7678, 40.9624])
	COM = np.array([13.2992, 34.7688, 40.9632])
	COV = np.array([-33.0769, -5.8676, -98.8213])
	COM_offset = np.array([13.3016, 34.7614, 40.9740])
	COV_offset = np.array([-34.6948,  6.1456, -95.8170])

	filename = '/home/awatts/Adam_PhD/models_fitting/GadgetSimulations/hydro_snapshot/snapshot_199.hdf5'
	file = h5py.File(filename, 'r')

	head = file['Header']
	DM = file['PartType1']
	stars = file['PartType5']
	gas = file['PartType0']

	DM_coordinates = np.array(DM['Coordinates'])
	stars_coordinates = np.array(stars['Coordinates'])
	gas_coordinates = np.array(gas['Coordinates'])

	DM_coordinates -= COM
	stars_coordinates -= COM
	gas_coordinates -= COM

	DM_radii = np.sqrt(np.nansum(DM_coordinates**2.e0, axis=1))
	stars_radii = np.sqrt(np.nansum(stars_coordinates**2.e0, axis=1))
	gas_radii = np.sqrt(np.nansum(gas_coordinates**2.e0, axis=1))

	DM_virial = np.where(DM_radii <= Rvir)[0]
	stars_virial = np.where(stars_radii <= Rvir)[0]
	gas_virial = np.where(gas_radii <= Rvir)[0]

	DM_coordinates = np.array(DM['Coordinates'])[DM_virial]
	DM_masses = np.array(DM['Masses'])[DM_virial] * unitmass
	DM_velocities = np.array(DM['Velocities'])[DM_virial]

	stars_coordinates = np.array(stars['Coordinates'])[stars_virial]
	stars_masses = np.array(stars['Masses'])[stars_virial]* unitmass
	stars_velocities = np.array(stars['Velocities'])[stars_virial]

	gas_coordinates = np.array(gas['Coordinates'])[gas_virial]
	gas_masses = np.array(gas['Masses'])[gas_virial]* unitmass
	gas_neutral_masses = np.array(gas['NeutralHydrogenAbundance'])[gas_virial] * gas_masses
	gas_velocities = np.array(gas['Velocities'])[gas_virial]
	gas_internal_energy = np.array(gas['InternalEnergy'])[gas_virial]
	gas_densities = np.array(gas['Density'])[gas_virial]


	print('DM mass[1.e12]',np.nansum(DM_masses)/1.e12)
	print('stellar mass[1.e10]',np.nansum(stars_masses)/1.e10)
	print('Total gas mass[1.e10]', np.nansum(gas_masses)/1.e10)
	print('Stellar fraction',np.nansum(stars_masses)/np.nansum(DM_masses))
	print('Total Gas fraction',np.nansum(gas_masses)/np.nansum(stars_masses))
				


	DM_radii = DM_radii[DM_virial]
	stars_radii = stars_radii[stars_virial]
	gas_radii = gas_radii[gas_virial]

	DM_coordinates -= COM
	stars_coordinates -= COM
	gas_coordinates -= COM

	COM_gas = calc_COM(gas_coordinates, gas_masses, Rvir)
	COM_stars = calc_COM(stars_coordinates, stars_masses, Rvir)

	gas_coordinates -= COM_gas
	stars_coordinates -= COM_stars

	gas_eigvec = diagonalise_inertia(gas_coordinates, gas_masses, Rvir)
	gas_coordinates = gas_coordinates @ gas_eigvec
	stars_coordinates = stars_coordinates @ gas_eigvec

	gas_coordinates *= 1.e3
	stars_coordinates *= 1.e3
	Rvir *=1.e3

	gas_velocities -= COV
	gas_velocities = gas_velocities @ gas_eigvec

	gas_radii =  np.sqrt(np.nansum(gas_coordinates**2.e0, axis=1))

	gas_coords_faceon = calc_coords_obs(gas_coordinates, 0, 0)
	gas_coords_edgeon = calc_coords_obs(gas_coordinates, 0, 90)
	gas_vel_edgeon = calc_vel_obs(gas_velocities, 0 , 90)


	# plt.scatter(stars_coordinates[:,0],stars_coordinates[:,2],s=0.01)
	# plt.xlim([-40,40])
	# plt.ylim([-40,40])
	# plt.show()


	radial_scaleheight(stars_coordinates, stars_masses, 0.1*Rvir)


	gas_Pextk = calc_Pextk_densityenergy(gas_densities, gas_internal_energy, lenunit = 'mpc')

	Rmol = calc_Rmol(gas_Pextk)
	HI_masses = gas_neutral_masses / (1.e0 + Rmol)
	H2_masses = gas_neutral_masses - HI_masses

	plt.hist(np.log10(gas_Pextk))
	plt.show()
	plt.close()

	gas_Pextk = calc_Pextk_midpressure(stars_coordinates, stars_masses, gas_coordinates, gas_neutral_masses)
	Rmol = calc_Rmol(gas_Pextk)
	HI_masses = gas_neutral_masses / (1.e0 + Rmol)
	H2_masses = gas_neutral_masses - HI_masses

	plt.hist(np.log10(gas_Pextk[gas_Pextk>0]))
	plt.show()

	print(np.nanmax(HI_masses),np.nanmin(HI_masses))
	print('total HI mass',np.nansum(HI_masses),'fraction=',np.nansum(HI_masses)/np.nansum(stars_masses))
	print('total H2 mass',np.nansum(H2_masses),'fraction=',np.nansum(H2_masses)/np.nansum(stars_masses))

	vel, spectrum = calc_spectrum(gas_coords_edgeon, gas_vel_edgeon, HI_masses)
	vel, spectrum_all = calc_spectrum(gas_coords_edgeon, gas_vel_edgeon, gas_neutral_masses)

	# PeaklocL, PeaklocR = locate_peaks(spectrum)
	# widths = locate_width(spectrum, [spectrum[PeaklocL],spectrum[PeaklocR]], 0.2)
	# Sint, Afr = areal_asymmetry(spectrum, widths, np.abs(np.diff(vel)[0]))

	widths = [0,0]

	plot_spatial_radial_spectrum(0, Rvir, gas_coordinates, gas_velocities, gas_neutral_masses,\
	Rmol, vel, spectrum, spectrum_all, widths)

def TNGsnap():

	Rvir = 250
	# unitmass = 1.e10

	base = '/home/awatts/Adam_PhD/models_fitting/GadgetSimulations/hydro_snapshot/556247'

	f = open(base + '_stars.txt')
	names = f.readline().split(', ')
	f.close()
	names[0] = names[0].split('# ')[-1]
	names[-1] = names[-1].split('\n')[0]

	stars = np.loadtxt(base + '_stars.txt',skiprows=2)
	stars_coordinates = stars[:,1:4]
	stars_velocities = stars[:,4:7]
	stars_masses = stars[:,7] #* unitmass

	gas = np.loadtxt(base + '_gas.txt',skiprows=2)
	gas_coordinates = gas[:,1:4]
	gas_velocities = gas[:,4:7]
	gas_masses = gas[:,7] #* unitmass
	gas_neutral_masses = gas[:,8] * gas_masses
	HImass_GK11 = gas[:,9]
	HImass_GD14 = gas[:,10]
	HImass_K13 = gas[:,11]
	H2mass_GK11 = gas[:,12]
	H2mass_GD14 = gas[:,13]
	H2mass_K13 = gas[:,14]



	COM_gas = calc_COM(gas_coordinates, gas_masses, Rvir)
	COM_stars = calc_COM(stars_coordinates, stars_masses, Rvir)
	gas_coordinates -= COM_gas
	stars_coordinates -= COM_stars

	gas_eigvec = diagonalise_inertia(gas_coordinates, gas_masses, Rvir)
	gas_coordinates = gas_coordinates @ gas_eigvec
	stars_coordinates = stars_coordinates @ gas_eigvec

	gas_velocities = gas_velocities @ gas_eigvec

	plt.scatter(gas_coordinates[:,0],gas_coordinates[:,1],s=0.05)
	plt.show()


	gas_radii =  np.sqrt(np.nansum(gas_coordinates**2.e0, axis=1))

	gas_coords_faceon = calc_coords_obs(gas_coordinates, 180, 0)
	gas_coords_edgeon = calc_coords_obs(gas_coordinates, 180, 90)
	gas_vel_edgeon = calc_vel_obs(gas_velocities, 180 , 90)

	spectrum_spatial_contribution(gas_coordinates, gas_velocities, HImass_K13)


	# gas_Pextk = calc_Pextk_midpressure(stars_coordinates, stars_masses, gas_coordinates, gas_neutral_masses)
	# Rmol = calc_Rmol(gas_Pextk)
	# HI_masses = gas_neutral_masses / (1.e0 + Rmol)
	# H2_masses = gas_neutral_masses - HI_masses

	# HImass_list = [HI_masses, HImass_GK11, HImass_K13, HImass_GD14]

	# compare_radialSigma(gas_coordinates, HImass_list,['BR06','GK11','K13','GD14'],40)

	# compare_spectra(gas_coords_edgeon, gas_vel_edgeon, HImass_list, ['BR06','GK11','K13','GD14'])

	# map_asymmetry_viewangle(gas_coordinates,gas_velocities,HI_masses)

	print(np.nanmax(HI_masses),np.nanmin(HI_masses))
	print('total HI mass',np.nansum(HI_masses),'fraction=',np.nansum(HI_masses)/np.nansum(stars_masses))
	print('total H2 mass',np.nansum(H2_masses),'fraction=',np.nansum(H2_masses)/np.nansum(stars_masses))

	gas_coords_edgeon = calc_coords_obs(gas_coordinates, 0, 17)
	gas_vel_edgeon = calc_vel_obs(gas_velocities, 0, 17)

	vel, spectrum = calc_spectrum(gas_coords_edgeon, gas_vel_edgeon, HI_masses)
	PeaklocL, PeaklocR = locate_peaks(spectrum)
	widths = locate_width(spectrum, [spectrum[PeaklocL],spectrum[PeaklocR]], 0.2)
	Sint, Afr = areal_asymmetry(spectrum, widths, np.abs(np.diff(vel)[0]))

	plt.plot(vel, spectrum)
	plt.plot([vel[PeaklocL],vel[PeaklocL]],[0,20])
	plt.plot([vel[PeaklocR],vel[PeaklocR]],[0,20])
	plt.show()
	print(PeaklocL, PeaklocR)
	print(widths)
	print(Afr)


	# plot_spatial_radial_spectrum(0, Rvir, gas_coordinates, gas_velocities, HI_masses, H2_masses)

def EAGLEsnap():
	
	# Rvir = 250
	unitmass = 1.e10
	filenames = glob.glob('/media/data/simulations/EAGLE_galaxies/*')
	# print(filenames)
	# exit()
	# filename = '/media/data/simulations/EAGLE_galaxies/EAGLE_galaxyID8253667.hdf5'
	# filename = '/media/data/simulations/EAGLE_galaxies/EAGLE_galaxyID8132670.hdf5'
	# filename = '/media/data/simulations/EAGLE_galaxies/EAGLE_galaxyID2948350.hdf5'
	# filename = '/media/data/simulations/EAGLE_galaxies/EAGLE_galaxyID2979098.hdf5'
	for filename in filenames:

		file = h5py.File(filename,'r')

		if 'PartType0' in list(file.keys()):
			head = file['Header']
			DM = file['PartType1']
			stars = file['PartType4']
			gas = file['PartType0']

			a = head.attrs['ExpansionFactor']
			h = head.attrs['HubbleParam']

			print(head.attrs['MassTable'][1] * a**(0) * h**(-1))

			[DM_coordinates, DM_velocities] = particle_info(a, h, DM, unitmass, ['Coordinates','Velocity'])
			DM_masses = np.ones(len(DM_coordinates))*head.attrs['MassTable'][1] * a**(0) * h**(-1)

			[gas_coordinates, gas_masses, gas_velocities, gas_densities, gas_internal_energy, gas_temperature] = \
					particle_info(a, h, gas, unitmass, ['Coordinates','Mass','Velocity','Density','InternalEnergy','Temperature'])

			[stars_coordinates, stars_masses, stars_velocities] = \
					particle_info(a, h, stars, unitmass, ['Coordinates','Mass','Velocity'])


			COM_DM = calc_COM(DM_coordinates, DM_masses)
			DM_coordinates -= COM_DM
			for ii in range(10):
				DM_radii = np.sqrt(np.nansum(DM_coordinates**2.e0, axis=1))
				COM = calc_COM(DM_coordinates[DM_radii < 0.15], DM_masses[DM_radii < 0.15])
				DM_coordinates -= COM
				COM_DM -= COM
				print(COM)
			print(COM_DM)


			gas_coordinates -= COM_DM
			stars_coordinates -= COM_DM


			stars_radii = np.sqrt(np.nansum(stars_coordinates**2.e0, axis=1))
			gas_radii = np.sqrt(np.nansum(gas_coordinates**2.e0, axis=1))
			DM_radii = np.sqrt(np.nansum(DM_coordinates**2.e0, axis=1))

			p_crit = 3 * (h*100)**2.e0 / (8 * np.pi * (4.3e-3 *1.e-6*1.e10 )  )		# in 1.e10Msun/Mpc^3
			print(p_crit)

			rad = 0.5
			rho = 250 * p_crit
			while(rho >= 200 * p_crit):
				rho = np.nansum(DM_masses[DM_radii < rad]) / (4. * np.pi * rad*rad*rad / 3.)
				rad += 0.01
			Rvir = rad

			DM_coordinates = DM_coordinates[DM_radii<Rvir]
			DM_masses = DM_masses[DM_radii<Rvir]
			DM_velocities = DM_velocities[DM_radii<Rvir]

			stars_coordinates = stars_coordinates[stars_radii<Rvir]
			stars_masses = stars_masses[stars_radii<Rvir]
			stars_velocities = stars_velocities[stars_radii<Rvir]

			gas_coordinates = gas_coordinates[gas_radii<Rvir]
			gas_masses = gas_masses[gas_radii<Rvir]
			gas_velocities = gas_velocities[gas_radii<Rvir]
			gas_densities = gas_densities[gas_radii<Rvir]
			gas_internal_energy = gas_internal_energy[gas_radii<Rvir]
			gas_temperature = gas_temperature[gas_radii<Rvir]

			print('DM mass[1.e12]',np.nansum(DM_masses)*unitmass / 1.e12)
			print('stellar mass [1.e10]',np.nansum(stars_masses) / 1.e10)
			print('Total gas mass [1.e10]', np.nansum(gas_masses) / 1.e10)
			print('Stellar fraction',np.nansum(stars_masses) / np.nansum(DM_masses))
			print('Total Gas fraction',np.nansum(gas_masses) / np.nansum(stars_masses))
			Rvir *= 1.e3
			print('Virial radius',Rvir, ' kpc')

			COV = calc_COV(DM_coordinates*1.e3, DM_velocities, DM_masses, Rvir)
			# print('COV')
			# print(COV)

			DM_coordinates *= 1.e3
			gas_coordinates *= 1.e3
			stars_coordinates *= 1.e3

			cold_gas = np.where(gas_temperature < 1.e5)[0]

			COM_gas = calc_COM(gas_coordinates, gas_masses, Rvir)
			COM_stars = calc_COM(stars_coordinates, stars_masses, Rvir)
			gas_coordinates -= COM_stars
			stars_coordinates -= COM_stars

			# plt.scatter(stars_coordinates[:,0],stars_coordinates[:,2],s=0.01)
			# plt.xlim([-40,40])
			# plt.ylim([-40,40])
			# plt.show()

			eigvec = orientation_matrix(stars_coordinates, stars_masses)

			stars_coordinates = stars_coordinates @ eigvec
			gas_coordinates = gas_coordinates @ eigvec
			DM_coordinates = DM_coordinates @ eigvec

			# plt.scatter(stars_coordinates[:,0],stars_coordinates[:,2],s=0.01)
			# plt.xlim([-40,40])
			# plt.ylim([-40,40])
			# plt.colorbar()
			# plt.show()

			
			gas_velocities -= COV
			gas_velocities = gas_velocities @ eigvec
			gas_coords_faceon = calc_coords_obs(gas_coordinates, 0, 0)
			gas_coords_edgeon = calc_coords_obs(gas_coordinates, 0, 90)

			stars_coords_faceon = calc_coords_obs(stars_coordinates, 0, 0)
			stars_coords_edgeon = calc_coords_obs(stars_coordinates, 0, 90)

			DM_coords_faceon = calc_coords_obs(DM_coordinates, 0, 0)
			DM_coords_edgeon = calc_coords_obs(DM_coordinates, 0, 90)

			gas_vel_edgeon = calc_vel_obs(gas_velocities, 0 , 90)

			# plt.scatter(gas_coords_edgeon[:,0],gas_coords_edgeon[:,1],s=0.05,c=gas_vel_edgeon)
			# plt.xlim([-40,40])
			# plt.ylim([-40,40])
			# plt.colorbar()
			# plt.show()


			# gas_Pextk = calc_Pextk_densityenergy(gas_densities, gas_internal_energy,lenunit='mpc')
			# Rmol_part = calc_Rmol(gas_Pextk)
			# HI_masses = gas_masses / (1.e0 + Rmol_part)
			# H2_masses = gas_masses - HI_masses 
			# plt.hist(np.log10(gas_Pextk))

			# gas_neutral_fraction, fH2 = calc_fH2_LGS(gas,unitmass,a,h)
			# H2_masses = gas_masses * gas_neutral_fraction * fH2
			# HI_masses = gas_masses * gas_neutral_fraction * (1 - fH2)

			HI_masses,H2_masses,gas_neutral_masses = calc_HI_H2_ARHS(gas,unitmass,a,h)

			# print(gas_neutral_fraction)
			# print(fH2)
			# exit()

			rad_stars, sigma_stars = calc_sigma(stars_coords_faceon, stars_masses, Rvir)
			rad_stars_left, sigma_stars_left = calc_sigma(stars_coords_faceon[stars_coords_faceon[:,0]<0], stars_masses[stars_coords_faceon[:,0]<0], Rvir)
			rad_stars_right, sigma_stars_right = calc_sigma(stars_coords_faceon[stars_coords_faceon[:,0]>0], stars_masses[stars_coords_faceon[:,0]>0], Rvir)

			rad_HI, sigma_HI = calc_sigma(gas_coords_faceon, HI_masses, Rvir)
			rad_HI_left, sigma_HI_left = calc_sigma(gas_coords_faceon[gas_coords_faceon[:,0]<0], HI_masses[gas_coords_faceon[:,0]<0], Rvir)
			rad_HI_right, sigma_HI_right = calc_sigma(gas_coords_faceon[gas_coords_faceon[:,0]>0], HI_masses[gas_coords_faceon[:,0]>0], Rvir)

			rad_H2, sigma_H2 = calc_sigma(gas_coords_faceon, H2_masses, Rvir)
			rad_DM, sigma_DM = calc_sigma(DM_coords_faceon, DM_masses, Rvir)

			rad_HI_RC, RC_HI = calc_RC(gas_coords_faceon, gas_velocities, Rvir)
			rad_HI_left_RC, RC_HI_left = calc_RC(gas_coords_faceon[gas_coords_faceon[:,0]<0], gas_velocities[gas_coords_faceon[:,0]<0], Rvir)
			rad_HI_right_RC, RC_HI_right = calc_RC(gas_coords_faceon[gas_coords_faceon[:,0]>0], gas_velocities[gas_coords_faceon[:,0]>0], Rvir)


			
			vel, spectrum = calc_spectrum(gas_coords_edgeon, gas_vel_edgeon, HI_masses)
			
			PeaklocL, PeaklocR = locate_peaks(spectrum)
			widths = locate_width(spectrum, [spectrum[PeaklocL],spectrum[PeaklocR]], 0.2)
			Sint, Afr = areal_asymmetry(spectrum, widths, np.abs(np.diff(vel)[0]))


			fig = plt.figure(figsize=(15,8))
			gs = gridspec.GridSpec(2,2, hspace=0) 
			sigma_ax = fig.add_subplot(gs[0,0])
			RC_ax = fig.add_subplot(gs[1,0], sharex = sigma_ax)
			spec_ax = fig.add_subplot(gs[:,1])
			
			beamsizes = [10,20,30,60]
			colors = ['Blue','Orange','Green','Red']
			for bb in range(len(beamsizes)):
				vel, spectrum_left = calc_spectrum(gas_coords_edgeon[gas_coords_faceon[:,0]<0], gas_vel_edgeon[gas_coords_faceon[:,0]<0], HI_masses[gas_coords_faceon[:,0]<0], beamsize = beamsizes[bb])
				vel, spectrum_right = calc_spectrum(gas_coords_edgeon[gas_coords_faceon[:,0]>0], gas_vel_edgeon[gas_coords_faceon[:,0]>0], HI_masses[gas_coords_faceon[:,0]>0], beamsize = beamsizes[bb] )

				spec_ax.plot(vel,spectrum_left, ls = '--',c=colors[bb], label = 'Beamsize = {b} kpc'.format(b = beamsizes[bb]))
				spec_ax.plot(vel,spectrum_right,ls = ':',c=colors[bb])
			
			spec_ax.set_xlabel('Velocity [km s$^{-1}$]', fontsize=15)
			spec_ax.set_ylabel('Spectral Flux [mJy]', fontsize=15)

			spec_ax.tick_params(axis = 'both', which='both', direction = 'in', labelsize=15, length = 8, width = 1.25)
			RC_ax.tick_params(axis = 'both', which='both', direction = 'in', labelsize=15, length = 8, width = 1.25)
			sigma_ax.tick_params(axis = 'both', which='both', direction = 'in', labelsize=15, length = 8, width = 1.25)
			sigma_ax.tick_params(axis = 'x', which='both', direction = 'in', labelsize=0, length = 8, width = 1.25)


			sigma_ax.plot(rad_stars,sigma_stars, label='Stars total',c = 'Red')
			sigma_ax.plot(rad_stars_left,sigma_stars_left, label='Stars LHS',c = 'Red', ls = '--')
			sigma_ax.plot(rad_stars_right,sigma_stars_right, label='Stars RHS',c = 'Red',ls = ':')
			sigma_ax.plot(rad_HI,sigma_HI,label = 'HI total',c = 'Blue')
			sigma_ax.plot(rad_HI_left,sigma_HI_left,label = 'HI LHS',c = 'Blue',ls = '--')
			sigma_ax.plot(rad_HI_right,sigma_HI_right,label = 'HI RHS',c = 'Blue',ls = ':')
			# sigma_ax.plot(rad_H2,sigma_H2,label = 'H$_{2}$')
			sigma_ax.plot(rad_DM,9+sigma_DM, label = 'DM + 9', color='Black')

			RC_ax.plot(rad_HI_RC, RC_HI, label = 'HI total',c = 'Blue')
			RC_ax.plot(rad_HI_left_RC, RC_HI_left, label = 'HI LHS',c = 'Blue',ls = '--')
			RC_ax.plot(rad_HI_right_RC, RC_HI_right, label = 'HI RHS',c = 'Blue',ls = ':')
			# sigma_ax.plot(rad_H2,sigma_H2,label = 'H$_{2}$')
			# sigma_ax.plot(rad_DM,9+sigma_DM, label = 'DM + 9', color='Black')


			sigma_ax.set_xlim([0,40])
			sigma_ax.set_ylim([-2.5,3])

			spec_ax.set_ylim([0,700])

			RC_ax.set_xlabel('Radius [kpc]', fontsize=15)
			RC_ax.set_ylabel('Vcirc [km s$^{-1}$]', fontsize=15)
			sigma_ax.set_ylabel('log$_{10} \Sigma$ [M$_{\odot}$ pc$^{-2}$]', fontsize=15)
			sigma_ax.legend(fontsize=12)
			spec_ax.legend(fontsize=12)
			spec_ax.set_title('EAGLE ID = {name}  A$_{{fr}}$ = {A:.2f}'.format(name=filename.split('ID')[-1].split('.')[0],A=Afr),fontsize=12)
			# plt.show()
			outname = './figures/sigma_RC_spec_EAGLE{name}.png'.format(name=filename.split('ID')[-1].split('.')[0])
			fig.savefig(outname, dpi=150)
			plt.close()
			# exit()


			# plot_spatial_radial_spectrum(0, Rvir, gas_coordinates, gas_velocities,HI_masses,H2_masses)




##############################################################################################################

# def read_TNGsnap(base)
	
def particle_info(a, h, part, unitmass, keys, cgs=False):

	data = []
	for key in keys:
		group = part[key]
		aexp = a**group.attrs['aexp-scale-exponent']
		hexp = h**group.attrs['h-scale-exponent']
		CGSconv = group.attrs['CGSConversionFactor']

		group = np.array(group) * aexp * hexp

		if cgs == True:
			group = group * CGSconv
		elif 'Mass' in key:
			group = group * unitmass

		data.append(group)
	return data	

def calc_COM(coordinates,masses, Rvir = None):

	radii = np.sqrt(np.nansum(coordinates**2.e0, axis=1))

	if Rvir != None:
		coordinates = coordinates[radii < 0.1*Rvir]
		masses = masses[radii < 0.1*Rvir]

	COM = np.array([np.nansum(coordinates[:,0]*masses, axis=0) / np.nansum(masses),\
					np.nansum(coordinates[:,1]*masses, axis=0) / np.nansum(masses),\
					np.nansum(coordinates[:,2]*masses, axis=0) / np.nansum(masses)])
	return COM

def calc_COV(coordinates, velocities, masses, Rvir = None):

	radii = np.sqrt(np.nansum(coordinates**2.e0, axis=1))

	if Rvir != None:
		coordinates = coordinates[radii < 0.1*Rvir]
		velocities = velocities[radii < 0.1*Rvir]
		masses = masses[radii < 0.1*Rvir]

	COV = np.array([np.nansum(velocities[:,0]*masses, axis=0) / np.nansum(masses),\
					np.nansum(velocities[:,1]*masses, axis=0) / np.nansum(masses),\
					np.nansum(velocities[:,2]*masses, axis=0) / np.nansum(masses)])
	return COV

def diagonalise_inertia(coordinates, masses, rad):

	radii = np.sqrt(np.nansum(coordinates**2.e0, axis=1))
	coordinates = coordinates[radii < rad]
	masses = masses[radii < rad]
	
	I = np.zeros([3,3])
	for ii in range(3):
		for jj in range(3):
			if ii == jj:
				I[ii,jj] = np.nansum( (coordinates[:,(ii + 1)%3]**2.e0 + 
											coordinates[:,(jj + 2)%3]**2.e0 )*masses )
			else:
				I[ii,jj] = -1.e0*np.nansum(coordinates[:,ii]*coordinates[:,jj]*masses)

	eigval, eigvec = np.linalg.eig(I)
	eigval_argsort = eigval.argsort()
	# eigval = eigval[eigval_argsort]
	# eigvec = eigvec[eigval_argsort]
	eigvec = np.linalg.inv(eigvec)
	eigvec = eigvec[eigval_argsort]

	return eigvec	

def orientation_matrix(coordinates, masses):

	Iprev = [[1,0,0],[0,1,0],[0,0,1]]
	eigvec_list = []

	rad = [1,2,3,4,5,6,8,10,12,14,16,18,20,25,30,35,40,45,50,55,60,80,100,150,200]
	ii = 0
	Idiff = 1
	while(Idiff > 1.e-4):
		eigvec = diagonalise_inertia(coordinates, masses, rad[ii])
		coordinates = coordinates @ eigvec
		eigvec_list.append(eigvec)

		I = np.zeros([3,3])
		for ii in range(3):
			for jj in range(3):
				if ii == jj:
					I[ii,jj] = np.nansum( (coordinates[:,(ii + 1)%3]**2.e0 + 
												coordinates[:,(jj + 2)%3]**2.e0 )*masses )
				else:
					I[ii,jj] = -1.e0*np.nansum(coordinates[:,ii]*coordinates[:,jj]*masses)

		Idiff = np.abs((I[2][2] - Iprev[2][2]) / Iprev[2][2])
		Iprev = I
		ii+=1

	eigvec = eigvec_list[0]
	for ii in range(1,len(eigvec_list)):
		eigvec = eigvec @ eigvec_list[ii]

	return eigvec

def create_gif(basedir):
	import glob
	import imageio
	files = glob.glob('{dir}/figures/snaps*.png'.format(dir=basedir))
	files = np.array(files)
	# print(files)
	nums = np.zeros(len(files))
	for ii in range(len(files)):
		nums[ii] = int('{}'.format(files[ii].split('snaps')[-1].split('.png')[0]))
	files = files[np.argsort(nums)]
	# print(files)

	images = []
	for file in files:
		print(file)
		images.append(imageio.imread(file))
	imageio.mimsave('{dir}/figures/gif.gif'.format(dir=basedir),images,duration = 0.005*len(files))

def compare_spectra(coordinates, velocities, gas_HI_masses, names):

	fig = fig = plt.figure(figsize=(8,10))
	gs = gridspec.GridSpec(1,1) 
	ax = fig.add_subplot(gs[0,0])
	ax.set_xlabel('Velocity')
	ax.set_ylabel('Spectral flux')

	for ii in range(len(gas_HI_masses)):
		vel, spectrum = calc_spectrum(coordinates, velocities, gas_HI_masses[ii])
		PeaklocL, PeaklocR = locate_peaks(spectrum)
		widths = locate_width(spectrum, [spectrum[PeaklocL],spectrum[PeaklocR]], 0.2)
		Sint, Afr = areal_asymmetry(spectrum, widths, np.abs(np.diff(vel)[0]))
		ax.plot(vel, spectrum, label='{name}: Sint = {Sint:.2f} Jy, $A_{{fr}}$= {Afr:.2f} '.format(
					name=names[ii],Sint=Sint/1.e3,Afr=Afr))

		# plt.title(filename.split('/')[-3])
	plt.legend()
	plt.show()

def compare_radialSigma(coordinates, gas_HI_masses, names, Rmax):

	fig = fig = plt.figure(figsize=(10,10))
	gs = gridspec.GridSpec(1,1) 
	ax = fig.add_subplot(gs[0,0])
	ax.set_xlabel('Radius',fontsize=15)
	ax.set_ylabel('$log_{{10}} \Sigma_{{HI}}$',fontsize=15)

	for ii in range(len(gas_HI_masses)):
		rad_points, sigma = calc_sigma(coordinates, gas_HI_masses[ii], Rmax)
		ax.plot(rad_points, sigma, label='{name}'.format(name=names[ii]))
	plt.legend()
	plt.show()

def plot_spatial_radial_spectrum(tt, Rvir, gas_coordinates, gas_velocities, HI_masses, H2_masses, save = None):

	gas_neutral_masses = HI_masses + H2_masses

	gas_coords_edgeon = calc_coords_obs(gas_coordinates, 0, 90)
	gas_vel_edgeon = calc_vel_obs(gas_velocities, 0, 90)

	vel, spectrum = calc_spectrum(gas_coords_edgeon, gas_vel_edgeon, HI_masses)
	vel, spectrum_all = calc_spectrum(gas_coords_edgeon, gas_vel_edgeon, gas_neutral_masses)

	rad_points, sigma_all = calc_sigma(gas_coordinates, gas_neutral_masses, 0.2*Rvir)
	rad_points, RC_all = calc_RC(gas_coordinates, gas_velocities, 0.2*Rvir)

	rad_points, sigma_HI= calc_sigma(gas_coordinates, HI_masses, 0.2*Rvir)
	rad_points,  RC_HI = calc_RC(gas_coordinates, gas_velocities, 0.2*Rvir)

	spacebins, H2faceon_mom0 = calc_spatial_dist(gas_coordinates, H2_masses, 0.2*Rvir)
	spacebins, H2edgeon_mom0 = calc_spatial_dist(gas_coords_edgeon, H2_masses, 0.2*Rvir)

	spacebins, HIfaceon_mom0 = calc_spatial_dist(gas_coordinates, HI_masses, 0.2*Rvir)
	spacebins, HIedgeon_mom0 = calc_spatial_dist(gas_coords_edgeon, HI_masses, 0.2*Rvir)


	fig = plt.figure(figsize=(18,8.9))
	gs = gridspec.GridSpec(3,4, left=0.05, top=0.9, bottom=0.11, right = 0.99, wspace=0.3,hspace=0,\
					height_ratios=[0.05,1,1], width_ratios = [1,1,1,1]) 
	
	sigma_ax = fig.add_subplot(gs[1,2])
	RC_ax = fig.add_subplot(gs[2,2],sharex = sigma_ax)
	spec_ax = fig.add_subplot(gs[:,3])
	HIfaceon_ax = fig.add_subplot(gs[1,0])
	HIedgeon_ax = fig.add_subplot(gs[2,0], sharex=HIfaceon_ax)

	H2faceon_ax = fig.add_subplot(gs[1,1], sharex=HIfaceon_ax)
	H2edgeon_ax = fig.add_subplot(gs[2,1], sharex=HIfaceon_ax, sharey = HIedgeon_ax)

	HI_colorbar = fig.add_subplot(gs[0,0])
	H2_colorbar = fig.add_subplot(gs[0,1])


	sigma_ax.set_ylim([-2.5,1.2])
	RC_ax.set_ylim([0,1.1*np.max(RC_HI)])
	# spec_ax.set_ylim([0,49])
	HIfaceon_ax.set_xlim([-40,40])
	HIfaceon_ax.set_ylim([-40,40])
	HIedgeon_ax.set_ylim([-40,40])
	HIedgeon_ax.set_xticks([-40,-20,0,20,40])


	H2faceon_ax.set_xlim([-40,40])
	H2faceon_ax.set_ylim([-40,40])
	H2edgeon_ax.set_ylim([-40,40])
	H2edgeon_ax.set_xticks([-40,-20,0,20,40])

	sigma_ax.set_xlim([0,40])
	RC_ax.set_xlim([0,40])
	spec_ax.set_xlim([-800,800])

	# HIfaceon_ax.set_xlabel('x [kpc]',fontsize = 18)
	HIfaceon_ax.set_ylabel('y [kpc]',fontsize = 18)
	HIedgeon_ax.set_xlabel('x [kpc]',fontsize = 18)
	HIedgeon_ax.set_ylabel('z [kpc]',fontsize = 18)

	# H2faceon_ax.set_xlabel('x [kpc]',fontsize = 18)
	H2faceon_ax.set_ylabel('y [kpc]',fontsize = 18)
	H2edgeon_ax.set_xlabel('x [kpc]',fontsize = 18)
	H2edgeon_ax.set_ylabel('z [kpc]',fontsize = 18)

	# sigma_ax.set_xlabel('Radius [kpc]',fontsize = 18)
	sigma_ax.set_ylabel('log$_{10}$ $\Sigma_{HI}$ [M$_{\odot}$ pc$^{-2}$]',fontsize = 18)

	RC_ax.set_xlabel('Radius [kpc]',fontsize = 18)
	RC_ax.set_ylabel('Rotational velocity [km s$^{-1}$]',fontsize = 18)

	spec_ax.set_xlabel('Velocity [km s$^{-1}$]',fontsize = 18)
	spec_ax.set_ylabel('Spectral Flux [mJy]',fontsize = 18)
	spec_ax.set_title('Prescription  = BR06',fontsize=15)

	sigma_ax.tick_params(axis = 'both', which='both', direction = 'in', labelsize=18, length = 8, width = 1.25)
	sigma_ax.tick_params(axis = 'x', which='both', direction = 'in', labelsize=0)
	RC_ax.tick_params(axis = 'both', which='both', direction = 'in', labelsize=18, length = 8, width = 1.25)
	spec_ax.tick_params(axis = 'both', which='both', direction = 'in', labelsize=18, length = 8, width = 1.25)
	HIfaceon_ax.tick_params(axis = 'both', which='both', direction = 'in', labelsize=18, length = 8, width = 1.25)
	HIfaceon_ax.tick_params(axis = 'x', which='both', direction = 'in', labelsize=0)
	HIedgeon_ax.tick_params(axis = 'both', which='both', direction = 'in', labelsize=18, length = 8, width = 1.25)
	H2faceon_ax.tick_params(axis = 'both', which='both', direction = 'in', labelsize=18, length = 8, width = 1.25)
	H2faceon_ax.tick_params(axis = 'x', which='both', direction = 'in', labelsize=0)
	H2edgeon_ax.tick_params(axis = 'both', which='both', direction = 'in', labelsize=18, length = 8, width = 1.25)
	

	sigma_ax.tick_params(axis = 'both', which='minor', direction = 'in', labelsize=18, length = 4, width = 1.25)
	RC_ax.tick_params(axis = 'both', which='minor', direction = 'in', labelsize=18, length = 4, width = 1.25)
	spec_ax.tick_params(axis = 'both', which='minor', direction = 'in', labelsize=18, length = 4, width = 1.25)
	HIfaceon_ax.tick_params(axis = 'both', which='minor', direction = 'in', labelsize=18, length = 4, width = 1.25)
	HIedgeon_ax.tick_params(axis = 'both', which='minor', direction = 'in', labelsize=18, length = 4, width = 1.25)
	H2faceon_ax.tick_params(axis = 'both', which='minor', direction = 'in', labelsize=18, length = 4, width = 1.25)
	H2edgeon_ax.tick_params(axis = 'both', which='minor', direction = 'in', labelsize=18, length = 4, width = 1.25)


	# dist = 50
	# mjy_conv = 1.e3 / (2.356e5  * (dist ** 2.e0))
	dx = np.abs(np.diff(spacebins)[0])
	print(dx)
	MHI = np.nansum(10.**HIfaceon_mom0[HIfaceon_mom0 > 0]) * dx*dx*1.e6

	RHI = 0.5*size_mass_relation(MHI)

	sigma_ax.plot(rad_points, sigma_HI)
	sigma_ax.plot(rad_points, sigma_all)
	sigma_ax.plot([np.min(rad_points),np.max(rad_points)],[0,0])
	sigma_ax.plot([RHI,RHI],[-2.5,1.2])
	RC_ax.plot(rad_points, RC_HI)
	RC_ax.plot(rad_points, RC_all)
	
	spec_ax.plot(vel, spectrum)
	spec_ax.plot(vel, spectrum_all)
	# spec_ax.plot([vel[PeaklocL],vel[PeaklocL]],[0,1.1*np.max(spectrum)],color='Red',ls='--')
	# spec_ax.plot([vel[PeaklocR],vel[PeaklocR]],[0,1.1*np.max(spectrum)],color='Red',ls='--')

	# spec_ax.plot([vel[int(widths[0])],vel[int(widths[0])]],[0,1.1*np.max(spectrum)],color='Black',ls=':')
	# spec_ax.plot([vel[int(np.ceil(widths[1]))],vel[int(np.ceil(widths[1]))]],[0,1.1*np.max(spectrum)],color='Black',ls=':')
	# spec_ax.text(0.05,0.92, '$A_{{fr}}$  = {Afr:.3f}'.format(Afr=Afr), fontsize=15, transform=spec_ax.transAxes, zorder=1)
	
	HIfaceon = HIfaceon_ax.imshow(HIfaceon_mom0, extent=[-40,40,-40,40],vmin=-2,vmax=2)
	HIedgeon = HIedgeon_ax.imshow(HIedgeon_mom0, extent=[-40,40,-40,40],vmin=-2,vmax=2)
	HI_cb = Colorbar(ax = HI_colorbar, mappable = HIfaceon, orientation = 'horizontal', ticklocation = 'top')
	HI_cb.set_label('log$_{10}$ $\Sigma_{HI}$ [M$_{\odot}$ pc$^{-2}$]')

	H2faceon = H2faceon_ax.imshow(H2faceon_mom0, extent=[-40,40,-40,40],vmin=-2,vmax=2)
	H2edgeon = H2edgeon_ax.imshow(H2edgeon_mom0, extent=[-40,40,-40,40],vmin=-2,vmax=2)
	H2_cb = Colorbar(ax = H2_colorbar, mappable = H2faceon, orientation = 'horizontal', ticklocation = 'top')
	H2_cb.set_label('log$_{10}$ $\Sigma_{H_2}$ [M$_{\odot}$ pc$^{-2}$]')

	if save != None:
		figname = '{dir}/figures/snaps{tt}.png'.format(dir=save,tt=str(tt).zfill(3))
		fig.savefig(figname, dpi=200)
		plt.close()
	else:
		plt.show()

def map_asymmetry_viewangle(tt, coordinates, velocities, HI_masses, save = None):

	if save == 'view':
		plt.imshow(tt,extent = [0,360,180,0])
		plt.colorbar()
		plt.show()
	else:
		phi_range = np.arange(0, 360, 5)
		theta_range = np.arccos(2. * np.arange(0, 1, 0.02) - 1.) * 180./np.pi

		Afr_grid = np.zeros([len(theta_range), len(phi_range)])

		for th in range(len(theta_range)):
			for ph in range(len(phi_range)):

				phi = phi_range[ph]
				theta = theta_range[th]
				# print(phi, theta)

				coords_obs = calc_coords_obs(coordinates, phi, theta)
				vel_LOS = calc_vel_obs(velocities, phi, theta)

				vel, spectrum = calc_spectrum(coords_obs, vel_LOS, HI_masses)
				PeaklocL, PeaklocR = locate_peaks(spectrum)
				widths = locate_width(spectrum, [spectrum[PeaklocL],spectrum[PeaklocR]], 0.2)
				Sint, Afr = areal_asymmetry(spectrum, widths, np.abs(np.diff(vel)[0]))
				Afr_grid[th,ph] = Afr

		if save != None:
			filename = '{dir}/data/snaps{tt}_Afr_viewgrid.dat'.format(dir=save, tt=str(tt).zfill(3))
			np.savetxt(filename, Afr_grid)
		else:
			plt.imshow(Afr_grid)
			plt.show()

def spectrum_spatial_contribution(coordinates, velocities, HI_masses):

	coords_obs = calc_coords_obs(coordinates, 0, 90)
	coords_faceon = calc_coords_obs(coordinates, 0, 45)
	vel_obs = calc_vel_obs(velocities, 0, 90)

	vel_bins, spectrum = calc_spectrum(coords_obs, vel_obs, HI_masses)
	vel_spatial_step = 80
	vel_spatial_range = np.arange(vel_bins[0], vel_bins[-1]+vel_spatial_step, vel_spatial_step)

	spec_segments = []
	part_IDs = []

	colors = ['darkred','limegreen','blue','red','aquamarine','blueviolet','brown','aqua','magenta','crimson',\
				'forestgreen','navy','darkorange','darkcyan','salmon','cornflowerblue','orange',\
					'indigo','chocolate','yellow','darkorchid','olive','slateblue','green','dodgerblue'\
					,'purple','royalblue']

	for ii in range(len(vel_spatial_range)-1):
		vel_low = vel_spatial_range[ii]
		vel_high = vel_spatial_range[ii + 1]
		invel  = np.where( (vel_obs  >= vel_low) &
				 			(vel_obs < vel_high) )[0]
		part_IDs.append(invel)
		vel_bins, spec_segment = calc_spectrum(coords_obs[invel], vel_obs[invel], HI_masses[invel])
		spec_segments.append(spec_segment)

	fig = plt.figure(figsize=(18,9))
	gs = gridspec.GridSpec(1,2, left=0.05, top=0.9, bottom=0.11, right = 0.99, wspace=0.3)
	space_ax = fig.add_subplot(gs[0,0])
	spec_ax = fig.add_subplot(gs[0,1])
	space_ax.set_xlim([-60,60])
	space_ax.set_ylim([-60,60])

				 
	for ii in range(len(part_IDs)):
		space_ax.scatter(coords_faceon[part_IDs[ii],0],coords_faceon[part_IDs[ii],1],s=1,color=colors[ii])
		spec_ax.plot(vel_bins,spec_segments[ii],color=colors[ii])
	spec_ax.plot(vel_bins, spectrum,color='Black',lw=3)
	plt.show()
	exit()

def gaussian_CDF(x,mu,sigma):
	prob = 0.5e0 * (1.e0 + erf( (x - mu) / (np.sqrt(2.e0) * sigma) ))
	return prob

def radial_scaleheight(coordinates, masses, Rmax):
	radii = np.sqrt(np.nansum(coordinates**2.e0, axis=1))
	inspace = np.where((radii <= Rmax) & (coordinates[:,2]<0.5*Rmax))[0]
	coordinates = coordinates[inspace]
	masses = masses[inspace]
	radii = radii[inspace]

	radii_argsort = np.argsort(radii)
	radii = radii[radii_argsort]
	coordinates = coordinates[radii_argsort]
	masses = masses[radii_argsort]
	Npart = len(coordinates)
	Nbins = 20
	rad_points = np.zeros(Nbins)
	scaleheight = np.zeros(Nbins)

	for ii in range(Nbins):
		low = (ii) * int(Npart / (Nbins))
		high = (ii + 1) * int(Npart / (Nbins))
		inbin_coordinates = coordinates[low:high,2]
		inbin_radii = radii[low:high]
		rad_points[ii] = np.median(inbin_radii)
		scaleheight[ii] = np.percentile(inbin_coordinates[inbin_coordinates>=0], 100* 1./np.exp(1)) * 1000
		scaleheight[ii] = np.percentile(np.abs(inbin_coordinates), 100* 1./(np.exp(1))) * 1000

	hstar = hstar_from_radfit(coordinates,masses,40)

	hstar1 = np.percentile(coordinates[coordinates[:,2]>=0,2], 100* 1./np.exp(1)) * 1000
	# hstar1 = np.percentile(np.abs(coordinates[:,2]), 100* 1./np.exp(1)) * 1000
	hstar2 = np.percentile(np.abs(coordinates[:,2]), 100* 2./(np.exp(1))) * 1000

	# return hstar1
	plt.plot(rad_points,scaleheight,label = 'Percentile')
	plt.plot([2.5,20],[hstar,hstar], label = 'Radial fit')
	plt.plot([2.5,20],[hstar1,hstar1], label = 'Percentile: all particles')
	plt.plot([2.5,20],[hstar2,hstar2])
	plt.xlabel('Radus [kpc]')
	plt.ylabel('Scaleheight [pc]')
	plt.title('B/T = 0')
	plt.legend()
	plt.show()
	exit()

def scaleheight_from_scalelength(Rstar):
	hstar = 10.e0**(-0.23 + 0.8 * np.log10(Rstar))
	return hstar

def hstar_from_radfit(coordinates, masses, Rmax):
	radii = np.sqrt(np.nansum(coordinates**2.e0, axis = 1))
	coordinates = coordinates[radii <= Rmax]
	masses = masses[radii <= Rmax]
	radii = radii[radii <= Rmax]

	rad_points, sigma = calc_sigma(coordinates, masses, Rmax)
	# print(np.diff(sigma))
	redchisq = 100
	minrad = 0
	maxrad = len(sigma)-2
	while(redchisq > 1.e-3):
		fit, covar = curve_fit(disk_exp,rad_points[minrad:maxrad],sigma[minrad:maxrad])
		redchisq = np.nansum((sigma[minrad:maxrad] - disk_exp(rad_points[minrad:maxrad],fit[0],fit[1]))**2.e0)
		minrad += 1
	# plt.plot(rad_points,sigma)
	# plt.plot(rad_points,disk_exp(rad_points,fit[0],fit[1]))
	# plt.show()
	Rstar = fit[1] * 1000.
	print('Disk radial scale length', Rstar)
	hstar = scaleheight_from_scalelength(Rstar)
	print('Disk scaleheight', hstar)
	return hstar

def disk_exp(rad, A, Rstar):
	d = A - (rad/Rstar)*np.log10(np.exp(1))
	return d

def calc_coords_obs(coordinates, view_phi, view_theta):
	view_theta *= np.pi/180.e0
	view_phi *= np.pi/180.e0

	coords_obs = np.zeros([len(coordinates),2])
	coords_obs[:,0] = (coordinates[:,0] * np.sin(view_phi) + 
							coordinates[:,1]*np.cos(view_phi))
	coords_obs[:,1] = ( (coordinates[:,0] * np.cos(view_phi) - 
							coordinates[:,1]*np.sin(view_phi)) * np.cos(view_theta) +
							coordinates[:,2] * np.sin(view_theta) )
	return coords_obs

def calc_vel_obs(velocities, view_phi, view_theta):
	view_theta *= np.pi/180.e0
	view_phi *= np.pi/180.e0

	vel_LOS = velocities[:,0]*np.cos(view_phi) * np.abs(np.sin(view_theta)) +\
				velocities[:,1] * -1.e0*np.sin(view_phi) * np.abs(np.sin(view_theta)) +\
				velocities[:,2] * np.cos(view_theta)								
	return vel_LOS

def calc_spatial_dist(coordinates, masses, Rmax):
	dim = 200
	image = np.zeros([dim,dim])
	dx = 2*Rmax/dim
	spacebins = np.arange(-1*Rmax,Rmax+dx,dx)
	area = dx*dx*1.e6
	for xx in range(len(spacebins)-1):
		xx_low = spacebins[xx]
		xx_high = spacebins[xx+1]
		for yy in range(len(spacebins)-1):
			yy_low = spacebins[yy]
			yy_high = spacebins[yy+1]

			image[yy,xx] = np.log10(np.nansum(masses[(coordinates[:,0] >= xx_low) & (coordinates[:,0]<xx_high) &\
									(coordinates[:,1]>=yy_low) & (coordinates[:,1]<yy_high)])/area)

	return spacebins[0:-1] + dx, image

def calc_sigma(coordinates, masses, Rmax):

	# masses = masses[coordinates[:,2]<0.5*Rmax]
	# coordinates = coordinates[coordinates[:,2]<0.5*Rmax]

	radii = np.sqrt(np.nansum(coordinates**2.e0, axis=1))
	coordinates = coordinates[radii <= Rmax]
	masses = masses[radii <= Rmax]
	radii = radii[radii <= Rmax]

	radii_argsort = np.argsort(radii)
	radii = radii[radii_argsort]
	masses = masses[radii_argsort]

	Npart = len(masses)
	Nbins = 20
	rad_points = np.zeros(Nbins)
	sigma = np.zeros(Nbins)

	for ii in range(Nbins):
		low = (ii) * int(Npart / (Nbins))
		high = (ii + 1) * int(Npart / (Nbins))
		inbin_radii = radii[low:high]
		inbin_masses = masses[low:high]

		minrad = np.min(inbin_radii)
		maxrad = np.max(inbin_radii)
		area = np.pi * (maxrad * maxrad - minrad * minrad) * 1.e6
		sigma[ii] = np.log10(np.nansum(inbin_masses)/area)
		rad_points[ii] = np.median(inbin_radii)

	return rad_points, sigma

def calc_RC(coordinates, velocities, Rmax):

	# velocities = velocities[coordinates[:,2]<1]
	# coordinates = coordinates[coordinates[:,2]<1]

	radii = np.sqrt(np.nansum(coordinates**2.e0, axis=1))

	coordinates = coordinates[radii <= Rmax]
	velocities = velocities[radii <= Rmax]
	radii = radii[radii <= Rmax]

	vcirc = np.sqrt(velocities[:,0]**2.e0 + velocities[:,1]**2.e0)

	radii_argsort = np.argsort(radii)
	radii = radii[radii_argsort]
	vcirc = vcirc[radii_argsort]

	Npart = len(coordinates)
	Nbins = 20
	rad_points = np.zeros(Nbins)
	rot_cur = np.zeros(Nbins)

	for ii in range(Nbins):
		low = (ii) * int(Npart / (Nbins))
		high = (ii + 1) * int(Npart / (Nbins))
		inbin_radii = radii[low:high]
		inbin_vcirc = vcirc[low:high]

		rot_cur[ii] = np.median(inbin_vcirc)
		rad_points[ii] = np.median(inbin_radii)

	return rad_points,rot_cur

def calc_Pextk_densityenergy(gas_densities, gas_internal_energy, lenunit ='kpc'):
	unitmass = 1.989e43						#1.e10 Msun in g
	if lenunit == 'kpc':
		unitlen = 3.086e21					#1 kpc in cm
	elif lenunit == 'mpc':
		unitlen = 3.086e24					#1 Mpc in cm
	unitvel = 1.e5							#1 km/s in cm/s
	kB = 1.381e-16							#Boltzmann constant
	gamma = 5./3.							#Adiabatic constant
	gas_densities *= unitmass / (unitlen ** 3.)
	gas_internal_energy *= unitvel * unitvel
	Pextk = (gamma - 1.) * gas_densities * gas_internal_energy / kB

	return Pextk

def calc_Pextk_midpressure(stars_coordinates, stars_masses, gas_coordinates, gas_neutral_masses):#,gas_zvel):
	
	hstar = hstar_from_radfit(stars_coordinates, stars_masses, 40)

	dim = 100
	imgphys = 80.		#kpc
	dx = imgphys/dim
	pixarea = dx*dx*1.e6	#pc^2
	spacebins = np.arange(-0.5*imgphys, 0.5*imgphys + dx, dx)
	Pextk = np.zeros(len(gas_neutral_masses))
	for xx in range(len(spacebins)-1):
		xx_low = spacebins[xx]
		xx_high = spacebins[xx+1]
		for yy in range(len(spacebins)-1):
			yy_low = spacebins[yy]
			yy_high = spacebins[yy+1]

			inspace_gas = np.where((gas_coordinates[:,0] >= xx_low) & (gas_coordinates[:,0]<xx_high) &\
									(gas_coordinates[:,1]>=yy_low) & (gas_coordinates[:,1]<yy_high) &\
									(np.abs(gas_coordinates[:,2] < 1*hstar/1.e3)))[0]

			if len(inspace_gas) != 0:
				sigma_gas = np.nansum(gas_neutral_masses[inspace_gas]) * 1.36 / pixarea
				Zdisp_gas = 7

				inspace_stars = np.where((stars_coordinates[:,0] >= xx_low) & (stars_coordinates[:,0]<xx_high) &\
									(stars_coordinates[:,1]>=yy_low) & (stars_coordinates[:,1]<yy_high) &\
									(np.abs(stars_coordinates[:,2] < 1*hstar/1.e3)))[0]

				if len(inspace_stars) != 0:
					sigma_stars = np.nansum(stars_masses[inspace_stars]) / pixarea
					Pextk[inspace_gas] = 272.e0*sigma_gas*(sigma_stars**0.5)*Zdisp_gas*(hstar**-0.5)
	Pextk[np.abs(gas_coordinates[:,2]) > 1*hstar/1.e3] = 0

	return Pextk

def calc_fH2_LGS(gas, unitmass, a, h):
	
	Hm2012_data = Find_H2_LGS.Read_PhotoIo_Table('Hm2012.txt')
	Rahmati2013 = Find_H2_LGS.Read_BestFit_Params('BF_Params_Rahmati2013.txt')


	[masses, densities, Z, Habundance, temperature, SFR] = \
					particle_info(a, h, gas, unitmass, ['Mass','Density','Metallicity','ElementAbundance/Hydrogen'\
														,'Temperature','StarFormationRate'],cgs=True)
	
	Z = Z/0.0127
	fH2 = np.zeros(len(masses))
	neut_frac = np.zeros(len(masses))
	redshift = 1./a - 1.
	Hmass = masses*Habundance

	for i in range(len(masses)): 
		neut_frac[i], fH2[i] = Find_H2_LGS.find_fH2(densities[i], Hmass[i], masses[i], Z[i], SFR[i], temperature[i], Hm2012_data, Rahmati2013, redshift)

	return neut_frac, fH2

def calc_HI_H2_ARHS(gas, unitmass, a, h):

	Hm2012_data = Find_H2_LGS.Read_PhotoIo_Table('Hm2012.txt')
	Rahmati2013 = Find_H2_LGS.Read_BestFit_Params('BF_Params_Rahmati2013.txt')


	[masses, densities, Z, Habundance, temperature, U, SFR] = \
					particle_info(a, h, gas, unitmass, ['Mass','Density','Metallicity','ElementAbundance/Hydrogen'\
														,'Temperature','InternalEnergy','StarFormationRate'],cgs=True)
	masses =  masses / 1.989e33	
	SFR = SFR * (3600*24*365.25) / 1.989e33
	Z = Z
	densities = densities * (3.086e18*3.086e18*3.086e18) / 1.989e33
	U = U/(1.e4)
	redshift = 1./a - 1.


	HI_masses, H2_masses, neutral_masses = galcalc_ARHS.HI_H2_masses(masses,SFR,Z,densities,temperature,None,redshift)
	return HI_masses,H2_masses,neutral_masses

def calc_Rmol(Pextk, coeff = 'LR08'):

	if coeff == 'BR06':
		div = 4.3e4
		power = 0.92
	elif coeff == 'LR08':
		div = 10.**4.23
		power = 0.8

	Rmol = (Pextk / div)**power
	return Rmol

def compare_pressures_spatial(filename,gas_coordinates, Pextk_MP, Pextk_part,neutral_masses):


	fig = fig = plt.figure(figsize=(18,7))
	gs = gridspec.GridSpec(1,3, left=0.05, top=0.9, bottom=0.11, right = 0.99, wspace=0.3)
	pext_ax = fig.add_subplot(gs[0,0])
	high_ax = fig.add_subplot(gs[0,1])
	low_ax = fig.add_subplot(gs[0,2])

	higherP = np.where((Pextk_MP /Pextk_part) > 10)[0]
	lowerP = np.where((Pextk_MP / Pextk_part) < 0.5)

	print(np.nansum(neutral_masses[np.log10(Pextk_MP)<-2]) / np.nansum(neutral_masses))


	mapp = pext_ax.scatter(np.log10(Pextk_part),np.log10(Pextk_MP), s=0.2,c=np.log10(neutral_masses))
	# pext_ax.scatter(np.log10(Pextk_part[higherP]),np.log10(Pextk_MP[higherP]),color='Red',s=0.5)
	# pext_ax.scatter(np.log10(Pextk_part[lowerP]),np.log10(Pextk_MP[lowerP]),color='Blue',s=0.5)
	pext_ax.plot([-1,5],[-1,5],color='Green')
	pext_ax.set_xlabel('log Pext/k particle')
	pext_ax.set_ylabel('log Pext/k midplane pressure')
	fig.colorbar(mapp)

	high_ax.scatter(gas_coordinates[:,0],gas_coordinates[:,1], s=0.025,color='Black',label='All')
	high_ax.scatter(gas_coordinates[higherP,0],gas_coordinates[higherP,1], s=0.07, color='Red', label='MP > 10 Part')
	high_ax.set_xlim([-40,40])
	high_ax.set_ylim([-2,2])
	high_ax.legend()
	high_ax.set_xlabel('x [kpc]')
	high_ax.set_ylabel('z [kpc]')
	high_ax.set_title(filename.split('/')[-3])


	low_ax.scatter(gas_coordinates[:,0],gas_coordinates[:,1], s=0.025,color='Black',label='All')
	low_ax.scatter(gas_coordinates[lowerP,0],gas_coordinates[lowerP,1], s=0.07, color='Blue', label='MP < 0.5 Part')
	low_ax.set_xlim([-40,40])
	low_ax.set_ylim([-2,2])
	low_ax.legend()
	low_ax.set_xlabel('x [kpc]')
	low_ax.set_ylabel('z [kpc]')

	plt.show()
	exit()

def calc_spectrum(coords_obs, gas_vel_LOS, HI_masses, Vres = 5, beamsize = 30):
	
	radii = np.sqrt(np.nansum(coords_obs**2.e0, axis=1))
	inbeam = np.where(radii <= beamsize)
	gas_vel_LOS = gas_vel_LOS[inbeam]
	HI_masses = HI_masses[inbeam]

	vlim = 300.e0
	vel_bins = np.arange(-vlim,vlim + Vres,Vres)
	vel_points = vel_bins[0:-1] + 0.5 * Vres
	spectrum = np.zeros([len(vel_bins) - 1])

	dist=50
	mjy_conv = 1.e3 / (2.356e5  * (dist ** 2.e0))

	for vv in range(len(vel_bins) - 1):
		vel_low = vel_bins[vv]
		vel_high = vel_bins[vv + 1]
		invel  = np.where( (gas_vel_LOS + 30 >= vel_low) &
				 			(gas_vel_LOS - 30 < vel_high) )[0]
				
		for part in invel:
			Mfrac = gaussian_CDF(vel_high, gas_vel_LOS[part], 7.e0) - \
					gaussian_CDF(vel_low, gas_vel_LOS[part], 7.e0)
			spectrum[vv] += HI_masses[part] * Mfrac * mjy_conv
	return vel_points, spectrum

def locate_peaks(spectrum):

	PeakL = 0
	PeaklocL = int(len(spectrum)/2.)
	chan=1
	while(chan< len(spectrum)/2 + 5):
		chan+=1
		grad = (spectrum[chan] - spectrum[chan-1]) * (spectrum[chan+1] - spectrum[chan])
		if grad<0 and spectrum[chan]>PeakL:
			PeaklocL = chan
			PeakL = spectrum[chan]

	PeakR = 0
	PeaklocR = int(len(spectrum)/2.)
	chan = len(spectrum)-1
	while(chan > len(spectrum)/2 - 5 ):
		chan-=1
		grad = (spectrum[chan] - spectrum[chan+1]) * (spectrum[chan-1] - spectrum[chan])
		if grad<0 and spectrum[chan]>PeakR:
			PeaklocR = chan
			PeakR = spectrum[chan]

	return PeaklocL,PeaklocR

def locate_width(spectrum, peaks, level):
	"""
	Locate the N% level of the peak on the left and right side of a spectrum

	Parameters
	----------
	spectrum : array
		Input spectrum
	peaks : list
		Value of each peak
	level : float
		N% of the peaks to measure

	Returns
	-------
	Wloc : list
		Location of N% of each peak in channels
	"""

	# channels = range(len(spectrum))	
	SpeakL = peaks[0]
	SpeakR = peaks[1]
	wL = -1
	wR = -1
	chan = 0
	while(chan < len(spectrum)-1 and spectrum[chan] < level * SpeakL):
		chan += 1
		wL = chan - 1 + ((level * SpeakL - spectrum[chan - 1]) / (
			spectrum[chan] - spectrum[chan - 1])) 

	chan = len(spectrum) - 2
	while(chan > 0 and spectrum[chan] < level * SpeakR):
		chan -= 1
		wR = chan + 1 + -1.e0 * ((level * SpeakR - spectrum[chan + 1]) / (
			spectrum[chan] - spectrum[chan + 1])) 

	Wloc = [wL,wR]
	return Wloc

def areal_asymmetry(spectrum, limits, Vres):
	"""
	Measure the asymmetry parameter and integrated flux of a spectrum between limits

	Parameters
	----------
	spectrum : array
		Input spectrum to measure
	limits : list
		Lower and upper channels to measure between
	Vres : float
		Velocity resolution

	Returns
	-------
	Sint : float
		Integrated flux in units of input spectrum * Vres
	Afr : float
		Asymmetry parameter
	"""

	minchan = limits[0]
	maxchan = limits[1]
	midchan = 0.5e0 * (minchan + maxchan)
	min_val = np.interp(minchan,[np.floor(minchan),np.ceil(minchan)],
			[spectrum[int(np.floor(minchan))],spectrum[int(np.ceil(minchan))]])
	max_val = np.interp(maxchan,[np.floor(maxchan),np.ceil(maxchan)],
			[spectrum[int(np.floor(maxchan))],spectrum[int(np.ceil(maxchan))]])
	mid_val = np.interp(midchan,[np.floor(midchan),np.ceil(midchan)],
			[spectrum[int(np.floor(midchan))],spectrum[int(np.ceil(midchan))]])

	Sedge_L = min_val * (np.ceil(minchan) - minchan)
	Sedge_R = max_val * (maxchan - np.floor(maxchan))
	Smid_L = mid_val * (midchan - np.floor(midchan))
	Smid_R = mid_val * (np.ceil(midchan) - midchan )

	S_L = spectrum[int(np.ceil(minchan)):int(np.floor(midchan) + 1)]
	# S_L = S_L[S_L > 0]
	S_L = np.nansum(S_L)
	S_R = spectrum[int(np.ceil(midchan)):int(np.floor(maxchan) + 1)]
	# S_R = S_R[S_R > 0]
	S_R = np.nansum(S_R)

	Sint_L = (Sedge_L + S_L + Smid_L) * Vres
	Sint_R = (Sedge_R + S_R + Smid_R) * Vres
	Sint = Sint_L + Sint_R

	Afr =  Sint_L / Sint_R 
	if Afr < 1.e0:
		Afr = 1.e0 / Afr

	return Sint, Afr	

def size_mass_relation(MHI):
	DHI = 10.e0**(0.506*np.log10(MHI)-3.293)	#kpc Wang+16
	return DHI

def create_datacube(tt, filename, incl, gas_coordinates, gas_velocities, gas_masses):

	gas_coords_obs = calc_coords_obs(gas_coordinates, 0, incl)
	vel_LOS = calc_vel_obs(gas_velocities, 0 , incl)


	cubephys = 100 		#kpc
	cubedim = 250
	dx = cubephys / cubedim
	print('Datacube spatial resolution:', dx *1000 , 'pc')
	spacebins = np.arange(-0.5*cubephys, 0.5*cubephys + dx, dx)
	dv = 5.e0
	print('Datacube velocity resolution:',dv, 'km/s')
	vlim = 300.e0
	vel_bins = np.arange(-vlim, vlim + dv, dv)
	dist = 50.
	mjy_conv = 1.e3 / (2.356e5  * (dist ** 2.e0))
	datacube = np.zeros([cubedim, cubedim, len(vel_bins)-1])

	for yy in range(cubedim):
		ylow = spacebins[yy]
		yhigh = spacebins[yy + 1]
		for xx in range(cubedim):
			xlow = spacebins[xx]
			xhigh = spacebins[xx + 1]
			vel_LOS_inspace = vel_LOS[(gas_coords_obs[:,0] >= xlow) & (gas_coords_obs[:,0] < xhigh) & \
						(gas_coords_obs[:,1] >= ylow) & (gas_coords_obs[:,1] < yhigh)]
			
			for vv in range(len(vel_bins) - 1):
				vel_low = vel_bins[vv]
				vel_high = vel_bins[vv + 1]
				invel  = np.where( (vel_LOS_inspace + 30 >= vel_low) &
						 			(vel_LOS_inspace - 30 < vel_high) )[0]
				for part in invel:
					Mfrac = gaussian_CDF(vel_high, vel_LOS_inspace[part], 7.e0) - \
							gaussian_CDF(vel_low, vel_LOS_inspace[part], 7.e0)
					datacube[yy,xx,vv] += gas_masses[part] * Mfrac * mjy_conv

	datacube = datacube.reshape((cubedim,cubedim*(len(vel_bins)-1)))
	savedir = '{dir}/data/datacube_{tt}_incl{incl}.dat'.format(dir = filename.split('/snaps/')[0], tt = tt, incl = incl)
	header = 'dist = {dist}\n cubephys = {cubephys}\n dx = {dx}\n vlim = {vlim}\n dv = {dv}\n'.format(
		dist = dist, cubephys = cubephys, dx = dx, dv = dv,vlim = vlim)
	np.savetxt(savedir, datacube, header = header,fmt = "%.6e")

	# return datacube

def read_datacube(tt, dir):

	filename1 = '{dir}data/datacube_{tt}_incl90.dat'.format(dir=dir, tt=str(tt))
	# filename2 = '{dir}data/datcube_{tt}_incl90.dat'.format(dir=basedir, tt=str(tt).zfill(3))

	f = open(filename1, 'r')
	for line in f:
		if 'dist' in line:
			dist = float(line.split(' ')[-1]) 
			print(dist)
		if 'cube' in line:
			cubephys = float(line.split(' ')[-1]) 
			print(cubephys)
		if 'dx' in line:
			dx = float(line.split(' ')[-1]) 
			print(dx)
		if 'vlim' in line:
			vlim = float(line.split(' ')[-1]) 
			print(vlim)
		if 'dv' in line:
			dv = float(line.split(' ')[-1]) 
			print(dv)
	f.close()

	mjy_conv = 1.e3 / (2.356e5  * (dist ** 2.e0))
	cubedim = int(cubephys/dx)
	cubespec = int(2*vlim  / dv)

	spacebins = np.arange(-0.5*cubephys + 0.5*dx , 0.5*cubephys + 0.5*dx, dx)
	velbins = np.arange(-vlim + 0.5*dv, vlim + 0.5*dv, dv)

	datacube = np.loadtxt(filename1).reshape((cubedim,cubedim,cubespec))

	return spacebins,velbins, datacube, mjy_conv

def plot_datacube(tt, basedir, spacebins, velbins, datacube, mjy_conv):

	dx = np.abs(np.diff(spacebins)[0])
	print(int(1.2/dx))
	dv = np.abs(np.diff(velbins)[0])
	levels = [-1, np.log10(0.3), 0,np.log10(3), 1]

	fig = plt.figure(figsize=(15,8))
	gs = gridspec.GridSpec(1,2) 
	mom0_ax = fig.add_subplot(gs[0,0])
	spec_ax = fig.add_subplot(gs[0,1])
	mom0_ax.set_xlabel('x [kpc]',fontsize=20)
	mom0_ax.set_ylabel('y [kpc]',fontsize=20)
	spec_ax.set_xlabel('Velocity',fontsize=20)
	spec_ax.set_ylabel('Spectral flux',fontsize=20)
	spec_ax.set_ylim([0,40])
	spec_ax.tick_params(axis = 'both', which='both', direction = 'in', labelsize=20, length = 8, width = 1.25)
	mom0_ax.tick_params(axis = 'both', which='both', direction = 'in', labelsize=20, length = 8, width = 1.25)


	spectrum = np.nansum(datacube,axis=(0,1)) 

	colors = ['Blue','Orange','Green','Red','Magenta']

	mom0_map = np.nansum(datacube,axis=2)* dv / (mjy_conv * dx * dx * 1.e6)		#Msun /pc^2
	mom0_map = convolve(mom0_map,Gaussian2DKernel(int(0.8/dx)))
	mom0_map = np.log10(mom0_map)

	mom0_ax.imshow(mom0_map, extent=[spacebins[0],spacebins[-1],spacebins[0],spacebins[-1]],
					 vmin=-1,vmax=np.log10(12),cmap='Greys')
	mom0_ax.contour(spacebins, spacebins, mom0_map, levels = levels, colors=colors, alpha=0.7)

	# mom0_map = np.log10(np.nansum(datacube,axis=2)* dv / (mjy_conv * dx * dx * 1.e6))		#Msun /pc^2

	for ii in range(len(levels)):
		lev= levels[ii]
		mask = np.where(mom0_map < lev)
		datacube[mask[0],mask[1],:] = np.nan
		spectrum = np.nansum(datacube,axis=(0,1)) 
		spec_ax.plot(velbins, spectrum, color=colors[ii],
				label = 'Min $\Sigma_{{HI}}$ = {lev:.2f}'.format(lev=10**lev))
	plt.legend()
	
	savedir = '{dir}/figures/mom0_spec_{tt}_i90.png'.format(dir=basedir, tt=str(tt).zfill(3))

	fig.savefig(savedir, dpi=150)
	# plt.legend()
	# plt.show()




if __name__ == '__main__':
	# measure_controlled_run()

	# plot_controlled_run()

	# asymmetry_time()

	# create_gif(sys.argv[1])

	# hydro_run()

	# TNGsnap()

	# analyse_datacube()

	EAGLEsnap()
