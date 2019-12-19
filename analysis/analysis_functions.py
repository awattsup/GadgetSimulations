import numpy as np 
from astropy.table import Table
# import matplotlib.pyplot as plt 
# import matplotlib.gridspec as gridspec
# from matplotlib.ticker import FormatStrFormatter
# from matplotlib.colorbar import Colorbar
# from matplotlib.lines import Line2D
import h5py
import glob
import numpy.random as nprand
from astropy.convolution import convolve, Gaussian2DKernel
from scipy.special import erf
from scipy.optimize import curve_fit
from mpi4py import MPI
import sys
# import Find_H2_LGS
import galcalc_ARHS

## fourier decomposition of datacube data

def fourier_decomp_datacube():
	sys.path.append('../../fourier-decomposition')
	import fourier_decomposition as fd

	# spacebins, velbins, datacube, params = read_datacube('/media/data/simulations/IllustrisTNG/data/datacube_incl60.dat')
	# spacebins, velbins, datacube, params = read_datacube('/media/data/simulations/EAGLE_galaxies/data/datacube_ID8253667_incl60.dat')
	controlled = genesis = TNG = False
	for ii in range(3):
		if ii == 0:
			controlled = True
		if ii == 1:
			genesis = True
		if ii == 2:
			TNG = True

		if controlled:
			spacebins, velbins, datacube, params = read_datacube('/media/data/simulations/parameterspace_models/iso_fbar0.01_BT0_FB0_GF10/data/datacube_incl60.dat')
			spec = np.loadtxt('/media/data/simulations/parameterspace_models/iso_fbar0.01_BT0_FB0_GF10/data/spectrum_incl60.dat')
			kinem_out = './data/controlled_kinem_incl60_annulartest3.png'
			spec_out = './data/controlled_spec_incl60.png'

		if genesis:
			spacebins, velbins, datacube, params = read_datacube('/media/data/simulations/Genesis/data/datacube_incl60.dat')
			spec = np.loadtxt('/media/data/simulations/Genesis/data/spectrum_incl60.dat')
			kinem_out = './data/genesis_kinem_incl60_fitV0.png'
			spec_out = './data/genesis_spec_incl60.png'

		if TNG:
			spacebins, velbins, datacube, params = read_datacube('/media/data/simulations/IllustrisTNG/data/datacube_incl60.dat')
			spec = np.loadtxt('/media/data/simulations/IllustrisTNG/data/spectrum_incl60.dat')
			kinem_out = './data/TNG_kinem_incl60_fitV0.png'
			spec_out = './data/TNG_spec_incl60.png'

		# vel = spec[:,0]
		# spec = spec[:,1]
		# peaks =	locate_peaks(spec)
		# widths = locate_width(spec, [spec[peaks[0]],spec[peaks[1]]], 0.2)
		# Sint, Afr = areal_asymmetry(spec,widths,np.abs(np.diff(vel)[0]))

		# fig,ax = plt.subplots()
		# ax.plot(vel,spec,color='Black')
		# ax.set_xlabel('Velocity',fontsize=15)
		# ax.set_ylabel('Spectral flux',fontsize=15)
		# ax.set_ylim([0,1.1*np.max(spec)])
		# ax.text(0.05,0.95,'Afr = {afr:.3f}'.format(afr=Afr),transform=ax.transAxes,fontsize=12)
		# fig.savefig(spec_out)


		dx = np.abs(np.diff(spacebins)[0])

		# print(dx)
		# exit()

		mom0  = np.nansum(datacube/mjy_conversion(params['dist'], params['Vres']), axis=2) /  (( dx * 1.e3) ** 2.e0)
		mom0 = convolve(mom0,Gaussian2DKernel(2/dx))
		bad_sens = np.where(mom0 < 0.1)


		velcube = np.zeros([len(spacebins)-1,len(spacebins)-1,len(velbins)-1])
		for ii in range(len(velbins)-1):
			velcube[:,:,ii] = velbins[ii] + 0.5e0 * np.abs(np.diff(velbins))[0]

		mom1 = np.nansum(datacube * velcube, axis = 2) / np.nansum(datacube, axis=2)

		mom0[bad_sens] = np.nan 
		mom1[bad_sens] = np.nan

		# plt.imshow(mom1)
		# plt.show()
		# exit()

		if controlled:
			centre = np.where(mom0 == np.nanmax(mom0))
			x0 = centre[1][0]
			y0 = centre[0][0]
		else:
			x0 = y0 = 0.5*len(mom0)
		# print(centre[0][0],centre[1][0])


		ellipse_params0, harmonic_coeffs_mom0 = fd.harmonic_decomposition(mom0, moment = 0,radii = 'linear', image = True)
		ellipse_params1, harmonic_coeffs_mom1 = fd.harmonic_decomposition(mom1, moment = 1,radii = 'linear', PAQ = [0,0.5],
												Vsys = False, centre = [x0,y0], image = True)


		mom0 = np.log10(mom0)
		fd.plot_radial_asym_measures(ellipse_params0, ellipse_params1, 
			harmonic_coeffs_mom0, harmonic_coeffs_mom1, mom0, mom1, dx, save=kinem_out)


		controlled = genesis = TNG = False
		exit()

### datacube and spectra creation /saving


def HI_datacube_spectrum_simulation():

	incl = 60

	filename = '/media/data/simulations/parameterspace_models/iso_fbar0.01_BT0_FB0_GF10/snaps/snapshot_000.hdf5'
	gas_coordinates, gas_velocities, HI_masses = read_controlled_run(filename)
	outname_dcb = '/media/data/simulations/parameterspace_models/iso_fbar0.01_BT0_FB0_GF10/data/datacube_incl{incl}.dat'.format(incl=incl)
	outname_spec = '/media/data/simulations/parameterspace_models/iso_fbar0.01_BT0_FB0_GF10/data/spectrum_incl{incl}.dat'.format(incl=incl)

	gas_coordinates_obs = calc_coords_obs(gas_coordinates, 0,incl)
	gas_vel_obs = calc_vel_obs(gas_velocities, 0, incl)

	# vel,spec = create_HI_spectrum(gas_coordinates_obs, gas_vel_obs, HI_masses, FWHM = [200,200])
	# plt.plot(vel,spec)

	# vel,spec = create_HI_spectrum(gas_coordinates_obs, gas_vel_obs, HI_masses, FWHM = [100,100])
	# plt.plot(vel,spec)

	# vel,spec = create_HI_spectrum(gas_coordinates_obs, gas_vel_obs, HI_masses, FWHM = [50,50])
	# plt.plot(vel,spec)
	# plt.show()
	# exit()


	create_HI_datacube(gas_coordinates_obs, gas_vel_obs, HI_masses, filename=outname_dcb)
	create_HI_spectrum(gas_coordinates_obs,gas_vel_obs,HI_masses, filename = outname_spec)

	filename = '/media/data/simulations/EAGLE_galaxies/EAGLE_galaxyID8253667.hdf5'
	gas_coordinates, gas_velocities, HI_masses = read_EAGLE(filename)
	outname_dcb = '/media/data/simulations/EAGLE_galaxies/data/datacube_ID8253667_incl{incl}.dat'.format(incl = incl)
	outname_spec = '/media/data/simulations/EAGLE_galaxies/data/spectrum_ID8253667_incl{incl}.dat'.format(incl = incl)

	gas_coordinates_obs = calc_coords_obs(gas_coordinates, 0,incl)
	gas_vel_obs = calc_vel_obs(gas_velocities, 0, incl)


	create_HI_datacube(gas_coordinates_obs, gas_vel_obs, HI_masses, filename=outname_dcb)
	create_HI_spectrum(gas_coordinates_obs,gas_vel_obs,HI_masses, filename = outname_spec)

	filename = '/media/data/simulations/IllustrisTNG/TNG_556247'
	gas_coordinates, gas_velocities, HI_masses = read_TNG(filename)
	outname_dcb = '/media/data/simulations/IllustrisTNG/data/datacube_incl{incl}.dat'.format(incl=incl)
	outname_spec = '/media/data/simulations/IllustrisTNG/data/spectrum_incl{incl}.dat'.format(incl=incl)

	gas_coordinates_obs = calc_coords_obs(gas_coordinates, 0,incl)
	gas_vel_obs = calc_vel_obs(gas_velocities, 0, incl)


	create_HI_datacube(gas_coordinates_obs, gas_vel_obs, HI_masses, filename=outname_dcb)
	create_HI_spectrum(gas_coordinates_obs,gas_vel_obs,HI_masses, filename = outname_spec)

	filename = '/media/data/simulations/Genesis/snapshot_199.hdf5'
	gas_coordinates, gas_velocities, HI_masses = read_Genesis(filename)
	outname_dcb = '/media/data/simulations/Genesis/data/datacube_incl{incl}.dat'.format(incl=incl)
	outname_spec = '/media/data/simulations/Genesis/data/spectrum_incl{incl}.dat'.format(incl=incl)


	gas_coordinates_obs = calc_coords_obs(gas_coordinates, 0, incl)
	gas_vel_obs = calc_vel_obs(gas_velocities, 0, incl)

	# bins, image = calc_spatial_dist(gas_coordinates_obs,HI_masses,50)
	# plt.imshow(image)
	# plt.show()
	# exit()



	create_HI_datacube(gas_coordinates_obs, gas_vel_obs, HI_masses, filename = outname_dcb)
	create_HI_spectrum(gas_coordinates_obs, gas_vel_obs, HI_masses, filename = outname_spec)


#### simulation reading###

def read_TNG(base):

	f = open(base + '_stars.txt')
	names = f.readline().split(', ')
	f.close()
	names[0] = names[0].split('# ')[-1]
	names[-1] = names[-1].split('\n')[0]

	stars = np.loadtxt('{}_stars.txt'.format(base),skiprows=2)
	stars_coordinates = stars[:,1:4]
	stars_velocities = stars[:,4:7]
	stars_masses = stars[:,7] #* unitmass

	gas = np.loadtxt('{}_gas.txt'.format(base),skiprows=2)
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

	COM_gas = calc_COM(gas_coordinates, gas_neutral_masses, Rmax=10)
	COM_stars = calc_COM(stars_coordinates, stars_masses)
	gas_coordinates -= COM_gas
	stars_coordinates -= COM_stars

	gas_eigvec = orientation_matrix(gas_coordinates, gas_neutral_masses)
	# gas_eigvec = diagonalise_inertia(gas_coordinates, gas_neutral_masses,10)
	gas_coordinates = gas_coordinates @ gas_eigvec
	stars_coordinates = stars_coordinates @ gas_eigvec

	# plt.scatter(gas_coordinates[:,0],gas_coordinates[:,2],s=0.1)
	# plt.show()
	# exit()

	gas_velocities = gas_velocities @ gas_eigvec


	return gas_coordinates, gas_velocities, HImass_K13

def read_EAGLE(filename):

	unitmass = 1.e10
	catalogue = Table.read('/media/data/simulations/EAGLE_galaxies/EAGLE_cat.ascii', format='ascii')
	ID = int(filename.split('ID')[-1].split('.')[0])
	catref = np.where(np.array(catalogue['GalaxyID']) == ID)[0]


	file = h5py.File(filename,'r')
	if 'PartType0' in list(file.keys()):
		head = file['Header']
		DM = file['PartType1']
		stars = file['PartType4']
		gas = file['PartType0']

		a = head.attrs['ExpansionFactor']
		h = head.attrs['HubbleParam']

		[DM_coordinates, DM_velocities] = particle_info(a, h, DM, unitmass, ['Coordinates','Velocity'])
		DM_masses = np.ones(len(DM_coordinates))*head.attrs['MassTable'][1] * a**(0) * h**(-1)

		[gas_coordinates, gas_masses, gas_velocities, gas_densities, gas_internal_energy, gas_temperature] = \
				particle_info(a, h, gas, unitmass, ['Coordinates','Mass','Velocity','Density','InternalEnergy','Temperature'])

		[stars_coordinates, stars_masses, stars_velocities] = \
				particle_info(a, h, stars, unitmass, ['Coordinates','Mass','Velocity'])

		COP = np.array(catalogue['CentreOfPotential_x','CentreOfPotential_y','CentreOfPotential_z'][catref])[0]
		COM = np.array(catalogue['CentreOfMass_x','CentreOfMass_y','CentreOfMass_z'][catref])[0]
		COV = np.array(catalogue['Velocity_x','Velocity_y','Velocity_z'][catref])[0]
		COP_group = np.array(catalogue['GroupCentreOfPotential_x','GroupCentreOfPotential_y','GroupCentreOfPotential_z'][catref])[0]
		
		COP = np.array([i for i in COP])
		COP_group = np.array([i for i in COP_group])
		COM = np.array([i for i in COM])
		COV = np.array([i for i in COV])

		DM_coordinates -= COP
		gas_coordinates -= COP
		stars_coordinates -= COP

		stars_radii = np.sqrt(np.nansum(stars_coordinates**2.e0, axis=1))
		gas_radii = np.sqrt(np.nansum(gas_coordinates**2.e0, axis=1))
		DM_radii = np.sqrt(np.nansum(DM_coordinates**2.e0, axis=1))

		p_crit = 3 * (h*100)**2.e0 / (8 * np.pi * (4.3e-3 *1.e-6*1.e10 )  )		# in 1.e10Msun/Mpc^3
		Rvir = 0.005
		rho = 200 * p_crit
		while(rho >= 200 * p_crit):
			rho = np.nansum(DM_masses[DM_radii < Rvir]) / (4. * np.pi * Rvir*Rvir*Rvir / 3.)
			Rvir += 0.01
	
		DM_coordinates = DM_coordinates[DM_radii<=Rvir]
		DM_masses = DM_masses[DM_radii<=Rvir]*1.e10
		DM_velocities = DM_velocities[DM_radii<=Rvir]

		stars_coordinates = stars_coordinates[stars_radii<=Rvir]
		stars_masses = stars_masses[stars_radii<=Rvir]
		stars_velocities = stars_velocities[stars_radii<=Rvir]

		gas_coordinates = gas_coordinates[gas_radii<=Rvir]
		gas_masses = gas_masses[gas_radii<=Rvir]
		gas_velocities = gas_velocities[gas_radii<=Rvir]
		gas_densities = gas_densities[gas_radii<=Rvir]
		gas_internal_energy = gas_internal_energy[gas_radii<=Rvir]
		gas_temperature = gas_temperature[gas_radii<=Rvir]

		print('DM mass[1.e12]',np.nansum(DM_masses) / 1.e12)
		print('stellar mass [1.e10]',np.nansum(stars_masses) / 1.e10)
		print('Total gas mass [1.e10]', np.nansum(gas_masses) / 1.e10)
		print('Stellar fraction',np.nansum(stars_masses) / np.nansum(DM_masses))
		print('Total Gas fraction',np.nansum(gas_masses) / np.nansum(stars_masses))
		Rvir *= 1.e3
		print('Virial radius',Rvir, ' kpc')
		
		DM_coordinates *= 1.e3
		gas_coordinates *= 1.e3
		stars_coordinates *= 1.e3

		DM_velocities -= COV
		gas_velocities -= COV
		stars_velocities -= COV

		COM_stars_tot = np.array([0.,0.,0.])
		
		COM_stars = calc_COM(stars_coordinates, stars_masses, Rmax = 0.1*Rvir)
		gas_coordinates -= COM_stars
		stars_coordinates -= COM_stars

		COM_stars_tot -= COM_stars

		eigvec = orientation_matrix(stars_coordinates, stars_masses)
		stars_coordinates = stars_coordinates @ eigvec
		gas_coordinates = gas_coordinates @ eigvec
		DM_coordinates = DM_coordinates @ eigvec

		COM_stars = calc_COM(stars_coordinates, stars_masses, Rmax = 0.2*Rvir, Zmax=0.1*Rvir)
		gas_coordinates -= COM_stars
		stars_coordinates -= COM_stars
		COM_stars_tot -= COM_stars

		# plt.scatter(stars_coordinates[:,0],stars_coordinates[:,2],s=0.1)
		# plt.scatter(gas_coordinates[:,0],gas_coordinates[:,2],s=0.1)
		# plt.show()
		# exit()

		HI_masses, H2_masses, gas_neutral_masses = calc_HI_H2_ARHS(gas,unitmass,a,h)
		HI_masses = HI_masses[gas_radii <= Rvir/1.e3]
		H2_masses = H2_masses[gas_radii <=Rvir/1.e3]

		gas_velocities = gas_velocities @ eigvec
		DM_velocities = DM_velocities @ eigvec
		stars_velocities = stars_velocities @ eigvec

	return gas_coordinates, gas_velocities, HI_masses

def read_Genesis(filename):
	unitmass = 1.e10
	Rvir = 0.2697

	COD = np.array([13.3031, 34.7678, 40.9624])
	COM = np.array([13.2992, 34.7688, 40.9632])
	COV = np.array([-33.0769, -5.8676, -98.8213])
	COM_offset = np.array([13.3016, 34.7614, 40.9740])
	COV_offset = np.array([-34.6948,  6.1456, -95.8170])

	file = h5py.File(filename, 'r')

	parttypes = list(file.keys())
	head = file['Header']
	DM = file['PartType1']
	stars = file['PartType5']
	gas = file['PartType0']

	a = head.attrs['Redshift']
	a = 1.e0 / (1.e0 + a)
	h = head.attrs['HubbleParam']

	[DM_coordinates, DM_masses, DM_velocities] = \
		particle_info(a, h ,DM, unitmass, ['Coordinates','Masses','Velocities'],comoving = False)
	[gas_coordinates, gas_masses, gas_velocities] =\
	 particle_info(a, h ,gas, unitmass, ['Coordinates','Masses','Velocities'],comoving = False)

	[stars_coordinates, stars_masses, stars_velocities] = \
		particle_info(a, h ,stars, unitmass, ['Coordinates','Masses','Velocities'],comoving = False)

	DM_radii = np.sqrt(np.nansum((DM_coordinates - COM)**2.e0, axis=1))
	stars_radii = np.sqrt(np.nansum((stars_coordinates - COM)**2.e0, axis=1))
	gas_radii = np.sqrt(np.nansum((gas_coordinates - COM)**2.e0, axis=1))

	DM_virial = np.where(DM_radii <= Rvir)[0]
	stars_virial = np.where(stars_radii <= Rvir)[0]
	gas_virial = np.where(gas_radii <= Rvir)[0]


	[DM_coordinates, DM_masses, DM_velocities] = \
		particle_info(a, h, DM, unitmass, ['Coordinates','Masses','Velocities'], 
		subset = DM_virial, comoving = False)
	
	[gas_coordinates, gas_masses, gas_velocities, gas_densities, gas_neutral_fraction, \
				gas_internal_energy, gas_Z] = particle_info(a, h, gas, unitmass, ['Coordinates','Masses',\
				'Velocities','Density','NeutralHydrogenAbundance','InternalEnergy','Metallicity'],
				subset = gas_virial, comoving = False)
	gas_neutral_masses = gas_masses * gas_neutral_fraction

	[stars_coordinates, stars_masses, stars_velocities] = \
		particle_info(a, h, stars, unitmass, ['Coordinates','Masses','Velocities'], 
		subset = stars_virial, comoving = False)

	DM_coordinates = (DM_coordinates - COM) * 1.e3
	stars_coordinates = (stars_coordinates - COM) * 1.e3
	gas_coordinates = (gas_coordinates - COM) * 1.e3
	gas_velocities -= COV

	COM_gas = calc_COM(gas_coordinates, gas_neutral_masses,Rmax = 100)
	COM_stars = calc_COM(stars_coordinates, stars_masses)
	gas_coordinates -= COM_gas
	stars_coordinates -= COM_stars

	# plt.scatter(gas_coordinates[:,0],gas_coordinates[:,2],s=0.1)
	# plt.show()

	# gas_eigvec = diagonalise_inertia(gas_coordinates, gas_neutral_masses, 20)

	gas_eigvec = orientation_matrix(gas_coordinates, gas_neutral_masses)
	gas_coordinates = gas_coordinates @ gas_eigvec
	stars_coordinates = stars_coordinates @ gas_eigvec

	# plt.scatter(gas_coordinates[:,0],gas_coordinates[:,2],s=0.1)
	# plt.show()
	# exit()


	gas_velocities = gas_velocities @ gas_eigvec


	gas_SFR = np.zeros(len(gas_masses))
	gas_density_unit = gas_densities * unitmass * 1.e-18

	gas_internal_energy_unit = gas_internal_energy * 1.e3 * 1.e3

	HI_masses, H2_masses = galcalc_ARHS.HI_H2_masses(
					gas_masses, gas_SFR, gas_Z, gas_density_unit, 
					gas_internal_energy_unit, gas_neutral_fraction, 0, mode='u')

	radii = np.sqrt(np.nansum(gas_coordinates**2.e0,axis=1))

	return gas_coordinates[radii<50], gas_velocities[radii<50], HI_masses[radii<50]

def read_controlled_run(filename):
	# filename = '/media/data/simulations/isolated_models/fstar1/BT0/GF10_fB0_fhalo0/snaps/snapshot_030.hdf5'
	# filename = '/media/data/simulations/disktest/iso_fstar0.01_BT0_fB10_GF10/snaps/snapshotGD2_011.hdf5'

	unitmass = 1.e10 									#1.e10 Msun

	file = h5py.File(filename,'r')
	parttypes = list(file.keys())
	head = file['Header']
	DM = file['PartType1']
	disk = file['PartType2']
	gas = file['PartType0']

	[DM_coordinates, DM_masses, DM_velocities] = \
		particle_info(1,1,DM, unitmass, ['Coordinates','Masses','Velocities'],comoving=False)
	[gas_coordinates, gas_masses, gas_velocities, gas_density, \
		gas_internal_energy] = particle_info(1,1,gas, unitmass, ['Coordinates','Masses',\
			'Velocities','Density','InternalEnergy'],comoving=False)
	[disk_coordinates, disk_masses, disk_velocities] = \
		particle_info(1,1,disk, unitmass, ['Coordinates','Masses','Velocities'],comoving=False)

	if 'PartType5' in parttypes:
		newstars = file['PartType5']

		[newstars_coordinates, newstars_masses, newstars_velocities] = \
		particle_info(1,1,newstars, unitmass, ['Coordinates','Masses','Velocities'],comoving=False)

		stars_coordinates = np.append(stars_coordinates, newstars_coordinates, axis=0)
		stars_masses = np.append(stars_masses, newstars_masses)
		stars_velocities = np.append(stars_velocities, newstars_velocities)
	else:
		stars_coordinates = disk_coordinates
		stars_masses = disk_masses
		stars_velocities = disk_velocities

	COM_DM = calc_COM(DM_coordinates,DM_masses,5.e3)

	DM_coordinates -= COM_DM
	stars_coordinates -= COM_DM
	disk_coordinates -= COM_DM
	gas_coordinates -= COM_DM - calc_COM(gas_coordinates,gas_masses)



	gas_SFR = np.zeros(len(gas_masses))
	gas_Z = np.zeros(len(gas_masses)) + 0.001
	gas_density *= unitmass * 1.e-9 							#1.e10 Msun/kpc^3 -> Msun/pc^3
	gas_internal_energy *= 1.e3 * 1.e3 							#(km/s)^2 -> (m/s)^2

	if 'GD2' in filename:
		HI_masses, H2_masses, gas_neutral_fraction = galcalc_ARHS.HI_H2_masses(
					gas_masses, gas_SFR, gas_Z, gas_density, 
					gas_internal_energy, None, 0, mode='u')
	else:
		[gas_neutral_fraction] = particle_info(1,1,gas,unitmass,['NeutralHydrogenAbundance'],comoving=False)
		HI_masses, H2_masses = galcalc_ARHS.HI_H2_masses(
					gas_masses, gas_SFR, gas_Z, gas_density, 
					gas_internal_energy, gas_neutral_fraction, 0, mode='u')

	return gas_coordinates, gas_velocities, HI_masses
		

## resolution tests

def resolution_test():

	comm = MPI.COMM_WORLD
	rank = comm.Get_rank()
	nproc = comm.Get_size()

	unitmass = 1.e10
	basedir = sys.argv[1]
	
	snaplist = [0,25,50,75]
	plot = True

	if plot:
		for tt in range(len(snaplist)):
			snap = snaplist[tt]
			phi_list = [0,45,90]
			theta_list = [90,50,20]
			Npart_list = [50, 100, 500, 1000, 5000,10000, 50000, 100000]
			Afr_list = np.zeros([len(Npart_list),1000,len(phi_list)*len(theta_list)])
			Afr_med_errs = np.zeros([len(Npart_list),len(phi_list)*len(theta_list),2])

			Nproc = len(glob.glob('{dir}/data/Afr_list_snap0_proc*'.format(dir=basedir)))
			for proc in range(Nproc):
				file = '{dir}/data/Afr_list_snap{snap}_proc{proc}.dat'.format(dir = basedir,snap=snap,proc=proc)
				data = np.loadtxt(file)
				data = data.reshape( (len(Npart_list),1000,len(phi_list)*len(theta_list)) )
				Afr_list += data

			for nn in range(len(Npart_list)):
				for pp in range(len(phi_list)):
					for tt in range(len(theta_list)):
						Afr_med_errs[nn,pp*len(phi_list)+tt,0] = np.median(Afr_list[nn,:,pp*len(phi_list)+tt])
						Afr_med_errs[nn,pp*len(phi_list)+tt,1] = median_absolute_deviation(Afr_list[nn,:,pp*len(phi_list)+tt])

			Afrfig = plt.figure(figsize=(16,8))
			Afr_gs = gridspec.GridSpec(3,1) 
			Afr_ax1 = Afrfig.add_subplot(Afr_gs[0,0])
			Afr_ax2 = Afrfig.add_subplot(Afr_gs[1,0], sharex = Afr_ax1)
			Afr_ax3 = Afrfig.add_subplot(Afr_gs[2,0], sharex = Afr_ax1)
			Afr_ax1.set_ylabel('Asymmetry measure A$_{fr}$',fontsize = 15)
			Afr_ax2.set_ylabel('Asymmetry measure A$_{fr}$',fontsize = 15)
			Afr_ax3.set_ylabel('Asymmetry measure A$_{fr}$',fontsize = 15)
			Afr_ax3.set_xlabel('Number of particles',fontsize = 15)
			Afr_ax1.set_xscale('log')
			Afr_ax2.set_xscale('log')
			Afr_ax3.set_xscale('log')
			Afr_ax3.tick_params(axis = 'both', which = 'both', direction = 'in', labelsize = 13)
			Afr_ax1.tick_params(axis = 'x', which = 'both', direction = 'in', labelsize = 0)
			Afr_ax1.tick_params(axis = 'x', which = 'both', direction = 'in', labelsize = 0)

			axes = [Afr_ax1, Afr_ax2, Afr_ax3]

			ls = ['-','--',':']


			for pp in range(len(phi_list)):
					for tt in range(len(theta_list)):
						axes[tt].errorbar(Npart_list, Afr_med_errs[:,pp*len(phi_list)+tt,0],
							yerr = Afr_med_errs[:,pp*len(phi_list)+tt,1], ls = ls[tt], color='C{}'.format(pp))

			leg = [Line2D([0],[0],ls='-',color='Black'),
					Line2D([0],[0],ls='--',color='Black'),
					Line2D([0],[0],ls=':',color='Black'),
					Line2D([0],[0],ls='-',color='C0'),
					Line2D([0],[0],ls='-',color='C1'),
					Line2D([0],[0],ls='-',color='C2')]				
			Afr_ax1.legend(leg,['i = 90', 'i = 50', 'i = 20','$\phi$ = 0','$\phi$ = 45','$\phi$ = 90'])
			# plt.show()
			Afr_figname = '{dir}/figures/snaps{snap}_Afr_Npart.png'.format(dir=basedir,snap = snap)
			Afrfig.savefig(Afr_figname, dpi=200)



		# specfig = plt.figure(figsize=(10,8))
		# spec_gs = gridspec.GridSpec(1,1) 
		# spec_ax = specfig.add_subplot(spec_gs[0,0])
		# spec_ax.set_xlabel('Velocity')
		# spec_ax.set_ylabel('Spectral flux')
		# spec_ax.set_ylim([0,40])
		# spec_ax.legend(fontsize=10)
		# spec_figname = '{dir}/figures/snaps{tt}_spec_Npart.png'.format(dir=basedir,tt=str(tt).zfill(3))
		# specfig.savefig(spec_figname, dpi=200)

	else:

		for ii in range(len(snaplist)):#range(rank,len(snaplist),nproc):
		
			if rank == 0:
				snap = snaplist[ii]

				filename = '{dir}snaps/snapshotGD2_{snap}.hdf5'.format(dir=basedir, snap=str(snap).zfill(3))
				file = h5py.File(filename,'r')
				parttypes = list(file.keys())
				head = file['Header']
				DM = file['PartType1']
				disk = file['PartType2']
				gas = file['PartType0']

				[DM_coordinates, DM_masses, DM_velocities] = \
					particle_info(1,1,DM, unitmass, ['Coordinates','Masses','Velocities'],comoving=False)
				[gas_coordinates, gas_masses, gas_velocities, gas_densities, \
					gas_internal_energy] = particle_info(1,1,gas, unitmass, ['Coordinates','Masses',\
						'Velocities','Density','InternalEnergy'],comoving=False)
				[disk_coordinates, disk_masses, disk_velocities] = \
					particle_info(1,1,disk, unitmass, ['Coordinates','Masses','Velocities'],comoving=False)

				if 'PartType5' in parttypes:
					newstars = file['PartType5']

					[newstars_coordinates, newstars_masses, newstars_velocities] = \
					particle_info(1,1,newstars, unitmass, ['Coordinates','Masses','Velocities'],comoving=False)

					stars_coordinates = np.append(stars_coordinates, newstars_coordinates, axis=0)
					stars_masses = np.append(stars_masses, newstars_masses)
					stars_velocities = np.append(stars_velocities, newstars_velocities)
				else:
					stars_coordinates = disk_coordinates
					stars_masses = disk_masses
					stars_velocities = disk_velocities

				COM_DM = calc_COM(DM_coordinates,DM_masses,5.e3)

				DM_coordinates -= COM_DM
				stars_coordinates -= COM_DM
				disk_coordinates -= COM_DM
				gas_coordinates -= COM_DM

				gas_coords_edgeon = calc_coords_obs(gas_coordinates, 0, 90)
				gas_vel_edgeon = calc_vel_obs(gas_velocities, 0, 90)


				gas_masses_unit = gas_masses *unitmass
				gas_SFR = np.zeros(len(gas_masses))
				gas_Z = np.zeros(len(gas_masses)) + 0.001
				gas_density_unit = gas_densities * unitmass * 1.e-9
				gas_internal_energy_unit = gas_internal_energy * 1.e3*1.e3


				HI_masses, H2_masses, gas_neutral_masses = galcalc_ARHS.HI_H2_masses(
					gas_masses_unit,gas_SFR,gas_Z,gas_density_unit,gas_internal_energy_unit,None,0, mode='u')
				HI_masses = np.array(HI_masses)
			else:
				HI_masses = np.array([])
				gas_coordinates = np.array([])
				gas_velocities = np.array([])
				snap = None

			HI_masses = comm.bcast(HI_masses, root=0)
			gas_coordinates = comm.bcast(gas_coordinates, root=0)
			gas_velocities = comm.bcast(gas_velocities, root=0)
			snap = comm.bcast(snap,root = 0)


			phi_list = [0,45,90]
			theta_list = [90,50,20]
			Npart_list = [50, 100, 500, 1000, 5000,10000, 50000, 100000]
			Afr_list = np.zeros([len(Npart_list),1000,len(phi_list)*len(theta_list)])
			Afr_med_errs = np.zeros([len(Npart_list),len(phi_list)*len(theta_list),2])

			for nn in range(len(Npart_list)):
				Npart = Npart_list[nn]
				if rank == 0:
					HI_masses_temp = HI_masses * len(HI_masses) / Npart
				else:
					HI_masses_temp = None
				HI_masses_temp = comm.bcast(HI_masses_temp, root=0)

				for pp in range(len(phi_list)):
					for tt in range(len(theta_list)):
						phi = phi_list[pp]
						theta = theta_list[tt]
						if rank == 0:
							gas_coords_temp = calc_coords_obs(gas_coordinates,phi,theta)
							gas_vel_temp = calc_vel_obs(gas_velocities,phi,theta)
						else:
							gas_coords_temp = None
							gas_vel_temp = None

						gas_coords_temp = comm.bcast(gas_coords_temp, root=0)
						gas_vel_temp = comm.bcast(gas_vel_temp, root=0)

						for ss in range(rank,1000,nproc):
							particle_sample = nprand.choice(range(len(HI_masses)), Npart)
								
							vel, spectrum = calc_spectrum(gas_coords_temp[particle_sample,:], 
							gas_vel_temp[particle_sample], HI_masses_temp[particle_sample], beamsize = 40)

							Peak = np.nanmax(spectrum)
							widths = locate_width(spectrum, [Peak,Peak], 0.2)
							Sint, Afr = areal_asymmetry(spectrum, widths, np.abs(np.diff(vel)[0]))
							Afr_list[nn,ss,pp*len(phi_list)+tt] = Afr
			Afr_list.flatten()
			np.savetxt('./data/Afr_list_snap{snap}_proc{rank}.dat'.format(snap = snap, rank=rank), 
				Afr_list.reshape(len(Npart_list)*1000*len(phi_list)*len(theta_list)))

			comm.Barrier()
			
def resolution_test_EAGLE():

	comm = MPI.COMM_WORLD
	rank = comm.Get_rank()
	nproc = comm.Get_size()

	unitmass = 1.e10
	gals = [8339149]#,2958109]

	snaplist = [0,25,50,75]
	plot = True

	basedir = '/media/data/simulations/EAGLE_galaxies/'

	if plot:

		for ID in gals:

			phi_list = [0,45,90]
			theta_list = [90,50,20]
			Npart_list = [50, 100, 500, 1000, 5000, 10000, 20000]
			Afr_list = np.zeros([len(Npart_list),1000,len(phi_list)*len(theta_list)])
			Afr_med_errs = np.zeros([len(Npart_list),len(phi_list)*len(theta_list),2])
			

			filename = '{basedir}EAGLE_galaxyID{ID}.hdf5'.format(basedir = basedir,ID=ID)

			Nproc = len(glob.glob('{basedir}/data/restest{ID}_proc*'.format(basedir = basedir,ID=ID)))
			for proc in range(Nproc):
				file = '{basedir}/data/restest{ID}_proc{proc}.dat'.format(basedir = basedir, ID=ID,proc=proc)
				data = np.loadtxt(file)
				data = data.reshape( (len(Npart_list),1000,len(phi_list)*len(theta_list)) )
				Afr_list += data

			for nn in range(len(Npart_list)):
				for pp in range(len(phi_list)):
					for tt in range(len(theta_list)):
						Afr_med_errs[nn,pp*len(phi_list)+tt,0] = np.median(Afr_list[nn,:,pp*len(phi_list)+tt])
						Afr_med_errs[nn,pp*len(phi_list)+tt,1] = median_absolute_deviation(Afr_list[nn,:,pp*len(phi_list)+tt])

			Afrfig = plt.figure(figsize=(16,8))
			Afr_gs = gridspec.GridSpec(3,1) 
			Afr_ax1 = Afrfig.add_subplot(Afr_gs[0,0])
			Afr_ax2 = Afrfig.add_subplot(Afr_gs[1,0], sharex = Afr_ax1)
			Afr_ax3 = Afrfig.add_subplot(Afr_gs[2,0], sharex = Afr_ax1)
			Afr_ax1.set_ylabel('Asymmetry measure A$_{fr}$',fontsize = 15)
			Afr_ax2.set_ylabel('Asymmetry measure A$_{fr}$',fontsize = 15)
			Afr_ax3.set_ylabel('Asymmetry measure A$_{fr}$',fontsize = 15)
			Afr_ax3.set_xlabel('Number of particles',fontsize = 15)
			Afr_ax1.set_xscale('log')
			Afr_ax2.set_xscale('log')
			Afr_ax3.set_xscale('log')
			Afr_ax3.tick_params(axis = 'both', which = 'both', direction = 'in', labelsize = 13)
			Afr_ax1.tick_params(axis = 'x', which = 'both', direction = 'in', labelsize = 0)
			Afr_ax1.tick_params(axis = 'x', which = 'both', direction = 'in', labelsize = 0)

			axes = [Afr_ax1, Afr_ax2, Afr_ax3]

			ls = ['-','--',':']


			for pp in range(len(phi_list)):
					for tt in range(len(theta_list)):
						axes[tt].errorbar(Npart_list, Afr_med_errs[:,pp*len(phi_list)+tt,0],
							yerr = Afr_med_errs[:,pp*len(phi_list)+tt,1], ls = ls[tt], color='C{}'.format(pp))


			leg = [Line2D([0],[0],ls='-',color='Black'),
					Line2D([0],[0],ls='--',color='Black'),
					Line2D([0],[0],ls=':',color='Black'),
					Line2D([0],[0],ls='-',color='C0'),
					Line2D([0],[0],ls='-',color='C1'),
					Line2D([0],[0],ls='-',color='C2')]				
			Afr_ax1.legend(leg,['i = 90', 'i = 50', 'i = 20','$\phi$ = 0','$\phi$ = 45','$\phi$ = 90'])
			# plt.show()
			Afr_figname = '{basedir}/figures/EAGLE{ID}_Afr_Npart.png'.format(basedir=basedir,ID=ID)
			Afrfig.savefig(Afr_figname, dpi=200)

	else:

		for ii in range(len(gals)):#range(rank,len(snaplist),nproc):
		
			if rank == 0:
				ID = gals[ii]
				filename = '/home/awatts/Adam_PhD/models_fitting/GadgetSimulations/simulations/EAGLE_galaxies/EAGLE_galaxyID{ID}.hdf5'.format(ID=ID)
				
				gas_coordinates, gas_velocities, HI_masses = read_EAGLE(filename)
			else:
				HI_masses = np.array([])
				gas_coordinates = np.array([])
				gas_velocities = np.array([])
				ID = None

			HI_masses = comm.bcast(HI_masses, root=0)
			gas_coordinates = comm.bcast(gas_coordinates, root=0)
			gas_velocities = comm.bcast(gas_velocities, root=0)
			ID = comm.bcast(ID, root = 0)

			# print(ID)
			# print(len(HI_masses))
			comm.Barrier()
			# exit()

			phi_list = [0,45,90]
			theta_list = [90,50,20]
			Npart_list = [50, 100, 500, 1000, 5000, 10000, 20000]
			Afr_list = np.zeros([len(Npart_list),1000,len(phi_list)*len(theta_list)])
			Afr_med_errs = np.zeros([len(Npart_list),len(phi_list)*len(theta_list),2])

			for nn in range(len(Npart_list)):
				Npart = Npart_list[nn]
				if rank == 0:
					HI_masses_temp = HI_masses * len(HI_masses) / Npart
				else:
					HI_masses_temp = None
				HI_masses_temp = comm.bcast(HI_masses_temp, root=0)

				for pp in range(len(phi_list)):
					for tt in range(len(theta_list)):
						phi = phi_list[pp]
						theta = theta_list[tt]
						if rank == 0:
							gas_coords_temp = calc_coords_obs(gas_coordinates,phi,theta)
							gas_vel_temp = calc_vel_obs(gas_velocities,phi,theta)
						else:
							gas_coords_temp = None
							gas_vel_temp = None

						gas_coords_temp = comm.bcast(gas_coords_temp, root=0)
						gas_vel_temp = comm.bcast(gas_vel_temp, root=0)

						for ss in range(rank,1000,nproc):
							particle_sample = nprand.choice(range(len(HI_masses)), Npart)
								
							vel, spectrum = calc_spectrum(gas_coords_temp[particle_sample,:], 
							gas_vel_temp[particle_sample], HI_masses_temp[particle_sample], beamsize = 40)

							Peak = np.nanmax(spectrum)
							widths = locate_width(spectrum, [Peak,Peak], 0.2)
							Sint, Afr = areal_asymmetry(spectrum, widths, np.abs(np.diff(vel)[0]))
							Afr_list[nn,ss,pp*len(phi_list)+tt] = Afr
			Afr_list.flatten()
			np.savetxt('/home/awatts/Adam_PhD/models_fitting/GadgetSimulations/simulations/EAGLE_galaxies/data/restest{ID}_proc{rank}.dat'.format(ID=ID, rank=rank), 
				Afr_list.reshape(len(Npart_list)*1000*len(phi_list)*len(theta_list)))

			comm.Barrier()

def resolution_test_TNG():

	comm = MPI.COMM_WORLD
	rank = comm.Get_rank()
	nproc = comm.Get_size()

	unitmass = 1.e10
	ID = 556247

	Nsamp = 10000


	# snaplist = [0, 25, 50, 75]
	plot1 = False
	plot2 = False
	plot3 = False
	plot4 = False

	# basedir = '/media/data/simulations/IllustrisTNG/'
	# basedir = '/media/data/simulations/IllustrisTNG/'
	basedir = '/home/awatts/Adam_PhD/models_fitting/GadgetSimulations/simulations/IllustrisTNG/'

	if plot1:

		phi_list = [0,45,90]
		theta_list = [90,50,20]
		# Npart_list = [50, 70, 100, 200, 500, 700, 1000, 2000, 5000, 7000, 10000, 14000]
		Npart_list = [50,500, 5000,  14000]
		Afr_list = np.zeros([len(Npart_list),1000,len(phi_list)*len(theta_list),2])
		Afr_med_errs = np.zeros([len(Npart_list),len(phi_list)*len(theta_list),2])
		

		Nproc = len(glob.glob('{basedir}/data/restest{ID}_proc*'.format(basedir = basedir,ID=ID)))
		for proc in range(Nproc):
			file = '{basedir}/data/restest{ID}_proc{proc}.dat'.format(basedir = basedir, ID=ID,proc=proc)
			data = np.loadtxt(file)
			data = data.reshape( (len(Npart_list),1000,len(phi_list)*len(theta_list),2) )
			Afr_list += data

		for nn in range(len(Npart_list)):
			for pp in range(len(phi_list)):
				for tt in range(len(theta_list)):
					Afr_med_errs[nn,pp*len(phi_list)+tt,0] = np.median(Afr_list[nn,:,pp*len(phi_list)+tt,0])
					Afr_med_errs[nn,pp*len(phi_list)+tt,1] = median_absolute_deviation(Afr_list[nn,:,pp*len(phi_list)+tt,0])

		print(Afr_med_errs[:,:,1])

		Afrfig = plt.figure(figsize=(14,10))
		Afr_gs = gridspec.GridSpec(3,1) 
		Afr_ax1 = Afrfig.add_subplot(Afr_gs[0,0])
		Afr_ax2 = Afrfig.add_subplot(Afr_gs[1,0], sharex = Afr_ax1,sharey = Afr_ax1)
		Afr_ax3 = Afrfig.add_subplot(Afr_gs[2,0], sharex = Afr_ax1,sharey = Afr_ax1)
		Afr_ax1.set_ylabel('Asymmetry measure A$_{fr}$',fontsize = 15)
		Afr_ax2.set_ylabel('Asymmetry measure A$_{fr}$',fontsize = 15)
		Afr_ax3.set_ylabel('Asymmetry measure A$_{fr}$',fontsize = 15)
		Afr_ax3.set_xlabel('Number of particles',fontsize = 15)
		Afr_ax1.set_xscale('log')
		Afr_ax2.set_xscale('log')
		Afr_ax3.set_xscale('log')
		Afr_ax3.tick_params(axis = 'both', which = 'both', direction = 'in', labelsize = 13)
		Afr_ax1.tick_params(axis = 'x', which = 'both', direction = 'in', labelsize = 0)
		Afr_ax1.tick_params(axis = 'x', which = 'both', direction = 'in', labelsize = 0)

		axes = [Afr_ax1, Afr_ax2, Afr_ax3]

		ls = ['-','--',':']


		for pp in range(len(phi_list)):
				for tt in range(len(theta_list)):
					axes[tt].errorbar(Npart_list, Afr_med_errs[:,pp*len(phi_list)+tt,0],
						yerr = Afr_med_errs[:,pp*len(phi_list)+tt,1],capsize=4, ls = ls[tt], color='C{}'.format(pp))


		leg = [Line2D([0],[0],ls='-',color='Black'),
				Line2D([0],[0],ls='--',color='Black'),
				Line2D([0],[0],ls=':',color='Black'),
				Line2D([0],[0],ls='-',color='C0'),
				Line2D([0],[0],ls='-',color='C1'),
				Line2D([0],[0],ls='-',color='C2')]				
		Afr_ax1.legend(leg,['i = 90', 'i = 50', 'i = 20','$\phi$ = 0','$\phi$ = 45','$\phi$ = 90'])
		# plt.show()
		Afr_figname = '{basedir}/figures/TNG_{ID}_Afr_Npart.png'.format(basedir=basedir,ID=ID)
		Afrfig.savefig(Afr_figname, dpi=200)

	elif plot2:
		particle_mass = 1.4e6

		phi_list = [0,45,90]
		theta_list = [90,50,20]
		Npart_list = [50, 70, 100, 200, 500, 700, 1000, 2000, 5000, 7000, 10000, 14000]
		# Npart_list = [50,500, 5000]
		percentiles = [10,20,30,40,50,60,70,80,90]
		Afr_list = np.zeros([len(Npart_list),Nsamp,len(phi_list)*len(theta_list),2])
		
		Afr_percentiles = np.zeros([len(Npart_list),len(phi_list)*len(theta_list),11])

		Nproc = len(glob.glob('{basedir}/data/restest{ID}_proc*'.format(basedir = basedir,ID=ID)))
		for proc in range(Nproc):
			file = '{basedir}/data/restest{ID}_proc{proc}.dat'.format(basedir = basedir, ID=ID,proc=proc)
			data = np.loadtxt(file)
			data = data.reshape( (len(Npart_list),Nsamp,len(phi_list)*len(theta_list),2) )
			Afr_list += data

		for nn in range(len(Npart_list)):
			for pp in range(len(phi_list)):
				for tt in range(len(theta_list)):
					Afr_percentiles[nn,pp*len(phi_list)+tt,0] = Npart_list[nn]
					
					Afr_percentiles[nn,pp*len(phi_list)+tt,1] = np.median(Afr_list[nn,:,pp*len(phi_list)+tt,1]/(5 * particle_mass * 14198 / Npart_list[nn]))

					for per in range(len(percentiles)):
						Afr_percentiles[nn,pp*len(phi_list)+tt,per+2] = np.percentile(Afr_list[nn,:,pp*len(phi_list)+tt,0],percentiles[per])
	
		for tt in range(len(theta_list)):
			Afrfig = plt.figure(figsize=(15,10))
			Afr_gs = gridspec.GridSpec(3,1) 
			Afr_ax1 = Afrfig.add_subplot(Afr_gs[0,0])
			Afr_ax2 = Afrfig.add_subplot(Afr_gs[1,0], sharex = Afr_ax1,sharey = Afr_ax1)
			Afr_ax3 = Afrfig.add_subplot(Afr_gs[2,0], sharex = Afr_ax1,sharey = Afr_ax1)
			Afr_ax1.set_ylabel('Asymmetry measure A$_{fr}$',fontsize = 15)
			Afr_ax2.set_ylabel('Asymmetry measure A$_{fr}$',fontsize = 15)
			Afr_ax3.set_ylabel('Asymmetry measure A$_{fr}$',fontsize = 15)
			Afr_ax3.set_xlabel('Number of particles',fontsize = 15)
			Afr_ax1.set_xscale('log')
			Afr_ax2.set_xscale('log')
			Afr_ax3.set_xscale('log')
			Afr_ax1.tick_params(axis = 'both', which = 'both', direction = 'in', labelsize = 13)
			Afr_ax2.tick_params(axis = 'both', which = 'both', direction = 'in', labelsize = 13)
			Afr_ax3.tick_params(axis = 'both', which = 'both', direction = 'in', labelsize = 13)
			Afr_ax1.tick_params(axis = 'x', which = 'both', direction = 'in', labelsize = 0)
			Afr_ax2.tick_params(axis = 'x', which = 'both', direction = 'in', labelsize = 0)


			Afr_ax1.plot(Afr_percentiles[:,tt,0],Afr_percentiles[:,tt,-1], color='Orange')
			Afr_ax1.plot(Afr_percentiles[:,tt,0],Afr_percentiles[:,tt,-5], color='Green')
			Afr_ax1.plot(Afr_percentiles[:,tt,1],Afr_percentiles[:,tt,-1], color='Orange',ls = '--')
			Afr_ax1.plot(Afr_percentiles[:,tt,1],Afr_percentiles[:,tt,-5], color='Green',ls = '--')
			Afr_ax1.set_title(r'$\theta$ = {th}  $\phi$ = {ph}'.format(th = theta_list[tt],ph = phi_list[0]))

			Afr_ax2.plot(Afr_percentiles[:,len(phi_list)+tt,0],Afr_percentiles[:,len(phi_list)+tt,-1], color='Orange')
			Afr_ax2.plot(Afr_percentiles[:,len(phi_list)+tt,0],Afr_percentiles[:,len(phi_list)+tt,-5], color='Green')
			Afr_ax2.plot(Afr_percentiles[:,len(phi_list)+tt,1],Afr_percentiles[:,len(phi_list)+tt,-1], color='Orange',ls = '--')
			Afr_ax2.plot(Afr_percentiles[:,len(phi_list)+tt,1],Afr_percentiles[:,len(phi_list)+tt,-5], color='Green',ls = '--')
			Afr_ax2.set_title(r'$\theta$ = {th}  $\phi$ = {ph}'.format(th = theta_list[tt],ph = phi_list[1]))

			Afr_ax3.plot(Afr_percentiles[:,2*len(phi_list)+tt,0],Afr_percentiles[:,2*len(phi_list)+tt,-1], color='Orange')
			Afr_ax3.plot(Afr_percentiles[:,2*len(phi_list)+tt,0],Afr_percentiles[:,2*len(phi_list)+tt,-5], color='Green')
			Afr_ax3.plot(Afr_percentiles[:,2*len(phi_list)+tt,1],Afr_percentiles[:,2*len(phi_list)+tt,-1], color='Orange',ls = '--')
			Afr_ax3.plot(Afr_percentiles[:,2*len(phi_list)+tt,1],Afr_percentiles[:,2*len(phi_list)+tt,-5], color='Green',ls = '--')
			Afr_ax3.set_title(r'$\theta$ = {th}  $\phi$ = {ph}'.format(th = theta_list[tt],ph = phi_list[2]))


			leg = [Line2D([0],[0],ls='-',color='Black'),
					Line2D([0],[0],ls='--',color='Black'),
					Line2D([0],[0],ls='-',color='Green'),
					Line2D([0],[0],ls='-',color='Orange')]

			Afr_ax1.legend(leg,['Raw Npart', 'Mass weighted Npart', 'P50', 'P90'])

			# Afr_figname = '{basedir}/figures/TNG_{ID}_Afr_Npart_percentile_th{th}.png'.format(basedir=basedir,ID=ID,th=theta_list[tt])
			# Afrfig.savefig(Afr_figname, dpi = 200)
			plt.close()

		names = ['Npart', 'Npart_HIweight']
		names.extend(['P{p}'.format(p=p) for p in percentiles])

		print(names)
		percentiles_sym = Table(Afr_percentiles[:,6,:],names = names)
		percentiles_sym.write('{basedir}/data/TNG_sym_percentile_stats.dat'.format(basedir=basedir),format='ascii')


	elif plot3:
		particle_mass = 1.4e6

		phi_list = [0,45,90]
		theta_list = [90,50,20]
		Npart_list = [50, 70, 100, 200, 500, 700, 1000, 2000, 5000, 7000, 10000, 14000]
		Afr_list = np.zeros([len(Npart_list),Nsamp,len(phi_list)*len(theta_list),2])
		
		Nproc = len(glob.glob('{basedir}/data/restest{ID}_proc*'.format(basedir = basedir,ID=ID)))
		for proc in range(Nproc):
			file = '{basedir}/data/restest{ID}_proc{proc}.dat'.format(basedir = basedir, ID=ID,proc=proc)
			data = np.loadtxt(file)
			data = data.reshape( (len(Npart_list),Nsamp,len(phi_list)*len(theta_list),2) )
			Afr_list += data

		Afr_500part = A_500part = Afr_list[4,:,6,0]
		Afr_1000part = A_1000part = Afr_list[6,:,6,0]
		Afr_5000part = A_5000part = Afr_list[8,:,6,0]


		A_500part[0:-1:2] = 1/Afr_500part[0:-1:2]
		A_1000part[0:-1:2] = 1/Afr_1000part[0:-1:2]
		A_5000part[0:-1:2] = 1/Afr_5000part[0:-1:2]




		fig = plt.figure(figsize = (15,9))
		gs  = gridspec.GridSpec(1, 2, left = 0.1, right = 0.99, top=0.97, bottom = 0.11)
		Aax = fig.add_subplot(gs[0,0])
		Afrax = fig.add_subplot(gs[0,1])

		Aax.set_xlim([-0.25,0.25])
		Afrax.set_xlim([0.99,1.85])

		Aax.tick_params( direction = 'in', labelsize = 25,length=8,width=1.5)
		Aax.tick_params( direction = 'in', labelsize = 25,length=8,width=1.5)

		Afrax.tick_params( direction = 'in', labelsize = 25,length=8,width=1.5)
		Afrax.tick_params( direction = 'in', labelsize = 25,length=8,width=1.5)


		Aax.hist(np.log10(A_5000part),bins=41,histtype= 'step',fill=False,density=True
			, lw = 3, color='Black', ls = ':')
		Aax.hist(np.log10(A_1000part),bins=41,histtype= 'step',fill=False,density=True
			, lw = 3, color='Blue', ls = '--')
		Aax.hist(np.log10(A_500part),bins=41,histtype= 'step',fill=False,density=True
			, lw = 3, color='Red', ls = '-')

		Afrax.hist(Afr_5000part,bins=np.arange(1,1.9,0.01),histtype= 'step',cumulative=True,fill=False,density=True
			, lw = 3, color='Black', ls = ':')
		Afrax.hist(Afr_1000part,bins=np.arange(1,1.9,0.01),histtype= 'step',cumulative=True,fill=False,density=True
			, lw = 3, color='Blue', ls = '--')
		Afrax.hist(Afr_500part,bins=np.arange(1,1.9,0.01),histtype= 'step',cumulative=True,fill=False,density=True
			, lw = 3, color='Red', ls = '-')
		


		Afrax.set_xlabel('Asymmetry measure A$_{fr}$',fontsize=27)
		Aax.set_xlabel('log$_{10}(A)$',fontsize=27)

		Afrax.set_ylabel('Cumulative Hisogram',fontsize=27)
		Aax.set_ylabel('Histogram Density',fontsize=27)



		legend = [Line2D([0], [0], color = 'Black',ls = ':', linewidth = 3),
					Line2D([0], [0], color = 'Blue',ls = '--', linewidth = 3),
					Line2D([0], [0], color = 'Red',ls = '-', linewidth = 3)]

		Aax.legend(legend,['N$_{part}$ = 5000','N$_{part}$ = 1000','N$_{part}$ = 500'],fontsize=24)

		fig.savefig('{basedir}/figures/logA_Afr_dist.png'.format(basedir=basedir))
		plt.show()


	elif plot4:
		particle_mass = 1.4e6
		phi_list = [0,45,90]
		theta_list = [90,50,20]
		Npart_list = [50, 70, 100, 200, 500, 700, 1000, 2000, 5000, 7000, 10000, 14000]
		# Npart_list = [50,500, 5000]
		percentiles = [10,20,30,40,50,60,70,80,90]
		Afr_list = np.zeros([len(Npart_list),Nsamp,len(phi_list)*len(theta_list),2])
		
		Afr_percentiles = np.zeros([len(Npart_list),len(phi_list)*len(theta_list),11])

		Nproc = len(glob.glob('{basedir}/data/restest{ID}_proc*'.format(basedir = basedir,ID=ID)))
		for proc in range(Nproc):
			file = '{basedir}/data/restest{ID}_proc{proc}.dat'.format(basedir = basedir, ID=ID,proc=proc)
			data = np.loadtxt(file)
			data = data.reshape( (len(Npart_list),Nsamp,len(phi_list)*len(theta_list),2) )
			Afr_list += data

		for nn in range(len(Npart_list)):
			for pp in range(len(phi_list)):
				for tt in range(len(theta_list)):
					Afr_percentiles[nn,pp*len(phi_list)+tt,0] = Npart_list[nn]
					
					Afr_percentiles[nn,pp*len(phi_list)+tt,1] = np.median(Afr_list[nn,:,pp*len(phi_list)+tt,1]/(5 * particle_mass * 14198 / Npart_list[nn]))

					for per in range(len(percentiles)):
						Afr_percentiles[nn,pp*len(phi_list)+tt,per+2] = np.percentile(Afr_list[nn,:,pp*len(phi_list)+tt,0],percentiles[per])
	
		Afrfig = plt.figure(figsize=(8,4))
		Afr_gs = gridspec.GridSpec(1,1) 
		Afr_ax1 = Afrfig.add_subplot(Afr_gs[0,0])
		Afr_ax1.set_ylabel('Asymmetry measure A$_{fr}$',fontsize = 16)
		Afr_ax1.set_xlabel('Number of particles',fontsize = 16)
		Afr_ax1.set_xscale('log')
		Afr_ax1.tick_params(axis = 'both', which = 'both', direction = 'in', labelsize = 16,length = 8, width = 1.25)
		Afr_ax1.tick_params(axis = 'both', which = 'minor', direction = 'in', labelsize = 16,length = 4, width = 1.25)


		Afr_ax1.plot(Afr_percentiles[:,6,0],Afr_percentiles[:,6,-1], color='Orange')
		Afr_ax1.plot(Afr_percentiles[:,6,0],Afr_percentiles[:,6,-5], color='Green')
		Afr_ax1.plot(Afr_percentiles[:,6,1],Afr_percentiles[:,6,-1], color='Orange',ls = '--')
		Afr_ax1.plot(Afr_percentiles[:,6,1],Afr_percentiles[:,6,-5], color='Green',ls = '--')

		

		leg = [Line2D([0],[0],ls='-',color='Black'),
				Line2D([0],[0],ls='--',color='Black'),
				Line2D([0],[0],ls='-',color='Green'),
				Line2D([0],[0],ls='-',color='Orange')]

		Afr_ax1.legend(leg,['Raw Npart', 'HI mass weighted', '50$^{th}$ percentile', '90$^{th}$ percentile'])

		plt.show()

		figname = '{basedir}/figures/sym_Npart_HI_percentiles.png'.format(basedir=basedir)
		Afrfig.savefig(figname, dpi = 200)
		plt.close()

	

	else:
	
		if rank == 0:
			filename = '{dir}TNG_{ID}'.format(dir = basedir, ID=ID)
			gas_coordinates, gas_velocities, HI_masses = read_TNG(filename)
			# print(np.median(HI_masses))
			# plt.hist(np.log10(HI_masses))
			# plt.show()
			# exit()
		else:
			HI_masses = np.array([])
			gas_coordinates = np.array([])
			gas_velocities = np.array([])

		HI_masses = comm.bcast(HI_masses, root=0)
		gas_coordinates = comm.bcast(gas_coordinates, root=0)
		gas_velocities = comm.bcast(gas_velocities, root=0)

		# print(ID)
		# print(len(HI_masses))
		comm.Barrier()
		# exit()


		phi_list = [74.8,53.48,7.06]
		theta = 90

		Npart_list = [50, 70, 100, 200, 500, 700, 1000, 2000, 5000, 7000, 10000, 14000]
		Afr_list = np.zeros([len(Npart_list),Nsamp,3,2])

		for nn in range(len(Npart_list)):
			Npart = Npart_list[nn]
			if rank == 0:
				HI_masses_temp = HI_masses * len(HI_masses) / Npart
			else:
				HI_masses_temp = None
			HI_masses_temp = comm.bcast(HI_masses_temp, root=0)

			for pp in range(len(phi_list)):
				phi = phi_list[pp]
				if rank == 0:
					gas_coords_temp = calc_coords_obs(gas_coordinates,phi,theta)
					gas_vel_temp = calc_vel_obs(gas_velocities,phi,theta)
				else:
					gas_coords_temp = None
					gas_vel_temp = None

				gas_coords_temp = comm.bcast(gas_coords_temp, root=0)
				gas_vel_temp = comm.bcast(gas_vel_temp, root=0)

				vel, spectrum = calc_spectrum(gas_coords_temp, gas_vel_temp, HI_masses_temp, beamsize = 40)

				Peaklocs = locate_peaks(spectrum)
				# print(Peaklocs)
				Peaks = [spectrum[Peaklocs[0]],spectrum[Peaklocs[1]]]
				widths = locate_width(spectrum, Peaks, 0.2)
				Sint, Afr_true = areal_asymmetry(spectrum, widths, np.abs(np.diff(vel)[0]))
				# print(Afr_true)

				# plt.plot(spectrum)
				# plt.plot([Peaklocs[0],Peaklocs[0]],[0,np.max(spectrum)])
				# plt.plot([Peaklocs[1],Peaklocs[1]],[0,np.max(spectrum)])
				# plt.show()

				for ss in range(rank,Nsamp,nproc):
					particle_sample = nprand.choice(range(len(HI_masses)), Npart)
						
					vel, spectrum = calc_spectrum(gas_coords_temp[particle_sample,:], 
					gas_vel_temp[particle_sample], HI_masses_temp[particle_sample], beamsize = 40)
					Sint, Afr = areal_asymmetry(spectrum, widths, np.abs(np.diff(vel)[0]))
					Afr_list[nn,ss,pp,0] = Afr
					Afr_list[nn,ss,pp,1] = Sint
		Afr_list.flatten()
		np.savetxt('{dir}/data/restest{ID}_proc{rank}_Afrset.dat'.format(dir=basedir,ID=ID, rank=rank), 
			Afr_list.reshape(len(Npart_list),Nsamp*3*2))

		comm.Barrier()			



# HI prescription comparisons
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



### kinda random intial codes
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

def EAGLEsnap():
	
	# Rvir = 250
	unitmass = 1.e10
	filenames = glob.glob('/media/data/simulations/EAGLE_galaxies/*.hdf5')
	# print(filenames)
	# exit()
	catalogue = Table.read('/media/data/simulations/EAGLE_galaxies/EAGLE_cat.ascii', format='ascii')
	# filenames = ['/media/data/simulations/EAGLE_galaxies/EAGLE_galaxyID8253667.hdf5']
	# filenames = ['/media/data/simulations/EAGLE_galaxies/EAGLE_galaxyID8339149.hdf5']
	# filename = '/media/data/simulations/EAGLE_galaxies/EAGLE_galaxyID8132670.hdf5'
	# filename = '/media/data/simulations/EAGLE_galaxies/EAGLE_galaxyID2948350.hdf5'
	# filename = '/media/data/simulations/EAGLE_galaxies/EAGLE_galaxyID2979098.hdf5'
	Afr_list = []
	Afr_err_list = []
	sersic_list = []
	DM_offsets = []
	gas_offsets = []
	stars_offsets = []
	Mgroup_list = []
	lgGF_list = []
	ID_list = []
	for filename in filenames:
		file = h5py.File(filename,'r')
		ID = int(filename.split('ID')[-1].split('.')[0])
		catref = np.where(np.array(catalogue['GalaxyID']) == ID)[0]
		print(ID,catref)
		if 'PartType0' in list(file.keys()):
			head = file['Header']
			DM = file['PartType1']
			stars = file['PartType4']
			gas = file['PartType0']

			a = head.attrs['ExpansionFactor']
			h = head.attrs['HubbleParam']

			# print(head.attrs['MassTable'][1] * a**(0) * h**(-1))

			[DM_coordinates, DM_velocities] = particle_info(a, h, DM, unitmass, ['Coordinates','Velocity'])
			DM_masses = np.ones(len(DM_coordinates))*head.attrs['MassTable'][1] * a**(0) * h**(-1)

			[gas_coordinates, gas_masses, gas_velocities, gas_densities, gas_internal_energy, gas_temperature] = \
					particle_info(a, h, gas, unitmass, ['Coordinates','Mass','Velocity','Density','InternalEnergy','Temperature'])

			[stars_coordinates, stars_masses, stars_velocities] = \
					particle_info(a, h, stars, unitmass, ['Coordinates','Mass','Velocity'])

			COP = np.array(catalogue['CentreOfPotential_x','CentreOfPotential_y','CentreOfPotential_z'][catref])[0]
			COM = np.array(catalogue['CentreOfMass_x','CentreOfMass_y','CentreOfMass_z'][catref])[0]
			COV = np.array(catalogue['Velocity_x','Velocity_y','Velocity_z'][catref])[0]
			COP_group = np.array(catalogue['GroupCentreOfPotential_x','GroupCentreOfPotential_y','GroupCentreOfPotential_z'][catref])[0]
			
			COP = np.array([i for i in COP])
			COP_group = np.array([i for i in COP_group])
			COM = np.array([i for i in COM])
			COV = np.array([i for i in COV])
			print(COP)
			print(COM)

			DM_coordinates -= COP
			gas_coordinates -= COP
			stars_coordinates -= COP

			stars_radii = np.sqrt(np.nansum(stars_coordinates**2.e0, axis=1))
			gas_radii = np.sqrt(np.nansum(gas_coordinates**2.e0, axis=1))
			DM_radii = np.sqrt(np.nansum(DM_coordinates**2.e0, axis=1))

			p_crit = 3 * (h*100)**2.e0 / (8 * np.pi * (4.3e-3 *1.e-6*1.e10 )  )		# in 1.e10Msun/Mpc^3
			Rvir = 0.005
			rho = 200 * p_crit
			while(rho >= 200 * p_crit):
				rho = np.nansum(DM_masses[DM_radii < Rvir]) / (4. * np.pi * Rvir*Rvir*Rvir / 3.)
				Rvir += 0.01
		
			DM_coordinates = DM_coordinates[DM_radii<=Rvir]
			DM_masses = DM_masses[DM_radii<=Rvir]*1.e10
			DM_velocities = DM_velocities[DM_radii<=Rvir]

			stars_coordinates = stars_coordinates[stars_radii<=Rvir]
			stars_masses = stars_masses[stars_radii<=Rvir]
			stars_velocities = stars_velocities[stars_radii<=Rvir]

			gas_coordinates = gas_coordinates[gas_radii<=Rvir]
			gas_masses = gas_masses[gas_radii<=Rvir]
			gas_velocities = gas_velocities[gas_radii<=Rvir]
			gas_densities = gas_densities[gas_radii<=Rvir]
			gas_internal_energy = gas_internal_energy[gas_radii<=Rvir]
			gas_temperature = gas_temperature[gas_radii<=Rvir]

			print('DM mass[1.e12]',np.nansum(DM_masses) / 1.e12)
			print('stellar mass [1.e10]',np.nansum(stars_masses) / 1.e10)
			print('Total gas mass [1.e10]', np.nansum(gas_masses) / 1.e10)
			print('Stellar fraction',np.nansum(stars_masses) / np.nansum(DM_masses))
			print('Total Gas fraction',np.nansum(gas_masses) / np.nansum(stars_masses))
			Rvir *= 1.e3
			print('Virial radius',Rvir, ' kpc')
			
			# print('COV')
			# print(COV)

			DM_coordinates *= 1.e3
			gas_coordinates *= 1.e3
			stars_coordinates *= 1.e3

			DM_velocities -= COV
			gas_velocities -= COV
			stars_velocities -= COV

			COM_stars_tot = np.array([0.,0.,0.])
			
			COM_stars = calc_COM(stars_coordinates, stars_masses, Rmax = 0.1*Rvir)
			gas_coordinates -= COM_stars
			stars_coordinates -= COM_stars

			COM_stars_tot -= COM_stars

			eigvec = orientation_matrix(stars_coordinates, stars_masses)
			exit()
			stars_coordinates = stars_coordinates @ eigvec
			gas_coordinates = gas_coordinates @ eigvec
			DM_coordinates = DM_coordinates @ eigvec


			COM_stars = calc_COM(stars_coordinates, stars_masses, Rmax = 0.2*Rvir,Zmax=0.1*Rvir)
			gas_coordinates -= COM_stars
			stars_coordinates -= COM_stars
			COM_stars_tot -= COM_stars


			HI_masses, H2_masses, gas_neutral_masses = calc_HI_H2_ARHS(gas,unitmass,a,h)
			HI_masses = HI_masses[gas_radii <= Rvir/1.e3]
			H2_masses = H2_masses[gas_radii <=Rvir/1.e3]

			gas_velocities = gas_velocities @ eigvec
			DM_velocities = DM_velocities @ eigvec
			stars_velocities = stars_velocities @ eigvec

			COV_gas = calc_COV(gas_coordinates, gas_velocities, gas_masses,Rmax = 0.1*Rvir,Zmax = 0.05*Rvir)
			gas_velocities -=  COV_gas

			rad_points, stellar_sigma = calc_sigma(stars_coordinates,stars_masses,Rmax=40,Zmax=20)

			fit_params, fit_covar = curve_fit(log_sersic,rad_points[rad_points<10],stellar_sigma[rad_points<10])

			sersic_N = fit_params[2]
			stars_radii = np.sqrt(np.nansum(stars_coordinates**2.e0, axis=1))

			sersic_N = np.nansum(stars_masses[stars_radii<5]) / (np.pi*5*5) # Msun/kpc^2

			M_group = catalogue['Group_M_Crit200'][catref][0]
			Rvir_group = catalogue['Group_R_Crit200'][catref][0]

			COM_DM = calc_COM(DM_coordinates, DM_masses, Rmax = Rvir)
			COM_gas = calc_COM(gas_coordinates, gas_masses, Rmax = Rvir)

			COM_offset_DM = np.sqrt(np.nansum((COP_group - COP - COM_DM)**2.e0))/Rvir_group
			COM_offset_gas = np.sqrt(np.nansum((COP_group - COP - COM_stars_tot - COM_gas)**2.e0))/Rvir_group
			COM_offset_stars = np.sqrt(np.nansum((COP_group - COP - COM_stars_tot)**2.e0))/Rvir_group


			theta_range = np.arccos(2. * np.arange(0, 1, 0.05) - 1.) * 180./np.pi
			phi_range = np.linspace(0, 360, 25)
			Afr_dist = []
			for theta in theta_range:
				for phi in phi_range:
					gas_coords_edgeon = calc_coords_obs(gas_coordinates, phi, theta)
					gas_vel_edgeon = calc_vel_obs(gas_velocities, phi, theta)

					vel, spectrum = calc_spectrum(gas_coords_edgeon, gas_vel_edgeon, HI_masses, beamsize=40)
					PeaklocL, PeaklocR = locate_peaks(spectrum)
					widths = locate_width(spectrum, [spectrum[PeaklocL],spectrum[PeaklocR]], 0.2)
					# widths = np.where(spectrum > 3)[0][[0,-1]]
					Sint, Afr = areal_asymmetry(spectrum, widths, np.abs(np.diff(vel)[0]))
					Afr_dist.extend([Afr])


			Afr_list.extend([np.median(Afr_dist)])
			Afr_err_list.extend([median_absolute_deviation(Afr_dist)])
			sersic_list.extend([np.log10(sersic_N)])
			DM_offsets.extend([COM_offset_DM])
			gas_offsets.extend([COM_offset_gas])
			stars_offsets.extend([COM_offset_stars])
			Mgroup_list.extend([M_group])
			lgGF_list.extend([np.log10(np.nansum(HI_masses)/np.nansum(stars_masses))])
			ID_list.extend([ID])
			# gas_neutral_fraction, fH2 = calc_fH2_LGS(gas,unitmass,a,h)
			# H2_masses = gas_masses * gas_neutral_fraction * fH2
			# HI_masses = gas_masses * gas_neutral_fraction * (1 - fH2)


			# plotgas_spatial_radial_profile(filename,DM_coordinates, DM_masses, stars_coordinates,stars_masses,\
				# gas_coordinates, gas_velocities, HI_masses, save=True)

			# COV_Per_offset(Rvir,filename,DM_coordinates,DM_velocities, DM_masses, stars_coordinates,stars_velocities,\
			 # stars_masses, gas_coordinates, gas_velocities, gas_masses, HI_masses, save=True)
			# COM_COP_offset(Rvir,filename,DM_coordinates, DM_masses, stars_coordinates, stars_masses, gas_coordinates, gas_masses,HI_masses,save=True)
			print(ID,Afr,np.log10(M_group),np.log10(np.nansum(HI_masses)/np.nansum(stars_masses)))
	
	fig = plt.figure(figsize=(16,8))
	gs = gridspec.GridSpec(2,2,wspace=0.25,hspace=0.25,left=0.05,right=0.95) 
	sersic_ax = fig.add_subplot(gs[0,0])
	offsets_ax = fig.add_subplot(gs[0,1],sharey = sersic_ax)
	Mgroup_ax = fig.add_subplot(gs[1,0],sharey = sersic_ax)
	lgGF_ax = fig.add_subplot(gs[1,1],sharey = sersic_ax)

	sersic_ax.set_ylabel('Asymmetry measure Afr',fontsize=13)
	offsets_ax.set_ylabel('Asymmetry measure Afr',fontsize=13)
	Mgroup_ax.set_ylabel('Asymmetry measure Afr',fontsize=13)
	lgGF_ax.set_ylabel('Asymmetry measure Afr',fontsize=13)

	sersic_ax.set_xlabel('$\mu_\star$ < 5kpc [Msun kpc$^{-2}$]',fontsize=13)
	offsets_ax.set_xlabel('COM offset',fontsize=13)
	Mgroup_ax.set_xlabel('Group Halo mass [Msun]',fontsize=13)
	lgGF_ax.set_xlabel('log10 HI gas fraction',fontsize=13)


	for ii in range(len(Afr_list)):
		sersic_ax.errorbar(sersic_list[ii],Afr_list[ii],marker='o',yerr = Afr_err_list[ii],c='C{}'.format(ii))
		offsets_ax.scatter(DM_offsets[ii],Afr_list[ii],marker='s',c='C{}'.format(ii))
		offsets_ax.scatter(gas_offsets[ii],Afr_list[ii],marker='o',c='C{}'.format(ii))
		offsets_ax.scatter(stars_offsets[ii],Afr_list[ii],marker='*',c='C{}'.format(ii))
		Mgroup_ax.errorbar(np.log10(Mgroup_list[ii]),Afr_list[ii],marker='o',yerr = Afr_err_list[ii],c='C{}'.format(ii))
		lgGF_ax.errorbar(lgGF_list[ii],Afr_list[ii],marker='o',yerr = Afr_err_list[ii],c='C{}'.format(ii))

	offsets_leg = [Line2D([0],[0],color='Black',marker='s',ls=''),
					Line2D([0],[0],color='Black',marker='o',ls=''),
					Line2D([0],[0],color='Black',marker='*',ls='')]
	offsets_ax.legend(offsets_leg,['DM','Gas','Stars'])


	leg = [Line2D([0],[0],ls='',marker='o',color='C{}'.format(ii)) for ii in range(len(Afr_list))]				
	lgGF_ax.legend(leg, ID_list)



	plt.show()
	exit()


##############################################################################################################



#### particle and structural stuff

	
def particle_info(a, h, part, unitmass, keys, subset = [], cgs=False, comoving=True):

	data = []
	if comoving == True:
		for key in keys:
			group = part[key]
			if len(subset)> 0:
				group = group[subset]
			aexp = a**group.attrs['aexp-scale-exponent']
			hexp = h**group.attrs['h-scale-exponent']
			CGSconv = group.attrs['CGSConversionFactor']

			group = np.array(group) * aexp * hexp

			if cgs == True:
				group = group * CGSconv
			elif 'Mass' in key:
				group = group * unitmass

			data.append(group)
	else:
		for key in keys:
			group = np.array(part[key])
			if len(subset)> 0:
				group = group[subset]

			if 'Mass' in key:
				group = group * unitmass

			data.append(group)
	return data	

def calc_COM(coordinates, masses, Rmax = None, Zmax = None):

	radii = np.sqrt(np.nansum(coordinates**2.e0, axis=1))
	z = coordinates[:,2]

	if Rmax != None:
		if Zmax != None:
			coordinates = coordinates[(radii < Rmax) & (z < Zmax)]
			masses = masses[(radii < Rmax) & (z < Zmax)]
		else:
			coordinates = coordinates[radii < Rmax]
			masses = masses[radii < Rmax]
	
	COM = np.array([np.nansum(coordinates[:,0]*masses, axis=0) / np.nansum(masses),\
					np.nansum(coordinates[:,1]*masses, axis=0) / np.nansum(masses),\
					np.nansum(coordinates[:,2]*masses, axis=0) / np.nansum(masses)])
	return COM

def calc_COV(coordinates, velocities, masses, Rmax = None, Zmax = None):

	radii = np.sqrt(np.nansum(coordinates**2.e0, axis=1))
	z = coordinates[:,2]

	if Rmax != None:
		if Zmax != None:
			velocities = velocities[(radii < Rmax) & (z < Zmax)]
			masses = masses[(radii < Rmax) & (z < Zmax)]
		else:
			velocities = velocities[radii < Rmax]
			masses = masses[radii < Rmax]

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
	# eigvec = np.linalg.inv(eigvec)
	eigvec = eigvec[eigval_argsort]

	return eigvec	

def orientation_matrix(coordinates, masses, show = False):

	Iprev = [[1,0,0],[0,1,0],[0,0,1]]
	eigvec_list = []

	rad = [1,2,3,4,5,6,8,10,11,12,13,14,15,16,17,18,19,20,25,30,35,40,45,50,55,60,80,100,150,200]		#kpc
	rr = 0
	Idiff = 1
	while(Idiff > 1.e-4):
		# print(rr,rad[rr])
		# eigvec = diagonalise_inertia(coordinates, masses, rad[rr])
		eigvec = diagonalise_inertia(coordinates, masses, 5)				#not sure why 5kpc works, but most galaxies converge to x-y orientation
		coordinates = coordinates @ eigvec 
		if show == True:
			plt.scatter(coordinates[:,0],coordinates[:,2],s=0.05)
			plt.xlim([-40,40])
			plt.ylim([-40,40])
			plt.show()
			plt.close()
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
		# print(Idiff)
		Iprev = I
		# rr+=1
		if show == True:
			plt.scatter(coordinates[:,0],coordinates[:,2],s=0.1)
			plt.xlim([-40,40])
			plt.ylim([-40,40])
			plt.show()
			plt.close()


	eigvec = eigvec_list[0]
	for ii in range(1,len(eigvec_list)):
		eigvec = eigvec @ eigvec_list[ii]

	return eigvec

def disk_exp(rad, A, Rstar):
	d = A - (rad/Rstar)*np.log(10)
	return d

def log_sersic(rad,A,Rstar,N):
	e = A - np.log(10)*((rad/Rstar)**(1./N))
	return e

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

def calc_sigma(coordinates, masses, Rmax = None, Zmax = None):

	z = coordinates[:,2]
	coordinates = calc_coords_obs(coordinates, 0, 0)
	radii = np.sqrt(np.nansum(coordinates**2.e0, axis=1))

	if Rmax != None:
		if Zmax != None:
			coordinates = coordinates[(radii < Rmax) & (z < Zmax)]
			masses = masses[(radii < Rmax) & (z < Zmax)]
		else:
			coordinates = coordinates[radii < Rmax]
			masses = masses[radii < Rmax]
	radii = np.sqrt(np.nansum(coordinates**2.e0, axis=1))
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

def calc_RC(coordinates, velocities, Rmax=None, Zmax=None):

	z = coordinates[:,2]
	coordinates = calc_coords_obs(coordinates, 0, 0)
	radii = np.sqrt(np.nansum(coordinates**2.e0, axis=1))
	
	if Rmax != None:
		if Zmax != None:
			coordinates = coordinates[(radii < Rmax) & (z < Zmax)]
			velocities = velocities[(radii < Rmax) & (z < Zmax)]
		else:
			coordinates = coordinates[radii < Rmax]
			velocities = velocities[radii < Rmax]
	vcirc = np.sqrt(velocities[:,0]**2.e0 + velocities[:,1]**2.e0)
	radii = np.sqrt(np.nansum(coordinates**2.e0, axis=1))
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



##### HI prescription stuff

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

def calc_HI_H2_ARHS(gas, unitmass, a, h, neutral_frac = False):

	# Hm2012_data = Find_H2_LGS.Read_PhotoIo_Table('Hm2012.txt')
	# Rahmati2013 = Find_H2_LGS.Read_BestFit_Params('BF_Params_Rahmati2013.txt')

	[masses, densities, Z, temperature, U, SFR] = \
					particle_info(a, h, gas, unitmass, ['Mass','Density','Metallicity'\
														,'Temperature','InternalEnergy','StarFormationRate'],cgs=True)
	masses =  masses / 1.989e33	
	SFR = SFR * (3600*24*365.25) / 1.989e33
	Z = Z
	densities = densities * (3.086e18*3.086e18*3.086e18) / 1.989e33
	U = U/(1.e4)
	redshift = 1./a - 1.

	if neutral_frac:
		neutral_fraction = particle_info(a,h,gas,unitmass,['NeutralHydrogenAbundance'], cgs = True)
		neutral_masses = neutral_fraction * unitmass
		HI_masses, H2_masses = galcalc_ARHS.HI_H2_masses(masses,SFR,Z,densities,temperature,neutral_masses,redshift)
	else:
		HI_masses, H2_masses, neutral_masses = galcalc_ARHS.HI_H2_masses(masses,SFR,Z,densities,temperature,None,redshift)
	return HI_masses,H2_masses, neutral_masses

def calc_Rmol(Pextk, coeff = 'LR08'):

	if coeff == 'BR06':
		div = 4.3e4
		power = 0.92
	elif coeff == 'LR08':
		div = 10.**4.23
		power = 0.8

	Rmol = (Pextk / div)**power
	return Rmol

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




##### spectrum and datacube stuff
def calc_spectrum(coords_obs, gas_vel_LOS, HI_masses, dist = 50, Vres = 5, beamsize = 30):
	
	radii = np.sqrt(np.nansum(coords_obs**2.e0, axis=1))
	inbeam = np.where(radii <= beamsize)
	gas_vel_LOS = gas_vel_LOS[inbeam]
	HI_masses = HI_masses[inbeam]

	vlim = 500.e0
	vel_bins = np.arange(-vlim,vlim + Vres,Vres)
	vel_points = vel_bins[0:-1] + 0.5 * Vres
	spectrum = np.zeros([len(vel_bins) - 1])

	mjy_conv = mjy_conversion(dist, Vres)

	for vv in range(len(vel_bins) - 1):
		vel_low = vel_bins[vv]
		vel_high = vel_bins[vv + 1]
		invel  = np.where( (gas_vel_LOS + 30 >= vel_low) &
				 			(gas_vel_LOS - 30 < vel_high) )[0]
				
		for part in invel:
			Mfrac = gaussian_CDF(vel_high, gas_vel_LOS[part], 7.e0) - \
					gaussian_CDF(vel_low, gas_vel_LOS[part], 7.e0)
			spectrum[vv] += HI_masses[part] * Mfrac #* mjy_conv
	return vel_points, spectrum

def locate_peaks(spectrum):

	PeakL = 0
	PeaklocL = int(len(spectrum)/2.)
	chan=1
	while(chan< len(spectrum)/2 + 10):
		chan+=1
		grad = (spectrum[chan] - spectrum[chan-1]) * (spectrum[chan+1] - spectrum[chan])
		if grad<0 and spectrum[chan]>PeakL:
			PeaklocL = chan
			PeakL = spectrum[chan]

	PeakR = 0
	PeaklocR = int(len(spectrum)/2.)
	chan = len(spectrum)-1
	while(chan > len(spectrum)/2 - 10 ):
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

def mjy_conversion(dist, Vres):
	conv = 1.e3 / (2.356e5  * (dist ** 2.e0) * Vres)
	return conv

def gaussian_CDF(x,mu,sigma):
	prob = 0.5e0 * (1.e0 + erf( (x - mu) / (np.sqrt(2.e0) * sigma) ))
	return prob

def gaussian_beam_response(coordinates, centre = [0,0], FWHM = [30,30]):
	xcoords = coordinates[:,0] - centre[0]
	ycoords = coordinates[:,1] - centre[1]

	sigma_x = FWHM[0]/2.355
	sigma_y = FWHM[1]/2.355

	response = np.exp( -1.e0 * ((xcoords*xcoords / (2 * sigma_x*sigma_x)) + (ycoords*ycoords / (2 * sigma_y*sigma_y))) )
	return response

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

def create_HI_spectrum(coordinates, velocities, HI_masses, Vres = 5, centre = [0,0], FWHM = [100,100], filename = None):
	
	HImasses_weighted = HI_masses * gaussian_beam_response(coordinates, centre = centre, FWHM = FWHM)

	vlim = 400.e0
	vel_bins = np.arange(-vlim,vlim + Vres,Vres)
	vel_points = vel_bins[0:-1] + 0.5 * Vres
	spectrum = np.zeros([len(vel_bins) - 1])

	for vv in range(len(vel_bins) - 1):
		vel_low = vel_bins[vv]
		vel_high = vel_bins[vv + 1]
		invel  = np.where( (velocities + 30 >= vel_low) &
				 			(velocities - 30 < vel_high) )[0]
				
		for part in invel:
			Mfrac = gaussian_CDF(vel_high, velocities[part], 7.e0) - \
					gaussian_CDF(vel_low, velocities[part], 7.e0)
			spectrum[vv] += HImasses_weighted[part] * Mfrac 
	if filename != None:
		data = np.array([vel_points,spectrum]).T
		np.savetxt(filename,data, fmt = "%.6e")
	else:
		return vel_points, spectrum

def create_HI_datacube(coordinates, velocities, HI_masses, params = None, filename = None):

	if params ==  None:							#default datacube parameters
		params = {'dist':50,					#Mpc
				'cubephys':100,					#kpc (left to right)
				'cubedim':250,					#pixels 
				'Vlim':800,						#km/s bandwidth
				'Vres':5,						#km/s
				'B_FWHM':1,						#kpc
				'rms':0}						#mJy RMS noise

	mjy_conv = mjy_conversion(params['dist'], params['Vres'])
	dx = params['cubephys'] / params['cubedim']

	spacebins = np.arange(-0.5e0 * params['cubephys'], 0.5e0 * params['cubephys'] + dx, dx)
	velbins = np.arange(-0.5e0 * params['Vlim'], 0.5e0 * params['Vlim'] + params['Vres'], params['Vres'])
	Nspace = len(spacebins) - 1
	Nvel = len(velbins) - 1
	datacube = np.zeros([Nspace, Nspace, Nvel])	

	for yy in range(Nspace):
		ylow = spacebins[yy]
		yhigh = spacebins[yy + 1]
		for xx in range(Nspace):
			xlow = spacebins[xx]
			xhigh = spacebins[xx + 1]
			vel_inspace = velocities[(coordinates[:,0] >= xlow) & (coordinates[:,0] < xhigh) & \
						(coordinates[:,1] >= ylow) & (coordinates[:,1] < yhigh)]
			
			if len(vel_inspace) > 0:
				minvel = np.nanmin(vel_inspace) - 30
				maxvel = np.nanmax(vel_inspace) + 30
				velrange_min = np.where(np.abs(velbins - minvel) ==  np.min(np.abs(velbins - minvel)))[0][0] - 10
				velrange_max = np.where(np.abs(velbins - maxvel) ==  np.min(np.abs(velbins - maxvel)))[0][0] + 10
				for vv in range(np.max([0,velrange_min]), np.min([velrange_max,Nvel])):
					vel_low = velbins[vv]
					vel_high = velbins[vv + 1]
					invel  = np.where( (vel_inspace + 30 >= vel_low) &
							 			(vel_inspace - 30 < vel_high) )[0]
					for part in invel:
						Mfrac = gaussian_CDF(vel_high, vel_inspace[part], 7.e0) - \
								gaussian_CDF(vel_low, vel_inspace[part], 7.e0)

						datacube[yy,xx,vv] += HI_masses[part] * Mfrac * mjy_conv

	datacube[:,:,:] += nprand.normal(np.zeros([Nspace, Nspace , Nvel]), params['rms'])				#add noise

	Bsigma = params['B_FWHM'] / (2.355 * dx)														#convert beam FWHM to Gausian stddev for convoluion
	# print(Bsigma)
	for vv in range(Nvel):
		datacube[:,:,vv]  = convolve(datacube[:,:,vv], Gaussian2DKernel(Bsigma))


	if filename == None:
		return spacebins, velbins, datacube
	else:
		datacube = datacube.reshape((Nspace , Nspace * Nvel))
		header = 'dist = {dist}\n cubephys = {cubephys}\n dx = {dx}\n Vlim = {Vlim}\n Vres = {Vres}\n'.format(
			dist = params['dist'], cubephys = params['cubephys'], dx = dx, Vres = params['Vres'], Vlim = params['Vlim'])
		np.savetxt(filename, datacube, header = header,fmt = "%.6e")

def read_datacube(filename):

	f = open(filename, 'r')
	for line in f:
		if 'dist' in line:
			params = {'dist':float(line.split(' ')[-1]) }
		if 'cubephys' in line:
			params['cubephys'] = float(line.split(' ')[-1]) 
		if 'dx' in line:
			dx = float(line.split(' ')[-1]) 
		if 'Vlim' in line:
			params['Vlim'] = float(line.split(' ')[-1]) 
		if 'Vres' in line:
			params['Vres'] = float(line.split(' ')[-1]) 
	f.close()

	spacebins = np.arange(-0.5e0 * params['cubephys'], 0.5e0 * params['cubephys'] + dx, dx)
	velbins = np.arange(-0.5e0 * params['Vlim'], 0.5e0 * params['Vlim'] + params['Vres'], params['Vres'])
	Nspace = len(spacebins) - 1
	Nvel = len(velbins) - 1

	print('reading datacube')
	datacube = np.loadtxt(filename).reshape((Nspace, Nspace, Nvel))

	return spacebins, velbins, datacube, params



########### plots
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

def plotgas_spatial_radial_profile(name, DM_coordinates, DM_masses, stars_coordinates, stars_masses, 
	gas_coordinates, gas_velocities, HI_masses,save=False):

		gas_coords_faceon = calc_coords_obs(gas_coordinates, 0, 0)
		gas_coords_edgeon = calc_coords_obs(gas_coordinates, 0, 90)
		gas_vel_edgeon = calc_vel_obs(gas_velocities,0,90)

		stars_coords_faceon = calc_coords_obs(stars_coordinates, 0, 0)
		stars_coords_edgeon = calc_coords_obs(stars_coordinates, 0, 90)

		DM_coords_faceon = calc_coords_obs(DM_coordinates, 0, 0)
		DM_coords_edgeon = calc_coords_obs(DM_coordinates, 0, 90)


		fig = plt.figure(figsize=(18,8))
		gs = gridspec.GridSpec(2,4, hspace=0,wspace=0.25,left=0.05,right=0.95) 
		stars_faceon_ax = fig.add_subplot(gs[0,0])
		stars_edgeon_ax = fig.add_subplot(gs[1,0],sharex = stars_faceon_ax, sharey = stars_faceon_ax)
		faceon_ax = fig.add_subplot(gs[0,1])
		edgeon_ax = fig.add_subplot(gs[1,1], sharex = faceon_ax, sharey = faceon_ax)
		sigma_ax = fig.add_subplot(gs[0,2])
		RC_ax = fig.add_subplot(gs[1,2], sharex = sigma_ax)
		spec_ax = fig.add_subplot(gs[:,3])

	
		stars_faceon_ax.set_ylabel('y [kpc]',fontsize=15)
		stars_edgeon_ax.set_xlabel('x [kpc]',fontsize=15)
		stars_edgeon_ax.set_ylabel('z [kpc]',fontsize=15)
		faceon_ax.set_ylabel('y [kpc]',fontsize=15)
		edgeon_ax.set_xlabel('x [kpc]',fontsize=15)
		edgeon_ax.set_ylabel('z [kpc]',fontsize=15)

		spec_ax.set_xlabel('Velocity [km s$^{-1}$]', fontsize=15)
		spec_ax.set_ylabel('Spectral Flux [mJy]', fontsize=15)

		spec_ax.tick_params(axis = 'both', which='both', direction = 'in', labelsize=15, length = 8, width = 1.25)
		RC_ax.tick_params(axis = 'both', which='both', direction = 'in', labelsize=15, length = 8, width = 1.25)
		sigma_ax.tick_params(axis = 'both', which='both', direction = 'in', labelsize=15, length = 8, width = 1.25)
		sigma_ax.tick_params(axis = 'x', which='both', direction = 'in', labelsize=0, length = 8, width = 1.25)
		faceon_ax.tick_params(axis = 'both', which='both', direction = 'in', labelsize=15, length = 8, width = 1.25)
		faceon_ax.tick_params(axis = 'x', which='both', direction = 'in', labelsize=0, length = 8, width = 1.25)
		edgeon_ax.tick_params(axis = 'both', which='both', direction = 'in', labelsize=15, length = 8, width = 1.25)
		stars_faceon_ax.tick_params(axis = 'both', which='both', direction = 'in', labelsize=15, length = 8, width = 1.25)
		stars_faceon_ax.tick_params(axis = 'x', which='both', direction = 'in', labelsize=0, length = 8, width = 1.25)
		stars_edgeon_ax.tick_params(axis = 'both', which='both', direction = 'in', labelsize=15, length = 8, width = 1.25)
		
		RC_ax.set_xlabel('Radius [kpc]', fontsize=15)
		RC_ax.set_ylabel('Vcirc [km s$^{-1}$]', fontsize=15)
		sigma_ax.set_ylabel('log$_{10} \Sigma$ [M$_{\odot}$ pc$^{-2}$]', fontsize=15)
		spec_ax.set_title('EAGLE ID = {name}'.format(name=name.split('ID')[-1].split('.')[0]),fontsize=12)


		rad_DM, sigma_DM = calc_sigma(DM_coordinates, DM_masses)

		rad_stars, sigma_stars = calc_sigma(stars_coordinates, stars_masses,Rmax = 40)
		rad_stars_left, sigma_stars_left = calc_sigma(stars_coordinates[stars_coords_faceon[:,0]<0], stars_masses[stars_coords_faceon[:,0]<0],Rmax = 40)
		rad_stars_right, sigma_stars_right = calc_sigma(stars_coordinates[stars_coords_faceon[:,0]>0], stars_masses[stars_coords_faceon[:,0]>0],Rmax = 40)

		rad_HI, sigma_HI = calc_sigma(gas_coordinates, HI_masses,Rmax = 40,Zmax=20)
		rad_HI_left, sigma_HI_left = calc_sigma(gas_coordinates[gas_coords_faceon[:,0]<0], HI_masses[gas_coords_faceon[:,0]<0],Rmax = 40,Zmax=20)
		rad_HI_right, sigma_HI_right = calc_sigma(gas_coordinates[gas_coords_faceon[:,0]>0], HI_masses[gas_coords_faceon[:,0]>0],Rmax = 40,Zmax=20)

		rad_HI_RC, RC_HI = calc_RC(gas_coordinates, gas_velocities,Rmax = 40,Zmax=20)
		rad_HI_left_RC, RC_HI_left = calc_RC(gas_coordinates[gas_coords_faceon[:,0]<0], gas_velocities[gas_coords_faceon[:,0]<0],Rmax = 40,Zmax=20)
		rad_HI_right_RC, RC_HI_right = calc_RC(gas_coordinates[gas_coords_faceon[:,0]>0], gas_velocities[gas_coords_faceon[:,0]>0],Rmax = 40,Zmax=20)

		spacebins, HIfaceon_mom0 = calc_spatial_dist(gas_coords_faceon, HI_masses, 40)
		spacebins, HIedgeon_mom0 = calc_spatial_dist(gas_coords_edgeon, HI_masses, 40)

		spacebins, stars_faceon = calc_spatial_dist(stars_coords_faceon, stars_masses, 40)
		spacebins, stars_edgeon = calc_spatial_dist(stars_coords_edgeon, stars_masses, 40)


		beamsizes = [10,15,20,30,40]
		colors = ['Blue','Orange','Green','Red','Black']
		for bb in range(len(beamsizes)):
			radii = np.sqrt(np.nansum(gas_coords_faceon**2.e0,axis=1))
			radmax = beamsizes[bb]
			
			left = np.intersect1d(np.where(radii<radmax)[0],np.where(gas_coords_faceon[:,0]< 0)[0])
			right = np.intersect1d(np.where(radii<radmax)[0],np.where(gas_coords_faceon[:,0]> 0)[0])

			vel, spectrum_left = calc_spectrum(gas_coords_edgeon[left], gas_vel_edgeon[left], HI_masses[left], beamsize = beamsizes[bb])
			vel, spectrum_right = calc_spectrum(gas_coords_edgeon[right], gas_vel_edgeon[right], HI_masses[right], beamsize = beamsizes[bb] )
			vel, spectrum = calc_spectrum(gas_coords_edgeon[radii < radmax], gas_vel_edgeon[radii < radmax], HI_masses[radii < radmax], beamsize = beamsizes[bb])
		
			PeaklocL, PeaklocR = locate_peaks(spectrum)
			widths = locate_width(spectrum, [spectrum[PeaklocL],spectrum[PeaklocR]], 0.2)
			Sint, Afr = areal_asymmetry(spectrum, widths, np.abs(np.diff(vel)[0]))

			spec_ax.plot(vel,spectrum_left, ls = '--',c=colors[bb], label = 'Beamsize = {b} kpc  Afr = {Afr:.2f}'.format(b = beamsizes[bb],Afr=Afr))
			spec_ax.plot(vel,spectrum_right,ls = ':',c=colors[bb])
		
		faceon_ax.imshow(HIfaceon_mom0, extent=[-40,40,-40,40], vmin=-2, vmax = 2)
		edgeon_ax.imshow(HIedgeon_mom0, extent=[-40,40,-40,40], vmin=-2, vmax = 2)

		stars_faceon_ax.imshow(stars_faceon, extent=[-40,40,-40,40], vmin=-2, vmax = 3)
		stars_edgeon_ax.imshow(stars_edgeon, extent=[-40,40,-40,40], vmin=-2, vmax = 3)

		sigma_ax.plot(rad_stars,sigma_stars, label='Stars total',c = 'Red')
		sigma_ax.plot(rad_stars_left,sigma_stars_left, label='Stars LHS',c = 'Red', ls = '--')
		sigma_ax.plot(rad_stars_right,sigma_stars_right, label='Stars RHS',c = 'Red',ls = ':')
		sigma_ax.plot(rad_HI,sigma_HI,label = 'HI total',c = 'Blue')
		sigma_ax.plot(rad_HI_left,sigma_HI_left,label = 'HI LHS',c = 'Blue',ls = '--')
		sigma_ax.plot(rad_HI_right,sigma_HI_right,label = 'HI RHS',c = 'Blue',ls = ':')
		sigma_ax.plot(rad_DM,sigma_DM, label = 'DM', color='Black')

		RC_ax.plot(rad_HI_RC, RC_HI, label = 'HI total',c = 'Blue')
		RC_ax.plot(rad_HI_left_RC, RC_HI_left, label = 'HI LHS',c = 'Blue',ls = '--')
		RC_ax.plot(rad_HI_right_RC, RC_HI_right, label = 'HI RHS',c = 'Blue',ls = ':')

		faceon_ax.set_xlim([-40,40])
		edgeon_ax.set_xlim([-40,40])
		stars_faceon_ax.set_xlim([-40,40])
		stars_edgeon_ax.set_xlim([-40,40])

		sigma_ax.set_xlim([0,40])
		sigma_ax.set_ylim([-2.5,3])

		spec_ax.set_ylim([0,1.5*np.nanmax([np.nanmax(spectrum_left),np.nanmax(spectrum_right)])])
		sigma_ax.legend(fontsize=12)
		spec_ax.legend(fontsize=12)
		if save == True:
			outname = './figures/sigma_RC_spec_EAGLE{name}_COVpercstarsCOM.png'.format(name=name.split('ID')[-1].split('.')[0])
			fig.savefig(outname, dpi=150)
		else:
			plt.show()
			plt.close()

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

def COM_COP_offset(Rvir,filename,DM_coordinates, DM_masses, stars_coordinates, stars_masses, gas_coordinates, gas_masses,HI_masses,save=False):
	COM_gas = calc_COM(gas_coordinates, gas_masses, Rmax = Rvir)
	COM_HI = calc_COM(gas_coordinates, HI_masses, Rmax = Rvir)
	COM_stars = calc_COM(stars_coordinates, stars_masses, Rmax = Rvir)
	COM_DM = calc_COM(DM_coordinates, DM_masses, Rmax = Rvir)

	COM_offset_gas = np.sqrt(np.nansum((np.array([0,0,0])- COM_gas)**2.e0))/Rvir
	COM_offset_HI = np.sqrt(np.nansum((np.array([0,0,0])- COM_HI)**2.e0))/Rvir
	COM_offset_stars = np.sqrt(np.nansum((np.array([0,0,0]) - COM_stars)**2.e0))/Rvir
	COM_offset_DM = np.sqrt(np.nansum((np.array([0,0,0]) - COM_DM)**2.e0))/Rvir

	COM_gas_disk = calc_COM(gas_coordinates, gas_masses, Rmax = 0.2*Rvir,Zmax = 0.1*Rvir)
	COM_HI_disk = calc_COM(gas_coordinates, HI_masses, Rmax = 0.2*Rvir,Zmax = 0.1*Rvir)
	COM_stars_disk = calc_COM(stars_coordinates, stars_masses, Rmax = 0.2*Rvir,Zmax = 0.1*Rvir)
	COM_DM_disk = calc_COM(DM_coordinates, DM_masses, Rmax = 0.2*Rvir,Zmax = 0.1*Rvir)

	COM_offset_gas_disk = np.sqrt(np.nansum((np.array([0,0,0]) - COM_gas_disk)**2.e0))/Rvir
	COM_offset_HI_disk = np.sqrt(np.nansum((np.array([0,0,0]) - COM_HI_disk)**2.e0))/Rvir
	COM_offset_stars_disk = np.sqrt(np.nansum((np.array([0,0,0]) - COM_stars_disk)**2.e0))/Rvir
	COM_offset_DM_disk = np.sqrt(np.nansum((np.array([0,0,0]) - COM_DM_disk)**2.e0))/Rvir

	fig, ax = plt.subplots()
	ax.set_ylim([0,0.2])
	ax.scatter([0,1,2,3],[COM_offset_gas,COM_offset_HI,COM_offset_DM, COM_offset_stars],color='Blue',label = 'Within Rvir')
	ax.scatter([0,1,2,3],[COM_offset_gas_disk,COM_offset_HI_disk,COM_offset_DM_disk, COM_offset_stars_disk],color='Red',label = 'Within disk')
	ax.set_xticks([0,1,2,3])
	ax.set_xticklabels(['Gas','HI','DM','Stars'],fontsize=12)
	ax.set_ylabel('Offset from COP / Rvir',fontsize=12)
	ax.set_title('EAGLE ID = {name}, Rvir = {Rvir:.2f}'.format(name=filename.split('ID')[-1].split('.')[0],Rvir=Rvir),fontsize=12)
	ax.legend(fontsize=12)
	if save == True:
		outname = './figures/COM_offsets_EAGLE{name}.png'.format(name=filename.split('ID')[-1].split('.')[0])
		fig.savefig(outname, dpi=150)
	else:
		plt.show()
		plt.close()

def COV_Per_offset(Rvir,filename,DM_coordinates,DM_velocities, DM_masses, stars_coordinates,stars_velocities, \
	stars_masses, gas_coordinates, gas_velocities, gas_masses, HI_masses, save=False):
	COV_gas = calc_COV(gas_coordinates, gas_velocities, gas_masses, Rmax = Rvir)
	COV_HI = calc_COV(gas_coordinates, gas_velocities, HI_masses, Rmax = Rvir)
	COV_stars = calc_COV(stars_coordinates, stars_velocities, stars_masses, Rmax = Rvir)
	COV_DM = calc_COV(DM_coordinates, DM_velocities, DM_masses, Rmax = Rvir)

	COV_offset_gas = np.sqrt(np.nansum((np.array([0,0,0])- COV_gas)**2.e0))
	COV_offset_HI = np.sqrt(np.nansum((np.array([0,0,0])- COV_HI)**2.e0))
	COV_offset_stars = np.sqrt(np.nansum((np.array([0,0,0]) - COV_stars)**2.e0))
	COV_offset_DM = np.sqrt(np.nansum((np.array([0,0,0]) - COV_DM)**2.e0))

	COV_gas_disk = calc_COV(gas_coordinates, gas_velocities, gas_masses, Rmax = 0.2*Rvir,Zmax = 0.1*Rvir)
	COV_HI_disk = calc_COV(gas_coordinates, gas_velocities, HI_masses, Rmax = 0.2*Rvir,Zmax = 0.1*Rvir)
	COV_stars_disk = calc_COV(stars_coordinates, stars_velocities, stars_masses, Rmax = 0.2*Rvir,Zmax = 0.1*Rvir)
	COV_DM_disk = calc_COV(DM_coordinates, DM_velocities, DM_masses, Rmax = 0.2*Rvir,Zmax = 0.1*Rvir)

	COV_offset_gas_disk = np.sqrt(np.nansum((np.array([0,0,0]) - COV_gas_disk)**2.e0))
	COV_offset_HI_disk = np.sqrt(np.nansum((np.array([0,0,0]) - COV_HI_disk)**2.e0))
	COV_offset_stars_disk = np.sqrt(np.nansum((np.array([0,0,0]) - COV_stars_disk)**2.e0))
	COV_offset_DM_disk = np.sqrt(np.nansum((np.array([0,0,0]) - COV_DM_disk)**2.e0))

	fig, ax = plt.subplots()
	# ax.set_ylim([0,0.2])
	ax.scatter([0,1,2,3],[COV_offset_gas,COV_offset_HI,COV_offset_DM, COV_offset_stars],color='Blue',label = 'Within Rvir')
	ax.scatter([0,1,2,3],[COV_offset_gas_disk,COV_offset_HI_disk,COV_offset_DM_disk, COV_offset_stars_disk],color='Red',label = 'Within disk')
	ax.set_xticks([0,1,2,3])
	ax.set_xticklabels(['Gas','HI','DM','Stars'],fontsize=12)
	ax.set_ylabel('Offset from Peculiar velocity [km/s]',fontsize=12)
	ax.set_title('EAGLE ID = {name}, Rvir = {Rvir:.2f}'.format(name=filename.split('ID')[-1].split('.')[0],Rvir=Rvir),fontsize=12)
	ax.legend(fontsize=12)
	if save == True:
		outname = './figures/COV_offsets_EAGLE{name}.png'.format(name=filename.split('ID')[-1].split('.')[0])
		fig.savefig(outname, dpi=150)
	else:
		plt.show()
		plt.close()




##### other stuff



def size_mass_relation(MHI):
	DHI = 10.e0**(0.506*np.log10(MHI)-3.293)	#kpc Wang+16
	return DHI

def median_absolute_deviation(data, scale=True):
	med = np.median(data)
	MAD = np.median(np.abs(data - med))
	if scale == True:
		MAD *= 1.4826
	return MAD

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



if __name__ == '__main__':
	# measure_controlled_run()

	# plot_controlled_run()

	# asymmetry_time()

	# create_gif(sys.argv[1])

	# hydro_run()

	# TNGsnap()

	# analyse_datacube()

	# EAGLEsnap()

	# resolution_test()

	# resolution_test_EAGLE()

	resolution_test_TNG()

	# read_Genesis()

	# read_controlled_run()

	# HI_datacube_spectrum_simulation()

	# fourier_decomp_datacube()

