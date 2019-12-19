import numpy as np 
import analysis_functions as af 
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec
import galread_ARHS as gr 
from astropy.table import Table
from mpi4py import MPI
from scipy.optimize import curve_fit
import dist_sampling_DAGJK as dsd


## some measurements

def add_extended_measurements():

	data = Table.read('/media/data/simulations/IllustrisTNG/TNG100_galdata_measured_v2.ascii',format='ascii')


	data_new = gr.TNG_HIlines('/media/data/simulations/IllustrisTNG/HIlines_ARHS_TNG100_extended.bin',mainbox=True,extended=True)


	keys_new = ['pos_rel','vel_rel']
	coords = ['_x','_y','_z']

	print(data)

	for ii in range(len(keys_new)):
		for jj in range(len(coords)):
			data['{key}{coord}'.format(key=keys_new[ii],coord=coords[jj])] = np.array(data_new[keys_new[ii]])[:,jj]

	print(data)
	# exit()
	data.write('/media/data/simulations/IllustrisTNG/TNG100_galdata_measured_v2.ascii',format='ascii',overwrite=True)

def split_TNGdata():
	data = gr.TNG_HIlines('/media/data/simulations/IllustrisTNG/HIlines_ARHS_TNG100-3.bin',mainbox=False)
	vel_bins = (data['vbins'] + np.diff(data['vbins'])[0])[0:-1]
	spectra_true = data['HIline_true']
	# spectra_mock = data['HIline_mock']

	spectra_true = np.append([vel_bins],spectra_true,axis=0)
	# spectra_mock = np.append([vel_bins],spectra_mock,axis=0)
	spectra_true = np.transpose(spectra_true)
	# spectra_mock = np.transpose(spectra_mock)





	keys_mock = ['subhaloes', 'groupnr','mass_stars', 'mass_stars_mock', 'mass_HI', 'mass_HI_mock', 
	'mass_halo', 'mass_halo_mock', 'SFR', 'SFR_mock',  'Type', 'Type_mock']

	keys = ['subhaloes', 'groupnr','mass_stars' , 'mass_HI', 
	'mass_halo', 'SFR',  'Type']

	# for key in keys

	galdata = [data[k] for k in keys]
	galdata = np.transpose(np.array(galdata))

	galdata = Table(galdata, names=keys)
	galdata.write('/media/data/simulations/IllustrisTNG/TNG100-3_galdata.ascii',format='ascii')
	# np.savetxt('/media/data/simulations/IllustrisTNG/TNG100-2_spectra_mock.dat',spectra_mock)
	np.savetxt('/media/data/simulations/IllustrisTNG/TNG100-3_spectra_true.dat',spectra_true)
	print(galdata)


def remeasure_Afr():
	data = Table.read('/media/data/simulations/IllustrisTNG/TNG_galdata_measured_v2_new.ascii',format='ascii')
	spectra = np.loadtxt('/media/data/simulations/IllustrisTNG/TNG_spectra_true.dat')
	vel_bins = spectra[:,0]
	spectra = spectra[:,1::]
	Vres = np.abs(np.diff(vel_bins))[0]

	good = np.where(data['fit_success'] == 0)[0]
	print(len(good))
	for ii in good:
		# print(ii)
		spectrum = spectra[:,ii]
		widths = [data['w20L'][ii],data['w20R'][ii]]
		Sint, Afr = areal_asymmetry(spectrum, widths, Vres)
		data['Sint'][ii] = Sint
		data['Afr'][ii] = Afr
	data.write('/media/data/simulations/IllustrisTNG/TNG_galdata_measured_v2_new.ascii',format='ascii',overwrite=True)

def fix_measurements(IDs,base):
	# base = '/media/data/simulations/IllustrisTNG/TNG100'

	data = Table.read('{}_galdata_measured_v2.ascii'.format(base),format='ascii')
	spectra = np.loadtxt('{}_spectra_true.dat'.format(base))
	vel_bins = spectra[:,0]
	spectra = spectra[:,1::]
	Vres = np.abs(np.diff(vel_bins))[0]
	good = np.where(data['fit_success'] == 0)[0]
	bad = np.where(data['fit_success'] != 0)[0]


	print(len(good))
	print(len(bad))
	# exit()


	# widths = (data['w20R'] - data['w20L']) * Vres
	# highW = np.where(widths>500)[0]
	# for ii in highW:
	# 	plt.plot(spectra[:,ii])
	# 	plt.plot([data['PeaklocL'][ii],data['PeaklocL'][ii]],[0,np.nanmax(spectra[:,ii])])
	# 	plt.plot([data['PeaklocR'][ii],data['PeaklocR'][ii]],[0,np.nanmax(spectra[:,ii])])

	# 	plt.plot([data['w20L'][ii],data['w20L'][ii]],[0,np.nanmax(spectra[:,ii])])
	# 	plt.plot([data['w20R'][ii],data['w20R'][ii]],[0,np.nanmax(spectra[:,ii])])

	# 	plt.show()
	# 	plt.close()

	# exit()


	original_check=False
	if original_check:
		peaks_diff = np.where(data['fit_success'] == 1)[0]
		peaks_bad = np.where(data['fit_success'] == 2)[0]
		width_bad = np.where(data['fit_success'] == 3)[0]

		print('Number of good',len(good))
		print('Number of peaks disagree',len(peaks_diff))
		print('Number of peaks are bad',len(peaks_bad))
		print('Number of widths are bad',len(width_bad))
		exit()
	# IDs = [5704,8593,13180]

	# IDs = bad[bad<210] #210
	# IDs = [1239,1295,1489,1511,2102,2206,2487,2654,2839,2973,3085,3679,3737,3738,3909,4207,4210,4943,5335,5669,5973,7013,7378,9636,10230,10767,10798,10830]
	# IDs = [91,242,288,674,987,1205,1453,1696,1802,2168,2213,4234,4420,4676,4941,5340,5534,5576,5638]
	# IDs = [5243,5347,5802,6040,6208]
	print(len(IDs))

	for ii in IDs:
		print(ii)
		spectrum = spectra[:,ii]
		level = 0.2
		answer = -1
		while(answer != '' and answer != 'b'):
			if answer != 'm':
				peaks = locate_peaks_v2(spectrum,level)
			else:
				peaks = locate_peaks_manual(spectrum,left,right)
			plt.plot(spectrum)
			plt.plot([peaks[0],peaks[0]],[0,np.nanmax(spectrum)])
			plt.plot([peaks[1],peaks[1]],[0,np.nanmax(spectrum)])
			plt.ion()
			plt.show()
			print('[enter] for good, [b] for bad or [p] to retry peak finding, [pp] for single peak, [m] for manual select')
			answer = input()
			print(answer)
			if answer == '':
				widths = locate_width(spectrum, peaks, 0.2)

				if all(w > 0 for w in widths) and all(w< len(spectrum) for w in widths):
					print(widths)
					Sint, Afr = areal_asymmetry(spectrum, widths, Vres)
					data['fit_success'][ii] = 0
					data['PeaklocL'][ii] = peaks[0]
					data['PeaklocR'][ii] = peaks[1]
					data['w20L'][ii] = widths[0]
					data['w20R'][ii] = widths[1]
				else:
					data['fit_success'][ii] = 3
					data['PeaklocL'][ii] = peaks[0]
					data['PeaklocR'][ii] = peaks[1]
			if answer == 'b':
				data['fit_success'][ii] = -1
			if answer == 'pp':
				print('inputting to fit one peak')
				level = 0.999
			if answer =='m':
				print('Choose left and right positions to start from')
				while True:
					pts = []
					while len(pts) < 2:
						pts.append(plt.ginput(1, timeout=-1))
						print(pts)
					if plt.waitforbuttonpress():
						break
				left = pts[0][0][0]
				right = pts[1][0][0]
				print(left)
				print(right)
			elif answer == 'p':
				print('input fraction of peak to try')
				level = float(input())
			plt.close()
		data.write('{}_galdata_measured_v2.ascii'.format(base),format='ascii',overwrite=True)

def find_bad_peaks():
	particle_mass = 1.4e6
	TNG100 = Table.read('/media/data/simulations/IllustrisTNG/TNG100_galdata_measured_v2.ascii',format='ascii')
	IDs = np.arange(len(TNG100))
	# IDs = IDs[TNG100['fit_success']==0]
	good = np.where(TNG100['fit_success']==0)[0]
	TNG100 = TNG100[good]

	spectra = np.loadtxt('/media/data/simulations/IllustrisTNG/TNG100_spectra_true.dat')
	vel = spectra[:,0]
	spectra = spectra[:,1::]
	spectra = spectra[:,good]
	IDlist = []
	for ii in range(len(TNG100)):
		spectrum = spectra[:,ii]
		PeaklocL = int(TNG100['PeaklocL'][ii])
		PeaklocR = int(TNG100['PeaklocR'][ii])
		w20L = int(TNG100['w20L'][ii])
		w20R = int(TNG100['w20R'][ii])

		if PeaklocL == PeaklocR:
			if any(spectrum[0:w20L] > spectrum[PeaklocL]*0.5) or  any(spectrum[w20R:-1] > spectrum[PeaklocL]*0.5):
				IDlist.extend([ii])
				# plt.plot(spectrum)
				# plt.show()

		else:
			if any(spectrum[PeaklocL-10:PeaklocL+10] > spectrum[PeaklocL]) or any(spectrum[PeaklocR-10:PeaklocR+10] > spectrum[PeaklocR]):
				IDlist.extend([ii])
				# plt.plot(spectrum)
				# plt.show()
	print(IDlist)
	print(len(IDlist))
	# print(np.where(np.array(IDlist) > 2600))
	# IDlist = IDlist[393::]
	# exit()


	base = '/media/data/simulations/IllustrisTNG/TNG100'
	fix_measurements(IDlist, base)

def measure_TNG_spectra():
	comm = MPI.COMM_WORLD
	rank = comm.Get_rank()
	nproc = comm.Get_size()

	base = '/media/data/simulations/IllustrisTNG/TNG100-3'

	spectra = np.loadtxt('{}_spectra_true.dat'.format(base))
	vel_bins = spectra[:,0]
	spectra = spectra[:,1::]
	Vres = np.diff(vel_bins)[0]
	Nspectra = len(spectra[0,:])

	measurements = []
	for specnum in range(rank, Nspectra, nproc):
		print(specnum)
		spectrum =  spectra[:,specnum]
		measurement = measure_spectrum_v2(spectrum, Vres)
		measurements.append([specnum] + measurement)

	measurements = np.array(measurements)
	comm.barrier()
	if rank == 0:
		measurements_all = np.empty([int(np.ceil(Nspectra/nproc)*nproc),8], dtype='d')
	else:
		measurements_all = None
	comm.Gather(measurements, measurements_all, root = 0)
	if rank == 0:
		measurements_all = measurements_all[0:Nspectra,:]
		sort = np.argsort(measurements_all[:,0])
		measurements_all = measurements_all[sort]
		data = Table.read('{}_galdata.ascii'.format(base),format='ascii')
		data['fit_success'] = measurements_all[:,1]
		data['PeaklocL'] = measurements_all[:,2]
		data['PeaklocR'] = measurements_all[:,3]
		data['w20L'] = measurements_all[:,4]
		data['w20r'] = measurements_all[:,5]
		data['Sint'] = measurements[:,6]
		data['Afr'] = measurements_all[:,7]

		data.write('{}_galdata_measured.ascii'.format(base),format='ascii')

def plot_Afr_env():

	particle_mass = 2.e6

	data = Table.read('/media/data/simulations/IllustrisTNG/TNG_galdata_measured_v2_new.ascii',format='ascii')
	print(len(np.where(data['fit_success'] == 0)[0]))
	good = np.where((data['Afr'] > 1) & (data['Afr'] < 3) & (data['fit_success'] == 0))[0]
	data = data[good]

	resolved = np.where((data['Sint']/particle_mass > 1.e3) & (data['mass_stars']/particle_mass > 1.e3))
	resolved_data = data[resolved]

	print(len(resolved_data))
	print(len(data))

	plt.hist(np.log10(data['mass_stars']),alpha=0.8)
	plt.hist(np.log10(resolved_data['mass_stars']),alpha=0.5)
	plt.show()
	# exit()

	print(len(data))
	isocent = np.where(resolved_data['Type'] == -1)[0]
	grpcent = np.where(resolved_data['Type'] == 0)[0]
	sat = np.where(resolved_data['Type'] == 1)[0]

	print(len(isocent))
	print(len(grpcent))
	print(len(sat))

	halomass = data['mass_halo']
	Afr = data['Afr']

	fig = plt.figure(figsize = (10,5))
	gs = gridspec.GridSpec(1, 1, top = 0.99, right = 0.98, bottom  = 0.148, left = 0.08)
	axes = fig.add_subplot(gs[0,0])
	axes.hist(data['Afr'][isocent],bins=np.arange(1,2.5,0.01),density=True,cumulative=True,color='Blue',histtype='step',label='Isolated centrals')
	axes.hist(data['Afr'][grpcent],bins=np.arange(1,2.5,0.01),density=True,cumulative=True,color='Green',histtype='step',label='Group Centrals')
	axes.hist(data['Afr'][sat],bins=np.arange(1,2.5,0.01),density=True,cumulative=True,color='Orange',histtype='step',label='Satellites')
	axes.set_xlabel('Asymmetry measure A$_{fr}$')
	axes.set_ylabel('Cumulative Histogram Density')
	axes.legend()
	plt.show()
	exit()


	plt.scatter(np.log10(halomass),Afr)
	plt.show()
	exit()


#resoltuion test stuff 

def resolution_completeness_sSFR():

	particle_mass = 1.4e6

	data = Table.read('/media/data/simulations/IllustrisTNG/TNG100-2_galdata_measured_v2.ascii',format='ascii')
	data = data[data['fit_success'] == 0]

	data['mass_stars'] = np.log10(data['mass_stars'])
	data['SFR'] = np.log10(data['SFR']) - data['mass_stars']

	plt.scatter(data['mass_stars'],data['SFR'],s=0.1)
	plt.show()
	# exit()


	resolved_asym = np.where((data['Sint']/particle_mass >= 5.e2))[0]
	unresolved_asym = np.where((data['Sint']/particle_mass < 5.e2))[0]

	print(len(resolved_asym))
	print(len(unresolved_asym))

	mstar_bins = np.arange(9,13,0.1)
	SFR_bins = np.arange(-15,-8,0.1)
	compl_grid = np.zeros([len(SFR_bins)-1,len(mstar_bins)-1])
	for mm in range(len(mstar_bins)-1):
		mbin_low = mstar_bins[mm]
		mbin_high = mstar_bins[mm + 1]
		inbin_mm = np.where((data['mass_stars'] > mbin_low) & (data['mass_stars'] < mbin_high))[0]
		inbin_mm_res = np.where((data['mass_stars'][resolved_asym] > mbin_low) & (data['mass_stars'][resolved_asym] < mbin_high))[0]
		for ss in range(len(SFR_bins)-1):
			sbin_low = SFR_bins[ss]
			sbin_high = SFR_bins[ss + 1]
			inbin_ss = np.where((data['SFR'] > sbin_low) & (data['SFR'] < sbin_high))[0]
			inbin_ss_res = np.where((data['SFR'][resolved_asym] > sbin_low) & (data['SFR'][resolved_asym] < sbin_high))[0]

			inbin = np.intersect1d(inbin_ss,inbin_mm)
			inbin_res = np.intersect1d(inbin_ss_res,inbin_mm_res)
			if len(inbin) == 0:
				compl_grid[ss,mm] = np.nan

			else:
				compl_grid[ss,mm] = len(inbin_res) / len(inbin)

	fig = plt.figure(figsize=(12,8))
	gs = gridspec.GridSpec(1,1, top=0.99, left = 0.1, right=0.99, bottom=0.1) 
	ax = fig.add_subplot(gs[0,0])
	ax.set_ylabel(' $\log_{10}$ sSFR [yr$^{-1}$]',fontsize = 16)
	ax.set_xlabel('$\log_{10}$ $M_{\star}$ [M$_{\odot}$]',fontsize = 16)
	ax.tick_params(axis = 'both', which = 'both', direction = 'in', labelsize = 16,length = 8, width = 1.25)
	ax.tick_params(axis = 'both', which = 'minor', direction = 'in', labelsize = 16,length = 4, width = 1.25)

	aa = ax.pcolormesh(mstar_bins,SFR_bins,compl_grid)
	fig.colorbar(aa)

	ax.contour(mstar_bins[0:-1],SFR_bins[0:-1],compl_grid,levels=[0.5],**{'linewidths':3,'colors':'Black'})

	ax.plot([9,12.5],[-0.344*(9-9) - 9.822, -0.344*(12.5-9) - 9.822],ls = '--',color='Red',linewidth=2)
	ax.plot([9,12.5],[-0.344*(9-9) - 9.822 - (0.088*(9-9) + 0.188), -0.344*(12.5-9) - 9.822 - (0.088*(12.5-9) + 0.188)],ls = ':',color='Red',linewidth=2)
	ax.plot([9,12.5],[-0.344*(9-9) - 9.822 + (0.088*(9-9) + 0.188), -0.344*(12.5-9) - 9.822 + (0.088*(12-9) + 0.188)],ls = ':',color='Red',linewidth=2)

	fig.savefig('/media/data/simulations/IllustrisTNG/figures/TNG100-2_sSFR_comp.png')

	plt.show()

	exit()

	all_grid = np.zeros([len(SFR_bins)-1,len(mstar_bins)-1])
	for mm in range(len(mstar_bins)-1):
		mbin_low = mstar_bins[mm]
		mbin_high = mstar_bins[mm + 1]
		inbin_mm = np.where((data['mass_stars'] > mbin_low) & (data['mass_stars'] < mbin_high))[0]
		for ss in range(len(SFR_bins)-1):
			sbin_low = SFR_bins[ss]
			sbin_high = SFR_bins[ss + 1]
			inbin_ss = np.where((data['SFR'] > sbin_low) & (data['SFR'] < sbin_high))[0]

			inbin = np.intersect1d(inbin_ss,inbin_mm)
			if len(inbin) == 0:
				all_grid[ss,mm] = np.nan

			else:
				all_grid[ss,mm] = len(inbin)

	# print(np.nansum(all_grid))


	plt.pcolormesh(mstar_bins,SFR_bins,all_grid)
	plt.colorbar()


	plt.plot([9,12.5],[-0.344*(9-9) - 9.822, -0.344*(12.5-9) - 9.822],ls = '--',color='Red',linewidth=2)
	plt.plot([9,12.5],[-0.344*(9-9) - 9.822 - (0.088*(9-9) + 0.188), -0.344*(12.5-9) - 9.822 - (0.088*(12.5-9) + 0.188)],ls = ':',color='Red',linewidth=2)
	plt.plot([9,12.5],[-0.344*(9-9) - 9.822 + (0.088*(9-9) + 0.188), -0.344*(12.5-9) - 9.822 + (0.088*(12-9) + 0.188)],ls = ':',color='Red',linewidth=2)

	plt.xlabel('log10 Stellar mass')
	plt.ylabel('log10 SFR')
	plt.title('TNG100 sSFR - lgMstar')
	plt.show()

def resolution_completeness_MHI():

	particle_mass = 1.4e6

	data = Table.read('/media/data/simulations/IllustrisTNG/TNG100_galdata_measured_v2.ascii',format='ascii')

	data['mass_stars'] = np.log10(data['mass_stars'])

	plt.scatter(data['mass_stars'],np.log10(data['Sint']) - data['mass_stars'],s=0.1)
	plt.show()
	# exit()


	resolved_asym = np.where((data['Sint']/particle_mass >= 1.e3))[0]
	unresolved_asym = np.where((data['Sint']/particle_mass < 1.e3))[0]


	data['Sint'] = np.log10(data['Sint']) - data['mass_stars']

	print(len(resolved_asym))
	print(len(unresolved_asym))


	mstar_bins = np.arange(9,13,0.1)
	resolved_frac = np.zeros(len(mstar_bins) - 1)
	for ii in range(len(mstar_bins)-1):
		mbin_low = mstar_bins[ii]
		mbin_high = mstar_bins[ii + 1]
		inbin = np.where((data['mass_stars'] > mbin_low) & (data['mass_stars'] < mbin_high))[0]
		inbin_res = np.where((data['mass_stars'][resolved_asym] > mbin_low) & (data['mass_stars'][resolved_asym] < mbin_high))[0]
		if len(inbin) != 0:
			resolved_frac[ii] = len(inbin_res) / len(inbin)
		else:
			resolved_frac[ii] = np.nan

	mstar_bins = np.arange(9,13,0.1)
	mHI_bins = np.arange(-6,1,0.1)

	all_grid = np.zeros([len(mHI_bins)-1,len(mstar_bins)-1])
	for mm in range(len(mstar_bins)-1):
		mbin_low = mstar_bins[mm]
		mbin_high = mstar_bins[mm + 1]
		inbin_mm = np.where((data['mass_stars'] > mbin_low) & (data['mass_stars'] < mbin_high))[0]
		for ss in range(len(mHI_bins)-1):
			sbin_low = mHI_bins[ss]
			sbin_high = mHI_bins[ss + 1]
			inbin_ss = np.where((data['Sint'] > sbin_low) & (data['Sint'] < sbin_high))[0]

			inbin = np.intersect1d(inbin_ss,inbin_mm)
			if len(inbin) == 0:
				all_grid[ss,mm] = np.nan

			else:
				all_grid[ss,mm] = len(inbin)

	# print(np.nansum(all_grid))


	fig = plt.figure(figsize = (8,12))
	gs = gridspec.GridSpec(4, 1, top = 0.95, right = 0.98, bottom  = 0.12, left = 0.08)

	ax1 = fig.add_subplot(gs[0:3,0])
	ax2 = fig.add_subplot(gs[3,0],sharex = ax1)

	ax1.tick_params(axis = 'x', which = 'both', direction = 'in', labelsize = 0)


	ax1.pcolormesh(mstar_bins,mHI_bins,all_grid)
	ax1.plot([9,12],[np.log10(1.e3*1.4e6 / 1.e9)/9 , np.log10(1.e3*1.4e6 / 1.e12)],color='Black',ls='--',label='1000 partcles')

	xGmed_lgMstar = [9.14,9.44,9.74,10.07,10.34,10.65,10.95,11.20]
	xGmed_MHI = [-0.092,-0.320,-0.656,-0.854,-1.278,-1.223,-1.707,-1.785]

	ax1.scatter(xGmed_lgMstar,xGmed_MHI, color='Red',s=20,label='xGASS weighted medians')
	ax1.legend()

	ax2.plot(mstar_bins[0:-1],resolved_frac, color='Black')
	ax2.plot([9,12],[0.5,0.5],color='Blue',ls = '--')

	ax2.set_xlabel('log10 Stellar mass')
	ax1.set_ylabel('log10 MHI/Mstar')
	ax2.set_ylabel('Resolved fraction')
	ax1.set_title('TNG100 MHI- lgMstar')

	ax2.set_ylim([0,1])
	plt.show()

def resolution_completeness_ratio():

	particle_mass = 1.4e6

	data = Table.read('/media/data/simulations/IllustrisTNG/TNG100_galdata_measured_v2.ascii',format='ascii')
	data['mass_stars'] = np.log10(data['mass_stars'])
	data['SFR'] = np.log10(data['SFR']) - data['mass_stars']
	data = data[data['fit_success']==0]
	resolved_asym = np.where((data['Sint']/particle_mass >= 5.e2))[0]
	mstar_bins = np.arange(9,13,0.1)
	SFR_bins = np.arange(-15,-8,0.1)
	compl_grid_1 = np.zeros([len(SFR_bins)-1,len(mstar_bins)-1])
	for mm in range(len(mstar_bins)-1):
		mbin_low = mstar_bins[mm]
		mbin_high = mstar_bins[mm + 1]
		inbin_mm = np.where((data['mass_stars'] > mbin_low) & (data['mass_stars'] < mbin_high))[0]
		inbin_mm_res = np.where((data['mass_stars'][resolved_asym] > mbin_low) & (data['mass_stars'][resolved_asym] < mbin_high))[0]
		for ss in range(len(SFR_bins)-1):
			sbin_low = SFR_bins[ss]
			sbin_high = SFR_bins[ss + 1]
			inbin_ss = np.where((data['SFR'] > sbin_low) & (data['SFR'] < sbin_high))[0]
			inbin_ss_res = np.where((data['SFR'][resolved_asym] > sbin_low) & (data['SFR'][resolved_asym] < sbin_high))[0]

			inbin = np.intersect1d(inbin_ss,inbin_mm)
			inbin_res = np.intersect1d(inbin_ss_res,inbin_mm_res)
			if len(inbin) == 0:
				compl_grid_1[ss,mm] = np.nan

			else:
				compl_grid_1[ss,mm] = len(inbin_res) / len(inbin)


	data = Table.read('/media/data/simulations/IllustrisTNG/TNG100-2_galdata_measured_v2.ascii',format='ascii')
	data['mass_stars'] = np.log10(data['mass_stars'])
	data['SFR'] = np.log10(data['SFR']) - data['mass_stars']
	data = data[data['fit_success']==0]
	resolved_asym = np.where((data['Sint']/particle_mass >= 5.e2))[0]
	mstar_bins = np.arange(9,13,0.1)
	SFR_bins = np.arange(-15,-8,0.1)
	compl_grid_2 = np.zeros([len(SFR_bins)-1,len(mstar_bins)-1])
	for mm in range(len(mstar_bins)-1):
		mbin_low = mstar_bins[mm]
		mbin_high = mstar_bins[mm + 1]
		inbin_mm = np.where((data['mass_stars'] > mbin_low) & (data['mass_stars'] < mbin_high))[0]
		inbin_mm_res = np.where((data['mass_stars'][resolved_asym] > mbin_low) & (data['mass_stars'][resolved_asym] < mbin_high))[0]
		for ss in range(len(SFR_bins)-1):
			sbin_low = SFR_bins[ss]
			sbin_high = SFR_bins[ss + 1]
			inbin_ss = np.where((data['SFR'] > sbin_low) & (data['SFR'] < sbin_high))[0]
			inbin_ss_res = np.where((data['SFR'][resolved_asym] > sbin_low) & (data['SFR'][resolved_asym] < sbin_high))[0]

			inbin = np.intersect1d(inbin_ss,inbin_mm)
			inbin_res = np.intersect1d(inbin_ss_res,inbin_mm_res)
			if len(inbin) == 0:
				compl_grid_2[ss,mm] = np.nan

			else:
				compl_grid_2[ss,mm] = len(inbin_res) / len(inbin)


	compl_ratio = compl_grid_1 / compl_grid_2
	compl_ratio[compl_ratio > 1] = 1.e0/ compl_ratio[compl_ratio>1]

	fig = plt.figure(figsize=(12,8))
	gs = gridspec.GridSpec(1,1, top=0.99, left = 0.1, right=0.99, bottom=0.1) 
	ax = fig.add_subplot(gs[0,0])
	ax.set_ylabel(' $\log_{10}$ sSFR [yr$^{-1}$]',fontsize = 16)
	ax.set_xlabel('$\log_{10}$ $M_{\star}$ [M$_{\odot}$]',fontsize = 16)
	ax.tick_params(axis = 'both', which = 'both', direction = 'in', labelsize = 16,length = 8, width = 1.25)
	ax.tick_params(axis = 'both', which = 'minor', direction = 'in', labelsize = 16,length = 4, width = 1.25)


	aa=ax.pcolormesh(mstar_bins,SFR_bins,compl_ratio)
	fig.colorbar(aa)

	ax.contour(mstar_bins[0:-1],SFR_bins[0:-1],compl_ratio,levels=[0.8],**{'linewidths':3,'colors':'Black'})
	
	ax.plot([9,12.5],[-0.344*(9-9) - 9.822, -0.344*(12.5-9) - 9.822],ls = '--',color='Red',linewidth=2)
	ax.plot([9,12.5],[-0.344*(9-9) - 9.822 - (0.088*(9-9) + 0.188), -0.344*(12.5-9) - 9.822 - (0.088*(12.5-9) + 0.188)],ls = ':',color='Red',linewidth=2)
	ax.plot([9,12.5],[-0.344*(9-9) - 9.822 + (0.088*(9-9) + 0.188), -0.344*(12.5-9) - 9.822 + (0.088*(12-9) + 0.188)],ls = ':',color='Red',linewidth=2)


	fig.savefig('/media/data/simulations/IllustrisTNG/figures/TNG100TNG100-2_complratio.png')
	plt.show()

	exit()


	resolved = np.where((data['Sint']/particle_mass > 1.e3) & (data['mass_stars']/particle_mass > 1.e3))
	resolved_data = data[resolved]

	print(len(resolved_data))
	print(len(data))

	plt.hist(np.log10(data['mass_stars']),alpha=0.8)
	plt.hist(np.log10(resolved_data['mass_stars']),alpha=0.5)	

def compare_Afrhist_TNGboxes():

	particle_mass = 1.4e6

	TNG100 = Table.read('/media/data/simulations/IllustrisTNG/TNG100_galdata_measured_v2.ascii',format='ascii')
	TNG100_2 = Table.read('/media/data/simulations/IllustrisTNG/TNG100-2_galdata_measured_v2.ascii',format='ascii')
	TNG100_3 = Table.read('/media/data/simulations/IllustrisTNG/TNG100-3_galdata_measured_v2.ascii',format='ascii')


	TNG100['Npart'] = TNG100['Sint']/particle_mass
	TNG100_2['Npart'] = TNG100_2['Sint']/particle_mass 
	TNG100_3['Npart'] = TNG100_3['Sint']/particle_mass


	# plt.hist(np.log10(TNG100['mass_stars']),bins=15,alpha=0.6,density=True)
	# plt.hist(np.log10(TNG100_2['mass_stars']),bins=15,alpha=0.4,density=True)
	# plt.xlabel('lgMstar')
	# plt.ylabel('Histogram Density')
	# plt.show()

	TNG100['mass_stars'] = np.log10(TNG100['mass_stars'])
	TNG100['sSFR'] = np.log10(TNG100['SFR']) - TNG100['mass_stars']

	TNG100_2['mass_stars'] = np.log10(TNG100_2['mass_stars'])
	TNG100_2['sSFR'] = np.log10(TNG100_2['SFR']) - TNG100_2['mass_stars']


	TNG100_3['mass_stars'] = np.log10(TNG100_3['mass_stars'])
	TNG100_3['sSFR'] = np.log10(TNG100_3['SFR']) - TNG100_3['mass_stars']




	TNG100TNG1002 = True
	TNG1002TNG1003 = False
	TNG100TNG1003 = False


	if TNG100TNG1002:
		
		plt.hist(TNG100['mass_stars'],bins=15,alpha=0.6,density=True)
		plt.hist(TNG100_2['mass_stars'],bins=15,alpha=0.4,density=True)
		plt.xlabel('lgMstar')
		plt.ylabel('Histogram Density')
		plt.show()


		lgMstar_low = 10
		lgMstar_high = 10.5

		sSFR_low = -10.5
		sSFR_high = -8



		TNG100 = TNG100[np.where( (TNG100['mass_stars'] < lgMstar_high) & (TNG100['mass_stars']> lgMstar_low) 
					& (TNG100['sSFR'] > sSFR_low) & (TNG100['sSFR']  < sSFR_high ))[0]]

		TNG100_2 = TNG100_2[np.where( (TNG100_2['mass_stars'] < lgMstar_high) & (TNG100_2['mass_stars']> lgMstar_low) 
					& (TNG100_2['sSFR'] > sSFR_low) & (TNG100_2['sSFR']  < sSFR_high ))[0]]

		TNG100 = TNG100[(TNG100['Npart'] > 5.e2)]
		TNG100_2 = TNG100_2[(TNG100_2['Npart'] >5.e2)]


		TNG100_cent = TNG100[TNG100['Type'] != 1]
		TNG100_sat = TNG100[TNG100['Type'] == 1]

		TNG100_2_cent = TNG100_2[TNG100_2['Type'] != 1]
		TNG100_2_sat = TNG100_2[TNG100_2['Type'] == 1]
		
		plt.hist(np.log10(TNG100_sat['Npart']),bins=15,alpha=0.6,density=True)
		plt.hist(np.log10(TNG100_cent['Npart']),bins=15,alpha=0.4,density=True)
		plt.xlabel('lgNpart')
		plt.ylabel('Histogram Density')
		plt.show()


		fig = plt.figure(figsize = (10,8))
		gs = gridspec.GridSpec(2, 1, top = 0.9, right = 0.98, bottom  = 0.08, left = 0.08)
		axes_1 = fig.add_subplot(gs[0,0])
		axes_2 = fig.add_subplot(gs[1,0],sharex = axes_1,sharey = axes_1)
		axes_1.set_ylabel('Cumulative Histogram Density')
		axes_2.set_ylabel('Cumulative Histogram Density')
		axes_2.set_xlabel('Asymmetry measure A$_{fr}$')


		names = ['Satellites ({})'.format(len(TNG100_sat)), 'Centrals ({})'.format(len(TNG100_cent))]


		Afr_bins = np.arange(1,2.5,0.05)
		Npart_bins = np.arange(np.log10(500),np.log10(5000)+0.1,0.1)
		Npart_bins = np.append(Npart_bins,np.array([np.log10(1.e6)]))

		samples = [TNG100_sat['Afr'],TNG100_cent['Afr']]
		controls = [np.log10(TNG100_sat['Npart']),np.log10(TNG100_cent['Npart'])]

		samples_all_hists, samples_all_DAGJKsigma, indexes_all_iter = \
								dsd.control_samples(samples,Afr_bins,controls,Npart_bins)

		dsd.plot_controlled_cumulative_histograms(samples_all_hists,Afr_bins,samples_all_DAGJKsigma,
			 axis = axes_1, names=names)
		# axes_1.legend()

		names = ['Satellites ({})'.format(len(TNG100_2_sat)), 'Centrals ({})'.format(len(TNG100_2_cent))]

		samples = [TNG100_2_sat['Afr'],TNG100_2_cent['Afr']]
		controls = [np.log10(TNG100_2_sat['Npart']),np.log10(TNG100_2_cent['Npart'])]

		samples_all_hists, samples_all_DAGJKsigma, indexes_all_iter = \
								dsd.control_samples(samples,Afr_bins,controls,Npart_bins)

		dsd.plot_controlled_cumulative_histograms(samples_all_hists,Afr_bins,samples_all_DAGJKsigma,
			axis = axes_2, names=names)

		# axes_2.legend()

		# fig.suptitle('TNG100 & TNG100-2  lgMstar = [9.,9.8] sSFR = [-10.1,-9.6]',fontsize = 15)


		plt.show()
		exit()


	if TNG1002TNG1003:
		

		plt.hist(np.log10(TNG100_3['mass_stars']),bins=15,alpha=0.6,density=True)
		plt.hist(np.log10(TNG100_2['mass_stars']),bins=15,alpha=0.4,density=True)
		plt.xlabel('lgMstar')
		plt.ylabel('Histogram Density')
		plt.show()

		TNG100_3 = TNG100_3[np.where((np.log10(TNG100_3['mass_stars'])<10.1) & (np.log10(TNG100_3['mass_stars'])> 9.3) 
					& (np.log10(TNG100_3['SFR'] / TNG100_3['mass_stars']) > -10.7 ) & (np.log10(TNG100_3['SFR'] / TNG100_3['mass_stars']) < -9.7 ))[0]]
		TNG100_2 = TNG100_2[np.where((np.log10(TNG100_2['mass_stars'])<10.1) & (np.log10(TNG100_2['mass_stars'])> 9.3) 
					& (np.log10(TNG100_2['SFR'] / TNG100_2['mass_stars']) > -10.7) & (np.log10(TNG100_2['SFR'] / TNG100_2['mass_stars']) < -9.7)) [0]]

		TNG100_3_cent = np.where(TNG100_3['Type'] <= 0)[0]
		TNG100_3_sat = np.where(TNG100_3['Type'] == 1)[0]

		TNG100_2_cent = np.where(TNG100_2['Type'] <= 0)[0]
		TNG100_2_sat = np.where(TNG100_2['Type'] == 1)[0]


		fig = plt.figure(figsize = (10,8))
		gs = gridspec.GridSpec(2, 1, top = 0.9, right = 0.98, bottom  = 0.08, left = 0.08)
		axes_1 = fig.add_subplot(gs[0,0])
		axes_1.hist(TNG100_3['Afr'][TNG100_3_cent],bins=np.arange(1,2.5,0.01),density=True,cumulative=True,color='Green',histtype='step',label='Centrals ({})'.format(len(TNG100_3_cent)))
		axes_1.hist(TNG100_3['Afr'][TNG100_3_sat],bins=np.arange(1,2.5,0.01),density=True,cumulative=True,color='Orange',histtype='step',label='Satellites ({})'.format(len(TNG100_3_sat)))
		# axes_1.set_xlabel('Asymmetry measure A$_{fr}$')
		axes_1.set_ylabel('Cumulative Histogram Density')
		axes_1.legend()

		axes_2 = fig.add_subplot(gs[1,0])
		axes_2.hist(TNG100_2['Afr'][TNG100_2_cent],bins=np.arange(1,2.5,0.01),density=True,cumulative=True,color='Green',histtype='step',label='Centrals ({})'.format(len(TNG100_2_cent)))
		axes_2.hist(TNG100_2['Afr'][TNG100_2_sat],bins=np.arange(1,2.5,0.01),density=True,cumulative=True,color='Orange',histtype='step',label='Satellites ({})'.format(len(TNG100_2_sat)))
		axes_2.set_xlabel('Asymmetry measure A$_{fr}$')
		axes_2.set_ylabel('Cumulative Histogram Density')
		axes_2.legend()

		fig.suptitle('TNG100-3 & TNG100-2  lgMstar = [9.3,10.1] sSFR = [-10.7,-9.7]',fontsize = 15)


		plt.show()
		exit()


	if TNG100TNG1003:
		
		plt.hist(np.log10(TNG100['mass_stars']),bins=15,alpha=0.6,density=True)
		plt.hist(np.log10(TNG100_3['mass_stars']),bins=15,alpha=0.4,density=True)
		plt.xlabel('lgMstar')
		plt.ylabel('Histogram Density')
		plt.show()

		TNG100_3 = TNG100_3[np.where((np.log10(TNG100_3['mass_stars'])<9.6) & (np.log10(TNG100_3['mass_stars'])> 9.1) 
					& (np.log10(TNG100_3['SFR'] / TNG100_3['mass_stars']) > -9.8 ) & (np.log10(TNG100_3['SFR'] / TNG100_3['mass_stars']) < -9.2 ))[0]]
		TNG100 = TNG100[np.where((np.log10(TNG100['mass_stars'])<9.6) & (np.log10(TNG100['mass_stars'])> 9.1) 
					& (np.log10(TNG100['SFR'] / TNG100['mass_stars']) > -9.8) & (np.log10(TNG100['SFR'] / TNG100['mass_stars']) < -9.2)) [0]]

		TNG100_3_cent = np.where(TNG100_3['Type'] <= 0)[0]
		TNG100_3_sat = np.where(TNG100_3['Type'] == 1)[0]

		TNG100_cent = np.where(TNG100['Type'] <= 0)[0]
		TNG100_sat = np.where(TNG100['Type'] == 1)[0]


		fig = plt.figure(figsize = (10,8))
		gs = gridspec.GridSpec(2, 1, top = 0.9, right = 0.98, bottom  = 0.08, left = 0.08)
		axes_1 = fig.add_subplot(gs[0,0])
		axes_1.hist(TNG100_3['Afr'][TNG100_3_cent],bins=np.arange(1,2.5,0.01),density=True,cumulative=True,color='Green',histtype='step',label='Centrals ({})'.format(len(TNG100_3_cent)))
		axes_1.hist(TNG100_3['Afr'][TNG100_3_sat],bins=np.arange(1,2.5,0.01),density=True,cumulative=True,color='Orange',histtype='step',label='Satellites ({})'.format(len(TNG100_3_sat)))
		# axes_1.set_xlabel('Asymmetry measure A$_{fr}$')
		axes_1.set_ylabel('Cumulative Histogram Density')
		axes_1.legend()

		axes_2 = fig.add_subplot(gs[1,0])
		axes_2.hist(TNG100['Afr'][TNG100_cent],bins=np.arange(1,2.5,0.01),density=True,cumulative=True,color='Green',histtype='step',label='Centrals ({})'.format(len(TNG100_cent)))
		axes_2.hist(TNG100['Afr'][TNG100_sat],bins=np.arange(1,2.5,0.01),density=True,cumulative=True,color='Orange',histtype='step',label='Satellites ({})'.format(len(TNG100_sat)))
		axes_2.set_xlabel('Asymmetry measure A$_{fr}$')
		axes_2.set_ylabel('Cumulative Histogram Density')
		axes_2.legend()

		fig.suptitle('TNG100-3 & TNG100  lgMstar = [9.25,9.5] sSFR = [-10.5,-9.5]',fontsize = 15)


		plt.show()
		exit()

def compare_Afrhist_Npart():
	particle_mass = 1.4e6

	TNG100 = Table.read('/media/data/simulations/IllustrisTNG/TNG100_galdata_measured_v2.ascii',format='ascii')

	TNG100['mass_stars'] = np.log10(TNG100['mass_stars'])
	TNG100['mass_halo'] = np.log10(TNG100['mass_halo'])
	TNG100['SFR'] = np.log10(TNG100['SFR']) - TNG100['mass_stars']


	TNG100 = TNG100[TNG100['mass_stars']>10.5]
	
	fig = plt.figure(figsize = (10,12))

	gs = gridspec.GridSpec(5, 1, top = 0.92, right = 0.98, bottom  = 0.08, left = 0.08)

	ax1 = fig.add_subplot(gs[0,0])
	ax2 = fig.add_subplot(gs[1,0],sharex = ax1,sharey = ax1)
	ax3 = fig.add_subplot(gs[2,0],sharex = ax1,sharey = ax1)
	ax4 = fig.add_subplot(gs[3,0],sharex = ax1,sharey = ax1)
	ax5 = fig.add_subplot(gs[4,0],sharex = ax1,sharey = ax1)

	Npart_list = [100,500,1000,2500,5000]
	ax = [ax1,ax2,ax3,ax4,ax5]

	for ii in range(len(Npart_list)):
		axes = ax[ii]
		samp = TNG100[np.where((TNG100['Sint']/particle_mass >= Npart_list[ii]))[0]]

		axes.hist(samp['Afr'][samp['Type']<1],color = 'Green',bins=np.arange(1,2.5,0.01),density=True,cumulative=True,histtype='step',fill=False,label='centrals ({})'.format(len(samp['Afr'][samp['Type']<1])))
		axes.hist(samp['Afr'][samp['Type']==1],color = 'Orange',bins=np.arange(1,2.5,0.01),density=True,cumulative=True,histtype='step',fill=False,label='satellites ({})'.format(len(samp['Afr'][samp['Type']==1])))
		axes.legend()
		axes.set_title('{} particles'.format(Npart_list[ii]))
		if ii < 4:	
			axes.tick_params(axis = 'x', which = 'both', direction = 'in', labelsize = 0)
		else:
			axes.tick_params(axis = 'x', which = 'both', direction = 'in')
		axes.tick_params(axis = 'y', which = 'both', direction = 'in')
		axes.set_ylabel('Histogram Density')


	ax5.set_xlabel('Asymmetry measure A$_{fr}$')
	fig.suptitle('lgMstar>10.5',fontsize=18)
	fig.savefig('./data/test12.png')



######## visual checking stuff

def check_highmass_SFgals():
	particle_mass = 1.4e6

	TNG100 = Table.read('/media/data/simulations/IllustrisTNG/TNG100_galdata_measured_v2.ascii',format='ascii')
	IDs = np.arange(len(TNG100))
	TNG100['mass_stars'] = np.log10(TNG100['mass_stars'])
	TNG100['SFR'] = np.log10(TNG100['SFR']) - TNG100['mass_stars']

	# plt.scatter(data['mass_stars'],data['SFR'],s=0.1)
	# plt.show()
	# exit()

	highmassSF = np.where((TNG100['mass_stars']>11.5) & (TNG100['SFR'] > -11.5))[0]
	data = TNG100[highmassSF]
	IDs = IDs[highmassSF]

	resolved_asym = np.where((data['Sint']/particle_mass >= 1.e3))[0]
	unresolved_asym = np.where((data['Sint']/particle_mass < 1.e3))[0]

	print(IDs[unresolved_asym])


	spectra = np.loadtxt('/media/data/simulations/IllustrisTNG/TNG100_spectra_true.dat')
	vel = spectra[:,0]
	spectra = spectra[:,1::]
	for ID in IDs:
		print(ID,TNG100['Afr'][ID])
		fig, ax = plt.subplots()
		ax.plot(range(len(vel)),spectra[:,ID],color='Black')
		ax.plot([TNG100['PeaklocL'][ID],TNG100['PeaklocL'][ID]],[0,np.max(spectra[:,ID])],color = 'red')
		ax.plot([TNG100['PeaklocR'][ID],TNG100['PeaklocR'][ID]],[0,np.max(spectra[:,ID])],color = 'red')
		ax.plot([TNG100['w20L'][ID],TNG100['w20L'][ID]],[0,np.max(spectra[:,ID])],color='Blue',ls='--')
		ax.plot([TNG100['w20R'][ID],TNG100['w20R'][ID]],[0,np.max(spectra[:,ID])],color='Blue',ls='--')
		ax.set_xlabel('Channels')
		ax.set_ylabel('HI mass')
		ax.text(0.1,0.9,'Afr = {afr:.3f}'.format(afr=TNG100['Afr'][ID]),fontsize=12, transform=ax.transAxes)
		plt.show()


	exit()

	print(len(resolved_asym))
	print(len(unresolved_asym))

	mstar_bins = np.arange(9,13,0.1)
	SFR_bins = np.arange(-15,-8,0.1)
	compl_grid = np.zeros([len(SFR_bins)-1,len(mstar_bins)-1])
	for mm in range(len(mstar_bins)-1):
		mbin_low = mstar_bins[mm]
		mbin_high = mstar_bins[mm + 1]
		inbin_mm = np.where((data['mass_stars'] > mbin_low) & (data['mass_stars'] < mbin_high))[0]
		inbin_mm_res = np.where((data['mass_stars'][resolved_asym] > mbin_low) & (data['mass_stars'][resolved_asym] < mbin_high))[0]
		for ss in range(len(SFR_bins)-1):
			sbin_low = SFR_bins[ss]
			sbin_high = SFR_bins[ss + 1]
			inbin_ss = np.where((data['SFR'] > sbin_low) & (data['SFR'] < sbin_high))[0]
			inbin_ss_res = np.where((data['SFR'][resolved_asym] > sbin_low) & (data['SFR'][resolved_asym] < sbin_high))[0]

			inbin = np.intersect1d(inbin_ss,inbin_mm)
			inbin_res = np.intersect1d(inbin_ss_res,inbin_mm_res)
			if len(inbin) == 0:
				compl_grid[ss,mm] = np.nan

			else:
				compl_grid[ss,mm] = len(inbin_res) / len(inbin)


def plot_selected_spectra():
	
	particle_mass = 1.4e6
	TNG100 = Table.read('/media/data/simulations/IllustrisTNG/TNG100_galdata_measured_v2.ascii',format='ascii')
	

	TNG100['mass_stars'] = np.log10(TNG100['mass_stars'])
	TNG100['mass_halo'] = np.log10(TNG100['mass_halo'])
	TNG100['SFR'] = np.log10(TNG100['SFR']) - TNG100['mass_stars']

	IDs = np.arange(len(TNG100))
	good = np.where((TNG100['Sint']/particle_mass >= 1.e3))[0]
	TNG100_good = TNG100[good]
	IDs = IDs[good]
	# massrange_1 = np.where((TNG100['mass_stars'] > 10.5) & (TNG100['mass_stars'] <= 11))[0]
	massrange = np.where((TNG100_good['mass_stars'] > 10.5) & (TNG100_good['mass_stars'] <= 11) & (TNG100_good['mass_halo'] > 13.5))[0]

	TNG_samp = TNG100_good[massrange]
	IDs = IDs[massrange]

	highAfr = np.where(TNG_samp['Afr']>1.7)[0]
	IDs = IDs[highAfr]
	print(IDs)
	print(TNG_samp)
	print(TNG100[IDs]['subhaloes'])

	# IDs = [ 31348, 60749, 76090, 88675,  161166, 217487]
	IDs = [ 102, 197, 243, 300,  645, 1019]

	spectra = np.loadtxt('/media/data/simulations/IllustrisTNG/TNG100_spectra_true.dat')
	vel = spectra[:,0]
	spectra = spectra[:,1::]
	for ID in IDs:
		spectrum = spectra[:,ID]
		print(ID,TNG100['mass_stars'][ID],TNG100['Afr'][ID],TNG100['Type'][ID], len(np.where(TNG100['groupnr'] == TNG100['groupnr'][ID])[0]))
		fig, ax = plt.subplots()
		ax.plot(vel,spectrum,color='Black')

		vel_peak_L = vel[int(TNG100['PeaklocL'][ID])]
		vel_peak_R = vel[int(TNG100['PeaklocR'][ID])]
		vel_w20_L = np.interp(TNG100['w20L'][ID],np.arange(len(vel)),vel)
		vel_w20_R = np.interp(TNG100['w20R'][ID],np.arange(len(vel)),vel)


		ax.plot([vel_peak_L,vel_peak_L],[0,np.max(spectrum)],color = 'red')
		ax.plot([vel_peak_R,vel_peak_R],[0,np.max(spectrum)],color = 'red')
		ax.plot([vel_w20_L,vel_w20_L],[0,np.max(spectrum)],color='Blue',ls='--')
		ax.plot([vel_w20_R,vel_w20_R],[0,np.max(spectrum)],color='Blue',ls='--')
		ax.set_xlabel('Velocity')
		ax.set_ylabel('HI mass')
		ax.text(0.1,0.9,'Afr = {afr:.3f}'.format(afr=TNG100['Afr'][ID]),fontsize=12, transform=ax.transAxes)
		# fig.savefig('./figures/ITNG_{id}_spectrum.png'.format(id=ID))

	exit()


def compare_asym_gasfraction():
	particle_mass = 1.4e6

	TNG100 = Table.read('/media/data/simulations/IllustrisTNG/TNG100_galdata_measured_v2.ascii',format='ascii')
	IDs = np.arange(len(TNG100))
	TNG100['mass_stars'] = np.log10(TNG100['mass_stars'])
	TNG100['SFR'] = np.log10(TNG100['SFR']) - TNG100['mass_stars']

	TNG100['Npart'] = TNG100['Sint']/particle_mass

	TNG100['lgGF'] = np.log10(TNG100['Sint']) - TNG100['mass_stars']


	Npart = [500,1000,2000,5000]

	fig = plt.figure(figsize = (15,9))
	gs  = gridspec.GridSpec(4, 4, left = 0.05, right = 0.99, top=0.97, bottom = 0.06)

	for ii in range(4):
		TNG100 = TNG100[(TNG100['Npart'] >= Npart[ii]) & (TNG100['Npart'] <= Npart[-2])]

		halorange = np.where((TNG100['mass_stars'] > 10.5) & (TNG100['mass_stars'] <= 11) & (TNG100['mass_halo'] > 13.5))[0]
		massrange = np.where((TNG100['mass_stars'] > 10.5) & (TNG100['mass_stars'] <= 11))[0]


		TNG_samp_mass = TNG100[massrange]

		TNG_samp = TNG100[halorange]

		highAfr = np.where(TNG_samp['Afr'] > 1.7)[0]

		scat_ax = fig.add_subplot(gs[ii,0])
		hist_ax = fig.add_subplot(gs[ii,1])
		Npart_ax = fig.add_subplot(gs[ii,2])
		mstar_ax = fig.add_subplot(gs[ii,3])

		scat_ax.set_xlim([-3.5,1.3])
		scat_ax.set_ylim([0.95,3.7])

		scat_ax.set_ylabel('Asymmetry measure Afr')
		hist_ax.set_ylabel('Histogram Density')
		Npart_ax.set_ylabel('Histogram Density')
		mstar_ax.set_ylabel('Histogram Density')

		scat_ax.scatter(TNG100['lgGF'],TNG100['Afr'], c = np.log10(TNG100['Npart']),s=1)
		# scat_ax.colorbar()

		# scat_ax.scatter(TNG_samp_mass['lgGF'],TNG_samp_mass['Afr'], color='Orange',s=5)

		scat_ax.scatter(TNG_samp['lgGF'][np.where(TNG_samp['Afr'] <1.3)[0]],TNG_samp['Afr'][np.where(TNG_samp['Afr'] <1.3)[0]], color='Blue',s=5)

		scat_ax.scatter(TNG_samp[highAfr]['lgGF'],TNG_samp[highAfr]['Afr'],color='Red',s=5)


		hist_ax.hist(TNG_samp['lgGF'][np.where(TNG_samp['Afr'] <1.3)[0]],bins=np.arange(-2.2,0.2,0.1),color='Blue',density=True,histtype='step')
		# hist_ax.hist(TNG_samp_mass['lgGF'],color='Orange',density=True,histtype='step')
		hist_ax.hist(TNG_samp_mass[highAfr]['lgGF'],bins=np.arange(-2.2,0.2,0.1),color='Red',density=True,histtype='step')


		Npart_ax.hist(np.log10(TNG_samp['Npart'][np.where(TNG_samp['Afr'] <1.3)[0]]),bins=np.arange(2.5,4.6,0.1), color='Blue',density=True,histtype='step')
		Npart_ax.hist(np.log10(TNG_samp_mass[highAfr]['Npart']),bins=np.arange(2.5,4.6,0.1), color='Red',density=True,histtype='step')

		mstar_ax.hist(TNG_samp['mass_stars'][np.where(TNG_samp['Afr'] <1.3)[0]],bins=np.arange(10.4,11,0.05), color='Blue',density=True,histtype='step')
		mstar_ax.hist(TNG_samp_mass[highAfr]['mass_stars'],bins=np.arange(10.4,11,0.05), color='Red',density=True,histtype='step')





	scat_ax.set_xlabel('log10 GF')
	hist_ax.set_xlabel('log10 GF')
	Npart_ax.set_xlabel('log10 Npart')
	mstar_ax.set_xlabel('lgMstar')
	fig.savefig('./figures/ITNG_Afr_lgGF_3.png')
	plt.show()


def gas_spatial_distribution():

	import pickle
	f =  open('/media/data/simulations/IllustrisTNG/cell_HI_data_Adam.pkl', 'rb') 
	cell = pickle.load(f, encoding='bytes')

	# print(cell)
	newcell = {}
	for key in cell.keys():
		newcell[key.decode()] = cell[key]


	IDs = [ 31348, 60749, 76090, 88675,  161166, 217487]

	for ID in IDs:
		data = newcell[str(ID)]

		newdata = {}
		for key in data.keys():
			newdata[key.decode()] = data[key]


		coordinates = np.array(newdata['pos'])
		velocities = np.array(newdata['vel'])
		HI_masses = np.array(newdata['mHI'])


		# fig,ax  = plt.subplots()
		# img = ax.scatter(coordinates[:,0],coordinates[:,1],s=0.1,c=np.log10(HI_masses))
		# ax.set_xlabel('x [kpc]')
		# ax.set_ylabel('y [kpc]')
		# cbar = fig.colorbar(img)
		# # plt.show()
		# cbar.set_label('log10 HI mass',rotation=270)
		# fig.savefig('./figures/ITNG_{id}_LOSspatial_faceon_HImass.png'.format(id=ID))

		vel,spec = af.calc_spectrum(coordinates[:,0:2],velocities[:,2],HI_masses,dist=1,Vres=2,beamsize=[1000000000])
		plt.plot(vel,spec)
		vel,spec = af.create_HI_spectrum(coordinates[:,0:2],velocities[:,2],HI_masses,Vres=2,FWHM=[50,50])
		plt.plot(vel,spec)

		plt.show()
		exit()



		COM_gas = af.calc_COM(coordinates,HI_masses,Rmax=10)

		coordinates -= COM_gas

		gas_eigvec = af.orientation_matrix(coordinates, HI_masses)
		coordinates = coordinates @ gas_eigvec
		# velocities = velocities @ gas_eigvec




		fig,ax  = plt.subplots()
		img = ax.scatter(coordinates[:,0],coordinates[:,1],s=0.1,c=np.log10(HI_masses))
		ax.set_xlim([-100,100])
		ax.set_ylim([-100,100])
		ax.set_xlabel('x [kpc]')
		ax.set_ylabel('y [kpc]')
		cbar = fig.colorbar(img)
		# plt.show()
		cbar.set_label('log10 HI mass',rotation=270)
		fig.savefig('./figures/ITNG_{id}_ALIGNspatial_faceon_HImass.png'.format(id=ID))


		fig,ax  = plt.subplots()
		img = ax.scatter(coordinates[:,0],coordinates[:,2],s=0.1,c=np.log10(HI_masses))
		ax.set_xlim([-100,100])
		ax.set_ylim([-100,100])
		ax.set_xlabel('x [kpc]')
		ax.set_ylabel('z [kpc]')
		cbar = fig.colorbar(img)
		# plt.show()
		cbar.set_label('log10 HI mass',rotation=270)
		fig.savefig('./figures/ITNG_{id}_ALIGNspatial_edgeon_HImass.png'.format(id=ID))

def plot_disturbed_satellites():
	
	particle_mass = 1.4e6
	TNG100 = Table.read('/media/data/simulations/IllustrisTNG/TNG100_galdata_measured_v2.ascii',format='ascii')
	
	TNG100['Npart'] = TNG100['Sint'] / particle_mass

	good = np.where((TNG100['Sint']/particle_mass >= 1.e3))[0]

	TNG100['Rvir'] = np.cbrt(np.array(TNG100['mass_halo'])* 4.3e-3 / (100 * 70 * 70 * 1.e-6 * 1.e-6) )*1.e-6
	# exit()
	pos = ['pos_rel_x','pos_rel_y','pos_rel_z']
	vel = ['vel_rel_x','vel_rel_y','vel_rel_z']


	TNG100['rad'] = np.sqrt(np.nansum((np.array([TNG100[p] for p in pos]).T)**2.e0, axis=1))
	TNG100['vrel'] = np.sqrt(np.nansum((np.array([TNG100[v] for v in vel]).T)**2.e0, axis=1))



	TNG100['mass_stars'] = np.log10(TNG100['mass_stars'])
	TNG100['mass_halo'] = np.log10(TNG100['mass_halo'])
	TNG100['sSFR'] = np.log10(TNG100['SFR']) - TNG100['mass_stars']

	lowmass = np.where((TNG100['Npart'] > 5.e2) & 
			(TNG100['mass_halo'] < 12)  & (TNG100['mass_halo'] > 11.4)  &
			(TNG100['rad']*1.e-3 < TNG100['Rvir']) &
			(TNG100['Afr'] > 1.5) & 
			(TNG100['Type'] == 1))[0]

	highmass = np.where((TNG100['Npart'] > 5.e2) & 
			(TNG100['mass_halo'] > 13)  &
			(TNG100['rad']*1.e-3 < TNG100['Rvir']) &
			(TNG100['Afr'] > 1.5) &
			(TNG100['Type'] == 1))[0]

	lowmass = lowmass[np.argsort(TNG100['Afr'][lowmass])]
	highmass = highmass[np.argsort(TNG100['Afr'][highmass])]
	

	IDs_list = [lowmass,highmass]

	spectra = np.loadtxt('/media/data/simulations/IllustrisTNG/TNG100_spectra_true.dat')
	vel = spectra[:,0]
	spectra = spectra[:,1::]
	for ii in range(2):
		IDs = IDs_list[ii]
		for ID in IDs:
			spectrum = spectra[:,ID]
			# print(ID,TNG100['mass_stars'][ID],TNG100['Afr'][ID],TNG100['Type'][ID], len(np.where(TNG100['groupnr'] == TNG100['groupnr'][ID])[0]))
			fig, ax = plt.subplots()
			ax.plot(vel,spectrum,color='Black')

			vel_peak_L = vel[int(TNG100['PeaklocL'][ID])]
			vel_peak_R = vel[int(TNG100['PeaklocR'][ID])]
			vel_w20_L = np.interp(TNG100['w20L'][ID],np.arange(len(vel)),vel)
			vel_w20_R = np.interp(TNG100['w20R'][ID],np.arange(len(vel)),vel)


			ax.plot([vel_peak_L,vel_peak_L],[0,np.max(spectrum)],color = 'red')
			ax.plot([vel_peak_R,vel_peak_R],[0,np.max(spectrum)],color = 'red')
			ax.plot([vel_w20_L,vel_w20_L],[0,np.max(spectrum)],color='Blue',ls='--')
			ax.plot([vel_w20_R,vel_w20_R],[0,np.max(spectrum)],color='Blue',ls='--')
			ax.set_xlabel('Velocity')
			ax.set_ylabel('HI mass kms$^{-1}$')
			ax.text(0.1,0.9,'Afr = {afr:.3f}'.format(afr=TNG100['Afr'][ID]),fontsize=12, transform=ax.transAxes)
			if ii == 0:	
				fig.savefig('/media/data/simulations/IllustrisTNG/figures/lowMh/ITNG_{id}_spectrum_Afr{a:.2f}.png'.format(id=ID,a=TNG100['Afr'][ID]))
			else:
				fig.savefig('/media/data/simulations/IllustrisTNG/figures/highMh/ITNG_{id}_spectrum_Afr{a:.2f}.png'.format(id=ID,a=TNG100['Afr'][ID]))

			plt.close()


	exit()

def plot_TNG100_TN100_2():
	
	base = '/media/data/simulations/IllustrisTNG/TNG100-2'

	particle_mass = 1.4e6
	TNG = Table.read('{}_galdata_measured_v2.ascii'.format(base),format='ascii')
	
	TNG['Npart'] = TNG['Sint'] / particle_mass

	TNG['mass_stars'] = np.log10(TNG['mass_stars'])
	TNG['mass_halo'] = np.log10(TNG['mass_halo'])
	TNG['sSFR'] = np.log10(TNG['SFR']) - TNG['mass_stars']


	lgMstar_low = 9.0
	lgMstar_high = 9.5

	sSFR_low = -10.5
	sSFR_high = -9.8


	gals = np.where((TNG['Npart'] > 5.e2) & 
			(TNG['mass_stars'] < lgMstar_high) & (TNG['mass_stars']> lgMstar_low)  &
			(TNG['sSFR'] > sSFR_low) & (TNG['sSFR']  < sSFR_high ) )[0]

	# print(TNG)

	savedir = '/'.join(base.split('/')[0:-1]) + '/figures/' + base.split('/')[-1]


	spectra = np.loadtxt('{}_spectra_true.dat'.format(base))
	vel = spectra[:,0]
	spectra = spectra[:,1::]
	for ID in gals:
		spectrum = spectra[:,ID]
		# print(ID,TNG100['mass_stars'][ID],TNG100['Afr'][ID],TNG100['Type'][ID], len(np.where(TNG100['groupnr'] == TNG100['groupnr'][ID])[0]))
		fig, ax = plt.subplots()
		ax.plot(vel,spectrum,color='Black')

		vel_peak_L = vel[int(TNG['PeaklocL'][ID])]
		vel_peak_R = vel[int(TNG['PeaklocR'][ID])]
		vel_w20_L = np.interp(TNG['w20L'][ID],np.arange(len(vel)),vel)
		vel_w20_R = np.interp(TNG['w20R'][ID],np.arange(len(vel)),vel)


		ax.plot([vel_peak_L,vel_peak_L],[0,np.max(spectrum)],color = 'red')
		ax.plot([vel_peak_R,vel_peak_R],[0,np.max(spectrum)],color = 'red')
		ax.plot([vel_w20_L,vel_w20_L],[0,np.max(spectrum)],color='Blue',ls='--')
		ax.plot([vel_w20_R,vel_w20_R],[0,np.max(spectrum)],color='Blue',ls='--')
		ax.set_xlabel('Velocity')
		ax.set_ylabel('HI mass kms$^{-1}$')
		ax.text(0.1,0.9,'Afr = {afr:.3f}'.format(afr=TNG['Afr'][ID]),fontsize=12, transform=ax.transAxes)
	
		fig.savefig('{savedir}/ITNG_{id}_spectrum_Afr{a:.2f}.png'.format(savedir=savedir,id=ID,a=TNG['Afr'][ID]))
		
		plt.close()


	exit()





#halo and environment stuff

def Afr_satellites_Rvir_sSFR():
	particle_mass = 1.4e6
	TNG100 = Table.read('/media/data/simulations/IllustrisTNG/TNG100_galdata_measured_v2.ascii',format='ascii')
	
	TNG100['Npart'] = TNG100['Sint'] / particle_mass


	TNG100['Rvir'] = np.cbrt(np.array(TNG100['mass_halo'])* 4.3e-3 / (100 * 70*70 * 1.e-6 * 1.e-6) )*1.e-6
	TNG100['V200'] = np.sqrt(4.3e-3 * TNG100['mass_halo'] / (TNG100['Rvir']*1.e6))

	TNG100['sSFR'] = np.log10(TNG100['SFR']) - np.log10(TNG100['mass_stars'])
	# exit()
	pos = ['pos_rel_x','pos_rel_y','pos_rel_z']
	vel = ['vel_rel_x','vel_rel_y','vel_rel_z']


	TNG100['rad'] = np.sqrt(np.nansum((np.array([TNG100[p] for p in pos]).T)**2.e0, axis=1))


	TNG100['vrad'] = np.nansum(np.array([TNG100[p] for p in pos]).T * np.array([TNG100[v] for v in vel]).T, axis=1) /\
						np.sqrt(np.nansum((np.array([TNG100[p] for p in pos]).T)**2.e0, axis=1))

	sats = TNG100[(TNG100['Type'] == 1) & (TNG100['Npart'] > 5.e2) & (TNG100['mass_halo'] < 1.e12)]

	sats_inside = sats[sats['rad']*1.e-3 < sats['Rvir']]
	sats_outside = sats[sats['rad']*1.e-3 > sats['Rvir']]


	asym_sats_inside = sats_inside[(sats_inside['Afr']>1.4)]

	plt.hist(asym_sats_inside['sSFR'],bins=np.arange(-12,-8,0.25),alpha=0.5,density=True)
	plt.hist(sats_outside['sSFR'],bins=np.arange(-12,-8,0.25),alpha=0.5,density=True)
	plt.show()

def Afrhist_satellites_centrals():
	particle_mass = 1.4e6
	TNG100 = Table.read('/media/data/simulations/IllustrisTNG/TNG100_galdata_measured_v2.ascii',format='ascii')
	
	TNG100['Npart'] = TNG100['Sint'] / particle_mass

	# exit()
	pos = ['pos_rel_x','pos_rel_y','pos_rel_z']
	vel = ['vel_rel_x','vel_rel_y','vel_rel_z']


	TNG100['rad'] = np.sqrt(np.nansum((np.array([TNG100[p] for p in pos]).T)**2.e0, axis=1))
	TNG100['vrel'] = np.sqrt(np.nansum((np.array([TNG100[v] for v in vel]).T)**2.e0, axis=1))


	
	TNG100 = TNG100[(TNG100['Npart'] > 5.e2)]

	sats = TNG100[TNG100['Type'] == 1]
	cents = TNG100[TNG100['Type'] != 1]


	samples = [sats['Afr'],cents['Afr']]
	controls = [np.log10(sats['Npart']),np.log10(cents['Npart'])]
	names = ['Satellites ({})'.format(len(sats)), 'Centrals ({})'.format(len(cents))]

	Afr_bins = np.arange(1,2.5,0.05)
	Npart_bins = np.arange(np.log10(500),np.log10(5000)+0.1,0.1)
	Npart_bins = np.append(Npart_bins,np.array([np.log10(1.e6)]))

	

	samples_all_hists, samples_all_DAGJKsigma, indexes_all_iter = \
								dsd.control_samples(samples,Afr_bins,controls,Npart_bins,Niter=100)

	dsd.plot_controlled_cumulative_histograms(samples_all_hists,Afr_bins,samples_all_DAGJKsigma,
			 names=names,save='/media/data/simulations/IllustrisTNG/figures/TNG_Afrhist_sat_cen_2.png')

	# dsd.plot_compare_DAGJK_Nsamp_sigmas(samples_all_DAGJKsigma, samples_all_hists, Afr_bins, names=names, save=
		# '/media/data/simulations/IllustrisTNG/figures/sigma_hist_sat_cent.png')

	# fig.savefig('/media/data/simulations/IllustrisTNG/figures/TNG_Afrhist_sat_cen.png')
	plt.show()
	exit()

def compare_asymmetry_halomass():

	particle_mass = 1.4e6

	TNG100 = Table.read('/media/data/simulations/IllustrisTNG/TNG100_galdata_measured_v2.ascii',format='ascii')

	TNG100['Npart'] = TNG100['Sint'] / particle_mass

	TNG100['mass_stars'] = np.log10(TNG100['mass_stars'])
	TNG100['mass_halo'] = np.log10(TNG100['mass_halo'])
	TNG100['sSFR'] = np.log10(TNG100['SFR']) - TNG100['mass_stars']

	Afr_bins = np.arange(1,2.5,0.05)
	Npart_bins = np.arange(np.log10(500),np.log10(5000)+0.1,0.1)
	Npart_bins = np.append(Npart_bins,np.array([np.log10(1.e6)]))
	

	TNG100 = TNG100[TNG100['Npart'] > 1.e3]

	# plt.hist(TNG100['mass_halo'][(TNG100['mass_stars'] > 10.5) & (TNG100['mass_stars']< 11.5)],bins=20)
	# plt.xlabel('log10 Halo mass')
	# plt.ylabel('Histogram Count')
	# plt.show()
	# exit()

	massrange_1 = np.where((TNG100['mass_stars'] > 10.5) & (TNG100['mass_stars'] <= 11))[0]
	massrange_2 = np.where((TNG100['mass_stars'] > 11) & (TNG100['mass_stars'] <= 11.5))[0]

	TNG100 = TNG100[massrange_1]

	lowmass = np.where((TNG100['mass_halo'] <= 12))[0]
	medmass = np.where((TNG100['mass_halo'] > 12) & (TNG100['mass_halo'] <= 13))[0]
	highmass = np.where( (TNG100['mass_halo'] > 13))[0]


	samp1 = TNG100[lowmass]
	samp2 = TNG100[medmass]
	samp3 = TNG100[highmass]



	# print('Ncen', len(TNG100[(TNG100['mass_stars'] > 10.) & (TNG100['mass_stars'] < 11.5) & (TNG100['Type'] < 1)] ))
	# print('Nsat', len(TNG100[(TNG100['mass_stars'] > 10.) & (TNG100['mass_stars'] < 11.5) & (TNG100['Type']  == 1)]))

	fig = plt.figure(figsize = (10,8))

	gs = gridspec.GridSpec(2, 1, top = 0.95, right = 0.98, bottom  = 0.12, left = 0.08)

	ax1 = fig.add_subplot(gs[0,0])
	ax2 = fig.add_subplot(gs[1,0],sharex=ax1, sharey=ax1)


	# ax1.set_ylabel('Cumulative Historgram')
	# ax2.set_ylabel('Cumulative Historgram')
	# ax2.set_xlabel('Asymmetry measure A$_{fr}$')

	# ax1.set_title('lgMstar = (10.5,11]',fontsize=10)
	# ax2.set_title('lgMstar = (11,11.5]',fontsize=10)


	samples_all_hists, samples_all_DAGJKsigma, indexes_all_iter = \
								dsd.control_samples([samp1['Afr'],samp2['Afr']],Afr_bins,[np.log10(samp1['Npart']),np.log10(samp2['Npart'])],Npart_bins)

	dsd.plot_controlled_cumulative_histograms(samples_all_hists,Afr_bins,samples_all_DAGJKsigma,
			 axis = ax1,names=['lgM$_h$ < 12 ({})'.format(len(lowmass)),'lgM$_h$ = (12,13] ({})'.format(len(medmass))])

	samples_all_hists, samples_all_DAGJKsigma, indexes_all_iter = \
								dsd.control_samples([samp2['Afr'],samp3['Afr']],Afr_bins,[np.log10(samp2['Npart']),np.log10(samp3['Npart'])],Npart_bins)

	dsd.plot_controlled_cumulative_histograms(samples_all_hists,Afr_bins,samples_all_DAGJKsigma,
			 axis = ax2,names=['lgM$_h$ = (12,13] ({})'.format(len(medmass)),'lgM$_h$ > 13 ({})'.format(len(highmass))])

	
	ax1.tick_params(axis = 'both', which = 'both', direction = 'in', labelsize = 16)
	ax2.tick_params(axis = 'both', which = 'both', direction = 'in', labelsize = 16)

	ax1.tick_params(axis = 'x', which = 'both', direction = 'in', labelsize = 0)


	plt.show()

def compare_asymmetry_halomass_envsplit():

	particle_mass = 1.4e6

	TNG100 = Table.read('/media/data/simulations/IllustrisTNG/TNG100_galdata_measured_v2.ascii',format='ascii')

	TNG100['Npart'] = TNG100['Sint'] / particle_mass

	TNG100 = TNG100[TNG100['Npart'] > 1.e3]

	TNG100['mass_stars'] = np.log10(TNG100['mass_stars'])
	TNG100['mass_halo'] = np.log10(TNG100['mass_halo'])
	TNG100['SFR'] = np.log10(TNG100['SFR']) - TNG100['mass_stars']


	satellites = TNG100[TNG100['Type'] == 1]
	centrals = TNG100[TNG100['Type'] < 1]

	plt.hist(satellites['mass_halo'],bins=20, histtype='step',label='{}'.format(len(satellites['mass_halo'])))
	plt.hist(centrals['mass_halo'],bins=20, histtype='step',label='{}'.format(len(centrals['mass_halo'])))
	plt.xlabel('log10 Halo mass')
	plt.ylabel('Histogram Count')
	plt.legend()
	plt.show()
	# exit()


	Mhrange_1 = np.where((TNG100['mass_halo'] > 11) & (TNG100['mass_halo'] <= 12))[0]
	Mhrange_2 = np.where((TNG100['mass_halo'] > 12) & (TNG100['mass_halo'] <= 13))[0]

	fig = plt.figure(figsize=(12,6))
	gs = gridspec.GridSpec(1,4,left=0.08,right=0.99,wspace=0.3) 
	ax1 = fig.add_subplot(gs[0,0])
	ax2 = fig.add_subplot(gs[0,1])
	ax3 = fig.add_subplot(gs[0,2])
	ax4 = fig.add_subplot(gs[0,3])
	ax1.set_ylabel('Cumulative Histogram',fontsize = 15)
	ax2.set_ylabel('Histogram Density',fontsize = 15)
	ax3.set_ylabel('Histogram Density',fontsize = 15)
	ax4.set_ylabel('Histogram Density',fontsize = 15)

	ax1.set_xlabel('Asymmetry measure A$_{fr}$',fontsize=15)
	ax2.set_xlabel('lgMh',fontsize=15)
	ax3.set_xlabel('lgMstar',fontsize=15)
	ax4.set_xlabel('lgNpart',fontsize=15)



	massranges = [Mhrange_1,Mhrange_2]
	axes = [ax1,ax2]
	colors = ['Green','Orange']

	# ax1.set_title('lgMstar = (10.5,11]',fontsize=10)
	# ax2.set_title('lgMstar = (11,11.5]',fontsize=10)
	# ax2.set_xlabel('Asymmetry measure A$_{fr}$')

	for ii in range(len(massranges)):

		TNG_samp = TNG100[massranges[ii]]


		satellites = TNG_samp[TNG_samp['Type'] == 1]
		centrals = TNG_samp[TNG_samp['Type'] < 1]


		ax1.hist(satellites['Afr'],bins=np.arange(1,2.2,0.05)
							,histtype='step',density=True,cumulative=True,lw = 2,color = colors[ii],ls=':')
		ax1.hist(centrals['Afr'],bins=np.arange(1,2.2,0.05)
							,histtype='step',density=True,cumulative=True,lw = 2,color = colors[ii],ls='--')
		# ax1.hist(TNG_samp['Afr'],bins=np.arange(1,2.2,0.05)
							# ,histtype='step',density=True,cumulative=True,lw = 2,color = colors[ii],ls='-')

		ax2.hist(satellites['mass_halo'],bins=10
							,histtype='step',density=True,lw = 2,color = colors[ii],ls=':')
		ax2.hist(centrals['mass_halo'],bins=10
							,histtype='step',density=True,lw = 2,color = colors[ii],ls='--')
		# ax2.hist(TNG_samp['mass_halo'],bins=10
							# ,histtype='step',density=True,lw = 2,color = colors[ii],ls='-')

		ax3.hist(satellites['mass_stars'],bins=10
							,histtype='step',density=True,lw = 2,color = colors[ii],ls=':')
		ax3.hist(centrals['mass_stars'],bins=10
							,histtype='step',density=True,lw = 2,color = colors[ii],ls='--')
		# ax3.hist(TNG_samp['mass_stars'],bins=10
							# ,histtype='step',density=True,lw = 2,color = colors[ii],ls='-')


		ax4.hist(np.log10(satellites['Npart']),bins=10
							,histtype='step',density=True,lw = 2,color = colors[ii],ls=':')
		ax4.hist(np.log10(centrals['Npart']),bins=10
							,histtype='step',density=True,lw = 2,color = colors[ii],ls='--')
		# ax4.hist(np.log10(TNG_samp['Npart']),bins=10
							# ,histtype='step',density=True,lw = 2,color = colors[ii],ls='-')

		

	fig.savefig('/media/data/simulations/IllustrisTNG/figures/Afr_hist_Mh_envsplit.png')	
	plt.show()
	exit()

def Afr_satellites_Rvir():

	particle_mass = 1.4e6
	TNG100 = Table.read('/media/data/simulations/IllustrisTNG/TNG100_galdata_measured_v2.ascii',format='ascii')
	
	TNG100['Npart'] = TNG100['Sint'] / particle_mass

	good = np.where((TNG100['Sint']/particle_mass >= 1.e3))[0]

	TNG100['Rvir'] = np.cbrt(np.array(TNG100['mass_halo'])* 4.3e-3 / (100 * 70*70 * 1.e-6 * 1.e-6) )*1.e-6
	# exit()
	pos = ['pos_rel_x','pos_rel_y','pos_rel_z']
	vel = ['vel_rel_x','vel_rel_y','vel_rel_z']


	TNG100['rad'] = np.sqrt(np.nansum((np.array([TNG100[p] for p in pos]).T)**2.e0, axis=1))
	TNG100['vrel'] = np.sqrt(np.nansum((np.array([TNG100[v] for v in vel]).T)**2.e0, axis=1))


	
	TNG100 = TNG100[(TNG100['Npart'] > 5.e2)]

	sats = TNG100[TNG100['Type'] == 1]

	# sats = sats[(sats['Npart'] > 5.e2) & (sats['mass_halo'] < 1.e12)  & (sats['mass_halo'] > 10.**(11.4)) ]
	# sat_inside = np.where(sats['rad']*1.e-3 < sats['Rvir'])[0]
	# sat_outside = np.where(sats['rad']*1.e-3 > sats['Rvir'])[0]
	# print(sats)
	# grp_sizes = np.zeros(len(sats))
	# for ii in range(len(sats)):
	# 	grp = sats['groupnr'][ii]
	# 	ingrp = np.where(sats['groupnr'] == grp)[0]
	# 	grp_sizes[ii] = len(ingrp)
	# plt.hist(grp_sizes[sat_inside],bins=np.arange(0.5,9.5),alpha=0.6)
	# plt.hist(grp_sizes[sat_outside],bins=np.arange(0.5,9.5),alpha=0.4)
	# plt.show()
	# exit()

	# print(len(sats))
	# print(len(sats[sats['rad']*1.e-3 < sats['Rvir']]['Afr']))
	# print(len(sats[sats['rad']*1.e-3 > sats['Rvir']]['Afr']))
	# exit()

	plot1=False
	plot2=False
	plot_lowMh_mstar = True

	if plot1:

		fig = plt.figure(figsize=(12,6))
		gs = gridspec.GridSpec(1,4,left=0.08,right=0.99,wspace=0.3) 
		ax1 = fig.add_subplot(gs[0,0])
		ax2 = fig.add_subplot(gs[0,1])
		ax3 = fig.add_subplot(gs[0,2])
		ax4 = fig.add_subplot(gs[0,3])
		ax1.set_ylabel('Cumulative Histogram',fontsize = 15)
		ax2.set_ylabel('Histogram Density',fontsize = 15)
		ax3.set_ylabel('Histogram Density',fontsize = 15)
		ax4.set_ylabel('Histogram Density',fontsize = 15)

		ax1.set_xlabel('Asymmetry measure A$_{fr}$',fontsize=15)
		ax2.set_xlabel('lgMh',fontsize=15)
		ax3.set_xlabel('lgMstar',fontsize=15)
		ax4.set_xlabel('lgNpart',fontsize=15)


		ax1.hist(sats[sats['rad']*1.e-3 < sats['Rvir']]['Afr'],bins=np.arange(1,2.2,0.05)
								,histtype='step',density=True,cumulative=True,lw = 2)
		ax1.hist(sats[sats['rad']*1.e-3 > sats['Rvir']]['Afr'],bins=np.arange(1,2.2,0.05)
								,histtype='step',density=True,cumulative=True,lw = 2)
		
		ax2.hist(np.log10(sats[sats['rad']*1.e-3 < sats['Rvir']]['mass_halo']),bins=10
								,histtype='step',density=True,lw = 2)
		ax2.hist(np.log10(sats[sats['rad']*1.e-3 > sats['Rvir']]['mass_halo']),bins=10
								,histtype='step',density=True,lw = 2)

		ax3.hist(np.log10(sats[sats['rad']*1.e-3 < sats['Rvir']]['mass_stars']),bins=10
								,histtype='step',density=True,lw = 2)
		ax3.hist(np.log10(sats[sats['rad']*1.e-3 > sats['Rvir']]['mass_stars']),bins=10
								,histtype='step',density=True,lw = 2)

		ax4.hist(np.log10(sats[sats['rad']*1.e-3 < sats['Rvir']]['Npart']),bins=10
								,histtype='step',density=True,lw = 2,label='Inside Rvir')
		ax4.hist(np.log10(sats[sats['rad']*1.e-3 > sats['Rvir']]['Npart']),bins=10
								,histtype='step',density=True,lw = 2,label='Outside Rvir')

		ax4.legend()

		fig.savefig('/media/data/simulations/IllustrisTNG/figures/sats_Rvir_asym.png')
		plt.show()
		exit()


	if plot2:

		fig = plt.figure(figsize=(18,15))
		gs = gridspec.GridSpec(3,3,left=0.08,right=0.99,wspace=0.3,hspace=0.3,top=0.99) 
		

		for ii in range(3):
			if ii == 0:
				TNG_samp = TNG100[(TNG100['Npart'] > 5.e2) & (TNG100['mass_halo'] < 1.e12)  & (TNG100['mass_halo'] > 10.**(11.4))  ]
			if ii == 1:
				TNG_samp = TNG100[(TNG100['Npart'] > 5.e2) & (TNG100['mass_halo'] < 1.e13) & (TNG100['mass_halo'] > 1.e12)]
			if ii == 2:
				TNG_samp = TNG100[(TNG100['Npart'] > 5.e2) & (TNG100['mass_halo'] > 1.e13)]

			sats = TNG_samp[TNG_samp['Type'] == 1]

			ax1 = fig.add_subplot(gs[ii,0])
			ax2 = fig.add_subplot(gs[ii,1])
			ax3 = fig.add_subplot(gs[ii,2])
			
			ax1.set_ylabel('Cumulative Histogram',fontsize = 15)
			ax2.set_ylabel('Histogram Density',fontsize = 15)
			ax3.set_ylabel('Histogram Density',fontsize = 15)

			ax1.set_xlabel('Asymmetry measure A$_{fr}$',fontsize=15)
			ax2.set_xlabel('lgMh',fontsize=15)
			ax3.set_xlabel('lgMstar',fontsize=15)


			sat_inside = sats[sats['rad']*1.e-3 < sats['Rvir']]
			sat_outside = sats[sats['rad']*1.e-3 > sats['Rvir']]

			samples = [sat_inside['Afr'],sat_outside['Afr']]
			controls = [np.log10(sat_inside['Npart']),np.log10(sat_outside['Npart'])]

			Afr_bins = np.arange(1,2.5,0.05)
			Npart_bins = np.arange(np.log10(500),np.log10(5000)+0.1,0.1)
			Npart_bins = np.append(Npart_bins,np.array([np.log10(1.e6)]))

			

			samples_all_hists, samples_all_DAGJKsigma, indexes_all_iter = \
										dsd.control_samples(samples,Afr_bins,controls,Npart_bins)

			dsd.plot_controlled_cumulative_histograms(samples_all_hists,Afr_bins,samples_all_DAGJKsigma,
						axis = ax1, names=['Inside R$_{{vir}}$ ({})'.format(len(sat_inside)),'Outside R$_{{vir}}$ ({})'.format(len(sat_outside))])

			ax2.hist(np.log10(sat_inside['mass_halo']),bins=10
									,histtype='step',density=True,lw = 2, color='Orange')
			ax2.hist(np.log10(sat_outside['mass_halo']),bins=10
									,histtype='step',density=True,lw = 2, color='Green')

			ax3.hist(np.log10(sat_inside['mass_stars']),bins=np.arange(9,12.25,0.1)
									,histtype='step',density=True,lw = 2, color='Orange')
			ax3.hist(np.log10(sat_outside['mass_stars']),bins=np.arange(9,12.25,0.1)
									,histtype='step',density=True,lw = 2, color='Green')


		fig.savefig('/media/data/simulations/IllustrisTNG/figures/sats_Rvir_asym_manyhalo_.png')
		plt.show()
		exit()



	if plot_lowMh_mstar:
		
		


		fig = plt.figure(figsize=(18,15))
		gs = gridspec.GridSpec(3,3,left=0.08,right=0.99,wspace=0.3,hspace=0.3,top=0.99) 
		


		ax1 = fig.add_subplot(gs[0,0])
		ax2 = fig.add_subplot(gs[0,1])
		ax3 = fig.add_subplot(gs[0,2])

		ax4 = fig.add_subplot(gs[1,0])
		ax5 = fig.add_subplot(gs[1,1])
		ax6 = fig.add_subplot(gs[1,2])

		ax7 = fig.add_subplot(gs[2,0])
		ax8 = fig.add_subplot(gs[2,1])
		ax9 = fig.add_subplot(gs[2,2])
		
		ax1.set_ylabel('Cumulative Histogram',fontsize = 15)
		ax2.set_ylabel('Histogram Density',fontsize = 15)
		ax3.set_ylabel('Histogram Density',fontsize = 15)

		ax4.set_ylabel('Cumulative Histogram',fontsize = 15)
		ax5.set_ylabel('Histogram Density',fontsize = 15)
		ax6.set_ylabel('Histogram Density',fontsize = 15)

		ax7.set_ylabel('Cumulative Histogram',fontsize = 15)
		ax8.set_ylabel('Histogram Density',fontsize = 15)
		ax9.set_ylabel('Histogram Density',fontsize = 15)

		ax1.set_xlabel('Asymmetry measure A$_{fr}$',fontsize=15)
		ax2.set_xlabel('lgMh',fontsize=15)
		ax3.set_xlabel('lgMstar',fontsize=15)

		ax4.set_xlabel('Asymmetry measure A$_{fr}$',fontsize=15)
		ax5.set_xlabel('lgMh',fontsize=15)
		ax6.set_xlabel('lgMstar',fontsize=15)

		ax7.set_xlabel('Asymmetry measure A$_{fr}$',fontsize=15)
		ax8.set_xlabel('lgMh',fontsize=15)
		ax9.set_xlabel('lgMstar',fontsize=15)


		TNG_samp = TNG100[(TNG100['Npart'] > 5.e2) & (TNG100['mass_halo'] < 1.e12)  & (TNG100['mass_halo'] > 10.**(11.4))  ]

		sats = TNG_samp[TNG_samp['Type'] == 1]

		sat_inside = sats[sats['rad']*1.e-3 < sats['Rvir']]
		sat_outside = sats[sats['rad']*1.e-3 > sats['Rvir']]

		lgMstar_dist = [np.log10(sat_inside['mass_stars']),np.log10(sat_outside['mass_stars'])]
		Afr_dist = [sat_inside['Afr'],sat_outside['Afr']]

		Afr_bins = np.arange(1,2.5,0.05)
		lgMstar_bins = np.arange(9,11,0.1)
		lgMh_bins = np.arange(11,12,0.1)

		##### plot just Npart control
		samples = [sat_inside['Afr'],sat_outside['Afr']]
		controls = [np.log10(sat_inside['Npart']),np.log10(sat_outside['Npart'])]
		Npart_bins = np.arange(np.log10(500),np.log10(5000)+0.1,0.1)
		Npart_bins = np.append(Npart_bins,np.array([np.log10(1.e6)]))

		samples_all_hists, samples_all_DAGJKsigma, indexes_all_iter = \
									dsd.control_samples(samples,Afr_bins,controls,Npart_bins)
		dsd.plot_controlled_cumulative_histograms(samples_all_hists,Afr_bins,samples_all_DAGJKsigma,
					axis=ax1,names=['Inside R$_{{vir}}$ ({})'.format(len(samples[0])),'Outside R$_{{vir}}$ ({})'.format(len(samples[1]))])					

		ax3.hist(np.log10(sat_inside['mass_stars']),bins=lgMstar_bins,density = True,lw=2, histtype='step',color='Orange')
		ax3.hist(np.log10(sat_outside['mass_stars']),bins=lgMstar_bins,density = True,lw=2, histtype='step',color='Green')

		ax2.hist(np.log10(sat_inside['mass_halo']),bins=lgMh_bins,density = True,lw=2, histtype='step',color='Orange')
		ax2.hist(np.log10(sat_outside['mass_halo']),bins=lgMh_bins,density = True,lw=2, histtype='step',color='Green')


		###### plot Npart control after a lgMstar control

		samples_all_hists, samples_all_DAGJKsigma, indexes_all_iter = \
										dsd.control_samples(Afr_dist,Afr_bins,lgMstar_dist,lgMstar_bins, Niter=1)
	

		sat_inside = sat_inside[indexes_all_iter[0][0]]
		sat_outside = sat_outside[indexes_all_iter[1][0]]

		samples = [sat_inside['Afr'],sat_outside['Afr']]
		controls = [np.log10(sat_inside['Npart']),np.log10(sat_outside['Npart'])]
		Npart_bins = np.arange(np.log10(500),np.log10(5000)+0.1,0.1)
		Npart_bins = np.append(Npart_bins,np.array([np.log10(1.e6)]))

		samples_all_hists, samples_all_DAGJKsigma, indexes_all_iter = \
									dsd.control_samples(samples,Afr_bins,controls,Npart_bins)
		dsd.plot_controlled_cumulative_histograms(samples_all_hists,Afr_bins,samples_all_DAGJKsigma,
					axis=ax4,names=['Inside R$_{{vir}}$ ({})'.format(len(samples[0])),'Outside R$_{{vir}}$ ({})'.format(len(samples[1]))])					


		ax6.hist(np.log10(sat_inside['mass_stars']),bins=lgMstar_bins,density = True,lw=2, histtype='step',color='Orange')
		ax6.hist(np.log10(sat_outside['mass_stars']),bins=lgMstar_bins,density = True,lw=2, histtype='step',color='Green')

		ax5.hist(np.log10(sat_inside['mass_halo']),bins=lgMh_bins,density = True,lw=2, histtype='step',color='Orange')
		ax5.hist(np.log10(sat_outside['mass_halo']),bins=lgMh_bins,density = True,lw=2, histtype='step',color='Green')


		###### plot Npart control after a lgMstar control and an lgMh control

		lgMh_dist = [np.log10(sat_inside['mass_halo']),np.log10(sat_outside['mass_halo'])]

		samples_all_hists, samples_all_DAGJKsigma, indexes_all_iter = \
										dsd.control_samples(Afr_dist,Afr_bins,lgMh_dist,lgMh_bins, Niter=1)
	
		sat_inside = sat_inside[indexes_all_iter[0][0]]
		sat_outside = sat_outside[indexes_all_iter[1][0]]

		samples = [sat_inside['Afr'],sat_outside['Afr']]
		controls = [np.log10(sat_inside['Npart']),np.log10(sat_outside['Npart'])]
		Npart_bins = np.arange(np.log10(500),np.log10(5000)+0.1,0.1)
		Npart_bins = np.append(Npart_bins,np.array([np.log10(1.e6)]))

		samples_all_hists, samples_all_DAGJKsigma, indexes_all_iter = \
									dsd.control_samples(samples,Afr_bins,controls,Npart_bins)
		dsd.plot_controlled_cumulative_histograms(samples_all_hists,Afr_bins,samples_all_DAGJKsigma,
					axis=ax7,names=['Inside R$_{{vir}}$ ({})'.format(len(samples[0])),'Outside R$_{{vir}}$ ({})'.format(len(samples[1]))])					



		ax9.hist(np.log10(sat_inside['mass_stars']),bins=lgMstar_bins,density = True,lw=2, histtype='step',color='Orange')
		ax9.hist(np.log10(sat_outside['mass_stars']),bins=lgMstar_bins,density = True,lw=2, histtype='step',color='Green')

		ax8.hist(np.log10(sat_inside['mass_halo']),bins=lgMh_bins,density = True,lw=2, histtype='step',color='Orange')
		ax8.hist(np.log10(sat_outside['mass_halo']),bins=lgMh_bins,density = True,lw=2, histtype='step',color='Green')

		fig.savefig('/media/data/simulations/IllustrisTNG/figures/sats_Rvir_asym_lowMh_lgMstar_sample.png')
		plt.show()
		exit()





		



	# grp_sigma = np.zeros(len(TNG100))
	# for grp in np.array(np.unique(TNG100['groupnr'])):
	# 	ingrp = np.where(TNG100['groupnr'] == grp)[0]
	# 	if len(ingrp) > 4:
	# 		grp_sigma[ingrp] = np.std(vrel[ingrp][vrel[ingrp] != 0])
	# 	else:
	# 		grp_sigma[ingrp] = -1


	# rad = rad[grp_sigma != -1]
	# Rvir = Rvir[grp_sigma != -1]
	# vrel = vrel[grp_sigma != -1]
	# grp_sigma = grp_sigma[grp_sigma != -1]

	# plt.scatter(rad*1.e-3/Rvir, vrel/grp_sigma,s=2)
	# plt.show()

def Afr_halo_phasespace():
	particle_mass = 1.4e6
	TNG100 = Table.read('/media/data/simulations/IllustrisTNG/TNG100_galdata_measured_v2.ascii',format='ascii')
	
	TNG100['Npart'] = TNG100['Sint'] / particle_mass


	TNG100['Rvir'] = np.cbrt(np.array(TNG100['mass_halo'])* 4.3e-3 / (100 * 70*70 * 1.e-6 * 1.e-6) ) *1.e-6
	TNG100['V200'] = np.sqrt(4.3e-3 * TNG100['mass_halo'] / (TNG100['Rvir']*1.e6))

	# exit()
	pos = ['pos_rel_x','pos_rel_y','pos_rel_z']
	vel = ['vel_rel_x','vel_rel_y','vel_rel_z']


	# TNG100['vrel'] = np.sqrt(np.nansum((np.array([TNG100[v] for v in vel]).T)**2.e0, axis=1))

	positions = np.array([TNG100[p] for p in pos]).T
	velocities = np.array([TNG100[v] for v in vel]).T

	TNG100['rad'] = np.sqrt(np.nansum(positions**2.e0, axis=1))


	# print(positions)
	# print(velocities)
	# print(positions * velocities)
	# print(np.nansum(positions*velocities, axis=1))


	TNG100['vrad'] = (np.nansum(positions*velocities, axis=1) / TNG100['rad'] )+ 70 * TNG100['rad']*1.e-3 
						

	stats = Table.read('/media/data/simulations/IllustrisTNG/data/TNG_sym_percentile_stats.dat', format='ascii')					

	Npart_HIweight = stats['Npart_HIweight']
	Npart = stats['Npart']
	P40_Afr = stats['P40']
	P50_Afr = stats['P50']
	P60_Afr = stats['P60']
	P70_Afr = stats['P70']
	P80_Afr = stats['P80']
	P90_Afr = stats['P90']

	P40_fit, P40_fit_covar = curve_fit(func,Npart_HIweight,P40_Afr)
	P50_fit, P50_fit_covar = curve_fit(func,Npart_HIweight,P50_Afr)
	P60_fit, P60_fit_covar = curve_fit(func,Npart_HIweight,P60_Afr)
	P70_fit, P70_fit_covar = curve_fit(func,Npart_HIweight,P70_Afr)
	P80_fit, P80_fit_covar = curve_fit(func,Npart_HIweight,P80_Afr)	
	P90_fit, P90_fit_covar = curve_fit(func,Npart_HIweight,P90_Afr)	

	




	# plt.scatter(TNG100['rad'][(TNG100['mass_halo']<1.e12) & (TNG100['Type']==1)]*1e-3/TNG100['Rvir'][(TNG100['mass_halo']<1.e12) & (TNG100['Type']==1)],TNG100['vrad'][(TNG100['mass_halo']<1.e12)&  (TNG100['Type']==1)],c='Grey',alpha=0.5,s=10)

	TNG = TNG100[(TNG100['Npart'] > 5.e2)]
	
	# sats = TNG[(TNG['Type'] == 1) & (TNG['mass_halo'] > 1.e13)]
	# sats1 = TNG100[(TNG100['Type'] == 1) & (TNG100['mass_halo'] > 1.e13)]
	# sats_outside = TNG100[(TNG100['Type'] == 1) & (TNG100['mass_halo'] > 1.e13) & (TNG100['rad']*1.e-3 > TNG100['Rvir']) ]
	# sats_inside = TNG100[(TNG100['Type'] == 1) & (TNG100['mass_halo'] > 1.e13) & (TNG100['rad']*1.e-3 < TNG100['Rvir']) ]

	sats = TNG[(TNG['Type'] == 1) & (TNG['mass_halo'] < 1.e12)]
	sats1 = TNG100[(TNG100['Type'] == 1) & (TNG100['mass_halo'] < 1.e12)]

	# sats_outside = TNG100[(TNG100['Type'] == 1) & (TNG100['mass_halo'] < 1.e12) & (TNG100['rad']*1.e-3 > TNG100['Rvir']) ]
	# sats_inside = TNG100[(TNG100['Type'] == 1) & (TNG100['mass_halo'] < 1.e12) & (TNG100['rad']*1.e-3 < TNG100['Rvir']) ]




	sym_index = []
	asym_index = []
	ind = []
	# for ii in range(len(sats)):
	# 	if (sats['Afr'][ii] > func(sats['Npart'][ii],P80_fit[0],P80_fit[1]) + 0.5):
	# 		asym_index.extend([ii])
	# 	elif (sats['Afr'][ii] > func(sats['Npart'][ii],P80_fit[0],P80_fit[1]) + 0.2) & (sats['Afr'][ii] < func(sats['Npart'][ii],P80_fit[0],P80_fit[1]) + 0.5):
	# 		sym_index.extend([ii])
	# 	elif sats['Afr'][ii]< func(sats['Npart'][ii],P80_fit[0],P80_fit[1]) + 0.2:
	# 		ind.extend([ii])
	for ii in range(len(sats)):
		if (sats['Afr'][ii] > func(sats['Npart'][ii],P80_fit[0],P80_fit[1]) + 0.1):
			asym_index.extend([ii])
		elif (sats['Afr'][ii] < func(sats['Npart'][ii],P60_fit[0],P60_fit[1])) :
			sym_index.extend([ii])



	sym = sats[sym_index]
	asym = sats[asym_index]
	# ind = sats[ind]

	plt.scatter(sats['Npart'],sats['Afr'],color='Grey')
	plt.scatter(sym['Npart'],sym['Afr'],color='Green')
	plt.scatter(asym['Npart'],asym['Afr'],color='Orange')
	# plt.scatter(ind['Npart'],ind['Afr'],color='Black')
	plt.plot(np.linspace(500,20000,5000),func(np.linspace(500,20000,5000),P90_fit[0],P90_fit[1])+0.1,color='black')
	plt.plot(np.linspace(500,20000,5000),func(np.linspace(500,20000,5000),P50_fit[0],P50_fit[1]),color='black')
	plt.xscale('log')
	plt.show()
	# exit()

	fig,ax  = plt.subplots()
	plt.scatter(sats1['rad']*1.e-3/sats1['Rvir'],sats1['vrad']/sats1['V200'],color='Grey',s=5)

	img = ax.scatter(sats['rad']*1e-3/sats['Rvir'],sats['vrad']/sats['V200'],c=np.log10(sats['Afr']),s=10)

	ax.plot([0.1,7],[0,0],ls=':',color='Black')
	ax.plot([1,1],[-3,3],ls=':',color='Black')
	# ax.set_xscale('log')
	ax.set_ylabel('Vrad / V200')
	ax.set_xlabel('R/Rvir')
	ax.set_ylim([-3,3])
	ax.set_xlim([0.05,7])
	# ax.set_title('lgMh > 1.e13')
	ax.set_title('lgMh < 1.e12')
	plt.colorbar(img)


	import splotch
	# splotch.sigma_cont(sym['rad']*1.e-3/sym['Rvir'],sym['vrad']/sym['V200'], percent = [68], ax = ax,
			# bin_type='edges',bins=[np.logspace(np.log10(0.05),np.log10(7),30),np.linspace(-3,3,10)],
			 # c='Red', output=True, plot_kw = {'linewidths':2})
	splotch.sigma_cont(asym['rad']*1.e-3/asym['Rvir'],asym['vrad']/asym['V200'], percent = [68], ax = ax,
			bin_type='edges',bins=[np.logspace(np.log10(0.05),np.log10(7),30),np.linspace(-3,3,10)],
			 c='Blue', output=True, plot_kw = {'linewidths':2})
	# splotch.sigma_cont(ind['rad']*1.e-3/ind['Rvir'],ind['vrad']/ind['V200'], percent = [68], ax = ax,
	# 		bin_type='edges',bins=[np.logspace(np.log10(0.05),np.log10(7),30),np.linspace(-3,3,10)],
	# 		 c='Black', output=True, plot_kw = {'linewidths':2})
	fig.savefig('/media/data/simulations/IllustrisTNG/figures/stacked_phasespace_lgMh12.png')
	plt.show()
	exit()

def compare_xGASS_TNG100():

	xG_filename = '/home/awatts/Adam_PhD/models_fitting/asymmetries/data/xGASS_asymmetries_catalogue.ascii'
	xGASS = Table.read(xG_filename,format = 'ascii')
	min_SN = 7
	xGASS = xGASS[np.where((xGASS['SN_HI'] >= min_SN) & (xGASS['HIconf_flag']  < 1))[0]]

	xGASS['sSFR'] = np.log10(xGASS['SFR']) - xGASS['lgMstar']

	particle_mass = 1.4e6
	TNG100 = Table.read('/media/data/simulations/IllustrisTNG/TNG100_galdata_measured_v2.ascii',format='ascii')
	
	TNG100 = TNG100[np.where((TNG100['Sint']/particle_mass >= 5.e3))[0]]
	TNG100['mass_stars'] = np.log10(TNG100['mass_stars'])
	TNG100['mass_halo'] = np.log10(TNG100['mass_halo'])
	TNG100['sSFR'] = np.log10(TNG100['SFR']) - TNG100['mass_stars']



	lgMstar_list = [9,9.5,10,10.5,11,11.5]

	lgMstar_low = 9.3
	lgMstar_high = 9.8
	sSFR_low = -10.8
	sSFR_high = -8.5

	fig = plt.figure(figsize = (20,20))
	gs  = gridspec.GridSpec(len(lgMstar_list)-1, 3, left = 0.05, right = 0.99, top=0.97, bottom = 0.08)


	for ii in range(len(lgMstar_list)-1):
		lgMstar_low = lgMstar_list[ii]
		lgMstar_high = lgMstar_list[ii+1]


		TNG100_samp = TNG100[np.where((TNG100['mass_stars'] > lgMstar_low) & (TNG100['mass_stars'] < lgMstar_high) &
							(TNG100['sSFR'] > sSFR_low) & (TNG100['sSFR'] < sSFR_high) )[0]]

		xGASS_samp = xGASS[np.where((xGASS['lgMstar'] > lgMstar_low) & (xGASS['lgMstar'] < lgMstar_high) &
						(xGASS['sSFR'] > sSFR_low) & (xGASS['sSFR'] < sSFR_high) & (xGASS['lgMh'] != 0))[0]]

		print('----')
		print(len(TNG100_samp[TNG100_samp['Type'] == -1]))
		print(len(TNG100_samp[TNG100_samp['Type'] == 0]))
		print(len(TNG100_samp[TNG100_samp['Type'] == 1]))
		print('----')
		print(len(xGASS_samp[xGASS_samp['env_code'] == 1]))
		print(len(xGASS_samp[xGASS_samp['env_code'] == 2]))
		print(len(xGASS_samp[xGASS_samp['env_code'] == 0]))

		print('----')

		sfms_ax = fig.add_subplot(gs[ii,0])
		lgmh_ax = fig.add_subplot(gs[ii,1])
		afr_ax = fig.add_subplot(gs[ii,2])

		sfms_ax.scatter(TNG100['mass_stars'],TNG100['sSFR'], s=0.1)
		sfms_ax.scatter(xGASS['lgMstar'],xGASS['sSFR'], s=0.5)
		sfms_ax.plot([lgMstar_low,lgMstar_high,lgMstar_high,lgMstar_low,lgMstar_low],[sSFR_low,sSFR_low,sSFR_high,sSFR_high,sSFR_low],color='Red')
		sfms_ax.set_ylim([-15,-8])
		sfms_ax.set_ylabel('sSFR')
		sfms_ax.set_xlabel('lgMstar')


		lgmh_ax.hist(TNG100_samp['mass_halo'],bins = np.arange(10.5,15,0.25),density=True,alpha = 0.6,label='TNG100 ({num})'.format(num=len(TNG100_samp)))
		lgmh_ax.hist(xGASS_samp['lgMh'],bins = np.arange(10.5,15,0.25),density=True,alpha = 0.4,label='xGASS ({num})'.format(num=len(xGASS_samp)))
		lgmh_ax.set_xlabel('lgMh')
		lgmh_ax.set_ylabel('Histogram Density')
		lgmh_ax.legend()


		afr_ax.hist(TNG100_samp['Afr'],bins=np.arange(1,2.5,0.01),density=True,cumulative=True,histtype='step')
		afr_ax.hist(xGASS_samp['Afr_spec_HI'],bins=np.arange(1,2.5,0.01),density=True,cumulative=True,histtype='step')
		afr_ax.set_xlabel('Asymmetry measure Afr')
		afr_ax.set_ylabel('Histogram Density')

	fig.savefig('./figures/test_sSFR_lgMh_Afr.png')
	plt.show()


def compare_sat_cent():
	particle_mass = 1.4e6
	TNG100 = Table.read('/media/data/simulations/IllustrisTNG/TNG100_galdata_measured_v2.ascii',format='ascii')
	
	TNG100['Npart'] = TNG100['Sint'] / particle_mass

	good = np.where((TNG100['Sint']/particle_mass >= 1.e3))[0]

	TNG100['Rvir'] = np.cbrt(np.array(TNG100['mass_halo'])* 4.3e-3 / (100 * 70*70 * 1.e-6 * 1.e-6) )*1.e-6
	# exit()
	pos = ['pos_rel_x','pos_rel_y','pos_rel_z']
	vel = ['vel_rel_x','vel_rel_y','vel_rel_z']


	TNG100['rad'] = np.sqrt(np.nansum((np.array([TNG100[p] for p in pos]).T)**2.e0, axis=1))
	TNG100['vrel'] = np.sqrt(np.nansum((np.array([TNG100[v] for v in vel]).T)**2.e0, axis=1))

	plot1=False
	plot2=True

	if plot1:

		fig = plt.figure(figsize=(18,15))
		gs = gridspec.GridSpec(3,3,left=0.08,right=0.99,wspace=0.3,hspace=0.3,top=0.99) 
		
		for ii in range(3):
			if ii == 0:
				TNG_samp = TNG100[(TNG100['Npart'] > 5.e2) & (TNG100['mass_halo'] < 1.e12)  & (TNG100['mass_halo'] > 10.**(11.4))  ]
			if ii == 1:
				TNG_samp = TNG100[(TNG100['Npart'] > 5.e2) & (TNG100['mass_halo'] < 1.e13) & (TNG100['mass_halo'] > 1.e12)]
			if ii == 2:
				TNG_samp = TNG100[(TNG100['Npart'] > 5.e2) & (TNG100['mass_halo'] > 1.e13)]

			sats = TNG_samp[TNG_samp['Type'] == 1]
			cents = TNG_samp[TNG_samp['Type'] != 1]

			ax1 = fig.add_subplot(gs[ii,0])
			ax2 = fig.add_subplot(gs[ii,1])
			ax3 = fig.add_subplot(gs[ii,2])
			
			ax1.set_ylabel('Cumulative Histogram',fontsize = 15)
			ax2.set_ylabel('Histogram Density',fontsize = 15)
			ax3.set_ylabel('Histogram Density',fontsize = 15)

			ax1.set_xlabel('Asymmetry measure A$_{fr}$',fontsize=15)
			ax2.set_xlabel('lgMh',fontsize=15)
			ax3.set_xlabel('lgMstar',fontsize=15)


			sat_inside = sats[sats['rad']*1.e-3 < sats['Rvir']]
			sat_outside = sats[sats['rad']*1.e-3 > sats['Rvir']]

			samples = [cents['Afr'],sat_outside['Afr']]
			controls = [np.log10(cents['Npart']),np.log10(sat_outside['Npart'])]

			Afr_bins = np.arange(1,2.5,0.05)
			Npart_bins = np.arange(np.log10(500),np.log10(5000)+0.1,0.1)
			Npart_bins = np.append(Npart_bins,np.array([np.log10(1.e6)]))

			

			samples_all_hists, samples_all_DAGJKsigma, indexes_all_iter = \
										dsd.control_samples(samples,Afr_bins,controls,Npart_bins)

			dsd.plot_controlled_cumulative_histograms(samples_all_hists,Afr_bins,samples_all_DAGJKsigma,
						axis = ax1, names=['Centrals ({})'.format(len(cents)),'Satellites > R$_{{vir}}$ ({})'.format(len(sat_outside))])

			ax2.hist(np.log10(cents['mass_halo']),bins=10
									,histtype='step',density=True,lw = 2, color='Orange')
			ax2.hist(np.log10(sat_outside['mass_halo']),bins=10
									,histtype='step',density=True,lw = 2, color='Green')

			ax3.hist(np.log10(cents['mass_stars']),bins=np.arange(9,12.25,0.1)
									,histtype='step',density=True,lw = 2, color='Orange')
			ax3.hist(np.log10(sat_outside['mass_stars']),bins=np.arange(9,12.25,0.1)
									,histtype='step',density=True,lw = 2, color='Green')


		fig.savefig('/media/data/simulations/IllustrisTNG/figures/sats_cents_asym_manyhalo_.png')
		plt.show()
		exit()

	if plot2:

		sats = TNG100[(TNG100['Type'] == 1) & (TNG100['Npart']>5.e2) ]
		cents = TNG100[(TNG100['Type'] != 1) & (TNG100['Npart']>5.e2)]
		sat_outside = sats[sats['rad']*1.e-3 > sats['Rvir']]


		Afr_bins = np.arange(1,2.5,0.05)
		Npart_bins = np.arange(np.log10(500),np.log10(5000)+0.1,0.1)
		Npart_bins = np.append(Npart_bins,np.array([np.log10(1.e6)]))


		fig = plt.figure(figsize=(18,6))
		gs = gridspec.GridSpec(1,4,left=0.08,right=0.99,wspace=0.3,hspace=0.3,top=0.99) 

		ax1 = fig.add_subplot(gs[0,0])
		ax2 = fig.add_subplot(gs[0,1])
		ax3 = fig.add_subplot(gs[0,2])
		ax4 = fig.add_subplot(gs[0,3])

		samples = [cents['Afr'],sat_outside['Afr']]
		controls = [np.log10(cents['Npart']),np.log10(sat_outside['Npart'])]

		# samples_all_hists, samples_all_DAGJKsigma, indexes_all_iter = \
										# dsd.control_samples(samples,Afr_bins,controls,Npart_bins)

		# dsd.plot_controlled_cumulative_histograms(samples_all_hists,Afr_bins,samples_all_DAGJKsigma,
						# axis = ax1, names=['Centrals ({})'.format(len(cents)),'Satellites > R$_{{vir}}$ ({})'.format(len(sat_outside))])


		ax1.hist(cents['Afr'],bins=Afr_bins,lw=2,histtype='step',density=True,cumulative=True,color='Orange')
								
		ax1.hist(sat_outside['Afr'],bins=Afr_bins,lw=2,histtype='step',density=True,cumulative=True,color='Green')
								

		ax2.hist(np.log10(cents['mass_halo']),bins=10
									,histtype='step',density=True,lw = 2, color='Orange')

		ax2.hist(np.log10(sat_outside['mass_halo']),bins=10
									,histtype='step',density=True,lw = 2, color='Green')

		ax3.hist(np.log10(cents['mass_stars']),bins=np.arange(9,12.25,0.1)
									,histtype='step',density=True,lw = 2, color='Orange',label = 'Centrals ({})'.format(len(cents)))

		ax3.hist(np.log10(sat_outside['mass_stars']),bins=np.arange(9,12.25,0.1)
									,histtype='step',density=True,lw = 2, color='Green',label = 'Satellites > Rvir ({})'.format(len(sat_outside)))

		ax4.hist(np.log10(cents['Npart']),bins=np.arange(2.6,5,0.1)
									,histtype='step',density=True,lw = 2, color='Orange')

		ax4.hist(np.log10(sat_outside['Npart']),bins=np.arange(2.6,5,0.1)
									,histtype='step',density=True,lw = 2, color='Green')



		ax1.set_ylabel('Cumulative Histogram',fontsize = 15)
		ax2.set_ylabel('Histogram Density',fontsize = 15)
		ax3.set_ylabel('Histogram Density',fontsize = 15)
		ax4.set_ylabel('Histogram Density',fontsize = 15)

		ax1.set_xlabel('Asymmetry measure A$_{fr}$',fontsize=15)
		ax2.set_xlabel('lgMh',fontsize=15)
		ax3.set_xlabel('lgMstar',fontsize=15)
		ax4.set_xlabel('lgNpart',fontsize=15)

		ax3.legend()

		fig.savefig('/media/data/simulations/IllustrisTNG/figures/cents_sats_allhaloes.png')
		plt.show()
		exit()

###########

def SFMS(lgMstar, sigma=0):

	sSFR = -0.344*(lgMstar - 9) - 9.822
	
	sSFR += sigma * 0.088 * (lgMstar-9) + 0.188

	return sSFR

	
def measure_spectrum(spectrum, Vres):
	peaks = locate_peaks(spectrum)
	if peaks[0]  !=  False and peaks[1] != False:
		success = 1
		widths = locate_width(spectrum,peaks,0.2)
	else:
		success = 0
		peaks = [-1,-1]
		widths = [-1,-1]
	if widths[0]>0 and widths[1]>0 and widths[0]<len(spectrum) and widths[1]<len(spectrum):
		Sint, Afr = areal_asymmetry(spectrum, widths, Vres)
	else:
		widths = [-1,-1]
		success = 0
		Sint = -1
		Afr = -1
	return [success, peaks[0], peaks[1], widths[0], widths[1], Afr]

def measure_spectrum_v2(spectrum, Vres):
	peaks20 = locate_peaks_v2(spectrum,0.2)
	peaks50 = locate_peaks_v2(spectrum,0.5)
	if peaks20 != peaks50:
		success = 1
		peaks = [-1,-1]
		widths = [-1,-1]
		Sint = -1
		Afr = -1
	elif peaks20[0] == -1 or peaks20[1] == -1:
		success = 2
		peaks = [-1,-1]
		widths = [-1,-1]
		Sint = -1
		Afr = -1
	else:
		peaks = peaks50
		widths = locate_width(spectrum, peaks, 0.2)
		if all(w > 0 for w in widths) and all(w< len(spectrum) for w in widths):
			success = 0
			Sint, Afr = areal_asymmetry(spectrum, widths, Vres)
		else:
			success = 3
			widths = [-1,-1]
			Sint = -1
			Afr = -1
	
	return [success, peaks[0], peaks[1], widths[0], widths[1], Sint, Afr]

def locate_peaks(spectrum):

	if len(np.where(spectrum > 1.e-10)[0]) != 0:

		minchan = np.nanmax([np.nanmin(np.where(spectrum > 1.e-10)) - 5, 0])
		maxchan = np.nanmin([np.nanmax(np.where(spectrum > 1.e-10)) + 5, len(spectrum) - 2])

		left_found = False
		right_found = False

		PeaklocL = chanL = minchan
		PeaklocR = chanR = maxchan
		narrow_factor_L = narrow_factor_R = 0

		iteration_num = 0
		while((left_found == False or right_found == False) and iteration_num<100):
			iteration_num+=1
			#print(iteration_num)
			if left_found == False:
				while(chanL < maxchan and (spectrum[chanL] - spectrum[chanL - 1]) * (spectrum[chanL + 1] - spectrum[chanL]) > 0):
					chanL += 1
				if chanL == maxchan:
					PeaklocL = False
			
			if right_found == False:
				while(chanR > minchan and (spectrum[chanR] - spectrum[chanR - 1]) * (spectrum[chanR + 1] - spectrum[chanR]) > 0):
					chanR -= 1
				#print((spectrum[chanR] - spectrum[chanR - 1]) * (spectrum[chanR + 1] - spectrum[chanR]))
				if chanR == minchan:
					PeaklocR = False

			if PeaklocL != False and PeaklocR != False:
				#print('still good ')
				midpeak = int(0.5*(chanL + chanR))
				# #print('location of middle of peaks', midpeak)
				# #print((np.nanmax(spectrum[midpeak:chanR]), spectrum[chanR]),any(spectrum[midpeak:chanR] > spectrum[chanR]))

				if any(spectrum[chanL+1:midpeak+1 - narrow_factor_L] > spectrum[chanL]):
					left_found = False
					chanL += 1

				else:
					#print('Left peak found',chanL)
					left_found = True
					PeaklocL = chanL

				if any(spectrum[midpeak+narrow_factor_R:chanR] > spectrum[chanR]):
					right_found = False
					chanR -= 1
				else:
					#print('Right peak found',chanR)

					right_found = True
					PeaklocR = chanR

				if left_found == True and right_found == True:
					if PeaklocR < len(spectrum)/2 and PeaklocL < len(spectrum)/2:
						right_found = False
						chanR = maxchan
						narrow_factor_R += 10
					elif PeaklocR > len(spectrum)/2 and PeaklocL > len(spectrum)/2:
						left_found = False
						chanL = maxchan
						narrow_factor_L += 10

					if PeaklocR - PeaklocL > 200:
						if PeaklocR - len(spectrum)/2 > len(spectrum)/2 - PeaklocL:
							right_found = False
							chanR = PeaklocR - 2
						else:
							left_found = False
							charL = PeaklocL + 2
			else:
				left_found = right_found = True

			# plt.plot(spectrum)
			# plt.plot([chanL,chanL],[0,np.max(spectrum)], color='Red')
			# plt.plot([chanR,chanR],[0,np.max(spectrum)], color='Green')
			# plt.plot([midpeak,midpeak],[0,np.max(spectrum)], color='Black')
			# plt.plot((np.diff(spectrum[0:-1]) * np.diff(spectrum[1::]))/1.e5)
			# plt.show()
			# plt.close()

		if iteration_num == 100:
			# print('Error finding peaks')
			PeaklocL = PeaklocR = False
	else:
		PeaklocL = PeaklocR = False
	
	return [PeaklocL,PeaklocR]

def locate_peaks_v2(spectrum, level):

	specmax_loc = np.where(spectrum == np.nanmax(spectrum))[0][0]
	specmax = spectrum[specmax_loc]

	chanR = chanL = specmax_loc												#find where the spectrum equals 10% of the peak
	while(chanR < len(spectrum)-1 and spectrum[chanR] > level * specmax):
		chanR += 1
	
	while(chanL > 0 and spectrum[chanL] > level * specmax):
		chanL -= 1


	#now iterate back up to find the first turning point
	grad = 1
	while(chanR > 1 and grad > 0):
		chanR -= 1
		grad = (spectrum[chanR] - spectrum[chanR - 1]) * (spectrum[chanR + 1] - spectrum[chanR])
		
	if grad < 0 and chanR != 1:
		PeaklocR = chanR
	else:
		PeaklocR = -1

	grad = 1
	while(chanL < len(spectrum)-2 and grad > 0):
		chanL += 1
		grad = (spectrum[chanL] - spectrum[chanL - 1]) * (spectrum[chanL + 1] - spectrum[chanL])

	if grad < 0 and chanL != len(spectrum) - 1:
		PeaklocL = chanL
	else:
		PeaklocL = -1

	return [PeaklocL, PeaklocR]

def locate_peaks_manual(spectrum, chanL, chanR):

	chanR = int(chanR)
	#now iterate back up to find the first turning point
	grad = 1
	while(chanR > 1 and grad > 0):
		chanR -= 1
		grad = (spectrum[chanR] - spectrum[chanR - 1]) * (spectrum[chanR + 1] - spectrum[chanR])
		
	if grad < 0 and chanR != 1:
		PeaklocR = chanR
	else:
		PeaklocR = -1

	chanL = int(chanL)

	grad = 1
	while(chanL < len(spectrum)-2 and grad > 0):
		chanL += 1
		grad = (spectrum[chanL] - spectrum[chanL - 1]) * (spectrum[chanL + 1] - spectrum[chanL])

	if grad < 0 and chanL != len(spectrum) - 1:
		PeaklocL = chanL
	else:
		PeaklocL = -1

	return [PeaklocL, PeaklocR]


def locate_width(spectrum, peaklocs, level):
	"""
	Locate the N% level of the peak on the left and right side of a spectrum

	Parameters
	----------
	spectrum : array
		Input spectrum
	peaklocs : list
		location of each peak
	level : float
		N% of the peaks to measure

	Returns
	-------
	Wloc : list
		Location of N% of each peak in channels
	"""

	SpeakL = spectrum[peaklocs[0]]
	SpeakR = spectrum[peaklocs[1]]
	# print(SpeakL,SpeakR)
	wL = -1
	wR = -1
	chanR = peaklocs[1]
	while(chanR < len(spectrum) - 1 and spectrum[chanR] > level * SpeakR):
		chanR += 1
		wR = chanR - 1.e0 * ((level * SpeakR - spectrum[chanR]) / (
			spectrum[chanR] - spectrum[chanR - 1])) 

	chanL = peaklocs[0]
	# print(chanL)
	while(chanL > 0 and spectrum[chanL] > level * SpeakL):
		chanL -= 1

		wL = chanL +  ((level * SpeakL - spectrum[chanL]) / (
			spectrum[chanL + 1] - spectrum[chanL])) 
		# print(chanL,wL)

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
	# Sint_L =  S_L  * Vres
	# Sint_R =  S_R  * Vres

	Sint = Sint_L + Sint_R

	Afr =  Sint_L / Sint_R 
	if Afr < 1.e0:
		Afr = 1.e0 / Afr

	return Sint, Afr


def func(x,a,c):
	E = 1/(a*(x-c))**0.5 + 1.
	return E

if __name__ == '__main__':
	# add_extended_measurements()
	
	# measure_TNG_spectra()
	# plot_Afr_env()
	# fix_measurements(0,0)
	# remeasure_Afr()
	# resolution_completeness_sSFR()
	# resolution_completeness_MHI()
	# resolution_completeness_ratio()
	# compare_Afrhist_TNGboxes()

	# plot_selected_spectra()
	# check_highmass_SFgals()

	# find_bad_peaks()

	# Afr_halo_phasespace()

	# Afr_satellites_Rvir_sSFR()

	# plot_disturbed_satellites()

	# compare_Afrhist_Npart()

	# compare_asymmetry_halomass()
	# compare_asymmetry_halomass_envsplit()


	# gas_spatial_distribution()

	# compare_xGASS_TNG100()

	# compare_asym_gasfraction()

	Afrhist_satellites_centrals()

	# plot_TNG100_TN100_2()

	# compare_sat_cent()


