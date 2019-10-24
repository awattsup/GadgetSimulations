import numpy as np 
import analysis_functions as af 
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec
# import galread_ARHS as gr 
from astropy.table import Table
from mpi4py import MPI


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

def resolution_completeness_sSFR():

	particle_mass = 1.4e6

	data = Table.read('/media/data/simulations/IllustrisTNG/TNG100_galdata_measured_v2.ascii',format='ascii')
	data = data[data['fit_success'] == 0]

	data['mass_stars'] = np.log10(data['mass_stars'])
	data['SFR'] = np.log10(data['SFR']) - data['mass_stars']

	plt.scatter(data['mass_stars'],data['SFR'],s=0.1)
	plt.show()
	# exit()


	resolved_asym = np.where((data['Sint']/particle_mass >= 1.e3))[0]
	unresolved_asym = np.where((data['Sint']/particle_mass < 1.e3))[0]

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




	plt.pcolormesh(mstar_bins,SFR_bins,compl_grid)
	plt.colorbar()

	plt.contour(mstar_bins[0:-1],SFR_bins[0:-1],compl_grid,levels=[0.5],**{'linewidths':3,'colors':'Black'})

	plt.plot([9,12.5],[-0.344*(9-9) - 9.822, -0.344*(12.5-9) - 9.822],ls = '--',color='Red',linewidth=2)
	plt.plot([9,12.5],[-0.344*(9-9) - 9.822 - (0.088*(9-9) + 0.188), -0.344*(12.5-9) - 9.822 - (0.088*(12.5-9) + 0.188)],ls = ':',color='Red',linewidth=2)
	plt.plot([9,12.5],[-0.344*(9-9) - 9.822 + (0.088*(9-9) + 0.188), -0.344*(12.5-9) - 9.822 + (0.088*(12-9) + 0.188)],ls = ':',color='Red',linewidth=2)

	plt.xlabel('log10 Stellar mass')
	plt.ylabel('log10 SFR')
	plt.title('TNG100 fraction of HI asymmetry resolved galaxies (contour = 0.6)')
	plt.show()

	# exit()

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

	data = Table.read('/media/data/simulations/IllustrisTNG/TNG100-3_galdata_measured_v2.ascii',format='ascii')
	data['mass_stars'] = np.log10(data['mass_stars'])
	data['SFR'] = np.log10(data['SFR']) - data['mass_stars']
	resolved_asym = np.where((data['Sint']/particle_mass >= 1.e3))[0]
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
	resolved_asym = np.where((data['Sint']/particle_mass >= 1.e3))[0]
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


	plt.pcolormesh(mstar_bins,SFR_bins,compl_ratio)
	plt.colorbar()

	plt.contour(mstar_bins[0:-1],SFR_bins[0:-1],compl_ratio,levels=[0.8],**{'linewidths':3,'colors':'Black'})
	plt.xlabel('log10 Stellar mass')
	plt.ylabel('log10 SFR')
	plt.title('TNG100-3 / TNG100-2 completness ratio')
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


	TNG100 = TNG100[np.where((TNG100['Sint']/particle_mass >= 1.e3))[0]]
	TNG100_2 = TNG100_2[np.where((TNG100_2['Sint']/particle_mass >= 1.e3))[0]]
	TNG100_3 = TNG100_3[np.where((TNG100_3['Sint']/particle_mass >= 1.e3))[0]]


	# plt.hist(np.log10(TNG100['mass_stars']),bins=15,alpha=0.6,density=True)
	# plt.hist(np.log10(TNG100_2['mass_stars']),bins=15,alpha=0.4,density=True)
	# plt.xlabel('lgMstar')
	# plt.ylabel('Histogram Density')
	# plt.show()


	TNG100TNG1002 = False
	if TNG100TNG1002:
		
		plt.hist(np.log10(TNG100['mass_stars']),bins=15,alpha=0.6,density=True)
		plt.hist(np.log10(TNG100_2['mass_stars']),bins=15,alpha=0.4,density=True)
		plt.xlabel('lgMstar')
		plt.ylabel('Histogram Density')
		plt.show()

		TNG100 = TNG100[np.where((np.log10(TNG100['mass_stars'])<9.8) & (np.log10(TNG100['mass_stars'])> 9.3) 
					& (np.log10(TNG100['SFR'] / TNG100['mass_stars']) > -10.1 ) & (np.log10(TNG100['SFR'] / TNG100['mass_stars']) < -9.6 ))[0]]
		TNG100_2 = TNG100_2[np.where((np.log10(TNG100_2['mass_stars'])<9.8) & (np.log10(TNG100_2['mass_stars'])> 9.3) 
					& (np.log10(TNG100_2['SFR'] / TNG100_2['mass_stars']) > -10.1) & (np.log10(TNG100_2['SFR'] / TNG100_2['mass_stars']) < -9.6)) [0]]

		TNG100_cent = np.where(TNG100['Type'] <= 0)[0]
		TNG100_sat = np.where(TNG100['Type'] == 1)[0]

		TNG100_2_cent = np.where(TNG100_2['Type'] <= 0)[0]
		TNG100_2_sat = np.where(TNG100_2['Type'] == 1)[0]




		fig = plt.figure(figsize = (10,8))
		gs = gridspec.GridSpec(2, 1, top = 0.9, right = 0.98, bottom  = 0.08, left = 0.08)
		axes_1 = fig.add_subplot(gs[0,0])
		axes_1.hist(TNG100['Afr'][TNG100_cent],bins=np.arange(1,2.5,0.01),density=True,cumulative=True,color='Green',histtype='step',label='Centrals ({})'.format(len(TNG100_cent)))
		axes_1.hist(TNG100['Afr'][TNG100_sat],bins=np.arange(1,2.5,0.01),density=True,cumulative=True,color='Orange',histtype='step',label='Satellites ({})'.format(len(TNG100_sat)))
		# axes_1.set_xlabel('Asymmetry measure A$_{fr}$')
		axes_1.set_ylabel('Cumulative Histogram Density')
		axes_1.legend()

		axes_2 = fig.add_subplot(gs[1,0])
		axes_2.hist(TNG100_2['Afr'][TNG100_2_cent],bins=np.arange(1,2.5,0.01),density=True,cumulative=True,color='Green',histtype='step',label='Centrals ({})'.format(len(TNG100_2_cent)))
		axes_2.hist(TNG100_2['Afr'][TNG100_2_sat],bins=np.arange(1,2.5,0.01),density=True,cumulative=True,color='Orange',histtype='step',label='Satellites ({})'.format(len(TNG100_2_sat)))
		axes_2.set_xlabel('Asymmetry measure A$_{fr}$')
		axes_2.set_ylabel('Cumulative Histogram Density')
		axes_2.legend()

		fig.suptitle('TNG100 & TNG100-2  lgMstar = [9.3,9.8] sSFR = [-10.1,-9.6]',fontsize = 15)


		plt.show()
		exit()


	TNG1002TNG1003 = True
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


	TNG100TNG1003 = True
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

def compare_asymmetry_halomass():


	particle_mass = 1.4e6

	TNG100 = Table.read('/media/data/simulations/IllustrisTNG/TNG100_galdata_measured_v2.ascii',format='ascii')

	TNG100 = TNG100[np.where((TNG100['Sint']/particle_mass >= 1.e3))[0]]

	TNG100['mass_stars'] = np.log10(TNG100['mass_stars'])
	TNG100['mass_halo'] = np.log10(TNG100['mass_halo'])
	TNG100['SFR'] = np.log10(TNG100['SFR']) - TNG100['mass_stars']


	plt.hist(TNG100['mass_halo'][(TNG100['mass_stars'] > 10.5) & (TNG100['mass_stars']< 11.5)],bins=20)
	plt.xlabel('log10 Halo mass')
	plt.ylabel('Histogram Count')
	plt.show()
	# exit()

	massrange_1 = np.where((TNG100['mass_stars'] > 10.5) & (TNG100['mass_stars'] <= 11))[0]
	massrange_2 = np.where((TNG100['mass_stars'] > 11) & (TNG100['mass_stars'] <= 11.5))[0]

	
	print('Ncen', len(TNG100[(TNG100['mass_stars'] > 10.) & (TNG100['mass_stars'] < 11.5) & (TNG100['Type'] < 1)] ))
	print('Nsat', len(TNG100[(TNG100['mass_stars'] > 10.) & (TNG100['mass_stars'] < 11.5) & (TNG100['Type']  == 1)]))

	fig = plt.figure(figsize = (10,8))

	gs = gridspec.GridSpec(2, 1, top = 0.95, right = 0.98, bottom  = 0.12, left = 0.08)

	ax1 = fig.add_subplot(gs[0,0])
	ax2 = fig.add_subplot(gs[1,0])

	ax1.tick_params(axis = 'x', which = 'both', direction = 'in', labelsize = 0)


	massranges = [massrange_1,massrange_2]
	axes = [ax1,ax2]

	ax1.set_title('lgMstar = (10.5,11]',fontsize=10)
	ax2.set_title('lgMstar = (11,11.5]',fontsize=10)
	ax2.set_xlabel('Asymmetry measure A$_{fr}$')

	for ii in range(len(massranges)):

		TNG_samp = TNG100[massranges[ii]]

		lowmass = np.where((TNG_samp['mass_halo'] > 11) & (TNG_samp['mass_halo'] <= 12))[0]
		medmass = np.where((TNG_samp['mass_halo'] > 12) & (TNG_samp['mass_halo'] <= 12.5))[0]
		highmass = np.where((TNG_samp['mass_halo'] > 12.5) & (TNG_samp['mass_halo'] <= 13.5))[0]
		bigmass = np.where((TNG_samp['mass_halo'] > 13.5))

		axes[ii].set_ylabel('Histogram Density')

		axes[ii].hist(TNG_samp['Afr'][lowmass],bins=np.arange(1,2.5,0.01),density=True,
			cumulative=True,color='Purple',histtype='step',label = 'M_$h$ = (11,12] ({})'.format(len(TNG_samp['Afr'][lowmass])))
		

		axes[ii].hist(TNG_samp['Afr'][medmass],bins=np.arange(1,2.5,0.01),density=True,
			cumulative=True,color='Orange',histtype='step',label = 'M_$h$ = (12,12.5] ({})'.format(len(TNG_samp['Afr'][medmass])))
		
		axes[ii].hist(TNG_samp['Afr'][highmass],bins=np.arange(1,2.5,0.01),density=True,
			cumulative=True,color='Green',histtype='step',label = 'M_$h$ = (12.5,13.5] ({})'.format(len(TNG_samp['Afr'][highmass])))
		
		# axes[ii].hist(TNG_samp['Afr'][highmass][TNG_samp[highmass]['Type']<1],bins=np.arange(1,2.5,0.01),density=True,
			# cumulative=True,color='Green',histtype='step',label = 'M_$h$ = (12,13] ({})'.format(len(TNG_samp['Afr'][highmass][TNG_samp[highmass]['Type']<1])))
		# axes[ii].hist(TNG_samp['Afr'][highmass][TNG_samp[highmass]['Type']==1],bins=np.arange(1,2.5,0.01),ls = '--',density=True,
		# 	cumulative=True,color='Green',histtype='step',label = 'M_$h$ = (12,13] ({})'.format(len(TNG_samp['Afr'][highmass][TNG_samp[highmass]['Type']==1])))
		

		axes[ii].hist(TNG_samp['Afr'][bigmass],bins=np.arange(1,2.5,0.01),density=True,
			cumulative=True,color='Red',histtype='step',label = 'M_$h$ > 13.5 ({})'.format(len(TNG_samp['Afr'][bigmass])))
		axes[ii].legend()
	plt.show()
	exit()


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
	massrange = np.where((TNG100_good['mass_stars'] > 10.5) & (TNG100_good['mass_stars'] <= 11) & (TNG100_good['mass_halo'] < 13.5))[0]

	TNG_samp = TNG100_good[massrange]
	IDs = IDs[massrange]

	highAfr = np.where(TNG_samp['Afr']>1.7)[0]
	IDs = IDs[highAfr]
	print(IDs)
	# exit()

	spectra = np.loadtxt('/media/data/simulations/IllustrisTNG/TNG100_spectra_true.dat')
	vel = spectra[:,0]
	spectra = spectra[:,1::]
	for ID in IDs:
		spectrum = spectra[:,ID]
		print(ID,TNG100['Afr'][ID])
		fig, ax = plt.subplots()
		ax.plot(range(len(vel)),spectrum,color='Black')
		ax.plot([TNG100['PeaklocL'][ID],TNG100['PeaklocL'][ID]],[0,np.max(spectrum)],color = 'red')
		ax.plot([TNG100['PeaklocR'][ID],TNG100['PeaklocR'][ID]],[0,np.max(spectrum)],color = 'red')
		ax.plot([TNG100['w20L'][ID],TNG100['w20L'][ID]],[0,np.max(spectrum)],color='Blue',ls='--')
		ax.plot([TNG100['w20R'][ID],TNG100['w20R'][ID]],[0,np.max(spectrum)],color='Blue',ls='--')
		ax.set_xlabel('Channels')
		ax.set_ylabel('HI mass')
		ax.text(0.1,0.9,'Afr = {afr:.3f}'.format(afr=TNG100['Afr'][ID]),fontsize=12, transform=ax.transAxes)
		plt.show()

	exit()




	
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

if __name__ == '__main__':
	# measure_TNG_spectra()
	# plot_Afr_env()
	# fix_measurements(0,0)
	# remeasure_Afr()
	# resolution_completeness_sSFR()
	# resolution_completeness_MHI()
	# resolution_completeness_ratio()
	# compare_Afrhist_TNGboxes()
	# compare_asymmetry_halomass()

	# plot_selected_spectra()
	# check_highmass_SFgals()

	find_bad_peaks()





