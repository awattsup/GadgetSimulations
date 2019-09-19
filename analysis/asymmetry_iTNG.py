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


def fix_measurements():
	data = Table.read('/media/data/simulations/IllustrisTNG/TNG_galdata_measured_v2_new.ascii',format='ascii')
	spectra = np.loadtxt('/media/data/simulations/IllustrisTNG/TNG_spectra_true.dat')
	vel_bins = spectra[:,0]
	spectra = spectra[:,1::]
	Vres = np.abs(np.diff(vel_bins))[0]
	good = np.where(data['fit_success'] == 0)[0]
	bad = np.where(data['fit_success'] == -1)[0]

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


	# original_check=False
	# if original_check:
	# 	peaks_diff = np.where(data['fit_success'] == 1)[0]
	# 	peaks_bad = np.where(data['fit_success'] == 2)[0]
	# 	width_bad = np.where(data['fit_success'] == 3)[0]

	# 	print('Number of good',len(good))
	# 	print('Number of peaks disagree',len(peaks_diff))
	# 	print('Number of peaks are bad',len(peaks_bad))
	# 	print('Number of widths are bad',len(width_bad))
	
	IDs = [5704,8593,13180]

	for ii in IDs:
		print(ii)
		spectrum = spectra[:,ii]
		level = 0.2
		answer = -1
		while(answer != '' and answer != 'b'):
			peaks = locate_peaks_v2(spectrum,level)
			plt.plot(spectrum)
			plt.plot([peaks[0],peaks[0]],[0,np.nanmax(spectrum)])
			plt.plot([peaks[1],peaks[1]],[0,np.nanmax(spectrum)])
			plt.ion()
			plt.show()
			print('[enter] for good, [b] for bad or [p] to retry peak finding')
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
			elif answer == 'p':
				print('input fraction of peak to try')
				level = float(input())
			plt.close()
		data.write('/media/data/simulations/IllustrisTNG/TNG_galdata_measured_v2_new.ascii',format='ascii',overwrite=True)

def plot_Afr_env():

	data = Table.read('/media/data/simulations/IllustrisTNG/TNG_galdata_measured_v2_new.ascii',format='ascii')
	print(len(np.where(data['fit_success'] == 0)[0]))
	good = np.where((data['Afr'] > 1) & (data['Afr'] < 3) & (data['fit_success'] == 0))[0]
	data = data[good]
	print(len(data))
	isocent = np.where(data['Type'] == -1)[0]
	grpcent = np.where(data['Type'] == 0)[0]
	sat = np.where(data['Type'] == 1)[0]

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

	spectra = np.loadtxt('/media/data/simulations/IllustrisTNG/TNG_spectra_true.dat')
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
		data = Table.read('/media/data/simulations/IllustrisTNG/TNG_galdata.ascii',format='ascii')
		data['fit_success'] = measurements_all[:,1]
		data['PeaklocL'] = measurements_all[:,2]
		data['PeaklocR'] = measurements_all[:,3]
		data['w20L'] = measurements_all[:,4]
		data['w20r'] = measurements_all[:,5]
		data['Sint'] = measurements[:,6]
		data['Afr'] = measurements_all[:,7]

		data.write('/media/data/simulations/IllustrisTNG/TNG_galdata_measured_v2.ascii',format='ascii')

	
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
		if all(w > 0 for w in widths):
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
	fix_measurements()
	# remeasure_Afr()







