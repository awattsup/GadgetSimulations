import numpy as np 
import pafit as pf 
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit



def main():



	radius, costheta, R_opt = create_arrays(500,0)

	mom0 = create_mom0(radius, costheta, R_opt)
	mom1 = create_mom1(radius, costheta, -1, R_opt)


	plt.imshow(mom0)
	plt.show()
	plt.imshow(mom1)
	plt.show()

	sample_rad = 0.5 * R_opt


	mom0_samples = []
	mom1_samples = []

	for phi in range(360):
		x = sample_rad * np.cos(phi*np.pi/180.)
		y = sample_rad * np.sin(phi*np.pi/180.)

		x_sample = int(round(0.5*len(radius) + x))
		y_sample = int(round(0.5*len(radius) + y))

		mom0_sample = mom0[y_sample,x_sample]
		mom1_sample = mom1[y_sample,x_sample]

		# plt.scatter(phi, mom0_sample)
		mom0_samples.extend([mom0_sample])
		mom1_samples.extend([mom1_sample])


	mom0_fit, mom0_covar = curve_fit(harmonic_expansion, np.arange(360), mom0_samples)
	mom1_fit, mom1_covar = curve_fit(harmonic_expansion, np.arange(360), mom1_samples)

	print(mom0_fit)
	print(mom1_fit)

	plt.scatter(np.arange(360),mom0_samples)
	plt.plot(np.arange(360),harmonic_expansion(np.arange(360),mom0_fit[0],
		mom0_fit[1],mom0_fit[2],mom0_fit[3],mom0_fit[4],mom0_fit[5],mom0_fit[6]), color='Red')
	plt.show()

	plt.scatter(np.arange(360),mom1_samples)
	plt.plot(np.arange(360),harmonic_expansion(np.arange(360),mom1_fit[0],
		mom1_fit[1],mom1_fit[2],mom1_fit[3],mom1_fit[4],mom1_fit[5],mom1_fit[6]), color='Red')

	plt.show()






def create_arrays(dim, incl):
	"""
	Creates 2D arrays of radius and angle for the HI toy model

    Parameters
    ----------
    dim : int 	[pixels]
        Dimension N 
	params : list
		List of input parameters
			params[0] = Galaxy inclination 	[deg]
        	
    Returns
    -------
 	radius : N x N array 	[pixels]
 		2D array of galactocentric radii
 	costheta: N x N array
 		2D array of cos(theta) = [-pi, pi] values where theta is the angle counter clockwise from
 		the receding major axis (defined as the positive x-axis)
 	R_opt : float 	[pixels]
 		Value of the optical radius in pixels defined as N/4, making Rmax = 2 R_opt
    """

	radius = np.zeros([dim, dim])
	costheta = np.zeros([dim, dim])
	incl = 1.e0 / np.cos(incl * np.pi / 180.e0)								#inclination correction goes as 1/cos
	for yy in range(dim):
		for xx in range(dim):
			xcoord = (xx + 1.e0) - 0.5e0 * (dim + 1)
			ycoord = (yy + 1.e0) - 0.5e0 * (dim + 1)
			rad = np.sqrt( xcoord * xcoord + (ycoord * ycoord * incl * incl) )	#y coordinate is projected by inclination
			if rad <= 0.5e0 * (dim + 1.e0):
				radius[yy, xx] = rad
				if xcoord != 0:
					costheta[yy, xx] = (np.sign(xcoord) *
						np.cos(np.arctan((ycoord * incl) / xcoord)) )
				else:
					costheta[yy, xx] = (np.sign(xcoord) *
						np.cos(np.sign(ycoord) * np.pi * 0.5e0) )
			else:
				radius[yy, xx] = float('nan')							#no data outside galaxy radius
				costheta[yy, xx] = float('nan')							#further routines will conserve NaN
	R_opt = dim / 4.e0													#define image to cover 2 optical radii						
	return radius, costheta, R_opt

def create_mom0(radius, costheta, R_opt, R_scale = [1,1]):
	"""
	Generates a 2D HI mass map for symmetric or asymmetric distribution inputs

    Parameters
    ----------
    radius : N x N array 	[pixels]
        2D array of galactocentric radii
    costheta : N x N array
        2D array of cos(theta) values from the receding major axis
    params : list
    	List of model input parameters
    		params[0] = Galaxy inclination 	[deg]
	    	params[1] = Model type
	    	params[2] = Asymmetry flag
	    	params[3] = Total HI mass 		[Msun]
	    	params[4:7] = Receding side / symmetric input parameters
	    	params[8:11] = Approaching side parameters
	R_opt : float 	[pixels]
    	Optical radius 

    Returns
    -------
	mom0_map : N x N array 	[Msun/pc^2]
		2D array of projected HI surface densities
	rad1d : array 	[1 / R_opt]
		Radii bins for measured radial HI surface densities
	input_profile : 2 element list of arrays 	[Msun/pc^2]
		Radial projected HI surface density profiles of 
		receding and approaching side respectively
	"""
	dim  = len(radius)
	mom0_map = np.zeros([dim,dim])
	R_scale = [R*R_opt for R in R_scale]

	R_scale_map = R_scale[1] * (1.e0 + (((R_scale[0] - R_scale[1])/R_scale[1]) * 0.5e0* (costheta + 1.e0)))

	mom0_map = np.exp(-1.e0*radius/R_scale_map)
	

	# Rstep = (0.5e0 * dim) / 50.
	# rad1d = np.arange(0, int((dim) / 2) + 2.e0 * Rstep, Rstep)
	# input_receding = np.arange(len(rad1d) - 1)
	# input_approaching = np.arange(len(rad1d) - 1)

	# radius_temp = np.zeros([dim,dim])									#make approaching side have negative 
	# radius_temp[:, 0:int(dim / 2)] = -1.e0 * radius[:, 0:int(dim / 2)]	#radius to make summing easier
	# radius_temp[:, int(dim / 2)::] = radius[:, int(dim / 2)::]
	# for bb in range(len(rad1d) - 1):
	# 	bin_low = rad1d[bb]
	# 	bin_high  = rad1d[bb + 1]
	# 	inbin_app = mom0_map[(radius_temp <= -1.e0 * bin_low) & (radius_temp >
	# 				 -1.e0 * bin_high)]
	# 	inbin_rec = mom0_map[(radius_temp >= bin_low) & (radius_temp < bin_high)]

	# 	input_approaching[bb] = np.nansum(inbin_app) * incl / len(inbin_app)
	# 	input_receding[bb] = np.nansum(inbin_rec) * incl / len(inbin_rec)		#inclination corrected 
	# rad1d = rad1d[0:-1] / R_opt


	return mom0_map #, rad1d , input_profile

def create_mom1(radius, costheta, rad1d, R_opt, Vamp = [200,200], R_PE = [0.164,0.164], alpha = [0.002,0.002], Vdisp = 0):
	"""
	Generates a 2D gas velocity map for symmetric or asymmetric distribution inputs

    Parameters
    ----------
    radius : N x N array 	[pixels]
        2D array of galactocentric radii
    costheta : N x N array
        2D array of cos(theta) values from the receding major axis
    rad1d : array [1 / R_opt]
    	Radii bins for measuring input rotation curve
    params : list
    	List of model input parameters
    		params[0] = Galaxy inclination 	[deg]
	    	params[12] = Asymmetry flag
	    	params[13:15] = Receding side / symmetric input parameters
	    	params[16:18] = Approaching side parameters
	    	params[19] = Velocity dispersion 	[km/s]
	R_opt : float 	[pixels]
    	Optical radius 

    Returns
    -------
	mom1_map : N x N array 	[km/s]
		2D array of projected gas rotational velcoities
	input_RC : array 	[km/s]
		Projected input rotation curve
	"""

	incl = 90

	Vamp_map = Vamp[1] * (1.e0 + (((Vamp[0] - Vamp[1]) / Vamp[1]) 
			* 0.5e0 * (costheta + 1.e0)))
	R_PE_map = R_PE[1] * (1.e0 + (((R_PE[0] - R_PE[1]) / R_PE[1])
			* 0.5e0 * (costheta + 1.e0)))
	alpha_map = alpha[1] * (1.e0 + (((alpha[0] - alpha[1]) / alpha[1]) 
			* 0.5e0 * (costheta + 1.e0)))

	# RC_rec = polyex_RC(rad1d * R_opt, 1.e0, Vamp[0], R_PE[0], R_opt, alpha[0], incl)
	# RC_app = polyex_RC(rad1d * R_opt, 1.e0, Vamp[1], R_PE[1], R_opt, alpha[0], incl)
	# input_RC = [RC_rec, RC_app]

	mom1_map = polyex_RC(radius, costheta, Vamp_map, R_PE_map, R_opt, alpha_map, incl)

	if Vdisp >= 0:								#add velocity dispersion
		mom1_map = np.random.normal(mom1_map, Vdisp)

	return mom1_map#, input_RC

def polyex_RC(radius, costheta, V0, scalePE, R_opt, aa, incl):
	"""
	Creates a 2D projected velocity map using the Polyex rotation curve (RC) defined 
	by Giovanelli & Haynes 2002, and used by Catinella, Giovanelli & Haynes 2006

    Parameters
    ----------
    radius : N x N array 	[pixels]
        2D array of galactocentric radii
    costheta : N x N array
        2D array of cos(theta) values from the receding major axis
    V_0 : float 	[km/s]
    	Amplitude of RC
    scalePE : float 	[1 / R_opt]
    	Scale length of exponential inner RC
    R_opt : float 	[pixels]
    	Optical radius
    aa : float
    	Slope of outer, linear part of RC
    incl : float 	[deg]
    	Galaxy inclination

    Returns
    -------
	mom1 : N x N array 	[km/s]
		2D array of inclination corrected rotational velocity of each pixel
	"""

	incl = np.sin(incl * (np.pi / 180.e0))
	R_PE = 1.e0 / (scalePE * R_opt)											#rotation curve scale length Catinella+06
	mom1 = ( (V0 * (1.e0 - np.exp((-1.e0 * radius) * R_PE)) * 
		(1.e0 + aa * radius * R_PE)) * costheta * incl )
	return mom1


def harmonic_expansion(phi, A0, A1, B1, A2, B2, A3, B3):

	H = A0 + A1 * np.cos(phi*np.pi/180.) + B1 * np.sin(phi*np.pi/180.) + \
			A2 * np.cos(2.e0*phi*np.pi/180.) + B2 * np.sin(2.e0*phi*np.pi/180.) + \
			A3 * np.cos(3.e0*phi*np.pi/180.) + B3 * np.sin(3.e0*phi*np.pi/180.)

	return H



if __name__ == '__main__':

	main()


