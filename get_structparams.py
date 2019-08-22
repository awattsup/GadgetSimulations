import os
import glob
import sys
import numpy as np
import fileinput




path = sys.argv[1]

Nbody_output = '{}/Nbody.out'.format(path)
gas_params = '{}/gas.param'.format(path)

Nbdyfile = open(Nbody_output,'r')
for line in Nbdyfile:
	if 'M200' in line:
		m200 = np.float(line.split('=')[-1].split('(')[0])
	if 'A (halo)' in line:
		Ahalo = np.float(line.split('=')[-1])
	if 'H  (disk)' in line:
		Hdisk = np.float(line.split('=')[-1])
		break
Nbdyfile.close()

fdisk = Hdisk/Ahalo
# print(fdisk)

gasfile = open(gas_params,'r+')
lines = gasfile.readlines()
gasfile.close()
gasfile = open(gas_params,'w')
for line in lines:
	if '?m200' in line:
		line = line.replace('?m200','{m200:.3f}'.format(m200=m200))
	if '?fdisk' in line:
		line = line.replace('?fdisk','{fdisk:.4f}'.format(fdisk=fdisk))
	gasfile.write(line)
gasfile.close()








