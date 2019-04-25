from __future__ import division, print_function
import os
import os.path as fs
import numpy as np
import pandas as pd
import re

###	PURPOSE:	 Takes a directory containing N files of the form mXXXXXX.ovf	###
###			and imports them to an N x X x Y x Z x 3 numpy array		###
###			where X,Y,Z are the number of cells in x,y,z			###
###	Files will have the naming convention m*.ovf where * is 6 digit decimal number	###
###		eg. 000000, 000001, 000123, etc						###
###	So use regex to find something of the form m/<number>*/.ovf			###

def import_dir(path='.', which_files='all', skyrmion=False, core_slice='h', average=True):	
	#default path is this folder	
	#which files gives a range of files (default to all in dir)
	ls = sorted(os.listdir(path))	#list and sort all files in given path
	magnetisation_files=[]	#init list of filenames in this dir
	for el in ls:		#test the regex for magnetisation file format, if found add to filename list
		if re.match('m\d*\.ovf' ,el) is not None:
			magnetisation_files.append(el)
	
	file_name=fs.join(path, str(magnetisation_files[0]))	#creates the filename for the first mag field 
	data_dimensions=getOvfAttributes(file_name)	#gets the file attributes x,y,z nodes (eg 2x2x128)


	num_files_to_import=len(magnetisation_files)

	if which_files!='all':
		print("not importing all files, importing files ",which_files[0], " to ", which_files[1])
		num_files_to_import=which_files[1]-which_files[0]

	all_mag_data=np.empty((num_files_to_import, data_dimensions[2]), dtype=(float, 3) )
	i=0
	first_time=True
	percentages=[]
	for n, fname in enumerate(magnetisation_files):
		if which_files!='all':
			if n<which_files[0]:
				continue
			if n>=which_files[1]:
				break
		if first_time:
			print("starting to read ",num_files_to_import," files")
			first_time=False

		this_filename=fs.join(path, fname)
		all_mag_data[i]=importOvfFilePandas(this_filename, data_dimensions, core_slice=core_slice, skyrmion=skyrmion, average_to_1D=average)

		if i/num_files_to_import*100%10<0.2:			
			if np.floor(i*100/num_files_to_import) not in percentages:
				print(np.floor(i*100.0/num_files_to_import),"% done")	
				percentages.append(np.floor(i*100/num_files_to_import))
		i+=1
		
	#print data_array.shape
	print("100% done!")
	print("read ",i," files")
	return all_mag_data	

def getOvfAttributes(filename):	
	if filename[-4:]!='.ovf':	#if filetype is not ovf, exit with error code 1
		print("FATAL ERROR, NOT AN OVF FILE")
		return -1

	f=open(filename, 'r')
	j=0

	for line in f:
		if re.match('.*Binary.*', line) is not None:	#if the data type is a binary, just exit with error code -2
			print("FATAL ERROR: BINARY NOT SUPPORTED")
			return -2	

		if j==20:
			x_nodes=int(line[10:])
		if j==21:
			y_nodes=int(line[10:])
		if j==22:
			z_nodes=int(line[10:])
			break
		#print (str(j)+'\t'+str(line))
		j+=1
	f.close()

	return(x_nodes, y_nodes, z_nodes)


# takes filename, imports ovf as pandas dataframe, takes data dimensions in (x,y,z) nodes format

def importOvfFilePandas(this_filename, data_dims, average_to_1D=False, skyrmion=False, core_slice='h'):
	ave_axis=None
	raw_data=pd.read_csv(this_filename, header=None, skiprows=28, skipfooter=2, delimiter="\s+")
	magnetisation_array=np.reshape(raw_data.as_matrix(), np.append(data_dims[::-1],3))
	if skyrmion:
		m1=int(data_dims[1]/2-1)
		m2=int(data_dims[1]/2+1)

		if core_slice=='h':
			magnetisation_array=magnetisation_array[:,m1:m2,:]
			ave_axis=1	

		elif core_slice=='v':
	
			magnetisation_array=magnetisation_array[:,:,m1:m2]
			ave_axis=2

		if average_to_1D:
			magnetisation_array=np.mean(magnetisation_array, axis=ave_axis)
			magnetisation_array=np.mean(magnetisation_array, axis=0)

	elif average_to_1D:
		for i in [1,2]:
			magnetisation_array=np.mean(magnetisation_array, axis=1)		
			
	#print(magnetisation_array.shape)
	return magnetisation_array
	
if __name__=="__main__":
	#test=importOvfFilePandas('/home/michael/Desktop/Honours/MuMax3/DataProcessing/SkyrmionData/ovfimporttest/m000035.ovf', (128,128,1), skyrmion=True, h_core_slice=True, average_to_1D=True)
	test=import_dir('/home/michael/Desktop/Honours/MuMax3/DataProcessing/HelicoidData/helicoidv8_mid.out/') 
	#test=importOvfFilePandas('/home/michael/Desktop/Honours/MuMax3/DataProcessing/SkyrmionData/ovfimporttest/m000035.ovf', (128,128,1), skyrmion=True, v_core_slice=True, average_to_1D=True)
	#test=importOvfFilePandas('/home/michael/Desktop/Honours/MuMax3/DataProcessing/HelicoidData/helicoidv6.out/m000035.ovf', (2,2,128), average_to_1D=True)
