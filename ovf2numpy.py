import os
import os.path as fs
import logging
import numpy as np
import pandas as pd
import re


class OVFReader:

	def __init__(self, ovf_dir):
		"""
        Initialize an instance of the OVFReader class.

        :param ovf_dir: the path to the directory containing the OVF files.
		"""
		self.data_dims = ()
		self.mag_data = []

		ls = sorted(os.listdir(ovf_dir))	#list and sort all files in given path
		self.files_to_import = [fs.join(ovf_dir, listing) for listing in ls if (re.match('m\d*\.ovf', listing) is not None)]	#init list of filenames in this dir
		self.read_data_dims()

	def import_dir(self):	
		"""
        Imports OVF magnetization files from the data directory and returns them as a numpy array stacked in the first dimension.
		If the data contains N files of 32*32*1 3-vectors, the resulting array will have shape (N, 32, 32, 1, 3)
		"""

		x_nodes, y_nodes, z_nodes = self.data_dims

		magnetization_over_time = [self.ovf_to_pandas(file, x_nodes, y_nodes, z_nodes) for file in self.files_to_import]

		self.mag_data = np.stack(magnetization_over_time, axis=0)

	def read_data_dims(self):	
		"""
        Reads the data dimensions (x_nodes, y_nodes, z_nodes) from the first OVF file in the OVF dir.

        :returns: A tuple of integers representing the data dimensions (x_nodes, y_nodes, z_nodes)
		"""
		with open(self.files_to_import[0]) as f:
			data = f.readlines() 		

		x_nodes = int(data[20][10:])
		y_nodes = int(data[21][10:])
		z_nodes = int(data[22][10:])

		return(x_nodes, y_nodes, z_nodes)

	def ovf_to_numpy(this_filename, x_nodes, y_nodes, z_nodes):
		"""
        Converts an OVF file to a numpy array via pandas.

        :param this_filename: the path to the OVF file to be converted.
        :param x_nodes: the number of cells in the x-direction.
        :param y_nodes: the number of cells in the y-direction.
        :param z_nodes: the number of cells in the z-direction.

        :returns: A numpy array of shape (x_nodes, y_nodes, z_nodes, 3).
        """
		raw_data=pd.read_csv(this_filename, header=None, skiprows=28, skipfooter=2, delimiter="\s+")
		print(f"{this_filename} has length {len(raw_data)} size {raw_data.size} shape {raw_data.shape} and expected dims ({x_nodes}, {y_nodes}, {z_nodes})")
		magnetisation_array=np.reshape(raw_data.values, [x_nodes, y_nodes, z_nodes, 3])
		
		return magnetisation_array
	
if __name__=="__main__":
	reader = OVFReader('/home/michael/Desktop/mumate/ovffiles')
	reader.import_dir() 
