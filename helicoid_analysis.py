from ovf2numpy import OVFReader
import os
import os.path as fs
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import scipy.fftpack as spft
import sys


def coords_rotcoords(ovf_files, Bmin, Bmax, Bstep, tmin, tmax, tstep):
	
	global save_directory	
	data_dim=ovf_files.shape[1]
	B_field_values=np.linspace(Bmin, Bmax, num=round((Bmax-Bmin)/Bstep)+1)
	i=0
	placeholder_array=np.array([]).reshape(0, data_dim, 3)

	all_displacements = placeholder_array.copy()
	all_displacements_rot = placeholder_array.copy()
	all_eq = placeholder_array.copy()

	for B_field in B_field_values:	###	Start of Bfield loop	###
		
		these_m_values=ovf_files[i*tstep:(i+1)*tstep]	#read values for this field
		i+=1
		displacements=np.ndarray([tstep-1, data_dim, 3])	#array to hold displacements from equilibrium
		
		### ALL ARE UNIT LENGTH WITHIN TOLERANCE
		
		equilibrium=these_m_values[0]

		for k, el in enumerate(these_m_values):
			if k==0:
				continue
			displacements[k-1]=el-equilibrium

		rho_hat= equilibrium[:,[0,1,2]]
		phi_hat= equilibrium[:,[1,0,2]]
		z_hat= equilibrium[:,[1,0,2]]
					
		rho_hat[:,2]=0
		phi_hat[:,2]=0; phi_hat[:,0]*=-1
		z_hat[:,:2]=0; z_hat[:,2]=1
		
		# ^^ CYLINDRICAL UNIT VECTORS ARE NORMALISED! checked ^^ #

		displacements_rho_comp=np.zeros((tstep-1,data_dim))
		displacements_phi_comp=np.zeros((tstep-1,data_dim))
		displacements_z_comp=np.zeros((tstep-1,data_dim))
	
		#print(displacements_rho_comp.shape)
		#print("from coords rot coords")	
		for idx, el in enumerate(displacements):

			for idx2, el2 in enumerate(el):

				displacements_rho_comp[idx][idx2]=np.inner(el2, rho_hat[idx2])
				displacements_phi_comp[idx][idx2]=np.inner(el2, phi_hat[idx2])
				displacements_z_comp[idx][idx2]=np.inner(el2, z_hat[idx2])
		
		# ^^ NO/NEGLIGIBLE CHANGE IN RADIUS! checked ^^ #

		displacements_rot=np.dstack((displacements_rho_comp, displacements_phi_comp, displacements_z_comp))
		all_displacements = np.concatenate((all_displacements, displacements), axis=0)
		all_displacements_rot = np.concatenate((all_displacements_rot, displacements_rot), axis=0) 
		all_eq = np.concatenate((all_eq, equilibrium[np.newaxis,:,:]), axis=0)
	
	return(all_displacements, all_displacements_rot, all_eq)

def fourier_displacements(ovf_files, Bmin, Bmax, Bstep, tmin, tmax, tstep):
	
	global conjs
	global save_directory_resonances
	global save_plot
	#define mag field and freq arrays
	
	B_field_values=np.linspace(Bmin, Bmax, num=round((Bmax-Bmin)/Bstep)+1)
	time_series=np.linspace(tmin, tmax, num=tstep-1)
	fft_freq=spft.fftfreq(len(time_series), d=time_series[1]-time_series[0])*1000
	triggered=False

	for idx, freq in enumerate(fft_freq):
		if freq>5:
			if not triggered:
				idx_lo=idx
				triggered=True
		if freq>30: 
			idx_hi=idx+1
			break
	freq_list=fft_freq[idx_lo:idx_hi]
	#print(idx_lo, idx_hi)
	#print(freq_list)

	rounded_list=np.round(freq_list*2)/2.0
	integer_freqs=[]
	freq_indices=[]
	residues=np.abs(rounded_list-freq_list)
	#print(rounded_list)
	for i, el in enumerate(rounded_list):
		if el==rounded_list[-1]:
			integer_freqs.append(el)
			freq_indices.append(i)
			break
		residual=[]
		j=i+1

		if el not in integer_freqs:
			while True:
				if rounded_list[j]!=el or j+1==len(rounded_list):
					break
				j+=1
			residual=residues[i:j]
			integer_freqs.append(rounded_list[i+np.argmin(residual)])
			freq_indices.append(i+np.argmin(residual))

	#print(integer_freqs)
	#print(freq_indices)
	#raw_input()
	#import arrays
	print("Calculating displacement data")
	displacements, displacements_rot, eqs = coords_rotcoords(ovf_files, Bmin, Bmax, Bstep, tmin, tmax, tstep)
	
	#get new shapes for arrays, split by B-field val
	bs, ts, ts_rot = (B_field_values.size,), (int(displacements.shape[0]/B_field_values.size),), (int(displacements_rot.shape[0]/B_field_values.size),)
	newshape=bs+ts+displacements.shape[1:]	
	newshape_rot=bs+ts_rot+displacements_rot.shape[1:]


	displacements=displacements.reshape(newshape)

	displacements_rot=displacements_rot.reshape(newshape_rot)
	
	#ordered according to B, f, z, component
	depth_vals=np.linspace(0,18,displacements.shape[2])

	print("Commencing fourier transforms")
	FM_disp=spft.fft(displacements, axis=1)[:,idx_lo:idx_hi,:,:]
	FM_disp_rot=spft.fft(displacements_rot, axis=1)[:,idx_lo:idx_hi,:,:]
	print("Fourier transforms completed")
	"""
	"""
	colorlist=['b','g','r']
	plot_titles=[
			["x-component: real", "x-component: imaginary","x-component: modulus"],
			["y-component: real", "y-component: imaginary","y-component: modulus"],
			["z-component: real", "z-component: imaginary","z-component: modulus"]]

	plot_titles_rot=[
			[r"$\phi$-component: real", r"$\phi$-component: imaginary",r"$\phi$-component: modulus"],
			["z-component: real", "z-component: imaginary","z-component: modulus"]]


	for Bidx, Bval in enumerate(B_field_values):
		print("Field value: ", Bval)
		for idx, freq in enumerate(integer_freqs):
			print("Freq value: ", freq)
			Fidx=freq_indices[idx]
			plot_title="Resonant response over depth at H="+str(Bval)+"HD and f="+str(Fidx)+"GHz"
			plot_filename="res_over_depth_this_field-this_freq_Bmin_Bmax-"+str(freq)+"-"+str(Bval)+"-"+str(Bmin)+"-"+str(Bmax)+".png"	


			fig1, axes  = plt.subplots(3, 3)

			save_dir=fs.join(save_directory_resonances,plot_filename)
			#print("Length of axes: ", len(axes), "highest index: ", len(axes)-1)
			for i, row in enumerate(axes):
				for j, graph in enumerate(row):
					this_data=FM_disp[Bidx, Fidx,:,i]
					graph.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
	
					if j==0:
						this_data=np.real(this_data)
					elif j==1:
						this_data=np.imag(this_data)
					elif j==2:
						this_data=np.abs(this_data)

					graph.plot(depth_vals, this_data, c=colorlist[j], ls='solid')
					graph.set_title(plot_titles[i][j],fontsize=24)
							
			plt.suptitle(plot_title, fontsize=24)
			fig1.text(0.5, 0.04, r'Depth(nm)', ha='center', fontsize=24)
			fig1.text(0.04, 0.5, r'$\Delta m_{\alpha}$ (dimensionless $\frac{M_{\alpha}}{M_s}$)', va='center', rotation='vertical', fontsize=24)	
			fig1.set_size_inches(16,9)	
			
			plt.subplots_adjust(top=0.85)
			
			if save_plot:
			
				fig1.set_size_inches(16,9)	
				plt.savefig(save_dir, format='png', dpi=100)	
				plt.close()
			else:
				plt.show()
			
			#plt.show()

			this_data=FM_disp_rot[Bidx, Fidx,:,:]

			fig2, axes = plt.subplots(2,3)
			plot_filename="res_over_depth_rot_this_field-this_freq_Bmin_Bmax-"+str(freq)+"-"+str(Bval)+"-"+str(Bmin)+"-"+str(Bmax)+".png"
			save_dir=fs.join(save_directory_resonances,plot_filename)

			for i, row in enumerate(axes):
				for j, graph in enumerate(row):	
					graph.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
	
					this_data=FM_disp_rot[Bidx, Fidx,:,i+1]	#+1 bc i=0 gives the r comp which we dont want

					if j==0:
						this_data=np.real(this_data)
					elif j==1:
						this_data=np.imag(this_data)
					elif j==2:
						this_data=np.abs(this_data)

					graph.plot(depth_vals, this_data, c=colorlist[j], ls='solid')
					graph.set_title(plot_titles_rot[i][j],fontsize=24)
							
			plt.suptitle(plot_title, fontsize=24)
			fig2.text(0.5, 0.04, r'Depth(nm)', ha='center', fontsize=24)
			fig2.text(0.04, 0.5, r'$\Delta m_{\alpha}$ (dimensionless $\frac{M_{\alpha}}{M_s}$)', va='center', rotation='vertical', fontsize=24)	
			fig2.set_size_inches(16,9)	
			
			plt.subplots_adjust(top=0.85)
			
			if save_plot:
			
				fig1.set_size_inches(16,9)	
				plt.savefig(save_dir, format='png', dpi=100)	
				plt.close()
			else:
				plt.show()
			
			#plt.show()
			#break
#"""
def helicoid_displacement_plotter(ovf_files):
	try:
		global save_directory_displacements	
		global save_plot
		global Bmin
		global Bmax
		global Bstep
		global tmin
		global tmax
		global tstep
		data_dim=ovf_files.shape[1]

		time_series=np.linspace(tmin, tmax, num=tstep-1)

		B_field_values=np.linspace(Bmin, Bmax, num=round((Bmax-Bmin)/Bstep)+1)
		i=0
		all_displacements, all_displacements_rot, all_eq  = coords_rotcoords(ovf_files, Bmin, Bmax, Bstep, tmin, tmax, tstep)
		q_depth	=	int(data_dim/4)
		h_depth	=	int(data_dim/2)
		tq_depth=	int(3*q_depth)

		for B_field in B_field_values:	###	Start of Bfield loop	###
			displacements = all_displacements[i*(tstep-1):(i+1)*(tstep-1)]
			displacements_rot = all_displacements_rot[i*(tstep-1):(i+1)*(tstep-1)]
			i+=1
			#""""
			###		UNROTATED REFERENCE FRAME		###
			M_ave_all	=	np.mean(displacements, axis=1)	#average over all spins
			M_middle	=	np.mean(displacements[:, h_depth-1 : h_depth+1 ], axis=1) #average over two middle spins
			M_b1		=	displacements[:,0]	#top boundary spin
			M_b2		=	displacements[:,-1]	#bottom boundary spin
			M_c1		=	np.mean(displacements[:, q_depth-1 : q_depth+1 ], axis=1)#average over two quarter depth spins
			M_c2		=	np.mean(displacements[:, tq_depth-1 : tq_depth+1 ], axis=1)#average over two three quarter depth spins

			###		ROTATED REFERENCE FRAME, RHO COMES OUT AS ZERO SO OMITTED		###
			M_ave_all_rot	=	np.mean(displacements_rot, axis=1)
			M_middle_rot	=	np.mean(displacements_rot[:, h_depth-1 : h_depth+1 ], axis=1) #average over two middle spins
			M_b1_rot	=	displacements_rot[:,0]
			M_b2_rot	=	displacements_rot[:,-1]
			M_c1_rot	=	np.mean(displacements_rot[:, q_depth-1 : q_depth+1 ], axis=1)#average over two quarter depth spins
			M_c2_rot	=	np.mean(displacements_rot[:, tq_depth-1 : tq_depth+1 ], axis=1)#average over two three quarter depth spins

			spin_position_list	=	[M_ave_all, 	M_middle, 	M_b1, 		M_b2,	M_c1,	M_c2]			
			rot_spin_position_list	=	[M_ave_all_rot,	M_middle_rot, 	M_b1_rot, 	M_b2_rot,	M_c1_rot,	M_c2_rot]

			plot_titles=		[r"Change in $\vec{m}=\frac{\vec{M}}{M_s}$ vs time for $H=$"+str(B_field)+"$H_D$: Average over all spins",
						r"Change in $\vec{m}=\frac{\vec{M}}{M_s}$ vs time for $H=$"+str(B_field)+"$H_D$: Half depth average",
						r"Change in $\vec{m}=\frac{\vec{M}}{M_s}$ vs time for $H=$"+str(B_field)+"$H_D$: Spin at top interface",
						r"Change in $\vec{m}=\frac{\vec{M}}{M_s}$ vs time for $H=$"+str(B_field)+"$H_D$: Spin at bottom interface",
						r"Change in $\vec{m}=\frac{\vec{M}}{M_s}$ vs time for $H=$"+str(B_field)+"$H_D$: Quarter depth average",
						r"Change in $\vec{m}=\frac{\vec{M}}{M_s}$ vs time for $H=$"+str(B_field)+"$H_D$: Three quarter depth average"]

			plot_filenames=		["Disp_ave_Bmin-Bmax-this_B-"+str(Bmin)+"-"+str(Bmax)+"-"+str(B_field)+".png",
						"Disp_mid_Bmin-Bmax-this_B-"+str(Bmin)+"-"+str(Bmax)+"-"+str(B_field)+".png",
						"Disp_b1_Bmin-Bmax-this_B-"+str(Bmin)+"-"+str(Bmax)+"-"+str(B_field)+".png",
						"Disp_b2_Bmin-Bmax-this_B-"+str(Bmin)+"-"+str(Bmax)+"-"+str(B_field)+".png",
						"Disp_c1_Bmin-Bmax-this_B-"+str(Bmin)+"-"+str(Bmax)+"-"+str(B_field)+".png",
						"Disp_c2_Bmin-Bmax-this_B-"+str(Bmin)+"-"+str(Bmax)+"-"+str(B_field)+".png"]


			rot_plot_titles=	[r"Change in $\vec{m}=\frac{\vec{M}}{M_s}$ vs time for $H=$"+str(B_field)+"$H_D$: Average over all spins: rotated coordinate system",
						r"Change in $\vec{m}=\frac{\vec{M}}{M_s}$ vs time for $H=$"+str(B_field)+"$H_D$: Half depth average: rotated coordinate system",
						r"Change in $\vec{m}=\frac{\vec{M}}{M_s}$ vs time for $H=$"+str(B_field)+"$H_D$: Spin at top interface: rotated coordinate system",
						r"Change in $\vec{m}=\frac{\vec{M}}{M_s}$ vs time for $H=$"+str(B_field)+"$H_D$: Spin at bottom interface: rotated coordinate system",
						r"Change in $\vec{m}=\frac{\vec{M}}{M_s}$ vs time for $H=$"+str(B_field)+"$H_D$: Quarter depth average: rotated coordinate system",
						r"Change in $\vec{m}=\frac{\vec{M}}{M_s}$ vs time for $H=$"+str(B_field)+"$H_D$: Three quarter depth average: rotated coordinate system"]

			rot_plot_filenames=		["Disp_ave_rot_Bmin-Bmax-this_B-"+str(Bmin)+"-"+str(Bmax)+"-"+str(B_field)+".png",
						"Disp_mid_rot_Bmin-Bmax-this_B-"+str(Bmin)+"-"+str(Bmax)+"-"+str(B_field)+".png",
						"Disp_b1_rot_Bmin-Bmax-this_B-"+str(Bmin)+"-"+str(Bmax)+"-"+str(B_field)+".png",
						"Disp_b2_rot_Bmin-Bmax-this_B-"+str(Bmin)+"-"+str(Bmax)+"-"+str(B_field)+".png",
						"Disp_c1_rot_Bmin-Bmax-this_B-"+str(Bmin)+"-"+str(Bmax)+"-"+str(B_field)+".png",
						"Disp_c2_rot_Bmin-Bmax-this_B-"+str(Bmin)+"-"+str(Bmax)+"-"+str(B_field)+".png"]


				
			for n, el in enumerate(spin_position_list):
				fig1, (x_comp, y_comp, z_comp) = plt.subplots(1, 3)#, sharex='col', sharey='row')
				plot_filename=plot_filenames[n]
				save_dir=fs.join(save_directory_displacements,plot_filename)
				x_comp.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
				y_comp.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
				z_comp.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))



				plt.suptitle(plot_titles[n], fontsize=24)
				x_comp.plot(time_series, el[:,0], c='r', ls='solid')	
				x_comp.set_title("x-component", fontsize=24)

				y_comp.plot(time_series, el[:,1], c='b', ls='dotted')
				y_comp.set_title("y-component", fontsize=24)

				z_comp.plot(time_series, el[:,2], c='g', ls='dashed')
				z_comp.set_title("z-component", fontsize=24)

				fig1.text(0.5, 0.04, r'Time (ps)', ha='center', fontsize=24)
				fig1.text(0.04, 0.5, r'$\Delta m_{\alpha}$ (dimensionless $\frac{M_{\alpha}}{M_s}$)', va='center', rotation='vertical', fontsize=24)	
				fig1.set_size_inches(16,9)	
			
				plt.subplots_adjust(top=0.85)


				if save_plot:
				
					fig1.set_size_inches(16,9)	
					plt.savefig(save_dir, format='png', dpi=100)	
					plt.close()
				else:
					plt.show()
			
			for n, el in enumerate(rot_spin_position_list):
				fig2, (phi_comp, z_comp) = plt.subplots(1, 2)#, sharex='col', sharey='row')
				plot_filename=rot_plot_filenames[n]
				save_dir=fs.join(save_directory_displacements, plot_filename)
				phi_comp.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
				z_comp.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))

				plt.suptitle(plot_titles[n], fontsize=24)

				phi_comp.plot(time_series, el[:,1], c='b', ls='dotted')
				phi_comp.set_title(r"$\phi$-component", fontsize=24)

				z_comp.plot(time_series, el[:,2], c='g', ls='dashed')
				z_comp.set_title("z-component", fontsize=24)

				fig2.text(0.5, 0.04, r'Time (ps)', ha='center', fontsize=24)
				fig2.text(0.04, 0.5, r'$\Delta m_{\alpha}$ (dimensionless $\frac{M_{\alpha}}{M_s}$)', va='center', rotation='vertical', fontsize=24)	
				fig2.set_size_inches(16,9)	
				plt.subplots_adjust(top=0.85)

				if save_plot:
				
					fig2.set_size_inches(16,9)	
					plt.savefig(save_dir, format='png', dpi=100)	
					plt.close()
				else:
					plt.show()
			
			min_val=0
			max_val=0
			for data in spin_position_list:
				for element in data:

					if float(np.max(element)) > max_val:
						max_val=np.max(element)
					elif float(np.min(element)) < min_val:
						min_val=np.min(element)

			if np.abs(min_val)>np.abs(max_val):
				max_val = np.abs(min_val)
			else:
				min_val = -max_val

			aspect_ratio=np.round((tmax-tmin)/18)
			displacement_x = displacements[:,:,0].T				
			displacement_y = displacements[:,:,1].T
			displacement_z = displacements[:,:,2].T

			fig3, (x_comp, y_comp, z_comp) = plt.subplots(1, 3)#, sharex='col', sharey='row')

			plot_title=str(r"Depth response of $\vec{m}=\frac{\vec{M}}{M_s}$ vs time for $H=$"+str(B_field)+"$H_D$")
			plot_filename="response_Bmin-Bmax-this_B-"+str(Bmin)+"-"+str(Bmax)+"-"+str(B_field)+".png"

			save_dir=fs.join(save_directory_displacements,plot_filename)

			plt.suptitle(plot_title, fontsize=24)

			imx=x_comp.imshow(displacement_x, vmin=min_val, vmax=max_val, origin="lower", extent=[tmin, tmax, 0, 18], aspect=aspect_ratio, cmap='seismic')
			x_comp.set_title("x-component", fontsize=24)

			imy=y_comp.imshow(displacement_y, vmin=min_val, vmax=max_val, origin="lower", extent=[tmin, tmax, 0, 18], aspect=aspect_ratio, cmap='seismic')
			y_comp.set_title("y-component", fontsize=24)

			imz=im=z_comp.imshow(displacement_z, vmin=min_val, vmax=max_val, origin="lower", extent=[tmin, tmax, 0, 18], aspect=aspect_ratio, cmap='seismic')
			z_comp.set_title("z-component", fontsize=24)

			fig3.text(0.5, 0.04, r'Time (ps)', ha='center', fontsize=24)
			fig3.text(0.04, 0.5, r'Depth (nm)', va='center', rotation='vertical', fontsize=24)	

			#cbar_ax=fig2.add_axes([1.00, 0.15, 0.01, 0.7])
			cbar_ax=fig3.add_axes([0.85, 0.15, 0.05, 0.7])
			fig3.colorbar(im, cax=cbar_ax)
			fig3.set_size_inches(16,9)	
			plt.subplots_adjust(top=0.85)

			if save_plot:
			
				fig3.set_size_inches(16,9)	
				plt.savefig(save_dir, format='png', dpi=100)	
				plt.close()
			else:
				plt.show()

			###		ROTATED COORDS		###
			min_val=0
			max_val=0
			for data in rot_spin_position_list:
				for element in data:

					if float(np.max(element)) > max_val:
						max_val=np.max(element)
					elif float(np.min(element)) < min_val:
						min_val=np.min(element)

			if np.abs(min_val)>np.abs(max_val):
				max_val = np.abs(min_val)
			else:
				min_val = -max_val

			aspect_ratio=np.round((tmax-tmin)/18)
			displacement_phi = displacements_rot[:,:,1].T
			displacement_z = displacements_rot[:,:,2].T


			fig4, (phi_comp, z_comp) = plt.subplots(1, 2)#, sharex='col', sharey='row')

			plot_title=str(r"Depth response of $\vec{m}=\frac{\vec{M}}{M_s}$ vs time for $H=$"+str(B_field)+"$H_D$: Rotated Coordinate System")
			plot_filename="response_rot_Bmin-Bmax-this_B-"+str(Bmin)+"-"+str(Bmax)+"-"+str(B_field)+".png"

			save_dir=fs.join(save_directory_displacements,plot_filename)

			plt.suptitle(plot_title, fontsize=24)

			imphi=phi_comp.imshow(displacement_phi, vmin=min_val, vmax=max_val, origin="lower", extent=[tmin, tmax, 0, 18], aspect=aspect_ratio, cmap='seismic')
			phi_comp.set_title(r"$\phi$-component", fontsize=24)

			imz=im=z_comp.imshow(displacement_z, vmin=min_val, vmax=max_val, origin="lower", extent=[tmin, tmax, 0, 18], aspect=aspect_ratio, cmap='seismic')
			z_comp.set_title("z-component", fontsize=24)

			fig4.text(0.5, 0.04, r'Time (ps)', ha='center', fontsize=24)
			fig4.text(0.04, 0.5, r'Depth (nm)', va='center', rotation='vertical', fontsize=24)	

			#cbar_ax=fig2.add_axes([1.00, 0.15, 0.01, 0.7])
			cbar_ax=fig4.add_axes([0.85, 0.15, 0.05, 0.7])
			fig4.colorbar(im, cax=cbar_ax)
			fig4.set_size_inches(16,9)	
			plt.subplots_adjust(top=0.85)

			if save_plot:
			
				fig4.set_size_inches(16,9)	
				plt.savefig(save_dir, format='png', dpi=100)	
				plt.close()
			else:
				plt.show()

		###	End of Bfield loop	###
		print("Displacements plotted")
		#"""

	except IOError as IO:	
		print("No such file!")

def helicoid_net_mag_plotter(ovf_files):
	try:
		global save_directory
		global save_plot
		global Bmin
		global Bmax
		global Bstep
		global tmin
		global tmax
		global tstep

		data_dim=ovf_files.shape[1]

		B_field_values=np.linspace(Bmin, Bmax, num=round((Bmax-Bmin)/Bstep)+1)
		net_magnetisation=np.ndarray([len(B_field_values), 3])
		i=0
		
		for B_field in B_field_values:	###	Start of Bfield loop	###

			this_m_value=ovf_files[i*tstep]	#read values for this field

			net_magnetisation[i]=np.sum(this_m_value, axis=0)
			i+=1

		magnitude = np.linalg.norm(net_magnetisation, axis=1)

		fig1, (mag, z_comp) = plt.subplots(1,2, sharex='col', sharey='row')
		mag.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
		z_comp.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))

		plot_filename='netmag.png'
		save_dir=fs.join(save_directory,plot_filename)

		plot_title=r"Net Magnetisation as a function of reduced magnetic field $\frac{H}{H_D}$"

		plt.suptitle(plot_title, fontsize=24)

		mag.plot(B_field_values, magnitude)
		mag.set_title(r"Magnitude: $|\vec{m}|$", fontsize=24)

		z_comp.plot(B_field_values, net_magnetisation[:,2], c='r', ls='solid')	
		z_comp.set_title("z-component", fontsize=24)

		fig1.text(0.5, 0.04, r'External Magnetic Field Strength (Dimensionless $\frac{H}{H_D}$)', ha='center', fontsize=24)
		fig1.text(0.04, 0.5, r'Net Magnetisation (Dimensionless $\frac{\vec{M}}{M_s}$)', va='center', rotation='vertical', fontsize=24)	

		if save_plot:
		
			fig1.set_size_inches(16,9)	
			plt.savefig(save_dir, format='png', dpi=100)	
			plt.close()
		else:
			plt.show()
		
		print("Net magnetisations plotted")
	except IOError as IO:	
		print("No such file!")


def helicoid_resonant_frequencies_plotter(ovf_files):
	try:

		global save_directory_resonances
		global conjs		
		global save_plot
		global Bmin
		global Bmax
		global Bstep
		global tmin
		global tmax
		global tstep

		data_dim=ovf_files.shape[1]

		time_series=np.linspace(tmin, tmax, num=tstep-1)
		fft_freq=spft.fftfreq(len(time_series), d=time_series[1]-time_series[0])*1000
		triggered=False

		for idx, freq in enumerate(fft_freq):
			if freq>5:
				if not triggered:
					idx_lo=idx
					triggered=True
			if freq>30: 
				idx_hi=idx
				break

		B_field_values=np.linspace(Bmin, Bmax, num=round((Bmax-Bmin)/Bstep)+1)
		i=0
		all_displacements, all_displacements_rot, all_eq  = coords_rotcoords(ovf_files, Bmin, Bmax, Bstep, tmin, tmax, tstep)
		all_ffts = []
		all_ffts_rot=[]		

		q_depth	=	int(data_dim/4)
		h_depth	=	int(data_dim/2)
		tq_depth=	int(3*q_depth)
		print("All displacements shape: ", all_displacements.shape)
		for B_field in B_field_values:	###	Start of Bfield loop	###
			print("Field value ",i+1," of ", B_field_values.size)

	
			displacements = all_displacements[i*(tstep-1):(i+1)*(tstep-1)]
			displacements_rot = all_displacements_rot[i*(tstep-1):(i+1)*(tstep-1)]
			#print("displacements shape: ",displacements.shape)
			i+=1

			###		FOURIER TRANSFORM ARRAYS		###

			FM_disp		= np.array(	[spft.fft(displacements[:,:,0], axis=0),
							spft.fft(displacements[:,:,1], axis=0),
							spft.fft(displacements[:,:,2], axis=0)]	)	

			FM_disp_rot	= np.array(	[spft.fft(displacements_rot[:,:,1], axis=0),
							spft.fft(displacements_rot[:,:,2], axis=0)] )
			if conjs==True:					
				FM_disp[np.imag(FM_disp) < 0] = np.conj(FM_disp[np.imag(FM_disp) < 0])
				FM_disp_rot[np.imag(FM_disp_rot) < 0] = np.conj(FM_disp_rot[np.imag(FM_disp_rot) < 0])
			#print("FM disp shape: ", FM_disp.shape)
			#print("FM disp rot shape: ", FM_disp_rot.shape)			
			###		GET CARTESIAN		###
			FM_ave_all	=	np.mean(FM_disp, axis=2)

			FM_middle	=	np.mean(FM_disp[:,:,h_depth-1:h_depth+1], axis=2)
			FM_quarter	=	np.mean(FM_disp[:,:,q_depth-1:q_depth+1], axis=2)
			FM_th_quarter	=	np.mean(FM_disp[:,:,tq_depth-1:tq_depth+1], axis=2)	

			FM_b1		=	FM_disp[:,:,0]
			FM_b2		=	FM_disp[:,:,-1]
			
			###		GET ROT ARRAYS		###
			FM_ave_all_rot		=	np.mean(FM_disp_rot, axis=2)
			FM_middle_rot		=	np.mean(FM_disp_rot[:,:,h_depth-1:h_depth+1], axis=2)
			FM_quarter_rot		=	np.mean(FM_disp_rot[:,:,q_depth-1:q_depth+1], axis=2)
			FM_th_quarter_rot	=	np.mean(FM_disp_rot[:,:,tq_depth-1:tq_depth+1], axis=2)	
			FM_b1_rot		=	FM_disp_rot[:,:,0]
			FM_b2_rot		=	FM_disp_rot[:,:,-1]

			spectra_list=np.asarray([FM_ave_all, FM_middle, FM_quarter, FM_th_quarter, FM_b1, FM_b2])
			spectra_rot_list=np.asarray([FM_ave_all_rot, FM_middle_rot, FM_quarter_rot, FM_th_quarter_rot, FM_b1_rot, FM_b2_rot])
			
			re_im_plot_titles=	[r"Fourier transform of $\vec{m}=\frac{\vec{M}}{M_s}$ for $H=$"+str(B_field)+"$H_D$: Average over all spins",
						r"Fourier transform of $\vec{m}=\frac{\vec{M}}{M_s}$ for $H=$"+str(B_field)+"$H_D$: Half depth average",
						r"Fourier transform of $\vec{m}=\frac{\vec{M}}{M_s}$ for $H=$"+str(B_field)+"$H_D$: Quarter depth average",
						r"Fourier transform of $\vec{m}=\frac{\vec{M}}{M_s}$ for $H=$"+str(B_field)+"$H_D$: Three quarter depth average",
						r"Fourier transform of $\vec{m}=\frac{\vec{M}}{M_s}$ for $H=$"+str(B_field)+"$H_D$: Spin at top interface",
						r"Fourier transform of $\vec{m}=\frac{\vec{M}}{M_s}$ for $H=$"+str(B_field)+"$H_D$: Spin at bottom interface"]	

			re_im_rot_plot_titles=	[r"Fourier transform of $\vec{m}=\frac{\vec{M}}{M_s}$ for $H=$"+str(B_field)+"$H_D$: Average over all spins: Rotated coordinate system",
						r"Fourier transform of $\vec{m}=\frac{\vec{M}}{M_s}$ for $H=$"+str(B_field)+"$H_D$: Half depth average: Rotated coordinate system",
						r"Fourier transform of $\vec{m}=\frac{\vec{M}}{M_s}$ for $H=$"+str(B_field)+"$H_D$: Quarter depth average: Rotated coordinate system",
						r"Fourier transform of $\vec{m}=\frac{\vec{M}}{M_s}$ for $H=$"+str(B_field)+"$H_D$: Three quarter depth average: Rotated coordinate system",
						r"Fourier transform of $\vec{m}=\frac{\vec{M}}{M_s}$ for $H=$"+str(B_field)+"$H_D$: Spin at top interface: Rotated coordinate system",
						r"Fourier transform of $\vec{m}=\frac{\vec{M}}{M_s}$ for $H=$"+str(B_field)+"$H_D$: Spin at bottom interface: Rotated coordinate system"]	



			re_im_plot_filenames	=	["Fourier_ave_Bmin-Bmax-this_B-"+str(Bmin)+"-"+str(Bmax)+"-"+str(B_field)+".png",
							"Fourier_mid_Bmin-Bmax-this_B-"+str(Bmin)+"-"+str(Bmax)+"-"+str(B_field)+".png",
							"Fourier_q_Bmin-Bmax-this_B-"+str(Bmin)+"-"+str(Bmax)+"-"+str(B_field)+".png",
							"Fourier_tq_Bmin-Bmax-this_B-"+str(Bmin)+"-"+str(Bmax)+"-"+str(B_field)+".png",
							"Fourier_b1_Bmin-Bmax-this_B-"+str(Bmin)+"-"+str(Bmax)+"-"+str(B_field)+".png",
							"Fourier_b2_Bmin-Bmax-this_B-"+str(Bmin)+"-"+str(Bmax)+"-"+str(B_field)+".png"]

			re_im_rot_plot_filenames=	["Fourier_ave_rot_Bmin-Bmax-this_B-"+str(Bmin)+"-"+str(Bmax)+"-"+str(B_field)+".png",
							"Fourier_mid_rot_Bmin-Bmax-this_B-"+str(Bmin)+"-"+str(Bmax)+"-"+str(B_field)+".png",
							"Fourier_q_rot_Bmin-Bmax-this_B-"+str(Bmin)+"-"+str(Bmax)+"-"+str(B_field)+".png",
							"Fourier_tq_rot_Bmin-Bmax-this_B-"+str(Bmin)+"-"+str(Bmax)+"-"+str(B_field)+".png",
							"Fourier_b1_rot_Bmin-Bmax-this_B-"+str(Bmin)+"-"+str(Bmax)+"-"+str(B_field)+".png",
							"Fourier_b2_rot_Bmin-Bmax-this_B-"+str(Bmin)+"-"+str(Bmax)+"-"+str(B_field)+".png"]

			po_ph_plot_titles=	[r"Power spectrum and phase angle of $\vec{m}=\frac{\vec{M}}{M_s}$ for $H=$"+str(B_field)+"$H_D$: Average over all spins",
						r"Power spectrum and phase angle of $\vec{m}=\frac{\vec{M}}{M_s}$ for $H=$"+str(B_field)+"$H_D$: Half depth average",
						r"Power spectrum and phase angle of $\vec{m}=\frac{\vec{M}}{M_s}$ for $H=$"+str(B_field)+"$H_D$: Quarter depth average",
						r"Power spectrum and phase angle of $\vec{m}=\frac{\vec{M}}{M_s}$ for $H=$"+str(B_field)+"$H_D$: Three quarter depth average",
						r"Power spectrum and phase angle of $\vec{m}=\frac{\vec{M}}{M_s}$ for $H=$"+str(B_field)+"$H_D$: Spin at top interface",
						r"Power spectrum and phase angle of $\vec{m}=\frac{\vec{M}}{M_s}$ for $H=$"+str(B_field)+"$H_D$: Spin at bottom interface"]	

			po_ph_rot_plot_titles=	[r"Power spectrum and phase angle of $\vec{m}=\frac{\vec{M}}{M_s}$ for $H=$"+str(B_field)+"$H_D$: Average over all spins: Rotated coordinate system",
						r"Power spectrum and phase angle of $\vec{m}=\frac{\vec{M}}{M_s}$ for $H=$"+str(B_field)+"$H_D$: Half depth average: Rotated coordinate system",
						r"Power spectrum and phase angle of $\vec{m}=\frac{\vec{M}}{M_s}$ for $H=$"+str(B_field)+"$H_D$: Quarter depth average: Rotated coordinate system",
						r"Power spectrum and phase angle of $\vec{m}=\frac{\vec{M}}{M_s}$ for $H=$"+str(B_field)+"$H_D$: Three quarter depth average: Rotated coordinate system",
						r"Power spectrum and phase angle of $\vec{m}=\frac{\vec{M}}{M_s}$ for $H=$"+str(B_field)+"$H_D$: Spin at top interface: Rotated coordinate system",
						r"Power spectrum and phase angle of $\vec{m}=\frac{\vec{M}}{M_s}$ for $H=$"+str(B_field)+"$H_D$: Spin at bottom interface: Rotated coordinate system"]	



			po_ph_plot_filenames	=	["PowPha_ave_Bmin-Bmax-this_B-"+str(Bmin)+"-"+str(Bmax)+"-"+str(B_field)+".png",
							"PowPha_mid_Bmin-Bmax-this_B-"+str(Bmin)+"-"+str(Bmax)+"-"+str(B_field)+".png",
							"PowPha_q_Bmin-Bmax-this_B-"+str(Bmin)+"-"+str(Bmax)+"-"+str(B_field)+".png",
							"PowPha_tq_Bmin-Bmax-this_B-"+str(Bmin)+"-"+str(Bmax)+"-"+str(B_field)+".png",
							"PowPha_b1_Bmin-Bmax-this_B-"+str(Bmin)+"-"+str(Bmax)+"-"+str(B_field)+".png",
							"PowPha_b2_Bmin-Bmax-this_B-"+str(Bmin)+"-"+str(Bmax)+"-"+str(B_field)+".png"]

			po_ph_rot_plot_filenames=	["PowPha_ave_rot_Bmin-Bmax-this_B-"+str(Bmin)+"-"+str(Bmax)+"-"+str(B_field)+".png",
							"PowPha_mid_rot_Bmin-Bmax-this_B-"+str(Bmin)+"-"+str(Bmax)+"-"+str(B_field)+".png",
							"PowPha_q_rot_Bmin-Bmax-this_B-"+str(Bmin)+"-"+str(Bmax)+"-"+str(B_field)+".png",
							"PowPha_tq_rot_Bmin-Bmax-this_B-"+str(Bmin)+"-"+str(Bmax)+"-"+str(B_field)+".png",
							"PowPha_b1_rot_Bmin-Bmax-this_B-"+str(Bmin)+"-"+str(Bmax)+"-"+str(B_field)+".png",
							"PowPha_b2_rot_Bmin-Bmax-this_B-"+str(Bmin)+"-"+str(Bmax)+"-"+str(B_field)+".png"]
			
			#print("Frequency list shape: ", fft_freq.shape)
			#print("Spectra list shape: ", spectra_list.shape)
			#print("Low freq idx: ",idx_lo,'/n',"High freq idx: ", idx_hi)
			#raw_input('press enter to crash')
			
			for n, el in enumerate(spectra_list):
				
				fig1, ((x_comp_re, y_comp_re, z_comp_re), (x_comp_im, y_comp_im, z_comp_im)) = plt.subplots(2, 3)#, sharex='col', sharey='row')
				plot_filename=re_im_plot_filenames[n]
				save_dir=fs.join(save_directory_resonances,plot_filename)

				plt.suptitle(re_im_plot_titles[n], fontsize=24)

				x_comp_re.plot(fft_freq[idx_lo:idx_hi], np.real(el[0,idx_lo:idx_hi]))
				x_comp_re.set_title("Re{x-component}", fontsize=24)
				
				y_comp_re.plot(fft_freq[idx_lo:idx_hi], np.real(el[1,idx_lo:idx_hi]))
				y_comp_re.set_title("Re{y-component}", fontsize=24)

				z_comp_re.plot(fft_freq[idx_lo:idx_hi], np.real(el[2,idx_lo:idx_hi]))
				z_comp_re.set_title("Re{z-component}", fontsize=24)

				x_comp_im.plot(fft_freq[idx_lo:idx_hi], np.imag(el[0,idx_lo:idx_hi]))
				x_comp_im.set_title("Im{x-component}", fontsize=24)
				
				y_comp_im.plot(fft_freq[idx_lo:idx_hi], np.imag(el[1,idx_lo:idx_hi]))
				y_comp_im.set_title("Im{y-component}", fontsize=24)

				z_comp_im.plot(fft_freq[idx_lo:idx_hi], np.imag(el[2,idx_lo:idx_hi]))
				z_comp_im.set_title("Im{z-component}", fontsize=24)

				fig1.text(0.5, 0.04, r'Frequency (GHz)', ha='center', fontsize=24)
				fig1.text(0.04, 0.5, r'Intensity (Dimensionless $\frac{\vec{M}}{M_s}$)', va='center', rotation='vertical', fontsize=24)		
				plt.subplots_adjust(top=0.85)

				if save_plot:
					fig1.set_size_inches(16,9)	
					plt.savefig(save_dir, format='png', dpi=100)	
					plt.close()
				else:
					plt.show()
				###	POW PHASE PLOTS, ADD IN SAME LOOP AS RE IM	STILL NEED TO TOUCH UP NP FUNCTIONS	###
				fig3, ((x_comp_pow, y_comp_pow, z_comp_pow), (x_comp_pha, y_comp_pha, z_comp_pha)) = plt.subplots(2, 3)#, sharex='col', sharey='row')
				plot_filename=po_ph_plot_filenames[n]
				save_dir=fs.join(save_directory_resonances,plot_filename)

				plt.suptitle(po_ph_plot_titles[n], fontsize=24)

				x_comp_pow.plot(fft_freq[idx_lo:idx_hi], np.absolute(el[0,idx_lo:idx_hi])**2)
				x_comp_pow.set_title("Power Spectrum (x-component)", fontsize=24)
				x_comp_pow.set_ylabel(r'Intensity (Dimensionless $\frac{\vec{M}}{M_s}$)', fontsize=24)
	
				y_comp_pow.plot(fft_freq[idx_lo:idx_hi], np.absolute(el[1,idx_lo:idx_hi])**2)
				y_comp_pow.set_title("Power Spectrum (y-component)", fontsize=24)

				z_comp_pow.plot(fft_freq[idx_lo:idx_hi], np.absolute(el[2,idx_lo:idx_hi])**2)
				z_comp_pow.set_title("Power Spectrum (z-component)", fontsize=24)

				x_comp_pha.plot(fft_freq[idx_lo:idx_hi], np.mod(np.angle(el[0,idx_lo:idx_hi],deg=True), 360))
				x_comp_pha.set_title("Phase Angle (x-component)", fontsize=24)	
				x_comp_pha.set_ylabel("Degrees", fontsize=24)

				y_comp_pha.plot(fft_freq[idx_lo:idx_hi], np.mod(np.angle(el[1,idx_lo:idx_hi],deg=True),360))
				y_comp_pha.set_title("Phase Angle (y-component)", fontsize=24)

				z_comp_pha.plot(fft_freq[idx_lo:idx_hi], np.mod(np.angle(el[2,idx_lo:idx_hi],deg=True),360))
				z_comp_pha.set_title("Phase Angle (z-component)", fontsize=24)

				fig3.text(0.5, 0.04, r'Frequency (GHz)', ha='center', fontsize=24)
				#fig3.text(0.04, 0.5, r'Relative Intensity', va='center', rotation='vertical', fontsize=24)		
				plt.subplots_adjust(top=0.85)

				if save_plot:
					fig3.set_size_inches(16,9)	
					plt.savefig(save_dir, format='png', dpi=100)	
					plt.close()
				else:
					plt.show()
				
			for n, el in enumerate(spectra_rot_list):
				
				fig2, ((phi_comp_re, z2_comp_re), (phi_comp_im, z2_comp_im)) = plt.subplots(2, 2)#, sharex='col', sharey='row')
				plot_filename=re_im_rot_plot_filenames[n]
				save_dir=fs.join(save_directory_resonances,plot_filename)

				plt.suptitle(re_im_rot_plot_titles[n], fontsize=24)

				phi_comp_re.plot(fft_freq[idx_lo:idx_hi], np.real(el[0,idx_lo:idx_hi]))
				phi_comp_re.set_title(r"Re{$\phi$-component}", fontsize=24)

				z2_comp_re.plot(fft_freq[idx_lo:idx_hi], np.real(el[1,idx_lo:idx_hi]))
				z2_comp_re.set_title("Re{z-component}", fontsize=24)

				phi_comp_im.plot(fft_freq[idx_lo:idx_hi], np.imag(el[0,idx_lo:idx_hi]))
				phi_comp_im.set_title(r"Im{$\phi$-component}", fontsize=24)

				z2_comp_im.plot(fft_freq[idx_lo:idx_hi], np.imag(el[1,idx_lo:idx_hi]))
				z2_comp_im.set_title("Im{z-component}", fontsize=24)

				fig2.text(0.5, 0.04, r'Frequency (GHz)', ha='center', fontsize=24)
				fig2.text(0.04, 0.5, r'Intensity (Dimensionless $\frac{\vec{M}}{M_s}$)', va='center', rotation='vertical', fontsize=24)		
				plt.subplots_adjust(top=0.85)

				if save_plot:
					fig2.set_size_inches(16,9)	
					plt.savefig(save_dir, format='png', dpi=100)	
					plt.close()
				else:
					plt.show()

				fig4, ((phi_comp_pow, z2_comp_pow), (phi_comp_pha, z2_comp_pha)) = plt.subplots(2, 2)
				plot_filename=po_ph_rot_plot_filenames[n]
				save_dir=fs.join(save_directory_resonances,plot_filename)

				plt.suptitle(po_ph_rot_plot_titles[n], fontsize=24)

				phi_comp_pow.plot(fft_freq[idx_lo:idx_hi], np.absolute(el[0,idx_lo:idx_hi])**2)
				phi_comp_pow.set_title(r"Power Spectrum ($\phi$-component)", fontsize=24)
				phi_comp_pow.set_ylabel(r'Intensity (Dimensionless $\frac{\vec{M}}{M_s}$)', fontsize=24)

				z2_comp_pow.plot(fft_freq[idx_lo:idx_hi], np.absolute(el[1,idx_lo:idx_hi])**2)
				z2_comp_pow.set_title("Power Spectrum (z-component)", fontsize=24)

				phi_comp_pha.plot(fft_freq[idx_lo:idx_hi], np.mod(np.angle(el[0,idx_lo:idx_hi],deg=True), 360))
				phi_comp_pha.set_title(r"Phase Angle ($\phi$-component)", fontsize=24)
				phi_comp_pha.set_ylabel("Degrees", fontsize=24)				

				z2_comp_pha.plot(fft_freq[idx_lo:idx_hi], np.mod(np.angle(el[1,idx_lo:idx_hi],deg=True),360))
				z2_comp_pha.set_title("Phase Angle (z-component)", fontsize=24)

				fig4.text(0.5, 0.04, r'Frequency (GHz)', ha='center', fontsize=24)
				plt.subplots_adjust(top=0.85)

				if save_plot:
					fig4.set_size_inches(16,9)	
					plt.savefig(save_dir, format='png', dpi=100)	
					plt.close()
				else:
					plt.show()
			
				
			all_ffts.append(FM_disp)
			all_ffts_rot.append(FM_disp_rot)	

		###	END OF B FIELD LOOP	###
		###	NEED TO BUILD ABOVE BUT FOR FREQ VS FIELD, ALL DIFF POINTS 	###
		###	ALSO PHASE V FIELD FOR FREQ VALUE	###

		all_ffts=np.asarray(all_ffts)
		all_ffts_rot=np.asarray(all_ffts_rot)
		freq_field = np.array([	np.mean(all_ffts, axis=3),
					np.mean(all_ffts[:,:,:,h_depth-1:h_depth+1], axis=3),
					np.mean(all_ffts[:,:,:,q_depth-1:q_depth+1], axis=3),
					np.mean(all_ffts[:,:,:,tq_depth-1:tq_depth+1], axis=3),
					all_ffts[:,:,:,0],
					all_ffts[:,:,:,-1]	])

		freq_field_rot = np.array([	np.mean(all_ffts_rot, axis=3),
						np.mean(all_ffts_rot[:,:,:,h_depth-1:h_depth+1], axis=3),
						np.mean(all_ffts_rot[:,:,:,q_depth-1:q_depth+1], axis=3),
						np.mean(all_ffts_rot[:,:,:,tq_depth-1:tq_depth+1], axis=3),
						all_ffts_rot[:,:,:,0],
						all_ffts_rot[:,:,:,-1]						])


		aspect_ratio=(Bmax-Bmin)/(fft_freq[idx_hi]-fft_freq[idx_lo])
				

		depth_field_rot_titles=	[r"Frequency response vs applied magnetic field: Average over all spins",
					r"Frequency response vs applied magnetic field: Half depth average",
					r"Frequency response vs applied magnetic field: Quarter depth average",
					r"Frequency response vs applied magnetic field: Three quarter depth average",
					r"Frequency response vs applied magnetic field: Spin at top interface",
					r"Frequency response vs applied magnetic field: Spin at bottom interface"]


		depth_field_rot_filenames=	["Freq_field_ave_Bmin-Bmax-this_B-"+str(Bmin)+"-"+str(Bmax)+".png",
						"Freq_field_mid_Bmin-Bmax-this_B-"+str(Bmin)+"-"+str(Bmax)+".png",
						"Freq_field_q_Bmin-Bmax-this_B-"+str(Bmin)+"-"+str(Bmax)+".png",
						"Freq_field_tq_Bmin-Bmax-this_B-"+str(Bmin)+"-"+str(Bmax)+".png",
						"Freq_field_b1_Bmin-Bmax-this_B-"+str(Bmin)+"-"+str(Bmax)+".png",
						"Freq_field_b2_Bmin-Bmax-this_B-"+str(Bmin)+"-"+str(Bmax)+".png"]



		freq_field_titles=	[r"Frequency response vs applied magnetic field: Average over all spins",
					r"Frequency response vs applied magnetic field: Half depth average",
					r"Frequency response vs applied magnetic field: Quarter depth average",
					r"Frequency response vs applied magnetic field: Three quarter depth average",
					r"Frequency response vs applied magnetic field: Spin at top interface",
					r"Frequency response vs applied magnetic field: Spin at bottom interface"]


		freq_field_filenames=	["Freq_field_ave_Bmin-Bmax-this_B-"+str(Bmin)+"-"+str(Bmax)+".png",
					"Freq_field_mid_Bmin-Bmax-this_B-"+str(Bmin)+"-"+str(Bmax)+".png",
					"Freq_field_q_Bmin-Bmax-this_B-"+str(Bmin)+"-"+str(Bmax)+".png",
					"Freq_field_tq_Bmin-Bmax-this_B-"+str(Bmin)+"-"+str(Bmax)+".png",
					"Freq_field_b1_Bmin-Bmax-this_B-"+str(Bmin)+"-"+str(Bmax)+".png",
					"Freq_field_b2_Bmin-Bmax-this_B-"+str(Bmin)+"-"+str(Bmax)+".png"]
		

		freq_field_phase_titles=	[r"Frequency response vs applied magnetic field: Average over all spins",
						r"Frequency response vs applied magnetic field: Half depth average",
						r"Frequency response vs applied magnetic field: Quarter depth average",
						r"Frequency response vs applied magnetic field: Three quarter depth average",
						r"Frequency response vs applied magnetic field: Spin at top interface",
						r"Frequency response vs applied magnetic field: Spin at bottom interface"]


		freq_field_phase_filenames=	["Freq_field_phase_ave_Bmin-Bmax-this_B-"+str(Bmin)+"-"+str(Bmax)+".png",
						"Freq_field_phase_mid_Bmin-Bmax-this_B-"+str(Bmin)+"-"+str(Bmax)+".png",
						"Freq_field_phase_q_Bmin-Bmax-this_B-"+str(Bmin)+"-"+str(Bmax)+".png",
						"Freq_field_phase_tq_Bmin-Bmax-this_B-"+str(Bmin)+"-"+str(Bmax)+".png",
						"Freq_field_phase_b1_Bmin-Bmax-this_B-"+str(Bmin)+"-"+str(Bmax)+".png",
						"Freq_field_phase_b2_Bmin-Bmax-this_B-"+str(Bmin)+"-"+str(Bmax)+".png"]

		freq_field_rot_titles=	[r"Frequency response vs applied magnetic field: Average over all spins: rotated coordinate system",
					r"Frequency response vs applied magnetic field: Half depth average: rotated coordinate system",
					r"Frequency response vs applied magnetic field: Quarter depth average: rotated coordinate system",
					r"Frequency response vs applied magnetic field: Three quarter depth average: rotated coordinate system",
					r"Frequency response vs applied magnetic field: Spin at top interface: rotated coordinate system",
					r"Frequency response vs applied magnetic field: Spin at bottom interface: rotated coordinate system"]


		freq_field_rot_filenames=	["Freq_field_ave_rot_Bmin-Bmax-this_B-"+str(Bmin)+"-"+str(Bmax)+".png",
						"Freq_field_mid_rot_Bmin-Bmax-this_B-"+str(Bmin)+"-"+str(Bmax)+".png",
						"Freq_field_q_rot_Bmin-Bmax-this_B-"+str(Bmin)+"-"+str(Bmax)+".png",
						"Freq_field_tq_rot_Bmin-Bmax-this_B-"+str(Bmin)+"-"+str(Bmax)+".png",
						"Freq_field_b1_rot_Bmin-Bmax-this_B-"+str(Bmin)+"-"+str(Bmax)+".png",
						"Freq_field_b2_rot_Bmin-Bmax-this_B-"+str(Bmin)+"-"+str(Bmax)+".png"]

		###	FFT ARRAY SHAPE CORRESPONDS TO (Field value;	Mag vector component;	frequency;	z position)
		#freq_field=np.swapaxes(freq_field (
		for m, el in enumerate(freq_field):
			for dim in el:
				dim/=np.max(np.abs(dim[:,idx_lo:idx_hi]))
			el_rot=freq_field_rot[m]
			for dim in el_rot:
				dim/=np.max(np.abs(dim[:,idx_lo:idx_hi]))

			fig5, ((x_comp_re, y_comp_re, z_comp_re), (x_comp_im, y_comp_im, z_comp_im), (x_comp_po, y_comp_po, z_comp_po)) = plt.subplots(3, 3)#, sharex='col', sharey='row')
			plot_filename=freq_field_filenames[m]
			save_dir=fs.join(save_directory_resonances,plot_filename)

			max_val= np.max(np.absolute(el[:,:,idx_lo:idx_hi].T))
			min_val= -max_val

			plt.suptitle(freq_field_titles[m], fontsize=24)

			x_comp_re.imshow(	np.real(el[:,0,idx_lo:idx_hi].T), origin="lower", vmin=min_val, vmax=max_val,
					extent=[Bmin, Bmax, fft_freq[idx_lo], fft_freq[idx_hi]], aspect=aspect_ratio, cmap='Spectral')
			x_comp_re.set_title("Re{x-component}", fontsize=24)


			y_comp_re.imshow(	np.real(el[:,1,idx_lo:idx_hi].T), origin="lower", vmin=min_val, vmax=max_val,
					extent=[Bmin, Bmax, fft_freq[idx_lo], fft_freq[idx_hi]], aspect=aspect_ratio, cmap='Spectral')
			y_comp_re.set_title("Re{y-component}", fontsize=24)


			z_comp_re.imshow(	np.real(el[:,2,idx_lo:idx_hi].T), origin="lower",vmin=min_val, vmax=max_val,
						extent=[Bmin, Bmax, fft_freq[idx_lo], fft_freq[idx_hi]], aspect=aspect_ratio, cmap='Spectral')
			z_comp_re.set_title("Re{z-component}", fontsize=24)

			x_comp_im.imshow(	np.imag(el[:,0,idx_lo:idx_hi].T), origin="lower", vmin=min_val, vmax=max_val,
					extent=[Bmin, Bmax, fft_freq[idx_lo], fft_freq[idx_hi]], aspect=aspect_ratio, cmap='Spectral')
			x_comp_im.set_title("Im{x-component}", fontsize=24)


			y_comp_im.imshow(	np.imag(el[:,1,idx_lo:idx_hi].T), origin="lower", vmin=min_val, vmax=max_val,
					extent=[Bmin, Bmax, fft_freq[idx_lo], fft_freq[idx_hi]], aspect=aspect_ratio, cmap='Spectral')
			y_comp_im.set_title("Im{y-component}", fontsize=24)


			z_comp_im.imshow(	np.imag(el[:,2,idx_lo:idx_hi].T), origin="lower",vmin=min_val, vmax=max_val,
						extent=[Bmin, Bmax, fft_freq[idx_lo], fft_freq[idx_hi]], aspect=aspect_ratio, cmap='Spectral')
			z_comp_im.set_title("Im{z-component}", fontsize=24)

			x_comp_po.imshow(	np.absolute(el[:,0,idx_lo:idx_hi].T), origin="lower",vmin=min_val, vmax=max_val,
					extent=[Bmin, Bmax, fft_freq[idx_lo], fft_freq[idx_hi]], aspect=aspect_ratio, cmap='Spectral')
			x_comp_po.set_title("Power spectrum, x-component", fontsize=24)


			y_comp_po.imshow(	np.absolute(el[:,1,idx_lo:idx_hi].T), origin="lower",vmin=min_val, vmax=max_val,
					extent=[Bmin, Bmax, fft_freq[idx_lo], fft_freq[idx_hi]], aspect=aspect_ratio, cmap='Spectral')
			y_comp_po.set_title("Power spectrum, y-component", fontsize=24)


			im=z_comp_po.imshow(	np.absolute(el[:,2,idx_lo:idx_hi].T), origin="lower",vmin=min_val, vmax=max_val,
						extent=[Bmin, Bmax, fft_freq[idx_lo], fft_freq[idx_hi]], aspect=aspect_ratio, cmap='Spectral')
			z_comp_po.set_title("Power spectrum, z-component", fontsize=24)



			#plt.suptitle(plot_titles[m])

			fig5.text(0.5, 0.04, r'External Field $\frac{H}{H_D}$', ha='center', fontsize=24)
			fig5.text(0.04, 0.5, r'Frequency (GHz)', va='center', rotation='vertical', fontsize=24)	

			cbar_ax=fig5.add_axes([0.91, 0.15, 0.05, 0.7])
			fig5.colorbar(im, cax=cbar_ax)
			fig5.set_size_inches(16,9)	
			if save_plot:
			
				fig5.set_size_inches(16,9)	
				plt.savefig(save_dir, format='png', dpi=100)	
				plt.close()
			else:
				plt.show()
	
			fig6, ((phi_comp_re, z2_comp_re), (phi_comp_im, z2_comp_im), (phi_comp_po, z2_comp_po)) = plt.subplots(3, 2)#, sharex='col', sharey='row')
			plot_filename=freq_field_rot_filenames[m]
			save_dir=fs.join(save_directory_resonances,plot_filename)

			plt.suptitle(freq_field_rot_titles[m], fontsize=24)

			max_val= np.max(np.absolute(el_rot[:,:,idx_lo:idx_hi].T))
			min_val= -max_val

			phi_comp_re.imshow(	np.real(el_rot[:,0,idx_lo:idx_hi].T), origin="lower", vmin=min_val, vmax=max_val,
					extent=[Bmin, Bmax, fft_freq[idx_lo], fft_freq[idx_hi]], aspect=aspect_ratio, cmap='Spectral')
			phi_comp_re.set_title(r"Re{$\phi$-component}", fontsize=24)


			z2_comp_re.imshow(	np.real(el_rot[:,1,idx_lo:idx_hi].T), origin="lower",vmin=min_val, vmax=max_val,
						extent=[Bmin, Bmax, fft_freq[idx_lo], fft_freq[idx_hi]], aspect=aspect_ratio, cmap='Spectral')
			z2_comp_re.set_title("Re{z-component}", fontsize=24)


			phi_comp_im.imshow(	np.imag(el_rot[:,0,idx_lo:idx_hi].T), origin="lower", vmin=min_val, vmax=max_val,
					extent=[Bmin, Bmax, fft_freq[idx_lo], fft_freq[idx_hi]], aspect=aspect_ratio, cmap='Spectral')
			phi_comp_im.set_title(r"Im{$\phi$-component}", fontsize=24)


			z2_comp_im.imshow(	np.imag(el_rot[:,1,idx_lo:idx_hi].T), origin="lower",vmin=min_val, vmax=max_val,
						extent=[Bmin, Bmax, fft_freq[idx_lo], fft_freq[idx_hi]], aspect=aspect_ratio, cmap='Spectral')
			z2_comp_im.set_title("Im{z-component}", fontsize=24)



			phi_comp_po.imshow(	np.absolute(el_rot[:,0,idx_lo:idx_hi].T), origin="lower", vmin=min_val, vmax=max_val,
					extent=[Bmin, Bmax, fft_freq[idx_lo], fft_freq[idx_hi]], aspect=aspect_ratio, cmap='Spectral')
			phi_comp_po.set_title(r"Power spectrum, $\phi$-component", fontsize=24)


			im=z2_comp_po.imshow(	np.absolute(el_rot[:,1,idx_lo:idx_hi].T), origin="lower",vmin=min_val, vmax=max_val,
						extent=[Bmin, Bmax, fft_freq[idx_lo], fft_freq[idx_hi]], aspect=aspect_ratio, cmap='Spectral')
			z2_comp_po.set_title("Power spectrum, z-component", fontsize=24)



			#plt.suptitle(plot_titles[m])

			fig6.text(0.5, 0.04, r'External Field $\frac{H}{H_D}$', ha='center', fontsize=24)
			fig6.text(0.04, 0.5, r'Frequency (GHz)', va='center', rotation='vertical', fontsize=24)	

			cbar_ax=fig6.add_axes([0.85, 0.15, 0.05, 0.7])
			fig6.colorbar(im, cax=cbar_ax)
			fig6.set_size_inches(16,9)	
			if save_plot:
			
				fig6.set_size_inches(16,9)	
				plt.savefig(save_dir, format='png', dpi=100)	
				plt.close()
			else:
				plt.show()
			
			fig7, ((x_comp_ph, y_comp_ph), (z_comp_ph, phi_comp_ph)) = plt.subplots(2, 2)
			plot_filename=freq_field_phase_filenames[m]
			save_dir=fs.join(save_directory_resonances,plot_filename)

			max_val= np.max(np.mod(np.angle(el[:,:,idx_lo:idx_hi].T, deg=True), 360))
			min_val= np.min(np.mod(np.angle(el[:,:,idx_lo:idx_hi].T, deg=True), 360))

			if np.max(np.mod(np.angle(el_rot[:,:,idx_lo:idx_hi].T, deg=True),360))>max_val:
				max_val=np.max(np.mod(np.angle(el_rot[:,:,idx_lo:idx_hi].T, deg=True),360))

			if np.min(np.mod(np.angle(el_rot[:,:,idx_lo:idx_hi].T, deg=True),360))<min_val:
				min_val=np.min(np.mod(np.angle(el_rot[:,:,idx_lo:idx_hi].T, deg=True),360))


			plt.suptitle(freq_field_phase_titles[m], fontsize=24)

			x_comp_ph.imshow(	np.mod(np.angle(el[:,0,idx_lo:idx_hi].T, deg=True),360), origin="lower", vmin=min_val, vmax=max_val,
					extent=[Bmin, Bmax, fft_freq[idx_lo], fft_freq[idx_hi]], aspect=aspect_ratio, cmap='hsv')
			x_comp_ph.set_title("Phase angle (deg), x-component", fontsize=24)


			y_comp_ph.imshow(	np.mod(np.angle(el[:,1,idx_lo:idx_hi].T, deg=True), 360), origin="lower", vmin=min_val, vmax=max_val,
					extent=[Bmin, Bmax, fft_freq[idx_lo], fft_freq[idx_hi]], aspect=aspect_ratio, cmap='hsv')
			y_comp_ph.set_title("Phase angle (deg), y-component", fontsize=24)


			z_comp_ph.imshow(	np.mod(np.angle(el[:,2,idx_lo:idx_hi].T, deg=True),360), origin="lower",vmin=min_val, vmax=max_val,
						extent=[Bmin, Bmax, fft_freq[idx_lo], fft_freq[idx_hi]], aspect=aspect_ratio, cmap='hsv')
			z_comp_ph.set_title("Phase angle (deg), z-component", fontsize=24)

			im=phi_comp_ph.imshow(	np.mod(np.angle(el_rot[:,0,idx_lo:idx_hi].T, deg=True),360), origin="lower", vmin=min_val, vmax=max_val,
					extent=[Bmin, Bmax, fft_freq[idx_lo], fft_freq[idx_hi]], aspect=aspect_ratio, cmap='hsv')
			phi_comp_ph.set_title(r"Phase angle (deg), $\phi$-component", fontsize=24)



			#plt.suptitle(plot_titles[m])

			fig7.text(0.5, 0.04, r'External Field $\frac{H}{H_D}$', ha='center', fontsize=24)
			fig7.text(0.04, 0.5, r'Frequency (GHz)', va='center', rotation='vertical', fontsize=24)	

			cbar_ax=fig7.add_axes([0.85, 0.15, 0.05, 0.7])
			fig7.colorbar(im, cax=cbar_ax)
			fig7.set_size_inches(16,9)	
			if save_plot:
			
				fig7.set_size_inches(16,9)	
				plt.savefig(save_dir, format='png', dpi=100)	
				plt.close()
			else:
				plt.show()
				
		aspect_ratio=(Bmax-Bmin)/18.0
		depth_field=np.swapaxes(all_ffts, 0,2 )
		depth_field_rot=np.swapaxes(all_ffts_rot, 0,2 )
		freq_list=[]
		###	FFT ARRAY SHAPE CORRESPONDS TO (frequency;	Mag vector component;	field value;	z position)
		for m, el in enumerate(depth_field[idx_lo:idx_hi]):
			for dim in el:
				dim/=np.max(np.abs(dim))

			el_rot=depth_field_rot[m]

			for dim in el_rot:
				dim/=np.max(np.abs(dim))

			if np.abs(fft_freq[m]%0.5-0.5)<0.1 and np.round(fft_freq[m]*2)/2.0 not in freq_list:	
	
				this_freq=np.round(fft_freq[m]*2)/2.0
				freq_list.append(this_freq)
					
				fig8, ((x_comp_re, y_comp_re, z_comp_re), (x_comp_im, y_comp_im, z_comp_im), (x_comp_po, y_comp_po, z_comp_po)) = plt.subplots(3, 3)#, sharex='col', sharey='row')
				plot_title="Resonance vs magnetic field at "+str(this_freq)+"GHz"
				plot_filename="res_depth_Bmin-Bmax-this_freq-"+str(Bmin)+"-"+str(Bmax)+"-"+str(this_freq)+".png"

				save_dir=fs.join(save_directory_resonances, plot_filename)

				max_val= np.max(np.absolute(el[:,:,idx_lo:idx_hi].T))
				min_val= -max_val

				plt.suptitle(plot_title, fontsize=24)

				x_comp_re.imshow(	np.real(el[0,:,:]).T, origin="lower", vmin=min_val, vmax=max_val,
							extent=[Bmin, Bmax, 0, 18], aspect=aspect_ratio, cmap='Spectral')
				x_comp_re.set_title("Re{x-component}", fontsize=24)


				y_comp_re.imshow(	np.real(el[1,:,:]).T, origin="lower", vmin=min_val, vmax=max_val,
							extent=[Bmin, Bmax, 0, 18], aspect=aspect_ratio, cmap='Spectral')
				y_comp_re.set_title("Re{y-component}", fontsize=24)


				z_comp_re.imshow(	np.real(el[2,:,:]).T, origin="lower",vmin=min_val, vmax=max_val,
							extent=[Bmin, Bmax, 0, 18], aspect=aspect_ratio, cmap='Spectral')
				z_comp_re.set_title("Re{z-component}", fontsize=24)

				x_comp_im.imshow(	np.imag(el[0,:,:]).T, origin="lower", vmin=min_val, vmax=max_val,
						extent=[Bmin, Bmax, 0, 18], aspect=aspect_ratio, cmap='Spectral')
				x_comp_im.set_title("Im{x-component}", fontsize=24)


				y_comp_im.imshow(	np.imag(el[1,:,:]).T, origin="lower", vmin=min_val, vmax=max_val,
							extent=[Bmin, Bmax, 0, 18], aspect=aspect_ratio, cmap='Spectral')
				y_comp_im.set_title("Im{y-component}", fontsize=24)


				z_comp_im.imshow(	np.imag(el[2,:,:]).T, origin="lower",vmin=min_val, vmax=max_val,
							extent=[Bmin, Bmax, 0, 18], aspect=aspect_ratio, cmap='Spectral')
				z_comp_im.set_title("Im{z-component}", fontsize=24)

				x_comp_po.imshow(	np.absolute(el[0,:,:]).T, origin="lower",vmin=min_val, vmax=max_val,
							extent=[Bmin, Bmax, 0, 18], aspect=aspect_ratio, cmap='Spectral')
				x_comp_po.set_title("Power spectrum, x-component", fontsize=24)


				y_comp_po.imshow(	np.absolute(el[1,:,:]).T, origin="lower",vmin=min_val, vmax=max_val,
							extent=[Bmin, Bmax, 0, 18], aspect=aspect_ratio, cmap='Spectral')
				y_comp_po.set_title("Power spectrum, y-component", fontsize=24)


				im=z_comp_po.imshow(	np.absolute(el[2,:,:]).T, origin="lower",vmin=min_val, vmax=max_val,
							extent=[Bmin, Bmax, 0, 18], aspect=aspect_ratio, cmap='Spectral')
				z_comp_po.set_title("Power spectrum, z-component", fontsize=24)

				fig8.text(0.5, 0.04, r'External Field $\frac{H}{H_D}$', ha='center', fontsize=24)
				fig8.text(0.04, 0.5, r'Depth (nm)', va='center', rotation='vertical', fontsize=24)	

				cbar_ax=fig8.add_axes([0.91, 0.15, 0.05, 0.7])
				fig8.colorbar(im, cax=cbar_ax)
				fig8.set_size_inches(16,9)	
				if save_plot:
				
					fig8.set_size_inches(16,9)	
					plt.savefig(save_dir, format='png', dpi=100)	
					plt.close()
				else:
					plt.show()

				fig9, ((phi_comp_re, z2_comp_re), (phi_comp_im, z2_comp_im), (phi_comp_po, z2_comp_po)) = plt.subplots(3, 2)#, sharex='col', sharey='row')
				plot_title="Resonance vs magnetic field at "+str(this_freq)+"GHz: Rotated coordinate system"
				plot_filename="res_depth_rot_Bmin-Bmax-this_freq-"+str(Bmin)+"-"+str(Bmax)+"-"+str(this_freq)+".png"

				save_dir=fs.join(save_directory_resonances, plot_filename)

				max_val= np.max(np.absolute(el_rot[:,:,idx_lo:idx_hi].T))
				min_val= -max_val

				plt.suptitle(plot_title, fontsize=24)

				phi_comp_re.imshow(	np.real(el_rot[0,:,:].T), origin="lower", vmin=min_val, vmax=max_val,
							extent=[Bmin, Bmax, 0, 18], aspect=aspect_ratio, cmap='Spectral')
				phi_comp_re.set_title(r"Re{$\phi$-component}", fontsize=24)


				z2_comp_re.imshow(	np.real(el_rot[1,:,:].T), origin="lower", vmin=min_val, vmax=max_val,
							extent=[Bmin, Bmax, 0, 18], aspect=aspect_ratio, cmap='Spectral')
				z2_comp_re.set_title("Re{z-component}", fontsize=24)



				phi_comp_im.imshow(	np.imag(el_rot[0,:,:].T), origin="lower", vmin=min_val, vmax=max_val,
							extent=[Bmin, Bmax, 0, 18], aspect=aspect_ratio, cmap='Spectral')
				phi_comp_im.set_title(r"Im{$\phi$-component}", fontsize=24)


				z2_comp_im.imshow(	np.imag(el_rot[1,:,:].T), origin="lower", vmin=min_val, vmax=max_val,
							extent=[Bmin, Bmax, 0, 18], aspect=aspect_ratio, cmap='Spectral')
				z2_comp_im.set_title("Im{z-component}", fontsize=24)


				phi_comp_po.imshow(	np.absolute(el_rot[0,:,:].T), origin="lower",vmin=min_val, vmax=max_val,
							extent=[Bmin, Bmax, 0, 18], aspect=aspect_ratio, cmap='Spectral')
				phi_comp_po.set_title(r"Power spectrum, $\phi$-component", fontsize=24)


				im=z2_comp_po.imshow(	np.absolute(el_rot[1,:,:].T), origin="lower",vmin=min_val, vmax=max_val,
							extent=[Bmin, Bmax, 0, 18], aspect=aspect_ratio, cmap='Spectral')
				z2_comp_po.set_title("Power spectrum, z-component", fontsize=24)


				fig9.text(0.5, 0.04, r'External Field $\frac{H}{H_D}$', ha='center', fontsize=24)
				fig9.text(0.04, 0.5, r'Depth (nm)', va='center', rotation='vertical', fontsize=24)	

				cbar_ax=fig9.add_axes([0.91, 0.15, 0.05, 0.7])
				fig9.colorbar(im, cax=cbar_ax)
				fig9.set_size_inches(16,9)	
				if save_plot:
				
					fig9.set_size_inches(16,9)	
					plt.savefig(save_dir, format='png', dpi=100)	
					plt.close()
				else:
					plt.show()
				
				fig10, ((x_comp_ph, y_comp_ph), (z_comp_ph, phi_comp_ph)) = plt.subplots(2, 2)#, sharex='col', sharey='row')
				plot_title="Phase angle at depth vs magnetic field at "+str(this_freq)+"GHz"
				plot_filename="phase_depth_Bmin-Bmax-this_freq-"+str(Bmin)+"-"+str(Bmax)+"-"+str(this_freq)+".png"

				save_dir=fs.join(save_directory_resonances, plot_filename)

				max_val= np.max(np.mod(np.angle(el, deg=True),360))
				min_val= np.min(np.mod(np.angle(el, deg=True),360))

				if np.max(np.mod(np.angle(el_rot, deg=True),360))>max_val:
					max_val=np.max(np.mod(np.angle(el_rot, deg=True),360))
				
				if np.min(np.mod(np.angle(el_rot, deg=True), 360))<min_val:
					min_val=np.min(np.mod(np.angle(el_rot, deg=True),360))
				
				plt.suptitle(plot_title, fontsize=24)


				x_comp_ph.imshow(	np.mod(np.angle(el[0,:,:], deg=True).T, 360), origin="lower", vmin=min_val, vmax=max_val,
							extent=[Bmin, Bmax, 0, 18], aspect=aspect_ratio, cmap='hsv')
				x_comp_ph.set_title("Phase angle (deg), x-component", fontsize=24)


				y_comp_ph.imshow(	np.mod(np.angle(el[1,:,:], deg=True).T, 360), origin="lower", vmin=min_val, vmax=max_val,
							extent=[Bmin, Bmax, 0, 18], aspect=aspect_ratio, cmap='hsv')
				y_comp_ph.set_title("Phase angle (deg), y-component", fontsize=24)


				z_comp_ph.imshow(	np.mod(np.angle(el[2,:,:], deg=True).T, 360), origin="lower",vmin=min_val, vmax=max_val,
							extent=[Bmin, Bmax, 0, 18], aspect=aspect_ratio, cmap='hsv')
				z_comp_ph.set_title("Phase angle (deg), z-component ", fontsize=24)


				im=phi_comp_ph.imshow(	np.mod(np.angle(el_rot[0,:,:], deg=True).T, 360), origin="lower", vmin=min_val, vmax=max_val,
						extent=[Bmin, Bmax, 0, 18], aspect=aspect_ratio, cmap='hsv')
				phi_comp_ph.set_title(r"Phase angle (deg), $\phi$ component", fontsize=24)

				fig10.text(0.5, 0.04, r'External Field $\frac{H}{H_D}$', ha='center', fontsize=24)
				fig10.text(0.04, 0.5, r'Depth (nm)', va='center', rotation='vertical', fontsize=24)	

				cbar_ax=fig10.add_axes([0.85, 0.15, 0.05, 0.7])
				fig10.colorbar(im, cax=cbar_ax)
				fig10.set_size_inches(16,9)	
				if save_plot:
				
					fig10.set_size_inches(16,9)	
					plt.savefig(save_dir, format='png', dpi=100)	
					plt.close()
				else:
					plt.show()
				
		print("Fourier Analysis done")

	except IOError as IO:	
		print("No such file!")




def main(data_dir):
	conjs=True
	save_plot=True

	Bmin=0.0
	Bmax=0.95
	Bstep=0.05
	tmin=0.0
	tmax=4450
	tstep=8902


	#"""
	###		PREPARE FILESYSTEM FOR SAVING PLOTS	###
	save_directory=data_dir[:-4]+"_plots_conjugates"#_h_trunc2"
	save_directory_resonances=save_directory+"/fourier/"
	save_directory_displacements=save_directory+"/displacements/"

	ls=os.listdir(fs.join(data_dir, '..'))
	if save_directory.split('/')[-1] not in ls:
		print("creating directory ",save_directory)
		os.mkdir(save_directory)	
		os.mkdir(save_directory_displacements)
		os.mkdir(save_directory_resonances)
	###########################################################



	###			ACTUAL				###
	ovf_reader = OVFReader(data_dir)
	ovf_reader.import_dir()
	ovf_files=ovf_reader.mag_data

	fourier_displacements(ovf_files, Bmin, Bmax, Bstep, tmin, tmax, tstep)
	#helicoid_resonant_frequencies_plotter(ovf_files, Bmin, Bmax, Bstep, tmin, tmax, tstep)

	###							###
	#"""
	# THIS ONE FLIPS THE SIGNS OF ALL THE IMAGINARY PARTS OF THE COMPLEX NUMBERS IF THEYRE LESS THAN ZERO

	print("Done!")

if __name__ == '__main__':
	main(*sys.argv[1:])


