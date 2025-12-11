import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import os

title_font = {'fontname':'Helvetica', 'size':'18', 'color':'black', 'weight':'normal','verticalalignment':'bottom'} # Bottom vertical alignment for more space
lable_font = {'fontname':'Helvetica', 'size':'16'}
fontP = FontProperties()
fontP.set_size('14')

receptor_name = 'M1'
drug_name = 'Clemastine'

data_0 = np.genfromtxt('replica1/galaxy_analysis/rmsd_clem.tabular', unpack=True)
plt.plot(data_0[0,:]/5, data_0[1,:], 'black',label='Replica 1', linewidth = 1)

data_1 = np.genfromtxt('replica2/galaxy_analysis/rmsd_clem.tabular', unpack=True)
plt.plot(data_1[0,:]/5, data_1[1,:], 'blue',label='Replica 2', linewidth = 1)

data_2 = np.genfromtxt('replica3/galaxy_analysis/rmsd_clem.tabular', unpack=True)
plt.plot(data_2[0,:]/5, data_2[1,:], 'red',label='Replica 3', linewidth = 1)


#plt.xlim([0.1,596])
plt.title(f"RMSD of {drug_name} ({receptor_name} Complex)", **title_font, fontsize=18)
plt.ylim([0,5])
plt.xlabel("Time (ns)",**lable_font)
plt.ylabel("RMSD ($\AA$)",**lable_font)


plt.tick_params(axis='x', labelsize=14)
plt.tick_params(axis='y', labelsize=14)


plt.legend(loc='upper right',fancybox=True,prop=fontP)

plt.tight_layout()
plt.savefig(f"{drug_name}_RMSD_{receptor_name}_complex.png", dpi=600, bbox_inches='tight')
#plt.show()
