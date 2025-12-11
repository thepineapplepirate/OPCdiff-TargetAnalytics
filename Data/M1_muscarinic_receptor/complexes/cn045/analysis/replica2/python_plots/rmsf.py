import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import os

title_font = {'fontname':'Helvetica', 'size':'14', 'color':'black', 'weight':'normal','verticalalignment':'bottom'} # Bottom vertical alignment for more space
lable_font = {'fontname':'Helvetica', 'size':'12'}
fontP = FontProperties()
fontP.set_size('small')


data_0 = np.genfromtxt('Galaxy43-[GROMACS_calculation_of_RMSF_on_data_1_and_data_25].xvg', unpack=True)
plt.plot(data_0[0,:]/16.198, data_0[1,:]*10, 'grey',label='MERS Helicase', linewidth = 2)

#data_1 = np.genfromtxt('rmsf_sarscov2.xvg', unpack=True)
#plt.plot(data_1[0,:]/15.567, data_1[1,:]*10, 'skyblue',label='SARS-CoV-2 Helicase', linewidth = 2)



#plt.xlim([0.1,596])
plt.title("RMSF of all C-alpha", **title_font)
#plt.ylim([1,10])
plt.xlabel("Residue Number",**lable_font)
plt.ylabel("RMSF ($\AA$)",**lable_font)


plt.tick_params(axis='x', labelsize=10)
plt.tick_params(axis='y', labelsize=10)


plt.legend(loc='upper right',fancybox=True,prop=fontP)

plt.tight_layout()

plt.show()
