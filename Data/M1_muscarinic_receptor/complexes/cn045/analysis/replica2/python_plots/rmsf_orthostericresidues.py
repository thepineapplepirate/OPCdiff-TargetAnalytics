import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import os

title_font = {'fontname':'Helvetica', 'size':'14', 'color':'black', 'weight':'normal','verticalalignment':'bottom'} # Bottom vertical alignment for more space
lable_font = {'fontname':'Helvetica', 'size':'12'}
fontP = FontProperties()
fontP.set_size('small')


data_0 = np.genfromtxt('orthosteric_wholeresidueRMSF.xvg', unpack=True)
plt.bar(data_0[0,:], data_0[1,:]*10, color='grey',label='Apo', width = 6)

data_1 = np.genfromtxt('orthosteric_wholeresidueRMSF_cn045.xvg', unpack=True)
plt.bar(data_1[0,:], data_1[1,:]*10, color='skyblue',label='CN045 bound', width = 6)
#plt.bar([86, 87, 90, 138, 170, 386, 387, 409, 413], data_1[1,:]*10, color='skyblue',label='CN045 bound', width = 4)


#plt.xlim([0.1,596])
plt.title("RMSF of orthosteric site residues", **title_font)
#plt.ylim([1,10])
plt.xlabel("Residue Number",**lable_font)
plt.ylabel("RMSF ($\AA$)",**lable_font)

plt.tick_params(axis='x', labelsize=10)
plt.tick_params(axis='y', labelsize=10)


plt.legend(loc='upper right',fancybox=True,prop=fontP)

plt.tight_layout()

plt.show()
