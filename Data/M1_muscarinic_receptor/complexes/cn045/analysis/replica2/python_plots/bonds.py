import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import os

title_font = {'fontname':'Helvetica', 'size':'14', 'color':'black', 'weight':'normal','verticalalignment':'bottom'} # Bottom vertical alignment for more space
lable_font = {'fontname':'Helvetica', 'size':'12'}
fontP = FontProperties()
fontP.set_size('small')


rawdata = np.genfromtxt('multiplot.dat', unpack=True)
plt.plot(rawdata[0,:]/5, rawdata[3,:], 'black',label='Asn387:HD22--N1:cn045(Hbond)', linewidth = 1)
plt.plot(rawdata[0,:]/5, rawdata[4,:], 'red',label='Tyr386:HH--N1:cn045(Hbond)', linewidth = 1)
plt.plot(rawdata[0,:]/5, rawdata[1,:], 'green',label='Tyr87:CE1--C5:cn045(hydrophobic and $\pi$-$\pi$ stacking)', linewidth = 1)
plt.plot(rawdata[0,:]/5, rawdata[2,:], 'blue',label='Trp138:CZ2--C5:cn045(hydrophobic and $\pi$-$\pi$ stacking)', linewidth = 1)
plt.plot(rawdata[0,:]/5, rawdata[5,:], 'orange',label='Ala177:CB--C5:cn045(hydrophobic)', linewidth = 1)
plt.plot(rawdata[0,:]/5, rawdata[6,:], 'purple',label='Ala177:CB--C11:cn045(hydrophobic)', linewidth = 1)




plt.xlim([0.1,1000])
#plt.title("", **title_font)
plt.ylim([1,14])
plt.xlabel("Time (ns)",**lable_font)
plt.ylabel("Bond length ($\AA$)",**lable_font)


plt.tick_params(axis='x', labelsize=10)
plt.tick_params(axis='y', labelsize=10)


plt.legend(loc='upper right',fancybox=True,prop=fontP)

plt.tight_layout()

plt.show()
