import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc
import os

#sort out where files are, should be numbered in order smallest to highest conc
directory = 'data/2023.03.17/reflect'
file_ex = pd.read_excel(directory+'/1_0g_reflect probe.xlsx')
wavelengths = file_ex.iloc[5:,0].to_numpy() #array of wavelengths

#make a list of all the np arrays
spectras_list = []
s_av_list = []
for filename in sorted(os.listdir(directory)):
    fn = os.path.join(directory,filename)
    df = pd.read_excel(fn)
    s = df.iloc[5:,1:6].to_numpy() #first 5 readings
    s_avg = np.average(s,1)
    s_av_list.append(s_avg)
    spectras_list.append(s) #list of np arrays


I_all = []
for spec in spectras_list:
    for i in range(spec.shape[-1]):
        I_skl = auc(wavelengths,spec[:,i])
        I_all.append(I_skl)

C = np.multiply([0,0,0,0,0,0.5,0.5,0.5,0.5,0.5,1,1,1,1,1,2,2,2,2,2,5,5,5,5,5],5)

plt.figure(figsize=[12,8])
plt.plot(C, I_all,'.')
plt.title('Intensity with concentration Bentonite Clay')
plt.xlabel('Conc. g/L')
plt.ylabel('I (arb.)')
plt.show()

s_av = np.array(s_av_list)
s_av_cl = np.divide((s_av - s_av[4,:]), np.max(s_av))

plt.figure(figsize=[12,8])
for spec in s_av_cl:
    plt.plot(wavelengths, spec)
plt.title('1st exp Intensity with Wavelength, cleaned + norm')
plt.xlabel('$\lambda$ (nm)')
plt.ylabel('I (arb.)')
plt.legend(['0', '2.5', '5', '10', '25'], title = 'C (g/L)')
plt.show()

# results = pd.DataFrame({'C (g/200ml)':C, 'I':I_all})
# results.to_csv('data/results/2023_03_17_I.csv', index=False)
