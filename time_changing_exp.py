import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc
import glob
import os
from matplotlib.gridspec import GridSpec

#dark file in date directory
directory = 'data/2023.03.28'
dark_files = glob.glob(directory+'/*.xlsx') #just take one of the darks
darks = []
wavelengths = []
for f in dark_files:
    df = pd.read_excel(f)
    darks.append(df.iloc[5:,1].to_numpy())
    wavelengths.append(df.iloc[5:,0].to_numpy())

#get the reference files, calculate max value and I0
ref_files = sorted(glob.glob(directory+'/0.0g/*.xlsx', recursive=True))
repeats = 5 #how many readings to average over

ref_max = [] #list of maximums for normalising later
I0 = [] #max intensity for normalising later
for i in range(len(ref_files)):
    df = pd.read_excel(ref_files[i])
    s = df.iloc[5:,1:repeats+1].to_numpy() 
    s_avg = np.average(s,1) #average of the repeat readings
    # s_avg = (avg_spec - dark[:,None]) 
    s_max = np.max(s_avg) #find max value for using later
    ref_max.append(s_max)
    I = auc(wavelengths[i],s_avg)
    I0.append(I)


#get the experiment files
exp_files = sorted(glob.glob(directory+'/ChangingConcExp/*.xlsx', recursive=True))

#get time gaps
t = 0
I_changing = []
for i in range(len(exp_files)):
    df = pd.read_excel(exp_files[i]) #they are big, this takes a while - better way?

    #get time gaps
    t_gap = (df.iloc[1,1]/1000)*df.iloc[2,1] #int time and num av
    if t_gap > t:
        t = t_gap #saves the larger int time

    #compute intensity measurements
    s_data = df.iloc[5:,1:].to_numpy() #takes all spectra data out of df
    n = s_data.shape[-1] #number of measurements
    I = []
    for j in range(n): #loop through each column (timestep)
        I_step = auc(wavelengths[i],s_data[:,j]) #calculate intensity
        I.append(I_step)
    I = np.array(I)
    I_changing.append(I)

t_steps = np.arange(1,n+1)*t

trans = I_changing[0]*100/(I0[0])
ab = 2 - np.log10(trans)


conc = (ab - c)/m
#plotting
fig = plt.figure(figsize=[16,7])
fig.suptitle('Experiment varying bentonite concentration with time')
G = fig.add_gridspec(1, 2, hspace=0.3, wspace = 0.3)
ax1 = plt.subplot(G[0,0])
ax2 = plt.subplot(G[0,1])

ax1.plot(t_steps, trans, 'r-')
ax1.set_title('% transmittance with time')
ax1.set_ylabel('% Transmittance')
ax1.set_xlabel('Time (s)')

ax2.plot(t_steps, conc.T, 'b-')
ax2.set_title('Concentration with time')
ax2.set_ylabel('Conc (g/L)')
ax2.set_xlabel('Time (s)')
lines = np.array([10,5,3.33,2.5,2])
for l in lines:
    ax2.axhline(y=l, color = 'k', linestyle = '--')

plt.tight_layout()
plt.show()




# T = np.divide(I_all, np.nanmax(I_all))*100 #percentage transmission
# A = 2 - np.log10(np.abs(T))

# #plotting
# fig1 = plt.figure(figsize=[16,7])
# G = fig1.add_gridspec(2, 4, hspace=0.5, wspace = 0.5)
# ax1 = plt.subplot(G[:,0:3])
# ax2 = plt.subplot(G[0,3])
# ax3 = plt.subplot(G[1,3])

# ax1.plot(wavelengths, avg_spec_norm, label = C_name)
# ax1.plot(wavelengths, avg_spec_norm, label = C_name)
# ax1.set_title('Absorbance Spectra')
# ax1.set_ylabel('Normalised scope')
# ax1.set_xlabel('$\lambda$ (nm)')
# ax1.legend()

# ax2.plot(C_all, T,'.')
# ax2.set_title('Transmittance')
# ax2.set_ylabel('% Transmittance')
# ax2.set_xlabel('Concentration (g/L)')

# ax3.plot(C_all, A,'.')
# ax3.set_title('Absorbance')
# ax3.set_ylabel('Absorbance')
# ax3.set_xlabel('Concentration (g/L)')

# # plt.tight_layout(pad=1.1)
# plt.show()


# results = pd.DataFrame({'C (g/L)':C_all, 'I':I_all})
# results.to_csv('data/results/2023_03_27_I_exp3.csv', index=False)
