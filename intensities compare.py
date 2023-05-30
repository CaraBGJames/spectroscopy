import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.linear_model import LinearRegression

#sort out where files are, should be numbered in order smallest to highest conc
directory = 'data/results'
I_df = pd.read_csv(directory+'/Best_dip_data.csv')#.sort_values(by=['C'])
# for col in I_df:
#     minimum = np.min(I_df[col])
#     if minimum < 0:
#         I_df[col] = I_df[col] + minimum + 1
C = I_df['C'].to_numpy()
I = I_df.iloc[:,1:4].to_numpy()
# I[I<0] = np.nan
T = np.divide(I, np.nanmax(I))*100 #percentage transmission
A = 2 - np.log10(T)

#linear regression
y = np.nansum(A,axis=1) #get rid of 0 values due to high conc
X = C[y != 0].reshape(-1,1)
Y = y[y != 0].reshape(-1,1)
lm = LinearRegression()
lm.fit(X,Y)
c = lm.intercept_
m = lm.coef_
y_pred = lm.predict(X)


#plotting
fig = plt.figure(figsize=[16,7])
fig.suptitle('Bentonite concentration calibration curve with dip probes')
names = ['exp1','exp2','exp3']#, 'exp4']
G = fig.add_gridspec(1, 2, hspace=0.3, wspace = 0.3)
ax1 = plt.subplot(G[0,0])
ax2 = plt.subplot(G[0,1])
ax1.plot(C, T,'.', label = names)
ax1.set_title('Transmission')
ax1.set_ylabel('% transmission')
ax1.set_xlabel('Conc (g/L)')
ax1.legend()

ax2.plot(C, A, '.')
ax2.plot(X,y_pred, 'k-')
ax2.set_title('Absorbance')
ax2.set_ylabel('Absorbance')
ax2.set_xlabel('Conc (g/L)')

plt.tight_layout()
plt.show()


# plt.figure(figsize=[12,8])
# plt.plot(C, T,'.')
# plt.title('Percentage transmittance')
# plt.xlabel('Conc. g/L')
# plt.ylabel('I (norm)')
# plt.show()

# work out slope of graph for absorbance --> conc



