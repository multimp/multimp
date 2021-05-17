import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

##ACC
path = 'E:/UNC-CS-Course/COMP 790-166/project/multimp/results/metrics/result_acc.xlsx'
df_frame = pd.read_csv(path)
ACC = df_frame['ACC'].values
missing_rate = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
plt.plot(missing_rate, ACC[0:5],color='blue',label='Mean')
plt.plot(missing_rate, ACC[5:10], color='pink',label='KNN')
plt.plot(missing_rate, ACC[10:15], color='red',label='Multi-View CPM-Nets with GANs')
plt.plot(missing_rate, ACC[15:20], color='yellow',label='Multi-View CPM-Nets without GANs')
plt.plot(missing_rate, ACC[20:25], color='green',label='Single-View CPM-Nets with GANs')
plt.plot(missing_rate, ACC[25:30], color='purple',label='Single-View CPM-Nets without GANs')
plt.ylabel('Accuracy')
plt.xlabel('Missing Rate')
plt.legend()
plt.savefig('E:/UNC-CS-Course/COMP 790-166/project/plots/Accuracy.png')
plt.show()
plt.close()











###########################################



path = 'E:/UNC-CS-Course/COMP 790-166/project/multimp/results/metrics/result_all.xlsx'
df_frame = pd.read_excel(path, sheet_name='ADNI')

missing_rate = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
RMSE = df_frame['RMSE'].values
AUC = df_frame['AUC'].values
plt.plot(missing_rate, RMSE[0:5], color='blue',label='Mean')
plt.plot(missing_rate, RMSE[5:10], color='pink',label='KNN')
plt.plot(missing_rate, RMSE[10:15], color='red',label='Multi-View CPM-Nets with GANs')
plt.plot(missing_rate, RMSE[15:20], color='yellow',label='Multi-View CPM-Nets without GANs')
plt.plot(missing_rate, RMSE[20:25], color='green',label='Single-View CPM-Nets with GANs')
plt.plot(missing_rate, RMSE[25:30], color='purple',label='Single-View CPM-Nets without GANs')
plt.ylabel('RMSE')
plt.xlabel('Missing Rate')
plt.legend()
plt.savefig('E:/UNC-CS-Course/COMP 790-166/project/plots/RMSE.png')

plt.show()
plt.close()

plt.plot(missing_rate, AUC[0:5],color='blue',label='Mean')
plt.plot(missing_rate, AUC[5:10], color='pink',label='KNN')
plt.plot(missing_rate, AUC[10:15], color='red',label='Multi-View CPM-Nets with GANs')
plt.plot(missing_rate, AUC[15:20], color='yellow',label='Multi-View CPM-Nets without GANs')
plt.plot(missing_rate, AUC[20:25], color='green',label='Single-View CPM-Nets with GANs')
plt.plot(missing_rate, AUC[25:30], color='purple',label='Single-View CPM-Nets without GANs')
plt.ylabel('AUC')
plt.xlabel('Missing Rate')
plt.legend()
plt.savefig('E:/UNC-CS-Course/COMP 790-166/project/plots/AUC.png')
plt.show()
plt.close()
