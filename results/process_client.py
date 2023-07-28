import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

latency_fn = "rx_feedbacks"

if latency_fn:
    df = pd.read_csv(latency_fn+".csv", header=None)

df_np = np.array(df)
latency_ms = df_np[:, 2]/1e6
tot_samples = latency_ms.shape[0]
print("The total number of samples is: ", tot_samples)

plt.figure()
dict_plt = plt.boxplot(latency_ms, notch=True, showmeans=True, meanline=True)
outliers = np.array(dict_plt['fliers'][0].get_data())
num_outliers = outliers.shape[1]
plt.grid()
plt.ylabel("Network Latency [ms]")
plt.title("Latency [percetange of outliers= "+str(round(num_outliers/tot_samples*100, 2))+"%]", fontsize=12)
# plt.legend()
plt.savefig(latency_fn+".png", dpi=500)
plt.show()
