import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


forcing = "output_forcing"
no_forcing = "output_no_forcing"
cols_idx = [5, 18]

if forcing:
	df_forc = pd.read_csv(forcing+".csv", header=None, usecols=cols_idx)

if no_forcing:
	df_no_forc = pd.read_csv(no_forcing+".csv", header=None, usecols=cols_idx)


np_forc = np.array(df_forc)
np_no_forc = np.array(df_no_forc)

np_forc = np_forc[np_forc[:, 1] == "I", :]
np_no_forc = np_no_forc[np_no_forc[:, 1] == "I", :]

it_forc = np_forc[1:, 0] - np_forc[:-1, 0]
it_no_forc = np_no_forc[1:, 0] - np_no_forc[:-1, 0]

plt.figure()
plt.title("Forcing an I-frame every 2s [GOP=16]", fontsize=12)
plt.stem(np_forc[:50, 0])
plt.grid()
plt.ylabel("Time [s]")
plt.xlabel("I-frame")
plt.savefig(forcing+".png", dpi=500)
plt.show()

plt.figure()
plt.title("No Forcing of I-frames [GOP=16]", fontsize=12)
plt.stem(np_no_forc[:50, 0])
plt.grid()
plt.xlabel("I-frame")
plt.ylabel("Time [s]")
plt.savefig(no_forcing+".png", dpi=500)
plt.show()


plt.figure()
plt.title("CDF of the inter-arrival time between I-frames [GOP=16]", fontsize=12)
plt.plot(np.sort(it_forc), np.linspace(0,1,it_forc.shape[0]), linewidth=2, label="Forcing an I-frame every 2s")
plt.plot(np.sort(it_no_forc), np.linspace(0,1,it_no_forc.shape[0]), linewidth=2, label="No forcing of I-frames")
plt.grid()
plt.ylabel("CDF")
plt.xlabel("Inter-arrival time")
plt.xlim([0, 1])
plt.legend()
plt.savefig("inter_arrival_iframe.png", dpi=500)
plt.show()
