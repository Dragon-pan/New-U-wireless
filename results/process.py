import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_rate_time(df):
    df_np = np.array(df)
    # print(df_np.shape)

    time = df_np[:,0]
    size = df_np[:,1]
    time = time - time[0] # normalize time with respect to first time value
    if var:
        targ = df_np[:,2] 

    step=1
    d_time = np.arange(0, np.max(time)+step, step)

    # average bitrate
    tot_rate = np.sum(size)*8
    max_time = np.max(time)
    av_bitrate = tot_rate/max_time

    d_rate = np.zeros(len(d_time)-1)
    t_rate = np.zeros(len(d_time)-1)
    for i,d in enumerate(d_time[1:]):
        pkts = size[(time >= d_time[i]) & (time < d)]
        if var:
            t_r  = targ[(time >= d_time[i]) & (time < d)]
        #print(d_time[i])
        #print(d)
        #print("-"*20)
        d_rate[i] = np.sum(pkts)*8/step
        if var:
            t_rate[i] = np.mean(t_r)

    return d_time, d_rate, av_bitrate, tot_rate, max_time, t_rate, time


tx_filename = None
rx_filename = "rx_pkt_trace"
est_band_filename = "est_band"
var = False
fec_overhead = 1.0

num_rx_pkts = None
if rx_filename:
    df_rx = pd.read_csv(rx_filename+".csv", header=None)
    num_rx_pkts = df_rx.shape[0]
    time_rx, rate_rx, _, tot_rate_rx, _, _, tot_time_rx = get_rate_time(df_rx)
    print("The number of RX packets is: ", num_rx_pkts)

if tx_filename:
    df_tx = pd.read_csv(tx_filename+".csv", header=None)
    num_tx_pkts = df_tx.shape[0]
    time_tx, rate_tx, av_bitrate_tx, _, max_time_tx, _, _ = get_rate_time(df_tx)
    print("The number of TX packets is: ", num_tx_pkts)

if tx_filename and rx_filename:
    print("The packet error rate is: {:.2f} %".format((1 - float(num_rx_pkts)/num_tx_pkts)*100))

'''plt.figure()
plt.title("The average bitrate is: "+str(round(av_bitrate_tx/1e6, 2))+" Mbps", fontsize=12)
plt.plot(time_tx[:-1], rate_tx/1e6, linewidth=2, label="Measured Bitrate")
if var:
    plt.plot(time_tx[:-1], t_rate, '--', linewidth=1.5, label="Target Bitrate")
plt.grid()
plt.xlabel("Time [s]")
plt.ylabel("Encoder Rate [Mbps]")
plt.ylim([0, 22])
plt.legend(loc="best")
#plt.savefig(tx_filename+".png", dpi=500)
#plt.show()

plt.figure()
plt.title("The average bitrate is: "+str(round(tot_rate_rx/max_time_tx/1e6, 2))+" Mbps", fontsize=12)
plt.plot(time_rx[:-1], rate_rx/1e6, linewidth=2, label="Measured Bitrate")
plt.grid()
plt.xlabel("Time [s]")
plt.ylabel("Rx Rate [Mbps]")
plt.ylim([0, 22])
plt.legend(loc="best")
plt.savefig(rx_filename+".png", dpi=500)
#plt.show()'''

# PLOT INTER-ARRIVAL TIME OF RX PACKETS
itime_rx_pkts = tot_time_rx[1:] - tot_time_rx[:-1]

plt.figure()
plt.title("CDF of the inter-arrival time between RX packets", fontsize=12)
plt.plot(np.sort(itime_rx_pkts)*1e3, np.linspace(0,1,itime_rx_pkts.shape[0]), linewidth=2)
plt.grid()
plt.xlabel("Time [ms]")
plt.ylabel("CDF")
plt.xlim([-1, 50])
# plt.savefig("inter_arrival_rx.png", dpi=500)

if est_band_filename:
    df_band = pd.read_csv(est_band_filename+".csv", header=None)

df_band = np.array(df_band)
time_est = df_band[:, 0] - df_band[0, 0]
band_est = df_band[:, 1]
# link_cap = df_band[:, 2]

plt.figure()
plt.title("The average estimated link-rate is: "+str(round(np.mean(band_est), 2))+" Mbps", fontsize=12)
plt.plot(time_est, band_est/fec_overhead, linewidth=2, label="Estimated Link Rate")
plt.plot(time_rx[:-1], rate_rx/1e6, linewidth=2, label="Measured Rx Bitrate")
# plt.plot(time_est, link_cap, linewidth=2, linestyle="--", label="True Link Rate")
plt.grid()
plt.xlabel("Time [s]")
plt.ylabel("Est. Link Rate [Mbps]")
plt.legend()
# plt.xlim([-1, 50])
plt.savefig(est_band_filename+".png", dpi=500)
plt.show()
