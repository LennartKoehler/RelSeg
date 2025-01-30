import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

lxt = np.load("comparison/lxt_segments_048f077a-a8a0-4dda-95f5-3c43be3a1274.npy")

signal = np.load("comparison/raw_signal_2_048f077a-a8a0-4dda-95f5-3c43be3a1274.npy")[0,0,:]


f5c = pd.read_table("../data/rna004/resquiggle/PNXRXX240011_r10k.tsv", sep="\\s+")
f5c = f5c.replace(".", 0).astype({"start_raw_idx": int, "end_raw_idx": int})[::-1]
f5c.rename(columns={"end_raw_idx":"end", "start_raw_idx":"start", "index":"global_index", "kmer_idx":"position"}, inplace=True)
groups = f5c.groupby('read_id')
f5c_read_dict = {key1: group for key1, group in groups}

f5c_read = f5c_read_dict["048f077a-a8a0-4dda-95f5-3c43be3a1274"]
f5c_segments_read = f5c_read["start"]


fig, axs = plt.subplots()
fig.set_size_inches(80,5)

axs.plot(signal, linewidth=0.3, color="black")

for segment in f5c_segments_read:
    axs.axvline(segment, -3, 3, color="red", alpha=0.5)
axs.axvline(0,0,0,color="red", alpha=0.5,label="f5c")

for segment in lxt:
    axs.axvline(segment, -3, 3, color="green", alpha=0.5)
axs.axvline(0,0,0,color="green", alpha=0.5,label="lxt")
axs.set_xlim(4000,8000)
axs.legend()

plt.savefig("comparison/test_comparison", dpi=300)