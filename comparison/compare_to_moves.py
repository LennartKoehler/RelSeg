import torch
import matplotlib.pyplot as plt
import torch.nn as nn

result = torch.load("test_result_6000.pkl")


read = torch.load("test_read.pkl")

print(read)
print(result[0])
result = result[1]


fig, axs = plt.subplots()
fig.set_size_inches(80,5)
axs.plot(read.signal)

moves_counter = 0
for i,move in enumerate(result["moves"]):
    if move ==1:
        moves_counter += 1
        x_pos = i*6+3
        axs.axvline(x_pos, -2, 2,linewidth=6, alpha=0.5, color="red")
        #axs.text(x_pos, -1, moves_counter, color="darkred")
axs.axvline(0,0,0,color="red", alpha=0.5, linewidth=6, label="moves")

lxt_counter = 0
for i,segment in enumerate(result["segments"]):
    if segment==1:
        lxt_counter += 1
        axs.axvline(i, -2, 2, alpha=0.5, color="green")
        #axs.text(i, 1, lxt_counter, color="darkgreen")
axs.axvline(0,0,0,color="green", alpha=0.5,label="lxt")

axs.set_xlim(0, 3000)
axs.legend()
plt.savefig("comparison/lxt_vs_moves.png", dpi=300)