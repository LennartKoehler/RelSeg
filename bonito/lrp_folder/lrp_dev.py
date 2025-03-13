import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_relevances(relevances, raw_signal):
    raw_signal = raw_signal.detach().cpu().numpy()
    relevances = relevances.detach().cpu().numpy()

    relevances = relevances.T
    fig, axs = plt.subplots((2))
    fig.set_size_inches(20,7)

    xmin = 400
    xmax = 1500
   
    axs[0].plot(raw_signal, color="black", linewidth=0.3, label="raw_data")
    for i,relevance in enumerate(relevances):
        # relevance = np.abs(relevance)
        # relevance = relevance/np.max(relevance)
        # if sum(relevance) < 500:
        axs[1].plot(relevance, label="relevance", alpha=0.7)
        segment = np.argmax(relevance)
        axs[0].vlines(segment, -3, 3)
        print(relevance)

    # axs[1].plot(np.abs(relevances[40])/np.max(np.abs(relevances[40])))

    axs[0].set_title("raw signal")
    axs[0].set_ylabel("current")
    axs[1].set_ylabel("absolute normed relevance")
    axs[1].set_title("relevance")
    axs[1].set_xlabel("datapoints")

    axs[1].set_xlim(xmin,xmax)
    axs[0].set_xlim(xmin,xmax)
    fig.tight_layout()

    plt.savefig("plots/lrp_test", dpi=300)

if __name__ == "__main__":
    relevances = torch.load("test_outputs/relevances.pkl")
    signal = torch.load("test_outputs/raw_signal.pkl")
    plot_relevances(relevances[3,:,:], signal[0,0,:])