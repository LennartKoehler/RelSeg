import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_relevances(relevances, raw_signal):
    raw_signal = raw_signal.detach().cpu().numpy()
    relevances = relevances.detach().cpu().numpy()

    relevances = relevances.T
    fig, axs = plt.subplots(2)
    fig.set_size_inches(80,5)

    xmin = 0
    xmax = 2000

    axs[0].plot(raw_signal, color="black", linewidth=0.3, label="raw_data")
    for i,relevance in enumerate(relevances):
        relevance = np.abs(relevance)
        relevance = relevance/np.max(relevance)
        axs[1].plot(relevance, label="lxt", alpha=0.7)
        start = np.argmax(np.abs(relevance))
        axs[0].axvline(start, -5, 5, color="red", alpha=0.5, linewidth=0.5)

    axs[0].set_xlim(xmin,xmax)
    axs[1].set_xlim(xmin,xmax)

    plt.savefig("plots/lxt_test", dpi=300)

if __name__ == "__main__":
    relevances = torch.load("relevances.pkl")
    signal = torch.load("raw_signal.pkl")
    plot_relevances(relevances[0,:,:], signal[0,0,:])