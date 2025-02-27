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
    old_segment = 0
    old_relevance=np.array([0,0,0])
    axs[0].plot(raw_signal, color="black", linewidth=0.3, label="raw_data")
    for i,relevance in enumerate(relevances):
        # relevance = np.abs(relevance)
        # relevance = relevance/np.max(relevance)
        # if sum(relevance) < 500:
        axs[1].plot(relevance, label="relevance", alpha=0.7)

            # segment = np.argmax(relevance)
            # if segment < old_segment and segment > 450:
            #     axs[1].plot(relevance, label="relevance", alpha=0.7)
            #     axs[1].plot(old_relevance, label="relevance", alpha=0.7)
            #     print(segment, old_segment)
            #     break

            # old_segment = segment
            # old_relevance = relevance
    # axs[1].plot(np.abs(relevances[40])/np.max(np.abs(relevances[40])))

    axs[0].set_title("raw signal")
    axs[0].set_ylabel("current")
    axs[1].set_ylabel("absolute normed relevance")
    axs[1].set_title("relevance")
    axs[1].set_xlabel("datapoints")

    axs[1].set_xlim(xmin,xmax)
    axs[0].set_xlim(xmin,xmax)
    fig.tight_layout()

    plt.savefig("plots/lrp_gamma_conv_0_01", dpi=300)

if __name__ == "__main__":
    relevances = torch.load("test_outputs/relevances.pkl")
    signal = torch.load("test_outputs/raw_signal.pkl")
    plot_relevances(relevances[2,:,:], signal[2,0,:])