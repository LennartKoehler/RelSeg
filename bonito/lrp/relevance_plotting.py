import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_relevances(relevances, raw_signal, out_name):
    raw_signal = raw_signal.detach().cpu().numpy()
    relevances = relevances.detach().cpu().numpy()

    relevances = relevances.T
    fig, axs = plt.subplots((2))
    fig.set_size_inches(10,5)

    xmin = 0
    xmax = 1000
    old_segment = 0
    old_relevance = 0
    counter = 0
    axs[0].plot(raw_signal, color="black", linewidth=0.3, label="raw_data")
    for i,relevance in enumerate(relevances):
        # relevance = np.abs(relevance)
        # relevance = relevance/np.max(relevance)
        # if sum(relevance) < 500:

        axs[1].plot(relevance, label="relevance", alpha=0.7)

        # if np.argmax(relevance)+20<old_segment:
        #     if counter==3:
        #         axs[1].plot(relevance, label="relevance", alpha=0.7, color="orange")
        #         axs[0].vlines(np.argmax(relevance), -3, 3, color="orange")

        #         axs[1].plot(old_relevance, label="relevance", alpha=0.7, color="blue")
        #         axs[0].vlines(old_segment, -3, 3, color="blue")

        #         axs[1].plot(old_old_relevance, label="relevance", alpha=0.7, color="green")
        #         axs[0].vlines(old_old_segment, -3, 3, color="green")       

        #     counter +=1

        # old_old_relevance = old_relevance
        # old_old_segment = old_segment
        # old_segment = np.argmax(relevance)
        # old_relevance = relevance

        # print(relevance)

    # axs[1].plot(np.abs(relevances[40])/np.max(np.abs(relevances[40])))

    axs[0].set_title("Signal")
    axs[0].set_ylabel("Normed Current")
    axs[1].set_ylabel("Relevance")
    axs[1].set_title("Relevance")
    axs[1].set_xlabel("Data Points")

    axs[1].set_xlim(xmin,xmax)
    axs[0].set_xlim(xmin,xmax)
    # axs[1].set_ylim(None, 3)
    fig.tight_layout()

    plt.savefig(out_name, dpi=200, format="pdf")

if __name__ == "__main__":
    relevances = torch.load("relevance/relevances_2.pkl")
    signal = torch.load("relevance/signals_2.pkl")

    plot_relevances(relevances[1,:,14:15], signal[1,0,:], "plots/test.pdf")

    # plot_relevances(relevances[1,:,14:15], signal[1,0,:], "plots/lrp_single.pdf")
    # plot_relevances(relevances[1,:,:], signal[1,0,:], "plots/lrp_all.pdf")

    # plot_relevances(relevances[1,:,:], signal[1,0,:], "plots/lrp_problem.pdf")
    # plot_relevances(relevances[1,:,:], signal[1,0,:], "plots/lrp_wrong_order.pdf") # counter == 3