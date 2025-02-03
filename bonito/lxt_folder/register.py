import operator
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np

from lxt.core import Composite
import lxt.functional as lf
import lxt.modules as lm
import lxt.rules as rules
from transformer.model import MultiHeadAttention, GatedMlp
import matplotlib.pyplot as plt

from bonito.lxt_folder.get_data import get_data, run_beam_search

from tqdm import tqdm

from PIL import Image
from zennit.composites import LayerMapComposite
import zennit.rules as z_rules
from zennit.image import gridify
from koi.ctc import SequenceDist, Max

#####################
### LRP Composite ###
#####################

# must be traced because of the functions, e.g. operator.add!
attnlrp = Composite({

        nn.SiLU: rules.IdentityRule,
        nn.GELU: rules.IdentityRule,
        nn.Tanh: rules.IdentityRule,
        nn.Linear: lm.LinearEpsilon,
        #nn.Conv1d: rules.EpsilonRule,

        #torch.exp: rules.identity(torch.exp),
        torch.softmax: lf.softmax,
        F.glu: lf.linear_epsilon,
        operator.add: lf.add2,
        torch.add: lf.add2,
        operator.mul: lf.mul2,
        operator.matmul: lf.matmul,
        torch.matmul: lf.matmul,
        F.normalize: lf.normalize,
    })

conv_gamma = 0.01

zennit_comp = LayerMapComposite([
    (nn.Conv1d, z_rules.Gamma(conv_gamma)),
    #(nn.Linear, z_rules.Gamma(lin_gamma)),
])

#####################
### Example Usage ###
#####################

def register(bonito_model, reads, use_koi):


    device = 'cuda'
    torch.backends.cudnn.enabled = False
    torch.autograd.set_detect_anomaly(True)


    model = bonito_model.encoder
    model.eval()
    model = model.to(device)


    read, data = get_data(reads, chunksize=12000, overlap=600) #adjust chunksize for different models
    data= data.to(device)
    #np.save(f"comparison/raw_signal_2_{read[0][0].read_id}.npy", data.detach().cpu().numpy())

    model(data)


    x = torch.randn_like(data, device=device)

    input_string = "x" if use_koi else "input"

    traced = attnlrp.register(model, dummy_inputs={input_string: x}, verbose=False) #IMPORTANT change "input" to "x" if necessary (use_koi!!!)

    zennit_comp.register(traced)

    y = traced(data.requires_grad_(True))

    if use_koi:
        positions = run_beam_search(y)["moves"]
    else:
        positions = run_viterbi(y, bonito_model.seqdist)
        y = y.permute([1,0,2])


    segments = get_segments(data, y, positions, segmentation)
    #plot_relevances(relevances, data)
    # np.save(f"comparison/beamsearch_move_positions_{read[0][0].read_id}.npy", positions)


def batch_positions(positions):
    result = torch.zeros_like(positions, dtype=torch.long)-1
    most_moves_in_batch = 0
    for i,sample in enumerate(positions):
        moves = torch.argwhere(sample==1).squeeze()
        result[i,:len(moves)] = moves
        if len(moves)>most_moves_in_batch:
            most_moves_in_batch = len(moves)
    result = result[:,:most_moves_in_batch]
    return result.T

def get_segments(data, y, positions, segmentation_function):
    batch_size, _ , seq_len = data.shape

    batched_positions = batch_positions(positions)

    segments_batch = torch.zeros(batch_size,1, device=data.device, dtype=torch.long)
    full_batch_indices = torch.arange(data.shape[0], dtype=torch.long)

    for i,motif_indices in tqdm(enumerate(batched_positions[:5,:]), total=batched_positions.shape[0]):
        data.grad = None
        batch_indices = full_batch_indices[motif_indices!=-1]
        motif_indices = motif_indices[motif_indices!=-1].to(torch.int64)
        
        y_current = y[batch_indices, motif_indices, :].sum()
        y_current.backward(retain_graph=True)
        relevance = data.grad[batch_indices, 0,:]

        z = torch.zeros((batch_size,1), device=data.device, dtype=torch.long)
        segment_indices = segmentation_function(relevance)
        z[batch_indices,:] = segment_indices # if not relevance because no more moves then just add segments add 0 untill the moves of the sample with the most moves is calculates

        segments_batch = torch.cat((segments_batch, z), dim=1)
    segments_batch_bit = torch.zeros((batch_size, seq_len))
    segments_batch_bit[full_batch_indices.unsqueeze(1),segments_batch] = 1
    return segments_batch_bit

# def get_relevances(data, y, positions):
#     relevances = []
#     for i,(position, *kmer) in tqdm(enumerate(positions), total=len(positions)):
#         data.grad = None
#         y_current = y[0, position, :].sum(-1) #  : positions[i+1][0]-1
#         y_current.backward(retain_graph=True)
#         relevance = data.grad[0, 0]

#         relevance = relevance / (abs(relevance).max())
#         relevance = relevance.detach().cpu().numpy()
#         relevances.append(relevance)
#     return relevances
    
def segmentation(relevances):
    indexes = torch.argmax(torch.abs(relevances), dim=1, keepdim=True)
    return indexes


def plot_relevances(relevances, raw_signal):
    raw_signal = raw_signal.detach().cpu().numpy()
    relevances = relevances.detach().cpu().numpy()

    fig, axs = plt.subplots(2)
    fig.set_size_inches(80,5)

    offset = 0
    range_end = 500


    axs[0].plot(raw_signal, color="black", linewidth=0.3, label="raw_data")
    for i,relevance in enumerate(relevances):
        print(relevance, raw_signal)
        axs[1].plot(relevance, label="lxt", alpha=0.7)
        start = np.argmax(np.abs(relevance))
        axs[0].axvline(start, -5, 5, color="red", alpha=0.5, linewidth=0.5)

    axs[0].set_xlim(offset,range_end)
    axs[1].set_xlim(offset,range_end)

    plt.savefig("plots/lxt_test", dpi=300)





def run_viterbi(y, seqdist):
    # only use with use_koi = False
    y_copy = y.detach().clone()
    traceback_kmer = decode(seqdist, y_copy)[0,:]
    kmers = [(position, kmer.detach().cpu()) for position, kmer in enumerate(traceback_kmer) if kmer % 5 != 0] # filter out those were new kmers are predicted, remove "N"
    return kmers


def decode(seqdist, scores):
    scores = seqdist.posteriors(scores.to(torch.float32)) + 1e-8 # probabilities
    tracebacks = viterbi(seqdist, scores.log()).to(torch.int16).T
    return tracebacks

    
def viterbi(seqdist, scores):
    traceback = seqdist.posteriors(scores, Max) # one motif (last dimension) in traceback has score 1 and the rest 0
    a_traceback = traceback.argmax(2) #IMPORTANT take a_traceback (index of kmer (or base))
    return a_traceback

def plot_raw_output(y):
    y = y.detach().cpu().numpy()
    pos = np.arange(len(y[0,:,0]))
    kmer = np.arange(len(y[0,0,:]))
    pos, kmer = np.meshgrid(kmer, pos)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(kmer, pos, y[0,:,:], cmap='viridis')
    fig.colorbar(surf, shrink=0.5, aspect=10)
    plt.savefig("plots/raw_output")


def varying_gamma(data, traced):
    offset = 100
    distance_between = 5
    relevances = []
    gammas = itertools.product([0.1, 0.5, 100], [0.01, 0.05, 0.1, 1])
    gammas, gammas2 = itertools.tee(gammas)
    for conv_gamma, lin_gamma in gammas:
        zennit_comp = LayerMapComposite([
            (nn.Conv1d, z_rules.Gamma(conv_gamma)),
            (nn.Linear, z_rules.Gamma(lin_gamma)),
        ])
        zennit_comp.register(traced)
        y = traced(data.requires_grad_(True))

        data.grad = None
        y_current = y[0,offset,:].sum()
        y_current.backward(retain_graph=True)

        relevance = data.grad[0, 0]
        relevance = relevance / (abs(relevance).max())
        relevance = relevance.detach().cpu().numpy()
        zennit_comp.remove()
        relevances.append(relevance)
    


    fig, axs = plt.subplots(len(relevances))
    offset *= 6
    distance_between *= 6
    fig.set_figheight(50)
    fig.set_figwidth(15)
    for i,relevance in enumerate(relevances):
        axs[i].plot(relevance[offset - 5*distance_between : offset + 15*distance_between])
        axs[i].set_title(f"gamma_conv, gamma_lin: {next(gammas2)}")
    plt.savefig("plots/gamma_variation", dpi=500)
