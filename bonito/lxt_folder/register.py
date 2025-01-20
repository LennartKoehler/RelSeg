import operator
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
print(os.getcwd())
from lxt.core import Composite
import lxt.functional as lf
import lxt.modules as lm
import lxt.rules as rules
from transformer.model import MultiHeadAttention, GatedMlp
import matplotlib.pyplot as plt

from bonito.lxt_folder.get_data import get_data
#####################
### LRP Composite ###
#####################

# must be traced because of the functions, e.g. operator.add!
attnlrp = Composite({
        #MultiHeadAttention: rules.EpsilonRule,
        #GatedMlp: rules.EpsilonRule,
        nn.SiLU: rules.IdentityRule,
        nn.GELU: rules.IdentityRule,
        nn.Tanh: rules.IdentityRule,
        #nn.Linear: lm.LinearEpsilon,
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

#####################
### Example Usage ###
#####################

def run_lxt(bonito_model, reads):
    from PIL import Image
    from zennit.composites import LayerMapComposite
    import zennit.rules as z_rules
    from zennit.image import gridify

    device = 'cuda'
    torch.backends.cudnn.enabled = False
    torch.autograd.set_detect_anomaly(True)


    model = bonito_model.encoder
    #del model[0][1] # IMPORTANT transformer is removed, work on that later
    model.eval()
    model = model.to(device)


    read, data = get_data(reads, chunksize=12000, overlap=600) #adjust chunksize for different models
    data= data.to(device)
    print(data.shape)

    model(data)
    # Load an image and tokenize a text


    # trace the model with a dummy input
    # verbose=True prints all functions/layers found and replaced by LXT
    # you will see at the last entry that e.g. tensor.exp() is not supported by LXT. This is not a problem in our case,
    # because this function is not used in the backward pass and therefore does not need to be replaced.
    # (look into the open_clip.transformer module code!)
    x = torch.randn_like(data, device=device)
    traced = attnlrp.register(model, dummy_inputs={'x': x}, verbose=True) #IMPORTANT change "input" to "x" if necessary
    # with open("graph.txt", "w") as f:
    #     f.write(str(traced.graph))

    # for Vision Transformer, we must perform a grid search for the best gamma hyperparameters
    # in general, it is enough to concentrate on the Conv2d and MLP layers
    # for simplicity we just use a few values that can be evaluated by hand & looking at the heatmaps
    heatmaps = []
    conv_gamma = 0.5
    lin_gamma= 0.5

    # print("Gamma Conv2d:", conv_gamma, "Gamma Linear:", lin_gamma)

    # we define rules for the Conv2d and Linear layers using 'Zennit'
    zennit_comp = LayerMapComposite([
            (nn.Conv1d, z_rules.Gamma(conv_gamma)),
            (nn.Linear, z_rules.Gamma(lin_gamma)),
        ])

    # register composite
    zennit_comp.register(traced)

    # forward & backward pass
    y = traced(data.requires_grad_(True))
    # explain the dog class ("a dog")
    for position in range(10):
        data.grad = None
        print(position)
        y_current = y[0,position*200,:].sum()
        y_current.backward(retain_graph=True)
        # normalize the heatmap
        heatmap = data.grad[0].sum(0)
        heatmap = heatmap.detach().cpu().numpy()


        # zennit composites can be removed so that we can register a new one!
        #zennit_comp.remove()

        heatmaps.append(heatmap)


    import numpy as np
    data = data.detach().cpu().numpy()
    plt.plot(np.arange(len(data[0,0,:])),data[0,0,:])
    plt.savefig("input_data")

    y = y.detach().cpu().numpy()
    pos = np.arange(len(y[0,:,0]))
    kmer = np.arange(len(y[0,0,:]))
    pos, kmer = np.meshgrid(kmer, pos)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(kmer, pos, y[0,:,:], cmap='viridis')
    fig.colorbar(surf, shrink=0.5, aspect=10)
    plt.savefig("raw_output")



    # save the heatmaps as a grid
    fig, axs = plt.subplots(len(heatmaps))
    fig.set_figheight(50)
    fig.set_figwidth(15)
    for i,heatmap in enumerate(heatmaps):
        axs[i].plot(heatmap)
    plt.savefig("lxt1", dpi=500)


