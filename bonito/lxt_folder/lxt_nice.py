import torch
from koi.decode import beam_search
import numpy as np
from lxt_folder.LRP_composites import lxt_comp, zennit_comp
from crf.basecall import *

# beamsearch, use_koi
def register(model, data): # the imported composites are used
    model.eval()
    dummy_input = torch.randn_like(data, device=data.device)
    traced_model = lxt_comp.register(model, dummy_inputs={"x": dummy_input}, verbose=False)
    zennit_comp.register(traced_model)
    return traced_model



# TODO how do i deal with batches, think i can just go over batches/ignore them
def get_relevances(input_signal, output_scores, positions):
    relevances = []
    for i,(position, *kmer) in enumerate(positions):
        input_signal.grad = None
        y_current = output_scores[0, position, :].sum() #  : moves[i+1][0]-1
        y_current.backward(retain_graph=True)
        relevance = input_signal.grad[0, 0]

        relevance = relevance / (abs(relevance).max())
        relevance = relevance.detach().cpu().numpy()
        relevances.append(relevance)
    return relevances

def run_beam_search(scores, beam_width=32, beam_cut=100.0, scale=1.0, offset=0.0, blank_score=2.0):
    with torch.cuda.device(scores.device):
        sequence, qstring, moves = beam_search( # IMPORTANT this is where the actual sequence is determined from the nn scores
            scores, beam_width=beam_width, beam_cut=beam_cut,
            scale=scale, offset=offset, blank_score=blank_score
        )
    return {
        'moves': moves,
        'qstring': qstring,
        'sequence': sequence,
    }

def get_segments(relevances):
    segments = np.zeros([len(relevances[0])])
    indexes = [np.argmax(np.abs(r)) for r in relevances] # could also be added to get_relevances() or use generators to speed up
    segments[indexes] = 1
    return segments


def forward_and_lrp(model, input_signal):
    device = next(model.parameters()).device
    dtype = torch.float16 if half_supported() else torch.float32
    input_signal = input_signal.to(dtype).to(device)

    model_enc = register(model.encoder, input_signal)
    scores = model_enc(input_signal.requires_grad_(True))
    beam_result = run_beam_search(scores)

    moves = beam_result["moves"][0,:]
    moves = torch.argwhere(moves==1).cpu().detach().numpy()

    moves = np.array([(1,0),(2,0)]) # TESTVALUE
    relevances = get_relevances(input_signal, scores, moves)
    segments = get_segments(relevances)
    beam_result["segments"] = segments
    return beam_result

def fmt(stride, attrs, rna=False):
    fliprna = (lambda x:x[::-1]) if rna else (lambda x:x)
    return {
        'stride': stride,
        'moves': attrs['moves'].numpy(),
        'segments': attrs['segments'],
        'qstring': fliprna(to_str(attrs['qstring'])),
        'sequence': fliprna(to_str(attrs['sequence'])), #IMPORTANT here int is translated to ascii (ACGT)
    }


def basecall_and_lrp(model, reads, chunksize=4000, overlap=100, batchsize=16,
             reverse=False, rna=False):
    """
    Basecalls a set of reads.
    """
    chunks = (
        ((read, 0, read.signal.shape[-1]), chunk(torch.from_numpy(read.signal), chunksize, overlap))
        for read in reads
    )

    batches = (batchify(chunks, batchsize=batchsize))

    scores = (
        (read, forward_and_lrp(model, batch)) for read, batch in batches # TODO reverse
    )

    results = (
        (read, stitch_results(scores, end - start, chunksize, overlap, model.stride, reverse))
        for ((read, start, end), scores) in unbatchify(scores)
    )

    return (
        (read, fmt(model.stride, attrs, rna))
        for read, attrs in results
    )