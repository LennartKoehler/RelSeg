"""
Bonito CRF basecalling
"""

import torch
import numpy as np
from koi.decode import beam_search, to_str
from torch.utils.tensorboard import SummaryWriter

from bonito.multiprocessing_bonito import thread_iter
from bonito.util import chunk, stitch, batchify, unbatchify, half_supported


def stitch_results(results, length, size, overlap, stride, reverse=False):
    """
    Stitch results together with a given overlap.
    """
    if isinstance(results, dict):
        return {
            k: stitch_results(v, length, size, overlap, stride, reverse=reverse)
            for k, v in results.items()
        }
    if length < size:
        return results[0, :int(np.floor(length / stride))]
    return stitch(results, size, overlap, length, stride, reverse=reverse)


def compute_scores(model, batch, beam_width=32, beam_cut=100.0, scale=1.0, offset=0.0, blank_score=2.0, reverse=False):
    """
    Compute scores for model.
    """

    with torch.inference_mode():
        device = next(model.parameters()).device
        dtype = torch.float16 if half_supported() else torch.float32
        scores = model(batch.to(dtype).to(device)) # IMPORTANT here the neural network is run

        if reverse:
            scores = model.seqdist.reverse_complement(scores)
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


def fmt(stride, attrs, rna=False):
    fliprna = (lambda x:x[::-1]) if rna else (lambda x:x)
    return {
        'stride': stride,
        'moves': attrs['moves'].numpy(),
        'qstring': fliprna(to_str(attrs['qstring'])),
        'sequence': fliprna(to_str(attrs['sequence'])), #IMPORTANT here int is translated to ascii (ACGT)
    }


def basecall(model, reads, chunksize=4000, overlap=100, batchsize=32,
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
        (read, compute_scores(model, batch, reverse=reverse)) for read, batch in batches
    )

    results = (
        (read, stitch_results(scores, end - start, chunksize, overlap, model.stride, reverse))
        for ((read, start, end), scores) in unbatchify(scores)
    )

    return (
        (read, fmt(model.stride, attrs, rna))
        for read, attrs in results
    )
    # chunks = thread_iter(
    #     ((read, 0, read.signal.shape[-1]), chunk(torch.from_numpy(read.signal), chunksize, overlap))
    #     for read in reads
    # )

    # batches = thread_iter(batchify(chunks, batchsize=batchsize))

    # scores = thread_iter(
    #     (read, compute_scores(model, batch, reverse=reverse)) for read, batch in batches
    # )

    # results = thread_iter(
    #     (read, stitch_results(scores, end - start, chunksize, overlap, model.stride, reverse))
    #     for ((read, start, end), scores) in unbatchify(scores)
    # )

    # return thread_iter(
    #     (read, fmt(model.stride, attrs, rna))
    #     for read, attrs in results
    # )
