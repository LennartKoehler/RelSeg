import torch
from koi.decode import beam_search
import numpy as np
from bonito.lxt_folder.LRP_composites import lxt_comp, zennit_comp
from bonito.crf.basecall import *
from tqdm import tqdm

# beamsearch, use_koi
def register(model, data): # the imported composites are used
    model.eval()
    dummy_input = torch.randn_like(data, device=data.device)
    traced_model = lxt_comp.register(model, dummy_inputs={"x": dummy_input}, verbose=False)
    zennit_comp.register(traced_model)
    return traced_model


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

def lrp(data, y, positions, segmentation_function):
    batch_size, _ , seq_len = data.shape

    batched_positions = batch_positions(positions)

    segments_batch = torch.zeros(batch_size,1, device=data.device, dtype=torch.long)
    full_batch_indices = torch.arange(data.shape[0], dtype=torch.long)

    for i,motif_indices in tqdm(enumerate(batched_positions[:30,:]), total=batched_positions.shape[0]):
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
    return segments_batch



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

def segmentation(relevances):
    indexes = torch.argmax(torch.abs(relevances), dim=1, keepdim=True)
    return indexes

def format_segments_scale(segments, stride, batches, downsampled_len):
    segments_bit = torch.zeros((batches, downsampled_len))
    segments_bit[torch.arange(batches), segments] = 1
    segments_bit = segments_bit.view(batches, -1, stride)
    return segments_bit

def forward_and_lrp(model, input_signal):
    device = next(model.parameters()).device
    dtype = torch.float16 if half_supported() else torch.float32
    input_signal = input_signal.to(dtype).to(device)

    model_enc = register(model.encoder, input_signal)
    scores = model_enc(input_signal.requires_grad_(True))
    beam_result = run_beam_search(scores)

    moves = beam_result["moves"]

    segments = lrp(input_signal, scores, moves, segmentation)
    segments = format_segments_scale(segments, model.stride, scores.shape[0], scores.shape[-1])

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


def basecall_and_lrp(model, reads, chunksize=4000, overlap=100, batchsize=4,
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