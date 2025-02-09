import torch
from koi.decode import beam_search
import numpy as np
from bonito.lrp_folder.LRP_composites import lxt_comp, zennit_comp
from bonito.util import chunk, stitch, batchify, unbatchify, half_supported, concat
from koi.decode import beam_search, to_str
from collections import deque
from scipy.signal import find_peaks
from tqdm import tqdm


def fmt(stride, attrs, rna=False):
    fliprna = (lambda x:x[::-1]) if rna else (lambda x:x)
    return {
        'stride': stride,
        'moves': attrs['moves'].numpy(),
        'segments': attrs['segments'].numpy(),
        'qstring': fliprna(to_str(attrs['qstring'])),
        'sequence': fliprna(to_str(attrs['sequence'])), #IMPORTANT here int is translated to ascii (ACGT)
    }

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



def save_relevance_and_signal(relevance_gen, input_signal, n_moves):
    batchsize, _, seq_len = input_signal.shape
    relevances = torch.empty((batchsize,seq_len, n_moves), dtype=input_signal.dtype, device=input_signal.device)

    for i, (relevance, batch_indices) in enumerate(relevance_gen):
        relevances[batch_indices, :, i] = relevance


    torch.save(relevances, "relevances.pkl")
    torch.save(input_signal, "raw_signal.pkl")

def register(model, dummy_input): # the imported composites are used
    model.eval()
    traced_model = lxt_comp.register(model, dummy_inputs={"x": dummy_input}, verbose=False)
    zennit_comp.register(traced_model)
    return traced_model

def stitch_results(results, length, size, overlap, stride, key, reverse=False):
    """
    Stitch results together with a given overlap.
    """
    if isinstance(results, dict):
        return {
            k: stitch_results(v, length, size, overlap, stride, k, reverse=reverse)
            for k, v in results.items()
        }
    if key == "segments":
        if length < size:
            return results[0, :int(np.floor(length))]
        return stitch_segments_indices(results, size, overlap, length, stride, reverse=reverse)
    else:
        if length < size:
            return results[0, :int(np.floor(length / stride))]
        
        return stitch(results, size, overlap, length, stride, reverse=reverse)




def stitch_segments_indices(chunks, chunksize, overlap, length, stride, reverse=False):
    """
    Stitch segments together with a given overlap
    """
    if chunks.shape[0] == 1: return chunks.squeeze(0)
    n_chunks = chunks.shape[0]

    semi_overlap = overlap // 2
    start, end = semi_overlap , (chunksize - semi_overlap) 
    stub = (length - overlap) % (chunksize - overlap)
    first_chunk_end = (stub + semi_overlap)  if (stub > 0) else end

    offset = torch.arange(0, (n_chunks-2)*(chunksize-overlap)+1, chunksize-overlap)
    offset += stub
    offset = torch.cat([torch.tensor([0]), offset]) # add offset for each chunk, since the segments are relative to the chunk start, not read start

    start_down = start // stride
    end_down = end // stride
    first_chunk_end_down = first_chunk_end // stride

    chunks = chunks.to(torch.float)
    chunks[chunks == -1] = float("nan")
    chunks += offset.unsqueeze(1)

    segments = concat([
            chunks[0, :first_chunk_end_down], *chunks[1:-1, start_down:end_down], chunks[-1, start_down:]
        ])

    return segments[~torch.isnan(segments)].to(torch.long)


def batch_positions(positions):
    result = torch.zeros_like(positions, dtype=torch.long)-1
    most_moves_in_batch = 0
    for i,sample in enumerate(positions):
        moves = torch.argwhere(sample==1).squeeze()
        result[i,:len(moves)] = moves
        if len(moves)>most_moves_in_batch:
            most_moves_in_batch = len(moves)
    result = result[:,:most_moves_in_batch]
    return result

def batched_lrp_loop(data, y, batched_positions):
    full_batch_indices = torch.arange(data.shape[0], dtype=torch.long)

    for motif_indices in batched_positions.T:

        data.grad = None
        batch_indices_filtered = full_batch_indices[motif_indices!=-1]
        motif_indices_filtered = motif_indices[motif_indices!=-1].to(torch.int64)
        y_current = y[batch_indices_filtered, motif_indices_filtered, :].sum()
        y_current.backward(retain_graph=True)
        relevance = data.grad[batch_indices_filtered, 0,:]
        yield (relevance, batch_indices_filtered, motif_indices)


def segmentation_loop(relevance_gen, segmentation_function, batchsize, downsampled_size):
    segments_batch = torch.zeros((batchsize, downsampled_size), dtype=torch.long)-1 # this tensor is longer than needed and will never be filled because n_moves is shorter than downsampled_size, but its easier to unbatchify

    for i, (relevance, batch_indices, motif_indices) in tqdm(enumerate(relevance_gen)):

        segment_indices = segmentation_function(relevance).to("cpu")
        z = torch.zeros(batchsize, dtype=torch.long)-1
        z[batch_indices] = segment_indices # if a sample in the batch no longer has moves then just keep adding 0 as segment until all samples in batch have no more moves
        segments_batch[torch.arange(batchsize),motif_indices] = z

    return segments_batch


def peak_segmentation(relevances):
    #indices = torch.argmax(torch.abs(relevances), dim=1, keepdim=False)
    result = []
    for relevance in relevances:
        relevance = relevance.abs()
        relevance = relevance / (relevance.max())
        peaks = find_peaks(relevance.detach().cpu(), distance=4, height=0.2)
        peaks = np.array([peaks[0], peaks[1]["peak_heights"]])
        peaks = peaks[:,np.argsort(peaks, 1)[1]]
        peaks = np.flip(peaks, axis=1).T

        assert peaks.size!=0, f"no peaks detected"
        result.append(peaks)
    return result

def argmax_segmentation(relevances):
    indices = torch.argmax(torch.abs(relevances), dim=1, keepdim=False)
    return indices



def forward_and_lrp(model_enc, input_signal, stride, save_relevance=False): #MAIN LRP FUNCTION
    device = next(model_enc.parameters()).device
    dtype = torch.float16 if half_supported() else torch.float32
    input_signal = input_signal.to(dtype).to(device)

    scores = model_enc(input_signal.requires_grad_(True))

    batchsize, downsampled_len, _ = scores.shape

    beam_result = run_beam_search(scores)

    moves = beam_result["moves"]
    batched_moves = batch_positions(moves) # -> shape [#moves, nbatches]

    relevance_gen = batched_lrp_loop(input_signal, scores, batched_moves)

    if save_relevance: # just for development
        save_relevance_and_signal(relevance_gen, input_signal, batched_moves.shape[1])
        assert False # TESTVALUE


    segments = segmentation_loop(relevance_gen, argmax_segmentation, batchsize, downsampled_len)
    #segments = format_segments_bit(segments, stride, batchsize, downsampled_len)

    beam_result["segments"] = segments
    return beam_result





def basecall_and_lrp(model, reads, chunksize=4000, overlap=100, batchsize=4,
             reverse=False, rna=False, save_test_relevance=False):
    """
    Basecalls a set of reads.
    """
    chunks = (
        ((read, 0, read.signal.shape[-1]), chunk(torch.from_numpy(read.signal), chunksize, overlap))
        for read in reads
    )

    batches = (batchify(chunks, batchsize=batchsize))

    dummy_input = torch.randn((batchsize,1,chunksize), device=next(model.parameters()).device)
    model_enc = register(model.encoder, dummy_input)

    scores = (
        (read, forward_and_lrp(model_enc, batch, model.stride, save_test_relevance)) for read, batch in batches # TODO reverse
    )

    results = (
        (read, stitch_results(scores, end - start, chunksize, overlap, model.stride, "", reverse))
        for ((read, start, end), scores) in unbatchify(scores)
    )

    return (
        (read, fmt(model.stride, attrs, rna))
        for read, attrs in results
    )


