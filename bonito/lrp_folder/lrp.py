import torch
import numpy as np
from bonito.lrp_folder.LRP_composites import lxt_comp, zennit_comp
from bonito.util import chunk, batchify, unbatchify, half_supported
from koi.decode import to_str
from scipy.signal import find_peaks
from bonito.lrp_folder.stitching import stitch_results
from bonito.lrp_folder.search import run_beam_search, run_viterbi

def fmt(stride, attrs, trimmed_samples, rna=False):
    segments = attrs['segments'].numpy()
    segments[:,:,0] += trimmed_samples
    return {
        'stride': stride,
        'moves': attrs['moves'].numpy(),
        'segments': segments,
        'qstring': to_str(attrs['qstring']), # IMPORTANT dont flip sequence etc like in basecaller because otherwirse they are not mappable to the raw signal
        'sequence': to_str(attrs['sequence']), #IMPORTANT here int is translated to ascii (ACGT)
    }



def save_relevance_and_signal(relevance_gen, input_signal, n_moves):
    batchsize, _, seq_len = input_signal.shape
    relevances = torch.zeros((batchsize,seq_len, n_moves), dtype=input_signal.dtype, device=input_signal.device)

    for i, (relevance, batch_indices, motif_indices) in enumerate(relevance_gen):
        relevances[batch_indices, :, i] = relevance

    torch.save(relevances, "test_outputs/relevances.pkl")
    torch.save(input_signal, "test_outputs/raw_signal.pkl")


def register(model, dummy_input, input_name): # the imported composites are used
    model.eval()
    traced_model = lxt_comp.register(model, dummy_inputs={input_name: dummy_input}, verbose=False) # "input" / "x" TESTVALUE
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
    return result



def batched_lrp_loop(data, y, batched_positions, traceback=None):
    full_batch_indices = torch.arange(data.shape[0], dtype=torch.long)

    for positions in batched_positions.T:

        batch_indices_filtered = full_batch_indices[positions!=-1]
        positions_filtered = positions[positions!=-1].to(torch.int64)

        if traceback != None:
            kmer = traceback[batch_indices_filtered, positions_filtered]
            y_current = y[batch_indices_filtered, positions_filtered, kmer].sum()

        else:
            y_current = y[batch_indices_filtered, positions_filtered, :].sum()
            
        data.grad = None
        y_current.backward(retain_graph=True)
        relevance = data.grad[batch_indices_filtered, 0,:]
        yield (relevance, batch_indices_filtered, positions)



def segmentation_loop(relevance_gen, segmentation_function_out_shape, batchsize, downsampled_size):
    segmentation_function, segment_shape = segmentation_function_out_shape

    segments_batch = torch.zeros((batchsize, downsampled_size, segment_shape[0], segment_shape[1]), dtype=torch.float) - 2 # this tensor is longer than needed and will never be filled because n_moves is shorter than downsampled_size, but its easier to unbatchify

    for i, (relevance, batch_indices, motif_indices) in enumerate(relevance_gen):

        segment_indices = segmentation_function(relevance).to("cpu")
        z = torch.zeros((batchsize, segment_shape[0], segment_shape[1]), dtype=torch.float) - 2
        z[batch_indices] = segment_indices # if a sample in the batch no longer has moves then just keep adding 0 as segment until all samples in batch have no more moves
        segments_batch[torch.arange(batchsize),motif_indices,:,:] = z

    return segments_batch

def peak_segmentation(number_peaks):
    segment_shape = (number_peaks, 2)
    def func(relevances):
        result = torch.zeros((relevances.shape[0], number_peaks, 2)) - 1 # not -1 because i dont want to delete later
        for i,relevance in enumerate(relevances):
            relevance = torch.nn.functional.pad(relevance.abs(), (1,1), "constant", 0)  # pad at beginning and end if peak is at position 0 then it will not find the peak because there is nothing to the left
            relevance = relevance / (relevance.max())
            peaks = find_peaks(relevance.detach().cpu(), distance=3, height=0.2)
            peaks = np.array([peaks[0]-1, peaks[1]["peak_heights"]]) # -1 because of the padding

            peaks = peaks[:,np.argsort(peaks, 1)[1]]
            peaks = np.flip(peaks, axis=1).T
            peaks = torch.from_numpy(peaks.copy())
            peaks = peaks[:number_peaks].to(torch.float)
            result[i,:peaks.shape[0],torch.arange(2)] = peaks
            
        return result
    return func, segment_shape

def argmax_segmentation():
    def func(relevances):
        indices = torch.argmax(torch.abs(relevances), dim=1, keepdim=False)
        return indices.view(-1,1,1).to(torch.float)
    return func, (1,1)



def forward_and_lrp(model_enc, input_signal, seqdist, search_algorithm, read,  save_relevance=False): #MAIN LRP FUNCTION
    print(read)
    device = next(model_enc.parameters()).device
    dtype = torch.float16 if half_supported() else torch.float32
    input_signal = input_signal.to(dtype).to(device)

    scores = model_enc(input_signal.requires_grad_(True))

    if search_algorithm == "viterbi":
        search_result = run_viterbi(scores, seqdist, len(seqdist.alphabet), seqdist.n_base)
        scores = scores.permute(1,0,2)
        kmers = search_result["traceback"]
    else:
        search_result = run_beam_search(scores) # chageable
        kmers = None
        
    batchsize, downsampled_len, _ = scores.shape
    moves = search_result["moves"]
    batched_moves = batch_positions(moves) # -> shape [#moves, nbatches]

    relevance_gen = batched_lrp_loop(input_signal, scores, batched_moves, kmers)

    if save_relevance: # just for development
        save_relevance_and_signal(relevance_gen, input_signal, batched_moves.shape[1])
        assert False # TESTVALUE


    segments = segmentation_loop(relevance_gen, peak_segmentation(5), batchsize, downsampled_len)
    search_result["segments"] = segments

    return search_result





def basecall_and_lrp(model, reads, search_algorithm, chunksize=4000, overlap=100, batchsize=4,
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
    input_name = "input" if search_algorithm == "viterbi" else "x"
    model_enc = register(model.encoder, dummy_input, input_name)

    scores = (
        (read, forward_and_lrp(model_enc, batch, model.seqdist, search_algorithm, read, save_test_relevance)) for read, batch in batches
    )

    results = (
        (read, stitch_results(scores, end - start, chunksize, overlap, model.stride, "", reverse))
        for ((read, start, end), scores) in unbatchify(scores)
    )

    return (
        (read, fmt(model.stride, attrs, read.trimmed_samples, rna))
        for read, attrs in results
    )


