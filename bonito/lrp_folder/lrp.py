import torch
import numpy as np
from bonito.lrp_folder.LRP_composites import lxt_comp, zennit_comp1,zennit_comp2, zennit_comp3, zennit_comp4, zennit_comp_first_conv
from bonito.util import chunk, batchify, unbatchify, half_supported
from koi.decode import to_str
from scipy.signal import find_peaks
from bonito.lrp_folder.stitching import stitch_results
from bonito.lrp_folder.search import run_beam_search, run_viterbi
import time

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




def register(model, dummy_input, input_name): # the imported composites are used
    model.eval()

    parent = lxt_comp.register(model)#, dummy_inputs={input_name: dummy_input}, verbose=False) # "input" / "x" TESTVALUE
    
    if "namedserial" == model[0].name: # using beamsearch adds a namedserial ontop of conv
        zennit_comp_first_conv.register(model[0][0][0])
        zennit_comp1.register(model[0][0][1])
        zennit_comp2.register(model[0][0][2])
        zennit_comp3.register(model[0][0][3])
        zennit_comp4.register(model[0][0][4])
    else:
        zennit_comp_first_conv.register(model[0][0])
        zennit_comp1.register(model[0][1])
        zennit_comp2.register(model[0][2])
        zennit_comp3.register(model[0][3])
        zennit_comp4.register(model[0][4])

    return model


def batch_positions(positions):
    """
    get move positions (index where a base is called) and create a tensor with those indices
    then backward() with one base for each sample of the batch instead of each individually
    if one sample of the batch has fewer bases/moves than the other then its filled with -1
    """
    result = torch.zeros_like(positions, dtype=torch.long)-1
    most_moves_in_batch = 0
    for i, sample in enumerate(positions):
        moves = torch.argwhere(sample==1).squeeze()
        l = len(moves)
        result[i, :l] = moves
        if l > most_moves_in_batch:
            most_moves_in_batch = l

    result = result[:,:most_moves_in_batch]
    return result



def batched_lrp_loop(data, y, batched_positions, traceback=None):
    full_batch_indices = torch.arange(data.shape[0], dtype=torch.long)

    for positions in batched_positions.T:
        batch_indices_filtered = full_batch_indices[positions!=-1] # batch_indices_filtered has the actually viable positions of bases for later
        # if there are no more bases in a sample of a batch, then it just computes the last one over and over, because it cant do backward() without a "full" batch
        if traceback != None:
            kmer = traceback[full_batch_indices, positions]
            y_current = y[full_batch_indices, positions, kmer].mean()*1000 # TESTVALUE so gradient doesnt go to 0 at end

        else:
            y_current = y[full_batch_indices, positions, :].mean()

        data.grad = None
        y_current.backward(retain_graph=True)
        relevance = data.grad[batch_indices_filtered, 0,:]

        yield (relevance, batch_indices_filtered, positions[positions!=-1])

def wrapped_batched_lrp_loop(data, y, batched_positions, traceback=None, save_relevance=False):
    if save_relevance:
        batchsize, _, seq_len = data.shape
        relevances = torch.zeros((batchsize,seq_len, batched_positions.shape[1]), dtype=data.dtype, device=data.device)
    for i, (relevance, batch_indices_filtered, positions) in enumerate(batched_lrp_loop(data, y, batched_positions, traceback)):
        if save_relevance:
            relevances[batch_indices_filtered, :, i] = relevance
        yield (relevance, batch_indices_filtered, positions)
    if save_relevance:
        torch.save(relevances, "test_outputs/relevances.pkl")
        torch.save(data, "test_outputs/raw_signal.pkl")
        assert False
    # TODO rethink this, how do i appropriately save it, now it keeps getting overwritten



# region
# def segmentation_loop(relevance_gen, segmentation_function_out_shape, batchsize, downsampled_size):
#     segmentation_function, segment_shape = segmentation_function_out_shape

#     segments_batch = torch.zeros((batchsize, downsampled_size, segment_shape[0], segment_shape[1]), dtype=torch.float) - 2 

#     start_time = time.time()
#     gen_time, seg_time, cpu_time, assign_time = 0, 0, 0, 0  # Track time for different steps

#     for i, (relevance, batch_indices, motif_indices) in enumerate(relevance_gen):

#         t0 = time.time()
#         segment_indices = segmentation_function(relevance)  # Time for segmentation function
#         t1 = time.time()
#         seg_time += t1 - t0

#         segment_indices = segment_indices.to("cpu")  # Time for transferring to CPU
#         t2 = time.time()
#         cpu_time += t2 - t1

#         z = torch.zeros((batchsize, segment_shape[0], segment_shape[1]), dtype=torch.float) - 2
#         z[batch_indices] = segment_indices  # Time for assigning segment indices
#         t3 = time.time()

#         segments_batch[torch.arange(batchsize), motif_indices, :, :] = z  # Assigning to main batch tensor

 

#         total_time = time.time() - start_time
#         print(f"Total Time: {total_time:.4f}s")
#         print(f"Segmentation Function Time: {seg_time:.4f}s")
#         print(f"CPU Transfer Time: {cpu_time:.4f}s")
#         print(f"Assignment Time: {assign_time:.4f}s")

#     return segments_batch
# endregion


def segmentation_loop(relevance_gen, segmentation_function_out_shape, batchsize, downsampled_size):
    segmentation_function, segment_shape = segmentation_function_out_shape
    segments_batch = torch.zeros((batchsize, downsampled_size, segment_shape[0], segment_shape[1]), dtype=torch.float) - 2 # this tensor is longer than needed and will never be filled because n_moves is shorter than downsampled_size, but its easier to unbatchify

    for i, (relevance, batch_indices, motif_indices) in enumerate(relevance_gen):

        segment_indices = segmentation_function(relevance).to("cpu")
        segments_batch[batch_indices, motif_indices,:,:] = segment_indices

    return segments_batch

def peak_segmentation(number_peaks):
    segment_shape = (number_peaks, 2)

    def func(relevances):
        result = torch.zeros((relevances.shape[0], number_peaks, 2)) - 1 # not -2 because i dont want to delete later

        for i,relevance in enumerate(relevances):

            relevance = torch.nn.functional.pad(relevance.abs(), (1,1), "constant", 0)  # pad at beginning and end if peak is at position 0 then it will not find the peak because there is nothing to the left
            relevance = relevance / (relevance.max())
            peaks = find_peaks(relevance.detach().cpu(), distance=3, height=0.2)
            peaks = np.array([peaks[0]-1, peaks[1]["peak_heights"]]) # -1 because of the padding

            peaks = peaks[:,np.argsort(peaks, 1)[1]]
            peaks = np.flip(peaks, axis=1).T
            peaks = torch.from_numpy(peaks.copy())
            peaks = peaks[:number_peaks].to(torch.float)
            result[i, :peaks.shape[0], torch.arange(2)] = peaks

            
        return result
    
    return func, segment_shape


def argmax_segmentation():
    def func(relevances):
        indices = torch.argmax(torch.abs(relevances), dim=1, keepdim=False)
        return indices.view(-1,1,1).to(torch.float)
    return func, (1,1)



def forward_and_lrp(model_enc, input_signal, seqdist, search_algorithm,  save_relevance=False): #MAIN LRP FUNCTION
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


    relevance_gen = wrapped_batched_lrp_loop(input_signal, scores, batched_moves, kmers, save_relevance)

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
        (read, forward_and_lrp(model_enc, batch, model.seqdist, search_algorithm, save_test_relevance)) for read, batch in batches
    )


    # scores = next(scores)[1]
    # moves = torch.argwhere(scores["moves"])
    # print(moves)
    # segments = scores["segments"]
    # for move in moves:
    #     print(segments[move[0], move[1], :, :])
    # print("done")

    results = (
        (read, stitch_results(scores, end - start, chunksize, overlap, model.stride, "", reverse))
        for ((read, start, end), scores) in unbatchify(scores)
    )

    return (
        (read, fmt(model.stride, attrs, read.trimmed_samples, rna))
        for read, attrs in results
    )


