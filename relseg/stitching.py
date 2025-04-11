import numpy as np
import torch
from bonito.util import stitch, concat

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
            return segments_single(results, size, overlap, length, stride, reverse=reverse)
   
        return stitch_segments_indices(results, size, overlap, length, stride, reverse=reverse)
    else:
        if length < size:
            return results[0, :int(np.floor(length / stride))]
        
        return stitch(results, size, overlap, length, stride, reverse=reverse)


def segments_single(chunks, chunksize, overlap, length, stride, reverse=False):
    n_chunks, l, n_peaks, per_peak = chunks.shape
    results = chunks[0, :int(np.floor(length / stride)),:,:]
    results = results[results != -2].view(-1, n_peaks, per_peak)
    return results


def stitch_segments_indices(chunks, chunksize, overlap, length, stride, reverse=False):
    """
    Stitch segments together with a given overlap
    """
    if chunks.shape[0] == 1: return chunks.squeeze(0)
    n_chunks, l, n_peaks, per_peak = chunks.shape

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

    chunks[chunks == -2] = float("nan")
    chunks[:,:,:,0] += offset.view(-1,1,1)

    segments = concat([
            chunks[0, :first_chunk_end_down,:,:], *chunks[1:-1, start_down:end_down,:,:], chunks[-1, start_down:,:,:]
        ])
    segments = segments[~torch.isnan(segments)].view(-1, n_peaks, per_peak)

    segments = untangle_segments(segments.numpy())

    segments = make_end_dim(segments, length)

    return segments

def make_end_dim(segments, length):
    segments = np.stack([segments, np.roll(segments, shift=-1)], axis=-1)
    segments[-1,1] = length
    mask = np.any(segments == -1, axis=1)
    segments[mask,:] = -1
    return segments

def untangle_segments(segments):
    def best_fit(left, peaks, right):
        mask = (peaks[:, 0] > left) & (peaks[:, 0] < right)
        peaks = peaks[mask]
        if len(peaks) == 0:
            return -1,-1
        best_idx = np.argmax(peaks[:,1])
        return peaks[best_idx,:]
    
    def get_sort_shifts(arr):
        sorted_indices = np.argsort(arr)  # Get the indices that would sort the array
        original_indices = np.argsort(sorted_indices)  # Get the original positions after sorting
        shifts = original_indices - np.arange(len(arr))  # Compute shifts
        return np.abs(shifts)
    
    shifts = get_sort_shifts(segments[:,0,0])
    while sum(shifts) != 0:
        highest_shift = np.argmax(shifts)
        shifts[highest_shift] = 0


        left = np.max(segments[:highest_shift, 0, 0], initial=0) # max because there may be a 0 (no segment) in the one directly before/after, so just take the one that is not 0( since they are sorted its the max)
        right = np.max(segments[:highest_shift+2, 0, 0], initial=0) # is it correct to set the initial for right to 0? i think this is never needed?
        peaks = segments[highest_shift,:,:]

        segments[highest_shift, 0, :] = best_fit(left, peaks, right)

    
    return segments[:,0,0]