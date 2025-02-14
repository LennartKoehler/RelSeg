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

    return segments