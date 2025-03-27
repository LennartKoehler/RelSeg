def batched_lrp(data, y, positions, segmentation_function):
    batchsize, _ , seq_len = data.shape
    device = data.device
    batched_positions = batch_positions(positions) # -> shape [#moves, nbatches]
    dupes_counter = 0
    segments_batch = torch.zeros(batchsize,batched_positions.shape[0], device=data.device, dtype=torch.long)
    full_batch_indices = torch.arange(data.shape[0], dtype=torch.long)

    memory_size = 5
    middle_peaks = memory_size//2
    previous_peaks = [deque([],maxlen=memory_size) for sample in range(batchsize)] # one queue for each batch

    for i,motif_indices in tqdm(enumerate(batched_positions), total=batched_positions.shape[0]):
        i_shift = i-middle_peaks
        data.grad = None
        batch_indices = full_batch_indices[motif_indices!=-1]
        motif_indices = motif_indices[motif_indices!=-1].to(torch.int64)
        y_current = y[batch_indices, motif_indices, :].sum()
        y_current.backward(retain_graph=True)
        relevance = data.grad[batch_indices, 0,:]


        segment_indices = segmentation_function(relevance)
        for j in full_batch_indices: # segment_indices doesnt always return something for every sample!
            if j in batch_indices:
                previous_peaks[j].append(segment_indices[j]) # to each deque append the newest peaks
            else:
                previous_peaks[j].append(np.array([[-1],[-1]])) # add -1 as filler variable for a batch that no longer has segments

        if i_shift>=0:
            z = torch.zeros(batchsize, device=data.device, dtype=torch.long)

            previous_peaks = [remove_duplicates(previous_peaks_sample) for previous_peaks_sample in previous_peaks]

            z[batch_indices] = torch.tensor([peaks[middle_peaks][0,0] for peaks in previous_peaks], dtype=torch.long, device=device ) # if a sample in the batch no longer has moves then just keep adding 0 as segment until all samples in batch have no more moves

            segments_batch[:,i_shift] = z



    print("duplicate_segments: ",dupes_counter)
    return segments_batch

def duplicate_indices(arr):
    arr = np.array(arr)
    unique_elements, counts = np.unique(arr, return_counts=True)
    duplicates = unique_elements[counts > 1]  # Find elements that appear more than once
    return {dup: np.where(arr == dup)[0]+1 for dup in duplicates if dup != -1}


def next_best_in_interval(peaks, lower_boundary_peak, upper_boundary_peak):
    for i,peak in enumerate(peaks[1:]):
        if peak[0] > lower_boundary_peak and peak[0] < upper_boundary_peak:
            return i,peak
    
    return 0,(0,0) # there is no peak in the interval


def remove_duplicates(previous_peaks): # init indices should be 0 (max value since peaks are sorted)
    #TODO dont check if newest value is duplicate because i want values to the right of it, so always check if the middle value is duplicate


    #TODO can probably speed up by always testing if new value is duplicate, only if yes run this elaborate function
    #DONE what if nbb is before the previous ones (or nba after the next), so that now i changed the order. I dont want to change the order, ideas:
    # 1. take nbb that is in between current and previous: might not exist
    # 2. if nbb before previous then set nbb to that peak and therefore add them to the dupe list and resolve it later, problem: a peak now exists that might not have existed before
    # 2.... and the peak might jump back and forth

    #IMPORTANT is previous_peaks by copy or reference? i want to change it in here, not sure if it needs to be by reference as i pass it on in recursion
    
    duplicates = duplicate_indices([peaks[0,0] for peaks in previous_peaks][1:-1]) # for the first and last there are no peaks before and after to use for the interval
    if not bool(duplicates):
        return previous_peaks
    else:
        for dupe_value, dupe_indices in duplicates.items():
            dupe_indices += 1 # because im using [1:-1] of previous peaks

            left_dupe = previous_peaks[dupe_indices[0]] # always compare pairs, even if three have the same value
            before_left_dupe = previous_peaks[dupe_indices[0]-1]
            right_dupe = previous_peaks[dupe_indices[1]]
            after_right_dupe = previous_peaks[dupe_indices[1]+1]

            index_peak_nbb, next_best_before = next_best_in_interval(left_dupe, before_left_dupe[0,0], left_dupe[0,0])
            index_peak_nba, next_best_after = next_best_in_interval(right_dupe, right_dupe[0,0], after_right_dupe[0,0])

            if not index_peak_nbb and not index_peak_nba: # if there is neither a peak before for the left or after for the right, then take the first peak of the neighbors
                left_dupe = np.insert(left_dupe, 1, before_left_dupe[0,:], axis=0)
                right_dupe = np.insert(right_dupe, 1, after_right_dupe[0,:], axis=0)
                index_peak_nbb = 1
                index_peak_nba = 1
                next_best_before = before_left_dupe[0,:]
                next_best_after = after_right_dupe[0,:]
            
            # if not index_peak_nbb and not index_peak_nba:
            #     # if there is no alternative peak to take then just move the peak
            #     # very simple
            #     left_dupe[0,0] -= 1
            #     right_dupe[0,0] += 1

            else:
                max_combination = np.argmax(next_best_before[1] + right_dupe[0,1], next_best_after[1] + left_dupe[0,1]) # from the dupes, change that peak of max_combination, as when this is changed the score is maximized
                
                # for one of the two remove the best peak and instead insert the next best at position 0
                if max_combination == 0:
                    left_dupe[0] = next_best_before
                    left_dupe = np.delete(left_dupe, index_peak_nbb, axis=0)
                if max_combination == 1:
                    right_dupe[0] = next_best_after
                    right_dupe = np.delete(right_dupe, index_peak_nba, axis=0)     
        return remove_duplicates(previous_peaks)

