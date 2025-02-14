import torch
import numpy as np
import pandas as pd

def data_loader(filename):
    data = pd.read_table(filename, header = 1)
    return data.iloc[:,2:].to_numpy().reshape((-1,5,2))


segments = data_loader("test_outputs/test2_segments.tsv")


last_value = [-1]




def get_sort_shifts(arr):
    sorted_indices = np.argsort(arr)  # Get the indices that would sort the array
    original_indices = np.argsort(sorted_indices)  # Get the original positions after sorting
    shifts = original_indices - np.arange(len(arr))  # Compute shifts
    return shifts
        
def segment_sort(segments):
    def get_max(current, upcoming):
        current = current[current[:,0]!=-1,:] # since the -1 are at the end i can actually do this without having to adjust the returned indices
        upcoming = upcoming[upcoming[:,0]!=-1,:]
        result_scores = np.zeros(5)
        result_indices = np.zeros(5, dtype=int)

        for i, (p_upcoming, s_upcoming) in enumerate(upcoming):
            max_score = 0
            index = -1

            for j, (p_current, s_current) in enumerate(current):
                score = s_upcoming + s_current
                if p_upcoming > p_current and score > max_score:
                    max_score = score
                    index = j
            result_scores[i] = max_score
            result_indices[i] = index
        return result_scores, result_indices
    

    dp_matrix_scores_positions = np.zeros_like(segments)
    dp_matrix_scores_positions[:,:,0] = segments[:,:,0]
    dp_matrix_traceback = np.zeros_like(segments[:,:,0], dtype=int)
    dp_matrix_scores_positions[0,:,1] = segments[0,:,1]

    invalid_segments = []
    i = 0
    max_segments = segments.shape[0]-1
    while i < max_segments:
        current = dp_matrix_scores_positions[i,:,:]
        upcoming = segments[i+1,:,:]
        scores, indices = get_max(current, upcoming)
        if sum(scores) ==0:
            invalid_segments.append(i+1)
            segments = np.delete(segments, i+1, axis=0)
            i -= 1
            max_segments -= 1
        else:
            dp_matrix_scores_positions[i+1,:,1] = scores
            dp_matrix_traceback[i+1,:] = indices
            i += 1


    dp_matrix_scores_positions = dp_matrix_scores_positions[:-len(invalid_segments)]
    dp_matrix_traceback = dp_matrix_traceback[:-len(invalid_segments)]
    positions = []
    def traceback(matrix, row, col):
        if row == 0:
            return
        positions.append(segments[row, col, 0])
        traceback(matrix, row-1, matrix[row,col])

    traceback(dp_matrix_traceback, dp_matrix_traceback.shape[0]-1, 0)
    return positions[::-1]
positions = segment_sort(segments)
print(get_sort_shifts(np.array(positions)))
print(get_sort_shifts(segments[:,0,0]))

