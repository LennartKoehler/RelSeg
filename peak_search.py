import torch
import numpy as np
import pandas as pd

def read_table(file_path):
    df = pd.read_csv(file_path, sep='\t', header=0, names=['read_id', 'base'] + [f'val_{i}' for i in range(10)])
    
    # Convert the last 10 columns into a 5x2 matrix for each row
    def create_matrix(row):
        # Extract the last 10 values and reshape into 5x2 matrix
        matrix_values = row[-10:].values.astype(float)
        return matrix_values.reshape(5, 2)
    
    # Create the matrix for each row
    df['start'] = df.apply(lambda row: create_matrix(row), axis=1)
    
    # Drop the original individual columns
    df = df.drop(columns=[f'val_{i}' for i in range(10)])
    return df



segments = read_table("test_outputs/viterbi_segments.tsv")
segments = segments.loc[segments["read_id"] ==  "048f077a-a8a0-4dda-95f5-3c43be3a1274"]
segments = segments["start"].values[:]
segments = np.stack(segments)




def get_sort_shifts(arr):
    sorted_indices = np.argsort(arr)  # Get the indices that would sort the array
    original_indices = np.argsort(sorted_indices)  # Get the original positions after sorting
    shifts = original_indices - np.arange(len(arr))  # Compute shifts
    return shifts
        
def segment_sort(segments):
    def get_max(current, upcoming):
        current = current[current[:,0]!=-1,:] # since the -1 are at the end i can actually do this without having to adjust the returned indices
        upcoming = upcoming[upcoming[:,0]!=-1,:]
        result_scores = np.zeros(5)-9999
        result_indices = np.zeros(5, dtype=int)+6

        for i, (p_upcoming, s_upcoming) in enumerate(upcoming):
            max_score = -9999
            index = 6

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
        if i == 396:
            print("starting")
        current = dp_matrix_scores_positions[i,:,:]
        upcoming = segments[i+1,:,:]
        scores, indices = get_max(current, upcoming)
        if (scores<0).all(): # if all paths "have ended" -> there is no way to connect paths here
            invalid_segments.append(i+1)
            segments = np.delete(segments, i+1, axis=0)
            #dp_matrix_scores_positions = np.delete(dp_matrix_scores_positions, i, axis=0)
            # i -= 1
            max_segments -= 1
        else:
            dp_matrix_scores_positions[i+1,:,1] = scores
            dp_matrix_traceback[i+1,:] = indices
            i += 1

    print(len(invalid_segments))
    # IMPORTANT some segments are wrong and are way ahead of the previous -> all segments between them are therefore removed
    dp_matrix_scores_positions = dp_matrix_scores_positions[:-len(invalid_segments)]
    dp_matrix_traceback = dp_matrix_traceback[:-len(invalid_segments)]
    print(dp_matrix_scores_positions[-1])
    positions = []
    def traceback(matrix, row, col):
        if row == 0:
            return
        print(row,col)
        if col == 6:
            print(row, col, segments[row-4:row+4, :, 0])
            pass
        positions.append(segments[row, col, 0])
        traceback(matrix, row-1, matrix[row,col])
    print(dp_matrix_traceback[-6:])
    print(dp_matrix_traceback[-1]) # TODO somehow i trace through i 6
    # traceback(dp_matrix_traceback, dp_matrix_traceback.shape[0]-1, 0)

    return positions[::-1]



positions = segment_sort(segments)
# print(get_sort_shifts(np.array(positions)))
# print(get_sort_shifts(segments[:,0,0]))

