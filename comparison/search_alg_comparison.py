import numpy as np


viterbi = np.load("comparison/viterbi_move_positions_048f077a-a8a0-4dda-95f5-3c43be3a1274.npy")
beamsearch = np.load("comparison/beamsearch_move_positions_048f077a-a8a0-4dda-95f5-3c43be3a1274.npy")
viterbi = [p[0] for p in viterbi]

total = len(viterbi)
same = 0

for position in viterbi:
    if position in beamsearch:
        same +=1

print(total, same)

total = len(beamsearch)
same = 0

for position in beamsearch:
    if position in viterbi:
        same +=1

print(same / total)