work on attention model

gatedmlp shouldnt be a problem, only need to think about activation function

need to probably replace all flash attention statements and the nwork out that the model can still be loaded from dict

for RoPE see equation (34) of RoPE paper

reverse flag removed, dont need



















when comparing the difference between relevance (heatmap) of the same move (kmer) e.g. move at position 10 and then comparing relevance obtained at position 10 and 11 are much more similar (10-fold) than comparing relevance 10 and 9 (assuming move is at 10, so 9 is the previous kmer)
below is the comparison of 10 and 9

data.grad = None
y_current = y[0, position, :].sum()
y_current.backward(retain_graph=True)

heatmap = data.grad[0, 0]

data.grad = None
y_current = y[0, position-1, :].sum()
y_current.backward(retain_graph=True)
heatmap2 = data.grad[0, 0]
print((heatmap - heatmap2).abs().sum())



relevance of setting
y_current = y[0, position : position + 2, :].sum()

is not the same as sum of relevances obtained when setting 
y_current = y[0, position, :].sum()
y_current = y[0, position+1, :].sum()
heatmaps.sum()

i thought it should be the same?


