work on attention model

gatedmlp shouldnt be a problem, only need to think about activation function

need to probably replace all flash attention statements and the nwork out that the model can still be loaded from dict

for RoPE see equation (34) of RoPE paper

reverse flag removed, dont need


residuals cause instability i think, maybe use the idea from explain transformer beyond... for residuals

the sum of inp_rel and out_rel of add2 is always the same, but since only one inp_relevance is passed on to the next function it is halfed in the out_relevance of that function, to see that one would alwys need to look at the first out_relevance


TIME COMPLEXITY:
what takes so long: forward takes 0.5s, backward takes 0.3s
since we go backward for each base (ca. 300) this adds up
-> lrp takes ~300x longer as basecalling (with chunksize 12000) -> 90 s/batch













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


