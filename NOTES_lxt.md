work on attention model

gatedmlp shouldnt be a problem, only need to think about activation function

need to probably replace all flash attention statements and the nwork out that the model can still be loaded from dict

for RoPE see equation (34) of RoPE paper

needed to change rotary embedding to not have any learnable parameters to make the import work