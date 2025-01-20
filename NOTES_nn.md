the crfencoder only balloons the channel dimension to the desired 6^4 (6-mers, this is set in the config.toml n_base = 4, state_len = 5 (+1)) from 512 (channel dimension before)
dimension on tensor of neural network: batches (e.g. 128) x length_of_sequence (2000) x channels (512/4096)
input is of size 128 x 1 x 12000
the convolutions get the input to the desired 128 x 1000 x 512 , transformers probably dont change that

bachsize (11 vs 41) is chosen by me
output_length = input_length / 6
channel dimension difference (5120 vs 4096) because of use_koi/ not use_koi:
for training the empty (ACGT N) is added and for basecalling it is not / it is added later in beamsearch

train
data_ shape: torch.Size([11, 1, 4000])
scores_ shape: torch.Size([11, 668, 5120])

basecall
batch shape: torch.Size([41, 1, 12000])
scores shape: torch.Size([41, 2000, 4096])