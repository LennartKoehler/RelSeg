the crfencoder only balloons the channel dimension to the desired 6^4 (6-mers, this is set in the config.toml n_base = 4, state_len = 5 (+1)) from 512 (channel dimension before)
dimension on tensor of neural network: batches (e.g. 128) x length_of_sequence (2000) x channels (512/4096)
input is of size 128 x 1 x 12000
the convolutions get the input to the desired 128 x 512 x 2000 , transformers probuably dont change that