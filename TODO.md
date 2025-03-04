IDEA: calc relevance for every fifth move at once, see what happens -> faster, can they be seperated?

search algorithm as argument and how do i propagate that information

in batch_lrp can i take the sum there? need to replace with lxt sum -> dont need to replace (shouldnt replace)

why are some relevance bugged, see lrp dev -> because not every batch is "full"



TODO:
basecall_and_lrp multiprocessing like original basecaller?

in the backward, maybe in the conservation wrap, set all negative inputs to zero

set rules for conv1d (gamma?) except for first conv (can take neg values -> should be wsquare or lrp-0)

check if forward pass stayed the same