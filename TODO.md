IDEA: calc relevance for every fifth move at once, see what happens -> faster, can they be seperated?

search algorithm as argument and how do i propagate that information

in batch_lrp can i take the sum there? need to replace with lxt sum -> dont need to replace (shouldnt replace)

why are some relevance bugged, see lrp dev -> because not every batch is "full"

TODO:
basecall_and_lrp multiprocessing like original basecaller?

in the backward, maybe in the conservation wrap, set all negative inputs to zero

somewhere in the transformer (not attention) relevance is already lost

replace attention with the attention of lxt, also use the initialization -> check that the forward stays the same