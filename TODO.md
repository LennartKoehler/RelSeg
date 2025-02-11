DONE:
check if output with use_koi is same as withotut except for the crfencoder (blanks) -> NO use_koi causes different moves???

check if stitching of segments is correct, theere is a slight mismatch between stride*downsampled_size and chunksize
SOLUTION: chunksize should always be divisible by stride!!!!! even if not useing lrp


TODO:

i think if a base is after the semioverlap, but the segmentation says the position is before, then this segment is lost
-> maybe i can add 0s in the segments for where there are no moves, then i can stitch the segments like the others, by just removing the beginning of the sequence and not filtering by size
-> maybe add another value to the segments which states from which sequence index it came from
    -> also return the batched_positions from the lrp loop, this states exactly where the segments came from

add trimmed to segment positions


IDEA: calc relevance for every fifth move at once, see what happens -> faster, can they be seperated

