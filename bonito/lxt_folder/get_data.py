
from bonito.crf.basecall import *


def get_data(reads, chunksize=4000, overlap=100, batchsize=32,
             reverse=False, rna=False):
    """
    Basecalls a set of reads.
    """

    chunks = (
        ((read, 0, read.signal.shape[-1]), chunk(torch.from_numpy(read.signal), chunksize, overlap))
        for read in reads
    )

    batches = (batchify(chunks, batchsize=batchsize))
    read, data = next(batches)
    read = read[0]
    data = data[0]
    data = data[None, :, :]
    dtype = torch.float16 if half_supported() else torch.float32
    data = data.to(dtype)
    return read, data

def run_beam_search(scores, beam_width=32, beam_cut=100.0, scale=1.0, offset=0.0, blank_score=2.0):
    with torch.cuda.device(scores.device):
        sequence, qstring, moves = beam_search( # IMPORTANT this is where the actual sequence is determined from the nn scores
            scores, beam_width=beam_width, beam_cut=beam_cut,
            scale=scale, offset=offset, blank_score=blank_score
        )
    return {
        'moves': moves,
        'qstring': qstring,
        'sequence': sequence,
    }