
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
    data = data[5]
    data = data[None, :, :]
    dtype = torch.float16 if half_supported() else torch.float32
    data = data.to(dtype)
    return read, data