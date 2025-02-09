def run_viterbi(y, seqdist):
    # only use with use_koi = False
    y_copy = y.detach().clone()
    traceback_kmer = decode(seqdist, y_copy)[0,:]
    kmers = [(position, kmer.detach().cpu()) for position, kmer in enumerate(traceback_kmer) if kmer % 5 != 0] # filter out those were new kmers are predicted, remove "N"
    return kmers


def decode(seqdist, scores):
    scores = seqdist.posteriors(scores.to(torch.float32)) + 1e-8 # probabilities
    tracebacks = viterbi(seqdist, scores.log()).to(torch.int16).T
    return tracebacks

    
def viterbi(seqdist, scores):
    traceback = seqdist.posteriors(scores, Max) # one motif (last dimension) in traceback has score 1 and the rest 0
    a_traceback = traceback.argmax(2) #IMPORTANT take a_traceback (index of kmer (or base))
    return a_traceback

def plot_raw_output(y):
    y = y.detach().cpu().numpy()
    pos = np.arange(len(y[0,:,0]))
    kmer = np.arange(len(y[0,0,:]))
    pos, kmer = np.meshgrid(kmer, pos)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(kmer, pos, y[0,:,:], cmap='viridis')
    fig.colorbar(surf, shrink=0.5, aspect=10)
    plt.savefig("plots/raw_output")


def varying_gamma(data, traced):
    offset = 100
    distance_between = 5
    relevances = []
    gammas = itertools.product([0.1, 0.5, 100], [0.01, 0.05, 0.1, 1])
    gammas, gammas2 = itertools.tee(gammas)
    for conv_gamma, lin_gamma in gammas:
        zennit_comp = LayerMapComposite([
            (nn.Conv1d, z_rules.Gamma(conv_gamma)),
            (nn.Linear, z_rules.Gamma(lin_gamma)),
        ])
        zennit_comp.register(traced)
        y = traced(data.requires_grad_(True))

        data.grad = None
        y_current = y[0,offset,:].sum()
        y_current.backward(retain_graph=True)

        relevance = data.grad[0, 0]
        relevance = relevance / (abs(relevance).max())
        relevance = relevance.detach().cpu().numpy()
        zennit_comp.remove()
        relevances.append(relevance)
    


    fig, axs = plt.subplots(len(relevances))
    offset *= 6
    distance_between *= 6
    fig.set_figheight(50)
    fig.set_figwidth(15)
    for i,relevance in enumerate(relevances):
        axs[i].plot(relevance[offset - 5*distance_between : offset + 15*distance_between])
        axs[i].set_title(f"gamma_conv, gamma_lin: {next(gammas2)}")
    plt.savefig("plots/gamma_variation", dpi=500)


def stitch_segments(chunks, chunksize, overlap, length, stride, reverse=False):
    """
    custom stitching function for segments, as these are in the original dimension, not downsampled dimension
    same as other stitch function just dont divide by stride
    this also doesnt introduce small offset if length of read is not divisible by stride

    note:
    tried to use original stitch function and pack the segments up into shape [NBatch, downsampled_len, stride] but if read_len is not divisible by stride
    then that might cause an offset after the first (not full) chunk that is stitched together. E.g. there might be an offset of 2 afterwards because read_len % stride = 2
    and therefore the stub is 2 shorter than it should be

    Stitch chunks together with a given overlap
    """
    if chunks.shape[0] == 1: return chunks.squeeze(0)

    semi_overlap = overlap // 2
    start, end = semi_overlap , (chunksize - semi_overlap) 
    stub = (length - overlap) % (chunksize - overlap)
    first_chunk_end = (stub + semi_overlap)  if (stub > 0) else end # IMPORTANT this is offset a bit if length % stride != 0
        
    if reverse:
        chunks = list(chunks)
        return concat([
            chunks[-1][:-start], *(x[-end:-start] for x in reversed(chunks[1:-1])), chunks[0][-first_chunk_end:]
        ])
    else:
        segments = concat([
            chunks[0, :first_chunk_end], *chunks[1:-1, start:end], chunks[-1, start:]
        ])
        segments_indices = torch.where(segments==1)[0] # get index of ones (convert back :) ) IMPORTANTTODO duplices etc are not included
        return segments_indices


def stitch_segments_indices(chunks, chunksize, overlap, length, stride, reverse=False):
    """
    Stitch chunks together with a given overlap
    1.  Chunks has two values for each segment, the first is where in the raw_signal the segment is
        the second is the index of the position of the base in the downsampled space which this segment corresponds to
    2.  The second value is therefore used for upper and lower cutoff, because this is the sapce where the bases are removed, therefore
        these are also the segments that need to be removed.
        E.g.: A base might be right after the semi_overlap that is removed from the chunk, but the segment_position in the raw signal space is
        before the semi_overlap. -> this segment should not be removed
    """
    if chunks.shape[0] == 1: return chunks.squeeze(0)
    n_chunks = chunks.shape[0]

    semi_overlap = overlap // 2
    start, end = semi_overlap , (chunksize - semi_overlap) 
    stub = (length - overlap) % (chunksize - overlap)
    first_chunk_end = (stub + semi_overlap)  if (stub > 0) else end # IMPORTANT this is offset a bit if length % stride != 0

    offset = torch.arange(0, (n_chunks-2)*(chunksize-overlap)+1, chunksize-overlap)
    offset += stub
    offset = torch.cat([torch.tensor([0]), offset])

    cutoffs_upper = torch.tensor([end // stride for _ in range(n_chunks-2)])
    cutoffs_upper = torch.cat([torch.tensor([first_chunk_end // stride]), cutoffs_upper, torch.tensor([chunksize // stride])]).unsqueeze(1)

    cutoffs_lower = torch.tensor([start // stride for _ in range(n_chunks-1)])
    cutoffs_lower = torch.cat([torch.tensor([0]),cutoffs_lower]).unsqueeze(1)

    condition_upper = chunks[:,:,1] >= cutoffs_upper
    condition_lower = chunks[:,:,1] < cutoffs_lower
    segments = chunks.to(torch.float)
    segments[condition_upper] = torch.nan
    segments[condition_lower] = torch.nan

    segments[:,:,0] += offset.unsqueeze(1)
    segments[:,:,1] += offset.unsqueeze(1) // stride

    segments_cleaned = segments[~torch.isnan(segments)].view(-1,2).to(torch.long)

    sorted_indices = torch.argsort(segments_cleaned[:,1])
    sorted_segments = segments_cleaned[sorted_indices].detach().cpu()

    return sorted_segments[:,0]

def format_segments_bit(segments, stride, batchsize, downsampled_len): # unused
    # segments of shape [batches, segment_position] (int position of segment) -> [NBatches, original_len] (1 and 0, one for segment location)
    segments_bit = torch.zeros((batchsize, stride*downsampled_len), device=segments.device, dtype=segments.dtype) # stride*downsampled_len != chunksize  but close
    segments_bit.index_put_((torch.arange(batchsize, dtype=torch.long).unsqueeze(1), segments), torch.ones_like(segments), accumulate=True)
    segments_bit[:,0] = 1
    return segments_bit