import torch
from koi.decode import beam_search
from koi.ctc import Max

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


def number_to_dna_ascii(sequence):
    mapping = torch.tensor([0, 65, 67, 71, 84])  # [0, ord('A'), ord('C'), ord('G'), ord('T')]
    # Use index_select to map the values
    return mapping[sequence.to(torch.long)].to(torch.int8)

def viterbi(seqdist, scores):
    scores = seqdist.posteriors(scores.to(torch.float32)) + 1e-8 # probabilities
    traceback = seqdist.posteriors(scores.log(), Max) # one motif (last dimension) in traceback has score 1 and the rest 0
    a_traceback = traceback.argmax(2) #IMPORTANT take a_traceback (index of kmer (or base))
    return a_traceback

def run_viterbi(scores, seqdist, alphabet_len, n_base):
    # only use with use_koi = False
    scores_copy = scores.detach().clone()
    traceback = viterbi(seqdist, scores_copy)
    moves = (traceback % alphabet_len) != 0
    paths = 1 + (torch.div(traceback, alphabet_len, rounding_mode="floor") % n_base) # return which of the 4 bases
    paths = torch.where(moves, paths, 0).to("cpu").T # only get base when move TRUE -> move != 0
    paths_encoded = number_to_dna_ascii(paths)
    return {
        "sequence": paths_encoded,
        "moves": moves.to(torch.int8).to("cpu").T,
        "qstring": torch.zeros_like(paths),
        "traceback": traceback.to("cpu").T,
    }



