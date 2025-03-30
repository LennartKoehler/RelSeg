from bonito.io import *

def write_segmentation(read_id, sequence, segments, filename):
    assert len(sequence) == segments.shape[0], f"problem (read_id: {read_id}), number of bases and number of segments doesn't match: (bases: {len(sequence)}, segments:{segments.shape[0]})"
    with open(filename, "a") as f:
        for base, segment in zip(sequence, segments):
            f.write(f"{read_id}\t{base}")
            f.write(f"\t{int(segment[0])}\t{int(segment[1])}")
            f.write("\n")

def get_segment_filename():
    """
    Return the filename to use for the segment tsv.
    """
    stdout = realpath('/dev/fd/1')
    if sys.stdout.isatty() or stdout.startswith('/proc'):
        return 'segments.tsv'
    return '%s_segments.tsv' % splitext(stdout)[0]  

class LRP_Writer(Writer):
    def run(self):
        with CSVLogger(summary_file(), sep='\t') as summary:

            segment_filename = get_segment_filename()
            with open(segment_filename, "w") as f:
                f.write(f"read_id\tbase\tstart\tend\n")

            for read, res in self.iterator:

                seq = res['sequence']
                qstring = res.get('qstring', '*')
                mean_qscore = res.get('mean_qscore', mean_qscore_from_qstring(qstring))
                mapping = res.get('mapping', False)
                mods_tags = res.get('mods', [])

                samples = len(read.signal)
                read_id = read.read_id

                self.log.append((read_id, samples))

                if mean_qscore < self.min_qscore:
                    continue

                tags = [
                    f'RG:Z:{read.run_id}_{self.group_key}',
                    f'qs:i:{round(mean_qscore)}',
                    f'ns:i:{read.num_samples}',
                    f'ts:i:{read.trimmed_samples}',
                    *read.tagdata(),
                    *mods_tags,
                ]
                tags.append(f'mv:B:c,{encode_moves(res["moves"], res["stride"])}')

                if len(seq):
                    segments = res["segments"]
                    write_segmentation(read_id, seq, segments, segment_filename)

                    write_fastq(read_id, seq[::-1], qstring[::-1], fd=self.fd, tags=tags) # reverse to have real orientation

                    summary.append(summary_row(read, len(seq), mean_qscore, alignment=mapping))
                else:
                    logger.warn("> skipping empty sequence %s", read_id)
