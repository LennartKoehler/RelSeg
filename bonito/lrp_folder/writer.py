from bonito.io import *

def write_segmentation(read_id, sequence, segments, fd=sys.stdout):
    assert len(sequence) == segments.shape[0], f"problem, number of bases and number of segments doesn't match: (bases: {len(sequence)}, segments:{segments.shape[0]})"
    fd.write(f"read_id \t base \t start_position")
    for base, segment in zip(sequence, segments):
        fd.write(f"{read_id}\t{base}\t{segment}\n")

class LRP_Writer(Writer):
    def run(self):
        with CSVLogger(summary_file(), sep='\t') as summary:
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


                segments = res["segments"]
                write_segmentation(read_id, seq, segments, fd=self.fd)


                # if len(seq):
                #     if self.mode == 'wfq':
                #         write_fastq(read_id, seq, qstring, fd=self.fd, tags=tags)
                #     else:
                #         self.output.write(
                #             AlignedSegment.fromstring(
                #                 sam_record(read_id, seq, qstring, mapping, tags=tags),
                #                 self.output.header
                #             )
                #         )
                #     summary.append(summary_row(read, len(seq), mean_qscore, alignment=mapping))
                # else:
                #     logger.warn("> skipping empty sequence %s", read_id)
