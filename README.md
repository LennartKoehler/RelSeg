# RelSeg

RelSeg aligns the basecalled sequence to the signal. It relies on the [bonito](https://github.com/nanoporetech/bonito) basecaller of ONT. The [lxt](https://github.com/rachtibat/LRP-eXplains-Transformers) and [zennit](https://github.com/chr5tphr/zennit) packages are used for the Layer-wise Relevance Propagation

```bash
$ pip install relseg
```

```bash
$ relseg rna004_130bps_sup@v5.0.0 data/reads --rna > basecall.txt
$ relseg rna004_130bps_sup@v5.0.0 data/reads --rna --save_relevance > basecall.txt
```

A transformer model which no longer requires `flash_attn` is implemented. 

## Output


| Column   | Description                                                                 |
|----------|-----------------------------------------------------------------------------|
| read_id  | Unique identifier for the read                                             |
| base     | The base (nucleotide) called at the specific position                      |
| start    | The start position of the base in the signal alignment (-1 for not aligned)            |
| end      | The end position of the base in the signal alignment (-1 for not aligned)                  |

Output example:
```tsv
read_id	base	start	end
5a729d16-b785-4e8c-ad91-314d862d980b	T	140	157
5a729d16-b785-4e8c-ad91-314d862d980b	C	157	157
5a729d16-b785-4e8c-ad91-314d862d980b	T	157	181
5a729d16-b785-4e8c-ad91-314d862d980b	C	-1	-1
```

The normal basecalls including the moves table are also output