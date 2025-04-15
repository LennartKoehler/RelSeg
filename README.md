# RelSeg
**Rel**evance Based **Seg**mentation of Nanopore Reads  
  
[![py310](https://img.shields.io/badge/python-3.10-brightgreen.svg)](https://img.shields.io/badge/python-3.10-brightgreen.svg)
[![DOI](https://zenodo.org/badge/956585138.svg)](https://doi.org/10.5281/zenodo.15195552)


RelSeg is used to align the basecalled sequence to the signal of nanopore reads. It relies on the [bonito](https://github.com/nanoporetech/bonito) basecaller of ONT. The [lxt](https://github.com/rachtibat/LRP-eXplains-Transformers) and [zennit](https://github.com/chr5tphr/zennit) packages are used for the Layer-wise Relevance Propagation.  
A transformer model which no longer requires `flash_attn` is implemented. 


## Installation
```bash
$ pip install relseg
```
## Usage

```bash
$ relseg rna004_130bps_sup@v5.0.0 /path/data/reads --rna > basecall.txt

$ relseg rna004_130bps_sup@v5.0.0 /path/data/reads --rna --save_relevance > basecall.txt
```


## Output


| Column   | Description                                                                 |
|----------|-----------------------------------------------------------------------------|
| read_id  | Unique identifier for the read                                             |
| base     | The base (nucleotide) called at the specific position                      |
| start    | The start position of the base in the signal alignment (-1 for not aligned)            |
| end      | The end position of the base in the signal alignment (-1 for not aligned)                  |

### Example Output
```tsv
read_id	base	start	end
5a729d16-b785-4e8c-ad91-314d862d980b	T	140	157
5a729d16-b785-4e8c-ad91-314d862d980b	C	157	157
5a729d16-b785-4e8c-ad91-314d862d980b	T	157	181
5a729d16-b785-4e8c-ad91-314d862d980b	C	-1	-1
```

The normal basecalls including the moves table are also output.

## Relevance for Consecutive Bases
![Description of the image](figures/relevance.png)


## Sequence Aligned to Signal
![Description of the image](figures/segmentation.png)



