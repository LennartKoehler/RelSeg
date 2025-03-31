# RelSeg

RelSeg aligns the basecalled sequence to the signal. It relies on the [Bonito] (https://github.com/nanoporetech/bonito) basecaller of ONT. The [LXT] (https://github.com/rachtibat/LRP-eXplains-Transformers) and [zennit] (https://github.com/chr5tphr/zennit) packages are used for the Layer-wise Relevance Propagation

```bash
$ pip install relseg
```

```bash
$ relseg rna004_130bps_sup@v5.0.0" data/reads --rna > basecall.txt
$ relseg rna004_130bps_sup@v5.0.0" data/reads --rna --save_relevance > basecall.txt

```

A transformer model which no longer requires `flash_attn`is implemented. 
