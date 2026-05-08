# ConvSplice

ConvSplice is a 5-model ensemble of dilated 1-D convolutional networks that
predicts donor and acceptor probabilities at every base in a 5kb window of
pre-mRNA, given 10kb of flanking context on each side (25kb total input). It is
used here as a stand-alone splice-site-region variant effect predictor: scoring
how a single-nucleotide variant changes the donor/acceptor probability at every
annotated splice site within +-10kb of the variant.

## Architecture

Input: one-hot encoded sequence of length `5kb + 2 * 10kb = 25kb`.

```
        10kb               5kb               10kb
[--- left flank ---][---  center  ---][--- right flank ---]
                            |
                            v
        20 dilated Conv1D layers (residual connection every 4 layers)
                            |
                            v
            crop to the central 5kb -> per-base softmax
                            |
                            v
     P(neither) | P(acceptor / 3'ss) | P(donor / 5'ss)  for each of 5,000 bases
```

Each `ConvBlock` is `BN -> SiLU -> Conv1D(N, W, D) -> BN -> SiLU -> Conv1D(N, W, D)`
with a residual skip; widths `W` and dilations `D` grow with depth
(`W = 11/21/41/51`, `D = 1/4/10/25`). Five models are trained with different
seeds and averaged.

## Layout

```
ConvSplice/
  convsplice/
    __init__.py
    model.py           # ConvSplice nn.Module + default architecture
    predictor.py       # ConvSplicePredictor (5-model ensemble)
    genome_utils.py    # Genome (FASTA), GTFReader, assign_variants_to_genes
  resources/
    model_weights/     # ConvSplice_model_{1..5}.pt (5x ~4.6 MB)
  predict_variants.py  # CLI for splice-site-region variant scoring
  predict_variants.sh  # SLURM/bash wrapper
  test.vcf, test.tsv   # Example inputs (hg38 coords)
  examples/            # End-to-end PTEN example with expected outputs
```

## Variant effect prediction

For each input variant the script:

1. Annotates the variant to overlapping genes using the GTF.
2. For every transcript of those genes, finds annotated splice sites within
   +-10kb of the variant.
3. Runs the 5-model ensemble on the 25kb window centered on each splice site,
   once with the reference allele and once with the alternate allele.
4. Reports the donor/acceptor probability for each splice site under both
   alleles and the per-site delta.

### Inputs

- `--variant_path` `.vcf`/`.vcf.gz` or `.tsv` with columns `chr, pos, ref, alt`
  (`pos` is 0-based for the TSV form, matching the source convention).
- `--gtf_path` GENCODE GTF (plain or gzipped). Must include `gene`,
  `transcript`, and `exon` features.
- `--genome_path` reference FASTA indexed by `pyfaidx`/`samtools faidx`.

### Usage

```bash
# Direct
python predict_variants.py \
    --variant_path test.vcf \
    --gtf_path /path/to/gencode.v47.basic.annotation.gtf.gz \
    --genome_path /path/to/hg38.fa.gz \
    --output_path ./test_outputs

# SLURM wrapper (positional: variant, output_dir, [gtf], [genome])
sbatch predict_variants.sh test.vcf ./test_outputs \
    /path/to/gencode.v47.basic.annotation.gtf.gz /path/to/hg38.fa.gz
```

### Splice sites: 5'ss vs 3'ss

A pre-mRNA intron is bounded by two splice sites that the spliceosome must
recognize for the intron to be excised:

- **5' splice site (5'ss / donor)** — the *exon -> intron* boundary at the
  upstream end of the intron. Almost always begins with the intronic
  dinucleotide `GT`. ConvSplice's "donor" channel scores how likely each base
  is to be a 5'ss.
- **3' splice site (3'ss / acceptor)** — the *intron -> exon* boundary at the
  downstream end of the intron. Almost always ends with the intronic
  dinucleotide `AG`. ConvSplice's "acceptor" channel scores how likely each
  base is to be a 3'ss.

For a transcript on the `+` strand the order along the chromosome is
`exon ... 5'ss | intron | 3'ss ... exon`; on the `-` strand the labels swap
relative to chromosome coordinates because the gene reads 3' -> 5' on the
reference. ConvSplice consumes 25kb of context (10kb flank + 5kb center + 10kb
flank), already strand-corrected, and predicts a per-base softmax over
`(neither, acceptor, donor)` for the central 5kb. A variant disrupts splicing
when its alternate allele lowers the donor probability of an annotated 5'ss or
the acceptor probability of an annotated 3'ss (negative `delta`), or, more
rarely, raises one elsewhere to create a cryptic site (positive `delta` at a
non-annotated position — not scored by this CLI, which only queries annotated
sites).

### Outputs

Two TSVs are written to `--output_path`:

- `splice_site_variant_effects.tsv` — one row per `(variant, transcript,
  splice_site)` triple, listing every annotated splice site within +-10kb of
  the variant. Columns:
  - `variant_id` — `chr_pos_ref_alt_hg38` (1-based pos)
  - `chr`, `pos`, `ref`, `alt` — variant fields (1-based pos)
  - `gene_id`, `transcript_id` — GENCODE IDs (version-stripped)
  - `splice_site_pos` — 1-based splice-site coordinate
  - `splice_site_type` — `5ss` (donor) or `3ss` (acceptor)
  - `distance_to_site` — `splice_site_pos - variant_pos` in genomic bp
    (sign = direction; magnitude <= 10000)
  - `ref_score` / `alt_score` — ensemble-mean ConvSplice probability at
    that splice site for the appropriate channel (donor for 5'ss, acceptor
    for 3'ss), in `[0, 1]`
  - `delta` — `alt_score - ref_score`. Strongly negative deltas at canonical
    sites suggest splice disruption.
- `max_variant_effects.tsv` — one row per variant, keeping the splice site
  with the largest `|delta|` across all transcripts. The `delta` column is
  renamed `max_delta`.

See [`examples/`](examples/README.md) for a runnable PTEN example with
pre-generated expected outputs.

## Programmatic use

```python
from convsplice import Genome, GTFReader, ConvSplicePredictor

genome = Genome('/path/to/hg38.fa.gz')
gtf = GTFReader('/path/to/gencode.v47.basic.annotation.gtf.gz',
                genome_path='/path/to/hg38.fa.gz', add_splice_site=True)
predictor = ConvSplicePredictor(genome, use_cuda=True)

transcript = gtf.transcripts['ENST00000367770']
ss_pos = next(iter(transcript.ss5))[2]  # any annotated 5'ss
ref = predictor.predict_batch(
    [(transcript.chrom, ss_pos, ss_pos + 1, transcript.strand)], transcript)
alt = predictor.predict_batch(
    [(transcript.chrom, ss_pos, ss_pos + 1, transcript.strand, variant_pos, 'A')],
    transcript)
delta = alt - ref
```

## Requirements

`pip install -r requirements.txt` (see `requirements.txt`). A CUDA-capable GPU
is recommended; pass `--cpu` to force CPU.

## Model weights

The 5 ensemble checkpoints
(`resources/model_weights/ConvSplice_model_{1..5}.pt`, ~4.6 MB each) are
bundled directly with this package.
