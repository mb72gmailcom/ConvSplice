# ConvSplice example

A minimal end-to-end run of the splice-site-region variant effect prediction
on two SNVs in the PTEN locus (hg38).

## Files

| File | Purpose |
| ---- | ------- |
| `example_variants.vcf` / `example_variants.tsv` | Two PTEN SNVs (chr10:87925512 G>C, chr10:87933252 G>T) |
| `pten_chr10.gtf` | GENCODE v40 GTF lines for PTEN only (73 lines) |
| `run_example.sh` | Runs `predict_variants.py` with the example inputs |
| `expected_output/` | Reference outputs from a CPU run, for comparison |

## Run

You need an hg38 FASTA indexed by `samtools faidx` / `pyfaidx` (the `.fa.fai`
must sit next to the `.fa`). Pass it as the first argument or via the
`HG38_FASTA` env var:

```bash
# from packages/ConvSplice/
bash examples/run_example.sh /path/to/hg38.fa
# or
HG38_FASTA=/path/to/hg38.fa bash examples/run_example.sh
```

Outputs go to `examples/output/` by default and should match
`examples/expected_output/`. On CPU the run takes ~70 s for these two variants.

## What the example exercises

For each variant, the script finds every annotated PTEN splice site within
+-10 kb and runs the 5-model ConvSplice ensemble twice (reference and alternate
allele). The alternate allele (`87925513 G>C`, 1-based) sits exactly at a 3'ss,
so its donor/acceptor delta there is large (~-0.12); the variant at 87933253
falls in an intronic region and produces near-zero deltas at the surrounding
sites.

## Outputs

- `splice_site_variant_effects.tsv` — one row per
  `(variant, transcript, splice_site)` with `ref_score`, `alt_score`, `delta`,
  `splice_site_type` (`5ss`/`3ss`), and `distance_to_site`.
- `max_variant_effects.tsv` — per-variant summary keeping the splice site with
  the largest `|delta|`.
