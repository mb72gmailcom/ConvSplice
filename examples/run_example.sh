#!/bin/bash
# Run ConvSplice variant effect prediction on the bundled PTEN example.
#
# Two SNVs in PTEN (chr10) are scored against a small PTEN-only GTF subset
# (`pten_chr10.gtf`). Provide a path to an indexed hg38 FASTA via $1 or via
# the HG38_FASTA env var.

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
PKG_DIR="$(dirname "$HERE")"

GENOME_PATH="${1:-${HG38_FASTA:-}}"
if [[ -z "$GENOME_PATH" ]]; then
    echo "Usage: $0 <hg38.fa>" >&2
    echo "   or: HG38_FASTA=/path/to/hg38.fa $0" >&2
    exit 1
fi

OUTPUT_DIR="${2:-$HERE/output}"
mkdir -p "$OUTPUT_DIR"

cd "$PKG_DIR"
python predict_variants.py \
    --variant_path "$HERE/example_variants.vcf" \
    --gtf_path "$HERE/pten_chr10.gtf" \
    --genome_path "$GENOME_PATH" \
    --output_path "$OUTPUT_DIR" \
    --cpu

echo
echo "Outputs written to: $OUTPUT_DIR"
echo "Compare against: $HERE/expected_output/"
