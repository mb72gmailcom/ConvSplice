#!/bin/bash
#SBATCH -t 0-01:00
#SBATCH -p gpu
#SBATCH --gpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=logfiles/slurm-%j.out
#SBATCH --mem=64G

set -euo pipefail
mkdir -p logfiles

variant_path="${1:-}"
output_dir="${2:-}"
gtf_path="${3:-resources/gencode.v47.basic.annotation.gtf.gz}"
genome_path="${4:-resources/hg38.fa.gz}"

if [[ -z "$variant_path" || -z "$output_dir" ]]; then
    echo "Usage: $0 <variant.vcf|tsv> <output_dir> [gtf_path] [genome_path]" >&2
    exit 1
fi

python predict_variants.py \
    --variant_path "$variant_path" \
    --gtf_path "$gtf_path" \
    --genome_path "$genome_path" \
    --output_path "$output_dir"
