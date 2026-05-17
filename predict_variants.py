"""ConvSplice splice-site-region variant effect prediction.

Given a VCF/TSV of SNVs and a GTF + reference FASTA, score each variant's effect
on every annotated splice site within +-10kb of the variant. For each affected
splice site, ConvSplice produces a reference and an alternate-allele probability
(donor for 5ss, acceptor for 3ss); the per-site delta is reported.
"""

import argparse
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


from convsplice import ConvSplicePredictor, Genome, GTFReader
from convsplice.genome_utils import assign_variants_to_genes


WINDOW = 10000  # ConvSplice variant context radius

# Model context length ~2 * predictor.WINDOW_SIZE (see convsplice.predictor); splice
# sites kept here are within WINDOW bp of the variant. Because WINDOW is comparable
# to half that context, the variant usually lies inside the 25kb window centered on
# each splice site — so ref vs alt inputs almost always differ and must both run.
# What repeats wastefully is the reference branch for the same (transcript, splice
# site) across many variants; see ``cache_ref_scores``.

def _qc_variants(variants):
    variants = variants.drop_duplicates(keep='first')
    variants = variants[variants['ref'].str.len() == 1]
    variants = variants[variants['alt'].str.len() == 1]
    variants = variants[~variants['chr'].astype(str).isin(['X', 'Y', 'M', 'chrX', 'chrY', 'chrM'])]
    return variants


def _load_variants(path):
    if path.endswith('.vcf') or path.endswith('.vcf.gz'):
        df = pd.read_csv(path, sep='\t', comment='#', header=None, usecols=range(5))
        df = df.rename(columns={0: 'chr', 1: 'pos', 3: 'ref', 4: 'alt'})
        return df[['chr', 'pos', 'ref', 'alt']]
    if path.endswith('.tsv') or path.endswith('.tsv.gz'):
        return pd.read_csv(path, sep='\t')
    raise ValueError(f'Unsupported variant file: {path}')


def _normalize_chrom(chrom, gtf_uses_chr):
    chrom = str(chrom)
    if gtf_uses_chr and not chrom.startswith('chr'):
        return f'chr{chrom}'
    if not gtf_uses_chr and chrom.startswith('chr'):
        return chrom[3:]
    return chrom


def _splice_sites_in_window(transcript, variant_pos):
    """Return (pos, ss_type) for every splice site of transcript within +-WINDOW
    of variant_pos. Sorted by ss_type then pos so the output ordering is stable
    across runs (Transcript.ss5 / .ss3 are sets, whose iteration order depends
    on Python's hash randomization)."""
    sites = []
    for info in transcript.ss5:
        ss_pos = info[2]
        if abs(ss_pos - variant_pos) <= WINDOW:
            sites.append((ss_pos, '5ss'))
    for info in transcript.ss3:
        ss_pos = info[2]
        if abs(ss_pos - variant_pos) <= WINDOW:
            sites.append((ss_pos, '3ss'))
    sites.sort(key=lambda s: (s[1], s[0]))
    return sites


def predict_variant_effects(variant_path, gtf_path, genome_path, output_path,
                            annotate=True, use_cuda=True, weights_dir=None,
                            cache_ref_scores=True):
    variants = _load_variants(variant_path)
    variants = _qc_variants(variants)
    print(f'Variants after QC: {len(variants)}')
    if len(variants) == 0:
        print('No variants to score.')
        return

    print('Loading GTF...')
    gtf_reader = GTFReader(gtf_path, genome_path=genome_path, add_splice_site=True)
    gtf_uses_chr = next(iter(gtf_reader.genes.values())).chrom.startswith('chr')

    print('Initializing ConvSplice predictor (5-model ensemble)...')
    genome = Genome(genome_path)
    predictor_kwargs = {'genome': genome, 'use_cuda': use_cuda}
    if weights_dir is not None:
        predictor_kwargs['weights_dir'] = weights_dir
    predictor = ConvSplicePredictor(**predictor_kwargs)

    if annotate:
        variants = assign_variants_to_genes(variants, gtf_reader, window=WINDOW)
        print(f'Variant-gene pairs after annotation: {len(variants)}')
    elif 'gene' not in variants.columns:
        raise ValueError('--annotate is False but the variant file has no `gene` column.')

    # Reference ensemble scores depend only on (transcript, splice site), not on
    # which variant is being scored — cache avoids recomputing ref for every variant.
    ref_pair_cache = {}


    rows = []
    for _, var in tqdm(variants.iterrows(), total=len(variants), desc='Variants'):
        gene_id = var['gene']
        var_chrom = _normalize_chrom(var['chr'], gtf_uses_chr)
        var_pos = int(var['pos'])  # 0-based, matches reference VCF semantics in source
        ref, alt = var['ref'], var['alt']
        variant_id = f"{str(var['chr']).replace('chr', '')}_{var_pos + 1}_{ref}_{alt}_hg38"

        if gene_id not in gtf_reader.genes:
            continue
        gene = gtf_reader.genes[gene_id]

        for transcript in gene.transcripts.values():
            if transcript.chrom != var_chrom:
                continue
            sites = list(_splice_sites_in_window(transcript, var_pos))
            if not sites:
                continue

            ref_coords_to_run = []
            ref_keys_pending = []
            ref_preds_by_idx = [None] * len(sites)
                

            for idx, (ss_pos, ss_type) in enumerate(sites):
                cache_key = (transcript.transcript_id, ss_pos, ss_type)
                if cache_ref_scores and cache_key in ref_pair_cache:
                    ref_preds_by_idx[idx] = ref_pair_cache[cache_key]
                else:
                    ref_coords_to_run.append(
                        (transcript.chrom, ss_pos, ss_pos + 1, transcript.strand))
                    ref_keys_pending.append((idx, cache_key))

            if ref_coords_to_run:
                batch_ref = predictor.predict_batch(ref_coords_to_run, transcript, verbose=0)
                for (idx, cache_key), ref_pair in zip(ref_keys_pending, batch_ref):
                    ref_preds_by_idx[idx] = ref_pair
                    if cache_ref_scores:
                        ref_pair_cache[cache_key] = ref_pair

            alt_coords = [
                (transcript.chrom, ss_pos, ss_pos + 1, transcript.strand, var_pos, alt)
                for ss_pos, _ in sites
            ]
            alt_preds = predictor.predict_batch(alt_coords, transcript, verbose=0)

            for (ss_pos, ss_type), ref_pair, alt_pair in zip(sites, ref_preds_by_idx, alt_preds):
                if ss_type == '5ss':
                    ref_score, alt_score = float(ref_pair[0]), float(alt_pair[0])
                else:
                    ref_score, alt_score = float(ref_pair[1]), float(alt_pair[1])
                rows.append({
                    'variant_id': variant_id,
                    'chr': var['chr'],
                    'pos': var_pos + 1,
                    'ref': ref,
                    'alt': alt,
                    'gene_id': gene_id,
                    'transcript_id': transcript.transcript_id,
                    'splice_site_pos': ss_pos + 1,
                    'splice_site_type': ss_type,
                    'distance_to_site': ss_pos - var_pos,
                    'ref_score': ref_score,
                    'alt_score': alt_score,
                    'delta': alt_score - ref_score,
                })

    out_df = pd.DataFrame(rows)
    detailed_path = os.path.join(output_path, 'splice_site_variant_effects.tsv')
    out_df.to_csv(detailed_path, sep='\t', index=False)
    print(f'Wrote {len(out_df)} variant-splice-site predictions to {detailed_path}')

    if len(out_df) > 0:
        summary = (out_df.assign(abs_delta=out_df['delta'].abs())
                         .sort_values('abs_delta', ascending=False)
                         .groupby('variant_id', as_index=False)
                         .first()[['variant_id', 'chr', 'pos', 'ref', 'alt',
                                   'gene_id', 'transcript_id', 'splice_site_pos',
                                   'splice_site_type', 'ref_score', 'alt_score',
                                   'delta']]
                         .rename(columns={'delta': 'max_delta'}))
        summary_path = os.path.join(output_path, 'max_variant_effects.tsv')
        summary.to_csv(summary_path, sep='\t', index=False)
        print(f'Wrote per-variant max-effect summary to {summary_path}')


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='ConvSplice splice-site-region variant effect prediction.')
    parser.add_argument('--variant_path', required=True,
                        help='Path to .vcf/.vcf.gz or .tsv with columns chr, pos, ref, alt.')
    parser.add_argument('--gtf_path', required=True,
                        help='Path to a GENCODE GTF (.gtf or .gtf.gz).')
    parser.add_argument('--genome_path', required=True,
                        help='Path to the reference FASTA (indexed by pyfaidx / samtools).')
    parser.add_argument('--output_path', required=True, help='Output directory.')
    parser.add_argument('--weights_dir', default=None,
                        help='Optional override for the ConvSplice model_weights directory.')
    parser.add_argument('--annotate', action='store_true', default=True,
                        help='Annotate variants to genes via the GTF (default: True).')
    parser.add_argument('--no_annotate', dest='annotate', action='store_false',
                        help='Disable gene annotation. The variant file must already have a `gene` column.')
    parser.add_argument('--cpu', action='store_true', default=False,
                        help='Run on CPU (default: GPU when available).')
    parser.add_argument(
        '--disable-ref-cache',
        action='store_true',
        default=False,
        help='Disable caching of reference splice-site scores across variants '
             '(for debugging / parity with older runs).',
    )

    args = parser.parse_args(argv)

    os.makedirs(args.output_path, exist_ok=True)

    use_cuda = (not args.cpu) and torch.cuda.is_available()
    if not use_cuda and not args.cpu:
        print('CUDA not available; falling back to CPU.', file=sys.stderr)

    predict_variant_effects(
        variant_path=args.variant_path,
        gtf_path=args.gtf_path,
        genome_path=args.genome_path,
        output_path=args.output_path,
        annotate=args.annotate,
        use_cuda=use_cuda,
        weights_dir=args.weights_dir,
        cache_ref_scores=not args.disable_ref_cache,
    )


if __name__ == '__main__':
    main()
