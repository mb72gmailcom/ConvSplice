import gzip

from tqdm import tqdm
from pysam import FastaFile
import pandas as pd
import numpy as np


class Genome:
    def __init__(self, genome_path):
        self.genome_path = genome_path
        self.genome = FastaFile(self.genome_path)
        self.reverse_complement = str.maketrans('ATCG', 'TAGC')
        self.map = np.asarray([[0, 0, 0, 0],
                               [1, 0, 0, 0],
                               [0, 1, 0, 0],
                               [0, 0, 1, 0],
                               [0, 0, 0, 1]])
        self.INDEX_TO_LATIN = {0: '\x01', 1: '\x02', 2: '\x03', 3: '\x04'}

    def get_sequence_from_coords(self, chrom, start, end, strand, mutate_pos=[], mutate_base=[], pad=False):
        start_pad, end_pad = 0, 0
        if pad:
            chr_len = self.genome.get_reference_length(chrom)
            if start < 0:
                start_pad = -1 * start
                start = 0
            if end > chr_len:
                end_pad = end - chr_len
                end = chr_len
            seq = 'N' * start_pad + self.genome.fetch(chrom, start, end).upper() + 'N' * end_pad
        else:
            seq = self.genome.fetch(chrom, start, end).upper()

        if isinstance(mutate_pos, int) and isinstance(mutate_base, str):
            mutate_pos = [mutate_pos]
            mutate_base = [mutate_base]

        if mutate_pos is not None and len(mutate_pos) > 0:
            seq = list(seq)
            for pos, base in zip(mutate_pos, mutate_base):
                if pos - start < 0 or pos - start >= len(seq):
                    continue
                seq[pos - start] = base
            seq = ''.join(seq)

        if strand == '-':
            seq = seq.translate(self.reverse_complement)[::-1]
        return seq

    def one_hot_encoding(self, seq, BASE_TO_INDEX={'A': 0, 'C': 1, 'G': 2, 'T': 3}, N_fill_value=0):
        seq = seq.upper().replace('A', self.INDEX_TO_LATIN[BASE_TO_INDEX['A']]).replace('C', self.INDEX_TO_LATIN[BASE_TO_INDEX['C']])
        seq = seq.replace('G', self.INDEX_TO_LATIN[BASE_TO_INDEX['G']]).replace('T', self.INDEX_TO_LATIN[BASE_TO_INDEX['T']]).replace('N', '\x00')
        seq_array = self.map[np.frombuffer(seq.encode('latin-1'), np.int8) % 5]
        is_N = seq_array.sum(axis=1) == 0
        seq_array = seq_array.astype('float32')
        seq_array[is_N] = N_fill_value
        return seq_array


class Transcript:
    def __init__(self, transcript_id, gene_id, chrom, strand, start, end):
        self.transcript_id = transcript_id
        self.gene_id = gene_id
        self.chrom = chrom
        self.strand = strand
        self.start = start
        self.end = end
        self.transcript_type = None
        self.exons = []
        self.ss5 = set()
        self.ss3 = set()

    def __repr__(self):
        return f'{self.transcript_id} {self.gene_id} {self.chrom} {self.strand} {self.start} {self.end}'

    def add_exon(self, start, end):
        self.exons.append((start, end))

    def add_ss5(self, ss5):
        self.ss5.add(ss5)

    def add_ss3(self, ss3):
        self.ss3.add(ss3)


class Gene:
    def __init__(self, gene_id, gene_name, chrom, start, end, strand):
        self.gene_id = gene_id
        self.gene_name = gene_name
        self.chrom = chrom
        self.start = start
        self.end = end
        self.transcripts = {}
        self.strand = strand
        self.gene_type = None

    def __len__(self):
        return len(self.transcripts)

    def add_transcript(self, transcript):
        self.transcripts[transcript.transcript_id] = transcript
        if self.strand is None:
            self.strand = transcript.strand

    def __getitem__(self, transcript_id):
        return self.transcripts[transcript_id]


class GTFReader:
    """Light-weight GTF parser that exposes genes / transcripts / splice sites.

    Supports both gzipped and plain-text GTFs. When ``add_splice_site`` is True the
    reader pulls 5'/3' splice-site dinucleotides from the genome.
    """

    def __init__(self, gtf_path, genome_path=None, add_splice_site=False):
        self.gtf_path = gtf_path
        self.genes = {}
        self.read_gtf()
        self.gene_id2name, self.gene_name2id = {}, {}
        self.transcripts = {}
        self._build_id_map()
        self._link_transcripts()
        self._sort_transcript_exons()
        self.genome = Genome(genome_path) if genome_path else None
        if add_splice_site:
            if self.genome is None:
                raise ValueError('Genome path is required to add splice site annotation')
            self._add_splice_site_annotation()

    def __len__(self):
        return len(self.genes)

    def __getitem__(self, gene_id):
        return self.genes[gene_id]

    def _open_gtf(self):
        if self.gtf_path.endswith('.gz'):
            return gzip.open(self.gtf_path, 'rt')
        return open(self.gtf_path, 'r')

    def _count_lines(self):
        with self._open_gtf() as f:
            return sum(1 for _ in f)

    def _build_id_map(self):
        for gene_id, gene in self.genes.items():
            self.gene_id2name[gene_id] = gene.gene_name
            self.gene_name2id[gene.gene_name] = gene_id

    def get_gene(self, gene_id):
        if gene_id in self.genes:
            return self.genes[gene_id]
        if gene_id in self.gene_name2id:
            return self.genes[self.gene_name2id[gene_id]]
        raise ValueError(f'Gene {gene_id} not found')

    def _link_transcripts(self):
        for gene in self.genes.values():
            for transcript_id, transcript in gene.transcripts.items():
                self.transcripts[transcript_id] = transcript

    def _sort_transcript_exons(self):
        for transcript in self.transcripts.values():
            reverse = transcript.strand == '-'
            transcript.exons = sorted(transcript.exons, key=lambda x: x[0], reverse=reverse)

    def read_gtf(self):
        total = self._count_lines()
        fp = self._open_gtf()
        for line in tqdm(fp, total=total, desc='Reading GTF file'):
            if line.startswith('#'):
                continue
            sp = line.strip().split('\t')
            if len(sp) < 9:
                continue
            annotate_info = sp[-1].split(';')
            chrom = sp[0]
            strand = sp[6]
            start, end = int(sp[3]) - 1, int(sp[4])

            if sp[2] == 'gene':
                gene_id, gene_name, gene_type = None, None, None
                for info in annotate_info:
                    if 'gene_id' in info:
                        gene_id = info.split('"')[1].split('.')[0]
                    if 'gene_name' in info:
                        gene_name = info.split('"')[1]
                    if 'gene_type' in info:
                        gene_type = info.split('"')[1]
                if gene_id is None:
                    raise ValueError('Gene ID not found')
                gene = Gene(gene_id, gene_name, chrom, start, end, strand)
                gene.gene_type = gene_type
                self.genes[gene_id] = gene

            elif sp[2] == 'transcript':
                transcript_id, transcript_type, gene_id = None, None, None
                for info in annotate_info:
                    if 'transcript_id' in info:
                        transcript_id = info.split('"')[1].split('.')[0]
                    if 'transcript_type' in info:
                        transcript_type = info.split('"')[1]
                    if 'gene_id' in info:
                        gene_id = info.split('"')[1].split('.')[0]
                if transcript_id is None:
                    raise ValueError('Transcript ID not found')
                transcript = Transcript(transcript_id, gene_id, chrom, strand, start, end)
                transcript.transcript_type = transcript_type
                self.genes[gene_id].add_transcript(transcript)

            elif sp[2] == 'exon':
                gene_id, transcript_id = None, None
                for info in annotate_info:
                    if 'gene_id' in info:
                        gene_id = info.split('"')[1].split('.')[0]
                    if 'transcript_id' in info:
                        transcript_id = info.split('"')[1].split('.')[0]
                self.genes[gene_id][transcript_id].add_exon(start, end)
        fp.close()

    def _add_splice_site_annotation(self):
        for gene in tqdm(self.genes.values(), desc='Adding splice site annotation'):
            for transcript in gene.transcripts.values():
                exons = transcript.exons
                if len(exons) < 2:
                    continue
                chrom = transcript.chrom
                strand = transcript.strand
                if strand == '+':
                    for exon in exons:
                        if exon[1] != transcript.end:
                            di = self.genome.get_sequence_from_coords(chrom, exon[1], exon[1] + 2, strand)
                            transcript.add_ss5((chrom, strand, exon[1], '5ss', di))
                        if exon[0] != transcript.start:
                            di = self.genome.get_sequence_from_coords(chrom, exon[0] - 2, exon[0], strand)
                            transcript.add_ss3((chrom, strand, exon[0], '3ss', di))
                else:
                    for exon in exons:
                        if exon[0] != transcript.start:
                            di = self.genome.get_sequence_from_coords(chrom, exon[0] - 2, exon[0], strand)
                            transcript.add_ss5((chrom, strand, exon[0], '5ss', di))
                        if exon[1] != transcript.end:
                            di = self.genome.get_sequence_from_coords(chrom, exon[1], exon[1] + 2, strand)
                            transcript.add_ss3((chrom, strand, exon[1], '3ss', di))


def assign_variants_to_genes(variants, gtf_reader, window=2000):
    """Annotate each variant with overlapping genes (gene body or +-window of TSS/TES).

    Args:
        variants: DataFrame with columns ``chr``, ``pos`` (0-based), ``ref``, ``alt``.
        gtf_reader: a populated :class:`GTFReader`.
        window: bp window around TSS/TES used as a soft boundary.

    Returns:
        DataFrame with an additional ``gene`` column (gene_id) and ``strand`` column.
        A variant overlapping multiple genes is duplicated, one row per gene.
    """
    rows = []
    for _, var in variants.iterrows():
        pos = var['pos'] + 1  # 1-based for GTF coordinates
        chrom = str(var['chr'])
        for gene in gtf_reader.genes.values():
            if gene.chrom != chrom and gene.chrom != f'chr{chrom}' and f'chr{gene.chrom}' != chrom:
                continue
            inside_body = gene.start <= pos <= gene.end
            tss = gene.start if gene.strand == '+' else gene.end
            tes = gene.end if gene.strand == '+' else gene.start
            near_tss = tss - window <= pos <= tss + window
            near_tes = tes - window <= pos <= tes + window
            if inside_body or near_tss or near_tes:
                annotated = var.copy()
                annotated['gene'] = gene.gene_id
                annotated['strand'] = gene.strand
                rows.append(annotated)
    return pd.DataFrame(rows)
