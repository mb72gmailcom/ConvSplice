import os

import numpy as np
import torch
from tqdm import tqdm

from .model import (
    ConvSplice,
    DEFAULT_KERNEL_SIZES,
    DEFAULT_DILATION_RATES,
    DEFAULT_CL,
    DEFAULT_FEAT_DIM,
)


PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DEFAULT_WEIGHTS_DIR = os.path.join(PACKAGE_DIR, 'resources', 'model_weights')
WINDOW_SIZE = 10000  # ConvSplice flanks each side with 10kb of context


class ConvSplicePredictor:
    """Predict ConvSplice donor/acceptor probabilities at splice sites.

    The released ConvSplice consumes a 25kb input window (10kb flank + 5kb center
    + 10kb flank) and predicts (neither, acceptor, donor) per base for the central
    5kb. This class wraps the 5-model ensemble for splice-site-region variant
    scoring: given a transcript and a list of splice-site coordinates, it returns
    the mean ensemble score at each site for the reference and (optionally) a
    mutated allele.
    """

    def __init__(self, genome, weights_dir=DEFAULT_WEIGHTS_DIR, num_models=5,
                 use_cuda=False, batch_size=64):
        self.genome = genome
        self.use_cuda = use_cuda
        self.batch_size = batch_size

        self.model_weights = [
            os.path.join(weights_dir, f'ConvSplice_model_{i + 1}.pt')
            for i in range(num_models)
        ]
        self.models = [self._load_model(path) for path in self.model_weights]

    def _load_model(self, model_path):
        model = ConvSplice(DEFAULT_FEAT_DIM, DEFAULT_CL, DEFAULT_KERNEL_SIZES, DEFAULT_DILATION_RATES)
        weight = torch.load(model_path, map_location='cpu')
        model.load_state_dict(weight['model'])
        if self.use_cuda:
            model = model.cuda()
        return model

    @staticmethod
    def _apply_mutation(reference_seq, transcript_start, transcript_end, strand,
                        mutation_pos, mutation_base, padding_window=WINDOW_SIZE,
                        reverse_complement=str.maketrans('ATCG', 'TAGC')):
        if isinstance(mutation_pos, int) and isinstance(mutation_base, str):
            mutation_pos = [mutation_pos]
            mutation_base = [mutation_base]
        if not mutation_pos:
            return reference_seq
        seq = list(reference_seq)
        if strand == '+':
            for pos, nc in zip(mutation_pos, mutation_base):
                seq[pos - transcript_start + padding_window] = nc
        else:
            for pos, nc in zip(mutation_pos, mutation_base):
                seq[transcript_end - 1 - pos + padding_window] = nc.translate(reverse_complement)
        return ''.join(seq)

    def _predict(self, sequence_encodings):
        inputs = torch.Tensor(sequence_encodings)
        if self.use_cuda:
            inputs = inputs.cuda()
        with torch.no_grad():
            preds = []
            for model in self.models:
                model.eval()
                preds.append(model(inputs))
            mean_preds = torch.mean(torch.stack(preds), axis=0)
            return mean_preds.cpu().numpy()

    @staticmethod
    def _splice_ann(pos, ss5_pos, ss3_pos):
        if pos in ss5_pos:
            return '5ss'
        if pos in ss3_pos:
            return '3ss'
        return None

    def predict_batch(self, coords, transcript, verbose=0):
        """Return (acceptor_score, donor_score) pairs for each coordinate.

        Args:
            coords: list of tuples. Each is either ``(chrom, start, end, strand)``
                (reference) or ``(chrom, start, end, strand, mutation_pos,
                mutation_base)`` (mutated). The midpoint of ``[start, end)`` must
                be an annotated splice site of ``transcript``; otherwise the score
                is ``[0, 0]``.
            transcript: a :class:`Transcript` with populated ``ss5``/``ss3`` sets.

        Returns:
            ``np.ndarray`` of shape ``(len(coords), 2)``. Column 0 is the donor
            (5ss) score (zero for 3ss sites); column 1 is the acceptor (3ss) score
            (zero for 5ss sites).
        """
        ss5_pos = {info[2] for info in transcript.ss5}
        ss3_pos = {info[2] for info in transcript.ss3}

        batch_size = self.batch_size
        window_size = WINDOW_SIZE

        progress = None
        if verbose:
            progress = tqdm(total=len(range(0, len(coords), batch_size)),
                            position=0, leave=True, desc='ConvSplice Predicting')

        reference_seq = self.genome.get_sequence_from_coords(
            transcript.chrom, transcript.start, transcript.end, transcript.strand, pad=True)
        reference_seq = 'N' * window_size + reference_seq + 'N' * window_size

        preds = []
        for i in range(0, len(coords), batch_size):
            batch_coords = coords[i:i + batch_size]
            batch_encodings = []
            none_idx = []
            ann_list = []
            for idx, coord in enumerate(batch_coords):
                if len(coord) == 4:
                    chrom, start, end, strand = coord
                    mutate_pos, mutate_base = [], []
                else:
                    chrom, start, end, strand, mutate_pos, mutate_base = coord
                mid_pos = start + ((end - start) // 2)
                ann = self._splice_ann(mid_pos, ss5_pos, ss3_pos)
                if ann is None:
                    none_idx.append(idx)
                    continue
                ann_list.append(ann)

                transcript_seq = self._apply_mutation(
                    reference_seq, transcript.start, transcript.end, transcript.strand,
                    mutate_pos, mutate_base, padding_window=window_size)

                if strand == '+':
                    if ann == '5ss':
                        context_seq = transcript_seq[mid_pos - transcript.start: mid_pos - transcript.start + 2 * window_size + 2]
                    else:
                        context_seq = transcript_seq[mid_pos - transcript.start - 2: mid_pos - transcript.start + 2 * window_size]
                else:
                    if ann == '5ss':
                        context_seq = transcript_seq[transcript.end - mid_pos: transcript.end - mid_pos + 2 * window_size + 2]
                    else:
                        context_seq = transcript_seq[transcript.end - mid_pos - 2: transcript.end - mid_pos + 2 * window_size]

                batch_encodings.append(self.genome.one_hot_encoding(context_seq))

            batch_encodings = np.array(batch_encodings)
            _preds = []
            if batch_encodings.shape[0] != 0:
                _preds = self._predict(batch_encodings)

            splice_preds = []
            pred_idx = 0
            for idx in range(len(batch_coords)):
                if idx in none_idx:
                    splice_preds.append([0, 0])
                else:
                    ann = ann_list[pred_idx]
                    if ann == '5ss':
                        splice_preds.append([_preds[pred_idx][0][1], 0])
                    else:
                        splice_preds.append([0, _preds[pred_idx][0][2]])
                    pred_idx += 1
            preds.extend(splice_preds)

            if progress is not None:
                progress.update(1)

        return np.array(preds)


# Backwards-compatible alias matching the otari naming.
ConvSplice_predictor = ConvSplicePredictor
