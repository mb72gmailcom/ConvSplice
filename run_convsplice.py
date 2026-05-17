"""Launch ConvSplice ``predict_variants.py`` using a fixed repo layout (venv + codedir).

Mirrors the otari/hedgehog pattern: ``topdir``, ``codedir``, ``venv_path``, ``run_pipeline``.
Override ``topdir`` / ``venv_path`` at deploy time if this file is not under the repo root.
"""
from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime

codedir = os.path.dirname(os.path.abspath(__file__))
venv_path = os.path.join(codedir, 'convsplice_venv')

# Default GTF / genome (same layout as predict_variants.sh). Set to absolute paths
# to pin refs cluster-wide; leave None to use codedir/resources/... below.
DEFAULT_GTF_PATH: str | None = '/mnt/home/mbershadsky/run_otari/resources/gencode.v47.basic.annotation.gtf.gz'
DEFAULT_GENOME_PATH: str | None = '/mnt/home/mbershadsky/run_otari/resources/hg38.fa.gz'

def _default_gtf_path() -> str:
    if DEFAULT_GTF_PATH is not None:
        return DEFAULT_GTF_PATH
    return os.path.join(codedir, 'resources', 'gencode.v47.basic.annotation.gtf.gz')


def _default_genome_path() -> str:
    if DEFAULT_GENOME_PATH is not None:
        return DEFAULT_GENOME_PATH
    return os.path.join(codedir, 'resources', 'hg38.fa.gz')


def log_time(msg: str) -> None:
    print(datetime.now().strftime('%H:%M:%S'), msg)


def run_pipeline(
    input_file: str,
    output_dir: str,
    gtf_path: str | None = None,
    genome_path: str | None = None,
    *,
    cpu: bool = False,
    weights_dir: str | None = None,
    annotate: bool = True,
) -> None:
    """Run ConvSplice variant scoring: VCF/TSV -> ``output_dir`` (TSVs).

    Uses ``venv_path``'s Python and runs ``predict_variants.py`` from ``codedir``.

    Args:
        input_file: Path to variant file (VCF or TSV).
        output_dir: Directory for outputs (created if missing).
        gtf_path: GENCODE GTF; if None, uses ``DEFAULT_GTF_PATH`` or
            ``codedir/resources/gencode.v47.basic.annotation.gtf.gz``.
        genome_path: Reference FASTA; if None, uses ``DEFAULT_GENOME_PATH`` or
            ``codedir/resources/hg38.fa.gz``.
        cpu: If True, pass ``--cpu`` to force CPU inference.
        weights_dir: Optional ``--weights_dir`` for model checkpoints.
        annotate: If False, pass ``--no_annotate`` (variants need a ``gene`` column).
    """
    os.makedirs(output_dir, exist_ok=True)

    inp = os.path.abspath(input_file)
    out = os.path.abspath(output_dir)
    gtf = os.path.abspath(gtf_path if gtf_path is not None else _default_gtf_path())
    fa = os.path.abspath(genome_path if genome_path is not None else _default_genome_path())

    for label, path in (
        ('Input file', inp),
        ('GTF', gtf),
        ('Genome FASTA', fa),
    ):
        if not os.path.isfile(path):
            sys.exit(f'{label} not found: {path}')

    predict_script = os.path.join(codedir, 'predict_variants.py')
    if not os.path.isfile(predict_script):
        sys.exit(f'predict_variants.py not found under codedir: {predict_script}')

    python_exe = os.path.join(venv_path, 'bin', 'python')
    if not os.path.isfile(python_exe):
        sys.exit(f'Venv python not found: {python_exe}')

    env = os.environ.copy()
    env['VIRTUAL_ENV'] = venv_path
    env['TQDM_DISABLE'] = '1'
    env['PYTHONUNBUFFERED'] = '1'
    env['PATH'] = os.path.join(venv_path, 'bin') + os.pathsep + env.get('PATH', '')

    cmd: list[str] = [
        python_exe,
        predict_script,
        '--variant_path',
        inp,
        '--gtf_path',
        gtf,
        '--genome_path',
        fa,
        '--output_path',
        out,
    ]
    if cpu:
        cmd.append('--cpu')
    if weights_dir:
        cmd.extend(['--weights_dir', os.path.abspath(weights_dir)])
    if not annotate:
        cmd.append('--no_annotate')

    log_time(f'Starting ConvSplice from {codedir}')
    p = subprocess.Popen(
        cmd,
        cwd=codedir,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding='utf-8',
        bufsize=1,
    )
    assert p.stdout is not None
    for line in iter(p.stdout.readline, ''):
        print(line, end='')
    p.stdout.close()
    rc = p.wait()
    if rc != 0:
        sys.exit(rc)

def main() -> None:
    import argparse

    p = argparse.ArgumentParser(
        description='Run predict_variants.py with the venv at venv_path (see run_convsplice.py).',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            'If gtf_path / genome_path are omitted, defaults match predict_variants.sh:\n'
            '  codedir/resources/gencode.v47.basic.annotation.gtf.gz\n'
            '  codedir/resources/hg38.fa.gz\n'
            'Override module-level DEFAULT_GTF_PATH / DEFAULT_GENOME_PATH for deploy-wide pins.'
        ),
    )
    p.add_argument('input_file', help='VCF or TSV (chr, pos, ref, alt).')
    p.add_argument('output_dir', help='Output directory for TSVs.')
    p.add_argument(
        'gtf_path',
        nargs='?',
        default=None,
        help='GENCODE GTF (default: resources/gencode.v47... under codedir or DEFAULT_GTF_PATH).',
    )
    p.add_argument(
        'genome_path',
        nargs='?',
        default=None,
        help='Reference FASTA (default: resources/hg38.fa.gz or DEFAULT_GENOME_PATH).',
    )
    p.add_argument('--cpu', action='store_true', help='Force CPU (--cpu).')
    p.add_argument('--weights-dir', default=None, dest='weights_dir', help='Optional --weights_dir.')
    p.add_argument(
        '--no-annotate',
        action='store_true',
        dest='no_annotate',
        help='Pass --no_annotate to predict_variants.py.',
    )
    args = p.parse_args()

    run_pipeline(
        args.input_file,
        args.output_dir,
        args.gtf_path,
        args.genome_path,
        cpu=args.cpu,
        weights_dir=args.weights_dir,
        annotate=not args.no_annotate,
    )


if __name__ == '__main__':
    main()

