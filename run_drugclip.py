#!/usr/bin/env python3
"""Verified NET demo and custom-pocket screening using the released DrugCLIP engine.

The host needs only Python's standard library, Docker and an NVIDIA GPU.
Heavy imports are confined to the task-owned inference container.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import pickle
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parent
IMAGE = 'drugclip-easy-start:20261023'
CORE_PREFIXES = ('mol_model.', 'pocket_model.', 'mol_project.', 'pocket_project.')


class UserError(RuntimeError):
    """An actionable error safe to present without a Python traceback."""


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def dump_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def reserve_output(path):
    path = Path(path).expanduser().resolve()
    try:
        path.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise UserError(f'Output already exists: {path}. Choose a new --out directory; existing results are preserved.') from exc
    return path


def verify_resource(path, expected):
    path = Path(path)
    if not path.is_file() or path.stat().st_size != expected['size']:
        raise UserError(f'Resource size mismatch: {path}')
    actual = sha256(path)
    if actual != expected['sha256']:
        raise UserError(f'Resource SHA256 mismatch: {path}. Do not reuse this cache.')
    return actual


def safe_extract(archive, destination, wanted=None):
    destination = Path(destination)
    with zipfile.ZipFile(archive) as source:
        members = source.infolist()
        for member in members:
            name = member.filename.replace('\\', '/')
            parts = PurePosixPath(name).parts
            mode = member.external_attr >> 16
            if (name.startswith('/') or '..' in parts or
                    any(':' in p for p in parts) or stat.S_ISLNK(mode)):
                raise UserError(f'Unsafe archive member: {member.filename}')
        destination.mkdir(parents=True, exist_ok=True)
        for member in members:
            if member.is_dir() or (wanted is not None and not wanted(member.filename)):
                continue
            target = destination.joinpath(*PurePosixPath(member.filename.replace('\\', '/')).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.open(member) as src, target.open('wb') as dst:
                shutil.copyfileobj(src, dst, 8 * 1024**2)


def select_ligand(candidates, code, chain=None, residue=None):
    matches = [c for c in candidates if c['code'] == code and
               (chain is None or c['chain'] == chain) and
               (residue is None or c['residue'] == str(residue))]
    if not matches:
        raise UserError(f'No matching ligand {code!r}. Check its residue code, chain and residue number.')
    if len(matches) != 1:
        choices = ', '.join(f"{c['chain']}:{c['residue']}" for c in matches)
        raise UserError(f'Ligand {code!r} is ambiguous ({choices}); set --ligand-chain and/or --ligand-residue.')
    return matches[0]


def validate_record(record):
    import numpy as np
    if not isinstance(record, dict):
        raise UserError('Each pocket LMDB value must be a pickled record dictionary.')
    name = record.get('pocket')
    if not isinstance(name, str) or not name.strip():
        raise UserError('Each pocket needs a nonempty pocket name.')
    atoms = np.asarray(record.get('pocket_atoms', []))
    try:
        coordinates = np.asarray(record.get('pocket_coordinates', []), dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise UserError(f'{name}: coordinates must be a numeric (N,3) array.') from exc
    if atoms.ndim != 1 or not len(atoms) or coordinates.shape != (len(atoms), 3):
        raise UserError(f'{name}: nonempty atoms and matching (N,3) coordinates are required.')
    if not np.isfinite(coordinates).all():
        raise UserError(f'{name}: coordinates contain nonfinite values.')
    unknown = set(atoms.tolist()) - {'C', 'N', 'O', 'S', 'H'}
    if unknown:
        raise UserError(f'{name}: unsupported element symbols {sorted(unknown)}; use a compatible protein pocket.')
    heavy = int(np.count_nonzero(atoms != 'H'))
    if not 0 < heavy <= 511:
        raise UserError(f'{name}: {heavy} heavy atoms; this quick-start accepts 1-511. Prepare a defined smaller pocket instead of silently sampling a whole protein.')
    return {'name': name, 'stored_atoms': len(atoms), 'heavy_atoms': heavy}


def check_model_keys(missing, unexpected):
    missing_core = [k for k in missing if k.startswith(CORE_PREFIXES)]
    if missing_core:
        raise UserError(f'Checkpoint leaves inference parameters uninitialized: {missing_core[:5]}')


def validate_records(records):
    summaries = [validate_record(record) for record in records]
    if not summaries:
        raise UserError('The pocket input is empty.')
    names = [item['name'] for item in summaries]
    if len(set(names)) != len(names):
        raise UserError('Duplicate pocket names; use distinct names for distinct templates.')
    return summaries


def validate_cache(embeddings, names, previous=None):
    import numpy as np
    if (not isinstance(embeddings, np.ndarray) or embeddings.ndim != 2 or
            embeddings.shape != (len(names), 128) or not len(names) or
            not np.isfinite(embeddings).all()):
        raise UserError('Invalid library embeddings: expected finite nonempty (N,128) array.')
    if any(not isinstance(name, str) or not name for name in names):
        raise UserError('Library names must be nonempty strings.')
    fingerprint = hashlib.sha256()
    for name in names:
        encoded = name.encode('utf-8')
        fingerprint.update(len(encoded).to_bytes(8, 'big'))
        fingerprint.update(encoded)
    identity = {'count': len(names), 'ordered_names_sha256': fingerprint.hexdigest()}
    if previous is not None and identity != previous:
        raise UserError('Molecule identity/order differs between fold caches.')
    return identity


def container_command(image, device, repo, assets, output):
    return ['docker', 'run', '--rm', '--gpus', f'device={device}', '--shm-size=2g',
            '--user', f'{os.getuid()}:{os.getgid()}' if hasattr(os, 'getuid') else '0:0',
            '-e', 'PYTHONPATH=/app', '-e', 'HOME=/tmp',
            '-v', f'{repo}:/app:ro', '-v', f'{assets}:/assets:ro',
            '-v', f'{output}:/out:rw', '-w', '/out', image,
            'python', '/app/run_drugclip.py', '_inside']


def environment_fingerprint(root=ROOT):
    return hashlib.sha256(''.join(sha256(root / name) for name in
                          ('docker/Dockerfile.inference', 'requirements-inference.txt')).encode()).hexdigest()


def read_results(path, library_count):
    rows = []
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        try:
            name, raw_score = line.rsplit(',', 1)
            score = float(raw_score)
        except ValueError as exc:
            raise UserError('Inference produced an invalid result row.') from exc
        if not name or not math.isfinite(score):
            raise UserError('Inference produced an empty compound identity or nonfinite score.')
        if rows and score > rows[-1][1]:
            raise UserError('Inference output is not numerically descending.')
        rows.append((name, score))
    expected = int(library_count * 0.02)
    if not rows or len(rows) != expected:
        raise UserError(f'Expected {expected} top-2% rows from {library_count} records, got {len(rows)}.')
    return rows


def parse_cli(argv=None):
    parser = argparse.ArgumentParser(description='Run the official DrugCLIP NET demo or screen a defined custom pocket.')
    commands = parser.add_subparsers(dest='command', required=True)
    for command in ('demo', 'screen'):
        sub = commands.add_parser(command)
        sub.add_argument('--device', default='0', help='Idle NVIDIA GPU index or UUID (default: 0).')
        sub.add_argument('--out', default=f'runs/{"net" if command == "demo" else "custom"}', help='New output directory; never overwritten.')
        sub.add_argument('--cache-dir', default=os.environ.get('DRUGCLIP_CACHE', str(Path.home() / '.cache' / 'drugclip-easy-start')))
        sub.add_argument('--image', default=IMAGE, help=argparse.SUPPRESS)
        sub.add_argument('--endpoint', default='https://huggingface.co', help='Resource endpoint; bytes must match the official manifest.')
        if command == 'screen':
            inputs = sub.add_mutually_exclusive_group(required=True)
            inputs.add_argument('--pocket-lmdb', help='An existing, validated single-file pocket LMDB.')
            inputs.add_argument('--pocket-pdb', help='An already extracted protein-only pocket PDB.')
            inputs.add_argument('--complex', help='An experimental PDB containing a known site-defining ligand.')
            sub.add_argument('--chains', help='Comma-separated protein chains to retain for a complex.')
            sub.add_argument('--ligand', help='Ligand residue code in the complex (for example DSP).')
            sub.add_argument('--ligand-chain')
            sub.add_argument('--ligand-residue')
    args = parser.parse_args(argv)
    if args.command == 'screen' and args.complex and (not args.chains or not args.ligand):
        raise UserError('A complex requires explicit --chains and --ligand; the tool does not guess your site.')
    if not re.fullmatch(r'(?:\d+|GPU-[0-9a-fA-F-]+)', args.device):
        raise UserError('--device must be an NVIDIA GPU index or UUID.')
    return args


def download_resource(url, destination, expected):
    destination = Path(destination)
    if destination.exists():
        verify_resource(destination, expected)
        print(f'Verified cached resource: {destination.name}', flush=True)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + '.part')
    offset = partial.stat().st_size if partial.exists() else 0
    if offset == expected['size']:
        verify_resource(partial, expected)
        partial.replace(destination)
        return
    headers = {'User-Agent': 'DrugCLIP-easy-start/1'}
    if offset:
        headers['Range'] = f'bytes={offset}-'
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            if offset and response.status == 206:
                if not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
                    raise UserError('Download server returned an inconsistent resume range.')
                mode = 'ab'
            else:
                offset, mode = 0, 'wb'
            written, last_report = offset, time.monotonic()
            with partial.open(mode) as stream:
                while True:
                    chunk = response.read(8 * 1024**2)
                    if not chunk:
                        break
                    stream.write(chunk)
                    written += len(chunk)
                    if time.monotonic() - last_report >= 10:
                        print(f'{destination.name}: {written / 1024**3:.2f}/{expected["size"] / 1024**3:.2f} GiB', flush=True)
                        last_report = time.monotonic()
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise UserError(f'Download interrupted for {destination.name}; partial bytes are retained for retry. {type(exc).__name__}: {exc}') from exc
    verify_resource(partial, expected)
    partial.replace(destination)
    print(f'Downloaded and verified: {destination.name}', flush=True)


def ensure_resources(cache, endpoint):
    manifest = json.loads((ROOT / 'inference-assets.json').read_text(encoding='utf-8'))
    release = Path(cache).expanduser().resolve() / manifest['revision']
    release.mkdir(parents=True, exist_ok=True)
    provenance = {'dataset': manifest['dataset'], 'revision': manifest['revision'], 'files': {}}
    for role, spec in manifest['files'].items():
        destination = release / spec['path'].rsplit('/', 1)[-1]
        url = f'{endpoint.rstrip("/")}/datasets/{manifest["dataset"]}/resolve/{manifest["revision"]}/{spec["path"]}'
        if role in ('weights', 'molecules'):
            extracted = release / role
            marker = extracted / 'verified.json'
            if marker.is_file():
                entries = json.loads(marker.read_text(encoding='utf-8'))
                if entries.get('archive_sha256') != spec['sha256']:
                    raise UserError(f'Extracted resource manifest mismatch: {marker}')
                for item in entries['members']:
                    verify_resource(extracted / item['path'], item)
                provenance['files'][role] = entries
                continue
            download_resource(url, destination, spec)
            print(f'Extracting six-fold {role} into the separate cache...', flush=True)
            if extracted.exists():
                raise UserError(f'Incomplete extracted cache exists: {extracted}; choose a fresh --cache-dir.')
            with tempfile.TemporaryDirectory(prefix='extract_', dir=release) as tmp:
                stage = Path(tmp) / 'unpacked'
                expected_names = {f'fold_{i}.pt' if role == 'weights' else f'fold{i}.pkl' for i in range(6)}
                wanted = lambda name: '6_folds' in PurePosixPath(name).parts and PurePosixPath(name).name in expected_names
                safe_extract(destination, stage, wanted)
                members = []
                for name in sorted(expected_names):
                    found = list(stage.rglob(name))
                    if len(found) != 1:
                        raise UserError(f'Expected exactly one six-fold {name} in {destination.name}.')
                    p = found[0]
                    members.append({'path': p.relative_to(stage).as_posix(), 'size': p.stat().st_size, 'sha256': sha256(p)})
                entries = {'archive_sha256': spec['sha256'], 'members': members}
                dump_json(stage / 'verified.json', entries)
                stage.rename(extracted)
            provenance['files'][role] = entries
        else:
            download_resource(url, destination, spec)
            provenance['files'][role] = dict(spec, local_name=destination.name)
    return release, provenance


def prepare_pdb(path, options, output):
    """Define one pocket, retaining original heavy-atom coordinates and residues."""
    import numpy as np
    from Bio.PDB import PDBParser, PDBIO, Select
    from Bio.PDB.Polypeptide import is_aa
    structure = PDBParser(QUIET=True).get_structure('input', str(path))
    models = list(structure)
    if len(models) != 1:
        raise UserError('Supply a single-model PDB; select your intended conformation first.')
    model = models[0]
    standard = [r for r in model.get_residues() if is_aa(r, standard=True) and r.id[0] == ' ']
    if not standard:
        raise UserError('No standard protein residues found in PDB.')
    ligand_info = None
    if options.get('complex'):
        chains = options['chains'].split(',')
        protein_chains = {r.parent.id for r in standard}
        if any(c not in protein_chains for c in chains) or len(set(chains)) != len(chains):
            raise UserError(f'Invalid protein --chains; available chains: {sorted(protein_chains)}')
        candidates = []
        for residue in model.get_residues():
            if residue.id[0].startswith('H_'):
                candidates.append({'code': residue.resname.strip(), 'chain': residue.parent.id,
                                   'residue': str(residue.id[1]) + residue.id[2].strip(), 'object': residue})
        ligand = select_ligand(candidates, options['ligand'], options.get('ligand_chain'), options.get('ligand_residue'))
        ligand_info = {k: v for k, v in ligand.items() if k != 'object'}
        ligand_coords = np.array([a.coord for a in ligand['object'] if a.element not in ('H', 'D')])
        if not len(ligand_coords):
            raise UserError('Selected ligand has no heavy atoms.')
        selected = [r for r in standard if r.parent.id in chains and any(
            np.any(np.linalg.norm(ligand_coords - a.coord, axis=1) <= 6.0)
            for a in r)]
    else:
        extra = [r for r in model.get_residues() if r not in standard and r.resname not in ('HOH', 'WAT')]
        if extra:
            raise UserError('Pocket PDB must contain only standard protein residues (waters are ignored); use --complex for a ligand-containing structure.')
        selected = standard
    atoms = []
    for residue in selected:
        for atom in residue:
            child = atom.selected_child if atom.is_disordered() == 2 else atom
            if child.element not in ('H', 'D'):
                atoms.append(child)
    record = {'pocket': path.stem, 'pocket_atoms': [a.element for a in atoms],
              'pocket_coordinates': np.asarray([a.coord for a in atoms], dtype=np.float32)}
    validate_record(record)
    chosen = {id(a) for a in atoms}
    class SelectedAtoms(Select):
        def accept_atom(self, atom):
            return id(atom) in chosen
    writer = PDBIO()
    writer.set_structure(structure)
    writer.save(str(output / 'prepared_pocket.pdb'), SelectedAtoms())
    metadata = {'method': 'whole standard residues with any stored atom within 6 A of selected ligand heavy atoms; protein heavy atoms serialized' if ligand_info else 'all standard protein heavy atoms in supplied pocket',
                'ligand': ligand_info, 'residues': [f'{r.parent.id}:{r.resname}:{r.id[1]}{r.id[2].strip()}' for r in selected],
                'alternate_locations': 'Bio.PDB selected conformer (highest occupancy)'}
    dump_json(output / 'pocket_definition.json', metadata)
    return [record]


def prepare_input(options, output):
    import lmdb
    source = output / ('input' + options['input_suffix'])
    destination = output / 'prepared_pocket.lmdb'
    if options['input_type'] == 'lmdb':
        env = lmdb.open(str(source), subdir=False, readonly=True, lock=False, readahead=False)
        try:
            with env.begin() as txn:
                records = [pickle.loads(value) for _, value in txn.cursor()]
        finally:
            env.close()
    else:
        records = prepare_pdb(source, options, output)
    summaries = validate_records(records)
    env = lmdb.open(str(destination), subdir=False, map_size=max(64 * 1024**2, source.stat().st_size * 4))
    try:
        with env.begin(write=True) as txn:
            for index, record in enumerate(records):
                txn.put(str(index).encode(), pickle.dumps(record, protocol=4))
        env.sync()
    finally:
        env.close()
    return summaries


def engine_argv():
    return ['retrieval.py', '/app/dict', '--user-dir', '/app/unimol', '--valid-subset', 'test',
            '--num-workers', '0', '--ddp-backend', 'c10d', '--batch-size', '4',
            '--task', 'drugclip', '--loss', 'in_batch_softmax', '--arch', 'drugclip',
            '--max-pocket-atoms', '511', '--fp16', '--fp16-init-scale', '4',
            '--fp16-scale-window', '256', '--seed', '1', '--log-interval', '100',
            '--log-format', 'simple', '--mol-path', 'unused.lmdb',
            '--pocket-path', '/out/prepared_pocket.lmdb', '--fold-version', '6_folds',
            '--use-cache', 'true', '--save-path', '/out/raw_results.txt']


def inside():
    import importlib.metadata
    import numpy as np
    import torch
    import unicore
    from unicore import checkpoint_utils
    import runpy
    output = Path('/out')
    options = json.loads((output / 'request.json').read_text())
    if not torch.cuda.is_available():
        raise UserError('PyTorch cannot access the selected CUDA GPU. Check NVIDIA Container Toolkit.')
    if torch.__version__.split('+')[0] != '2.1.2':
        raise UserError('This workflow requires the declared PyTorch 2.1.2 inference image.')
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    summary = prepare_input(options, output)
    work = output / 'work'
    work.mkdir()
    identity = None
    allowed_names = None
    manifest = {'pockets': summary, 'seed': 1, 'folds': [], 'torch': torch.__version__,
                'python': sys.version, 'numpy': np.__version__, 'gpu': torch.cuda.get_device_name(0),
                'engine_argv': engine_argv(), 'input_sha256': sha256(output / ('input' + options['input_suffix'])),
                'prepared_lmdb_sha256': sha256(output / 'prepared_pocket.lmdb')}
    manifest['packages'] = {d.metadata['Name']: d.version for d in importlib.metadata.distributions() if d.metadata['Name']}
    manifest['engine_source_sha256'] = {name: sha256(Path('/app') / name) for name in
                                       ['unimol/retrieval.py', 'unimol/tasks/drugclip.py', 'unimol/models/drugclip.py']}
    for role, prefix, pattern in [('weights', 'model_weights', 'fold_{}.pt'), ('molecules', 'encoded_mol_embs', 'fold{}.pkl')]:
        target = work / 'data' / prefix / '6_folds'
        target.mkdir(parents=True)
        for fold in range(6):
            matches = list((Path('/assets') / role).rglob(pattern.format(fold)))
            if len(matches) != 1:
                raise UserError(f'Expected one {pattern.format(fold)} in the verified cache.')
            (target / pattern.format(fold)).symlink_to(matches[0])
            if role == 'molecules':
                with matches[0].open('rb') as stream:
                    embeddings, names = pickle.load(stream)
                identity = validate_cache(embeddings, names, identity)
                if allowed_names is None:
                    allowed_names = set(names)
                del embeddings, names
    manifest['library'] = identity
    manifest['dictionary_sha256'] = {p.name: sha256(p) for p in Path('/app/dict').iterdir() if p.is_file()}
    dump_json(output / 'inference_manifest.json', manifest)
    original_load = checkpoint_utils.load_checkpoint_to_cpu
    original_state = torch.nn.Module.load_state_dict
    def observe_load(path, *args, **kwargs):
        state = original_load(path, *args, **kwargs)
        entry = {'file': Path(path).name, 'sha256': sha256(path)}
        manifest['folds'].append(entry)
        print('CHECKPOINT_LOADED', json.dumps(entry), flush=True)
        dump_json(output / 'inference_manifest.json', manifest)
        return state
    def observe_state(model, *args, **kwargs):
        result = original_state(model, *args, **kwargs)
        check_model_keys(result.missing_keys, result.unexpected_keys)
        if manifest['folds']:
            manifest['folds'][-1]['missing_keys'] = list(result.missing_keys)
            manifest['folds'][-1]['unexpected_keys'] = list(result.unexpected_keys)
        return result
    checkpoint_utils.load_checkpoint_to_cpu = observe_load
    torch.nn.Module.load_state_dict = observe_state
    os.chdir(work)
    sys.argv = engine_argv()
    try:
        runpy.run_path('/app/unimol/retrieval.py', run_name='__main__')
    finally:
        checkpoint_utils.load_checkpoint_to_cpu = original_load
        torch.nn.Module.load_state_dict = original_state
    if [f['file'] for f in manifest['folds']] != [f'fold_{i}.pt' for i in range(6)]:
        raise UserError('Inference did not load the expected six checkpoints in order.')
    rows = read_results(output / 'raw_results.txt', identity['count'])
    if any(name not in allowed_names for name, _ in rows):
        raise UserError('Result compound identity is absent from the verified library.')
    with (output / 'results.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['rank', 'molecule', 'score'])
        writer.writerows((index, name, score) for index, (name, score) in enumerate(rows, 1))
    manifest.update(result_count=len(rows), result_sha256=sha256(output / 'results.csv'),
                    scores_finite=True, identities_verified=True, status='success')
    dump_json(output / 'inference_manifest.json', manifest)
    print(f'SUCCESS: {len(rows)} ranked molecules from {identity["count"]}; six folds loaded.', flush=True)
    return 0


def checked(command):
    try:
        return subprocess.check_output(command, stderr=subprocess.STDOUT, text=True).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, 'output', '') or str(exc)
        raise UserError(f'Command failed: {command[:3]}\n{detail[-4000:]}') from exc


def run_host(args):
    if sys.platform != 'linux':
        raise UserError('Run this GPU workflow on Linux with Docker and NVIDIA Container Toolkit; see README prerequisites.')
    if Path(args.out).expanduser().exists():
        raise UserError('Output already exists; choose a new --out directory.')
    source = None
    options = vars(args).copy()
    if args.command == 'screen':
        source = Path(args.pocket_lmdb or args.pocket_pdb or args.complex).expanduser().resolve()
        if not source.is_file():
            raise UserError(f'Input file not found: {source}')
    checked(['docker', 'info', '--format', '{{.ServerVersion}}'])
    gpu = checked(['nvidia-smi', '-i', args.device, '--query-gpu=index,memory.used,utilization.gpu', '--format=csv,noheader,nounits'])
    fields = [f.strip() for f in gpu.split(',')]
    if len(fields) != 3 or int(fields[1]) > 2048 or int(fields[2]) > 5:
        raise UserError(f'Selected GPU is busy ({gpu}); choose an idle --device.')
    environment_key = environment_fingerprint()
    probe = subprocess.run(['docker', 'image', 'inspect', '--format', '{{json .Config.Labels}}', args.image],
                           stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    labels = (json.loads(probe.stdout) or {}) if probe.returncode == 0 else {}
    if labels.get('drugclip.environment') != environment_key:
        print('Building the inference environment (first run only)...', flush=True)
        code = subprocess.call(['docker', 'build', '-f', str(ROOT / 'docker/Dockerfile.inference'),
                                '--label', f'drugclip.environment={environment_key}', '--build-arg',
                                'PIP_INDEX_URL=' + os.environ.get('DRUGCLIP_PIP_INDEX_URL', 'https://pypi.org/simple'),
                                '-t', args.image, str(ROOT)])
        if code:
            raise UserError('Inference image build failed. Keep the build log; see README troubleshooting.')
    release, resources = ensure_resources(args.cache_dir, args.endpoint)
    gpu = checked(['nvidia-smi', '-i', args.device, '--query-gpu=index,memory.used,utilization.gpu', '--format=csv,noheader,nounits'])
    fields = [f.strip() for f in gpu.split(',')]
    if len(fields) != 3 or int(fields[1]) > 2048 or int(fields[2]) > 5:
        raise UserError(f'Selected GPU became busy during setup ({gpu}); choose an idle --device.')
    output = reserve_output(args.out)
    options['input_type'] = 'lmdb' if args.command == 'demo' or args.pocket_lmdb else 'pdb'
    options['input_suffix'] = '.lmdb' if options['input_type'] == 'lmdb' else '.pdb'
    if args.command == 'demo':
        source = release / 'pocket.lmdb'
        shutil.copy2(release / 'NET1_DSP.pdb', output / 'NET1_DSP.pdb')
    shutil.copy2(source, output / ('input' + options['input_suffix']))
    dump_json(output / 'request.json', options)
    command = container_command(args.image, args.device, ROOT, release, output)
    provenance = {'created_utc': datetime.now(timezone.utc).isoformat(), 'resources': resources,
                  'docker_command': command, 'image_id': checked(['docker', 'image', 'inspect', '--format', '{{.Id}}', args.image]),
                  'gpu_before': gpu, 'source_path': str(source), 'source_sha256': sha256(source),
                  'runner_sha256': sha256(__file__), 'status': 'running'}
    provenance['environment_definition_sha256'] = environment_key
    try:
        revision = subprocess.run(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'],
                                  stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        provenance['git_commit'] = revision.stdout.strip() if revision.returncode == 0 else None
    except OSError:
        provenance['git_commit'] = None
    dump_json(output / 'run_manifest.json', provenance)
    print(f'Running six-fold GPU screening; full log: {output / "run.log"}', flush=True)
    with (output / 'run.log').open('w', encoding='utf-8') as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            print(line, end='', flush=True)
            log.write(line)
            log.flush()
        code = process.wait()
    provenance['status'] = 'success' if code == 0 else 'failed'
    provenance['exit_code'] = code
    dump_json(output / 'run_manifest.json', provenance)
    if code:
        raise UserError(f'Inference failed (exit {code}); inspect {output / "run.log"}. Choose a new --out for the next run.')
    print(f'Results: {output / "results.csv"}', flush=True)
    return 0


def main(argv=None):
    if (argv or sys.argv[1:]) == ['_inside']:
        return inside()
    args = parse_cli(argv)
    return run_host(args)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except UserError as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        sys.exit(2)
