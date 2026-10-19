"""Output preflight and verified resume. Never trust a filename alone."""
from __future__ import annotations
import hashlib
import json
import os
import tempfile
from pathlib import Path
from PIL import Image

STATE_NAME = '.batchresize-state.json'
LOCK_NAME = '.batchresize.lock'

def path_key(path):
    return os.path.normcase(str(Path(path).resolve())).casefold()

def reject_links(path: Path, root: Path):
    """Reject output symlinks/junctions, including pre-existing leaf aliases.

    This is a local-user safety guard, not a hostile-filesystem TOCTOU defence.
    """
    path, root = Path(path).absolute(), Path(root).absolute()
    if not path.is_relative_to(root):
        raise ValueError(f'output escapes destination: {path}')
    current = path
    while True:
        if (current.is_symlink() or getattr(current, 'is_junction', lambda: False)() or
                (current.exists() and getattr(current.lstat(), 'st_file_attributes', 0) & 0x400)):
            raise ValueError(f'output link/reparse point is not allowed: {current}')
        if current == root:
            break
        current = current.parent

def check_roots(src: Path, dst: Path):
    if src == dst or src.is_relative_to(dst):
        raise ValueError('output must not equal or be an ancestor of input')

def source_identities(files):
    ids = set()
    for path in files:
        st = path.stat()
        ids.add((st.st_dev, st.st_ino))
    return ids

def protect_target(path: Path, root: Path, source_keys, identities):
    reject_links(path, root)
    if path_key(path) in source_keys:
        raise ValueError(f'output aliases an original: {path}')
    if path.exists():
        st = path.stat()
        if (st.st_dev, st.st_ino) in identities:
            raise ValueError(f'output is the same file as an original: {path}')

def atomic_write(path: Path, data: bytes, fsync=False, *, overwrite=True):
    path = Path(path)
    if path.is_symlink() or getattr(path, 'is_junction', lambda: False)():
        raise ValueError(f'refusing output link: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.' + path.name + '.', suffix='.tmp', dir=path.parent)
    tmp = Path(name)
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(data); f.flush()
            if fsync:
                os.fsync(f.fileno())
        if overwrite:
            os.replace(tmp, path)
        else:
            # Exclusive publication: never silently replace a concurrent result.
            # If the filesystem lacks hard links, fail safely instead of replacing.
            os.link(tmp, path)
            tmp.unlink()
        if fsync and os.name != 'nt':
            directory = os.open(path.parent, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    finally:
        tmp.unlink(missing_ok=True)

class ResumeState:
    def __init__(self, root: Path, config: dict):
        self.root = root
        self.path = root / STATE_NAME
        self.lock_path = root / LOCK_NAME
        self.entries = {}
        self.fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
        self.locked = False
        reject_links(self.path, root)
        if self.path.exists():
            obj = json.loads(self.path.read_text(encoding='utf8'))
            if obj.get('version') != 1 or not isinstance(obj.get('entries'), dict):
                raise ValueError('invalid resume manifest; preserve it and inspect before retry')
            self.entries = obj['entries']

    def acquire(self):
        self.root.mkdir(parents=True, exist_ok=True)
        reject_links(self.lock_path, self.root)
        # Reload only while holding the exclusive run lock.
        with self.lock_path.open('x', encoding='utf8') as f:
            f.write(str(os.getpid()))
        self.locked = True
        if self.path.exists():
            obj = json.loads(self.path.read_text(encoding='utf8'))
            if obj.get('version') != 1 or not isinstance(obj.get('entries'), dict):
                raise ValueError('invalid resume manifest')
            self.entries = obj['entries']

    def release(self):
        if self.locked:
            self.lock_path.unlink(missing_ok=True)
            self.locked = False

    @staticmethod
    def _key(relative: str, variant_id):
        return relative if variant_id is None else f'{relative}::{variant_id}'

    def previous_mapping(self, files, src, variant_id=None):
        out = {}
        for key, entry in self.entries.items():
            if not isinstance(key, str) or not isinstance(entry, dict):
                raise ValueError('invalid resume entry')
            if entry.get('variant') != variant_id:
                continue
            rel = entry.get('source', key)
            if not isinstance(rel, str) or Path(rel).is_absolute() or '..' in Path(rel).parts:
                raise ValueError('invalid manifest source path')
            f = src / rel
            name = entry.get('output')
            if name:
                target = self.root / name
                if Path(name).is_absolute() or '..' in Path(name).parts:
                    raise ValueError('invalid manifest output path')
                reject_links(target, self.root)
                if target.suffix.lower() != '.jpg':
                    raise ValueError('invalid manifest output extension')
                out[f] = target
        return out

    def valid(self, job, src, *, variant_id=None, config_digest=None,
              out=None, plan=None):
        relative = job.src.relative_to(src).as_posix()
        entry = self.entries.get(self._key(relative, variant_id), {})
        out = out or job.out
        plan = plan or job.plan
        if (entry.get('variant') != variant_id or
                entry.get('source', relative) != relative or
                entry.get('config') != (config_digest or self.fingerprint) or
                entry.get('source_sha256') != job.digest or
                entry.get('output') != out.relative_to(self.root).as_posix()):
            return False
        try:
            data = out.read_bytes()
            if hashlib.sha256(data).hexdigest() != entry.get('output_sha256'):
                return False
            import io
            with Image.open(io.BytesIO(data)) as im:
                im.load()
                return (im.format == 'JPEG' and im.mode == 'RGB' and
                        im.size == (plan.out_w, plan.out_h))
        except (OSError, ValueError):
            return False

    def record(self, job, jpeg, src, *, variant_id=None, config_digest=None,
               out=None, plan=None):
        relative = job.src.relative_to(src).as_posix()
        out = out or job.out
        plan = plan or job.plan
        entry = {
            'config': config_digest or self.fingerprint,
            'source_sha256': job.digest,
            'source': relative,
            'output': out.relative_to(self.root).as_posix(),
            'output_sha256': hashlib.sha256(jpeg).hexdigest(),
            'output_w': plan.out_w,
            'output_h': plan.out_h,
        }
        if variant_id is not None:
            entry['variant'] = variant_id
        self.entries[self._key(relative, variant_id)] = entry

    def save(self):
        payload = json.dumps({'version': 1, 'entries': self.entries}, ensure_ascii=False, indent=2)
        atomic_write(self.path, payload.encode('utf8'), True)
