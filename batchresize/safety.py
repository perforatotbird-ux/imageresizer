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

def _lock_payload() -> dict:
    """Lock content: PID plus a token that survives PID reuse."""
    pid = os.getpid()
    token = None
    try:
        import ctypes
        from ctypes import wintypes

        class _FILETIME(ctypes.Structure):
            _fields_ = [("lo", wintypes.DWORD), ("hi", wintypes.DWORD)]

        creation, exit_code, kernel, user = _FILETIME(), _FILETIME(), _FILETIME(), _FILETIME()
        k32 = ctypes.WinDLL('kernel32', use_last_error=True)
        k32.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(_FILETIME)] * 4
        if not k32.GetProcessTimes(k32.GetCurrentProcess(), ctypes.byref(creation),
                                   ctypes.byref(exit_code), ctypes.byref(kernel),
                                   ctypes.byref(user)):
            raise OSError(ctypes.get_last_error())
        token = (creation.hi << 32) | creation.lo
    except Exception:
        token = None
    return {'pid': pid, 'token': token}


def _read_lock(path: Path) -> Optional[dict]:
    try:
        text = path.read_text(encoding='utf8').strip()
    except OSError:
        return None
    if not text:
        return None
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and 'pid' in obj:
            return {'pid': int(obj['pid']), 'token': obj.get('token')}
    except (ValueError, TypeError):
        pass
    try:                                   # legacy payload: plain PID text
        return {'pid': int(text), 'token': None}
    except (ValueError, TypeError):
        return None


def _pid_alive(pid: int) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return False
    if os.name == 'nt':
        try:
            import ctypes
            from ctypes import wintypes

            k32 = ctypes.WinDLL('kernel32', use_last_error=True)
            k32.GetExitCodeProcess.argtypes = [wintypes.HANDLE,
                                               ctypes.POINTER(wintypes.DWORD)]
            k32.GetExitCodeProcess.restype = wintypes.BOOL
            handle = k32.OpenProcess(0x1000, False, pid)  # QUERY_LIMITED_INFORMATION
            if not handle:
                return False                     # no such process any more
            try:
                code = wintypes.DWORD()
                if not k32.GetExitCodeProcess(handle, ctypes.byref(code)):
                    return False
                # OpenProcess succeeds for exited-but-unreaped processes, so the
                # exit code - not the handle - decides liveness.
                return code.value == 259         # STILL_ACTIVE
            finally:
                k32.CloseHandle(handle)
        except Exception:
            return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def lock_is_live(info: Optional[dict]) -> bool:
    """True when the lock is genuinely held by a running instance."""
    if not info or not isinstance(info.get('pid'), int):
        return False
    if not _pid_alive(info['pid']):
        return False
    mine = _lock_payload()
    if info['pid'] == mine['pid']:
        # our own live lock, or a previous incarnation that reused our PID
        return info.get('token') == mine['token']
    return True


def clear_stale_lock(path: Path, *, force: bool = False) -> bool:
    """Remove a lock left behind by a dead instance. Never touches a live one."""
    path = Path(path)
    try:
        reject_links(path, path.parent)
    except ValueError:
        if not force:
            raise
    if not path.exists():
        return False
    if not force and lock_is_live(_read_lock(path)):
        raise FileExistsError(
            f'{path} is held by a running batchresize instance; stop it first')
    try:
        path.unlink()
    except FileNotFoundError:
        return False
    return True


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
        for _ in range(3):
            try:
                # Reload only while holding the exclusive run lock.
                with self.lock_path.open('x', encoding='utf8') as f:
                    json.dump(_lock_payload(), f)
                self.locked = True
                break
            except FileExistsError:
                info = _read_lock(self.lock_path)
                if lock_is_live(info):
                    raise FileExistsError(
                        f'another batchresize run is writing to {self.root} '
                        f'({LOCK_NAME} held by pid {info["pid"]}). If that run is '
                        f'gone, rerun with --unlock to clear the stale lock.')
                # Stale lock from a dead instance: take it over.
                if not clear_stale_lock(self.lock_path):
                    raise FileExistsError(
                        f'could not clear the stale lock {self.lock_path}; '
                        f'inspect it or rerun with --unlock')
        else:
            raise FileExistsError(
                f'output lock {self.lock_path} kept changing between runs; '
                f'inspect it manually')
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
                        im.size == (plan.target_w, plan.target_h))
        except (OSError, ValueError):
            return False

    def record(self, job, jpeg, src, *, variant_id=None, config_digest=None,
               out=None, plan=None, actual_quality=None, target_size_achieved=None):
        relative = job.src.relative_to(src).as_posix()
        out = out or job.out
        plan = plan or job.plan
        entry = {
            'config': config_digest or self.fingerprint,
            'source_sha256': job.digest,
            'source': relative,
            'output': out.relative_to(self.root).as_posix(),
            'output_sha256': hashlib.sha256(jpeg).hexdigest(),
            'output_w': plan.target_w,
            'output_h': plan.target_h,
            'output_bytes': len(jpeg),
        }
        if actual_quality is not None:
            entry['actual_quality'] = actual_quality
        if target_size_achieved is not None:
            entry['target_size_achieved'] = target_size_achieved
        if variant_id is not None:
            entry['variant'] = variant_id
        self.entries[self._key(relative, variant_id)] = entry

    def save(self):
        payload = json.dumps({'version': 1, 'entries': self.entries}, ensure_ascii=False, indent=2)
        atomic_write(self.path, payload.encode('utf8'), True)
