"""Desired-behaviour regressions for IR-01..26. GPU contracts use CPU mocks."""
import ast
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import types
import numpy as np
import pytest
from PIL import Image, ImageOps, ImageCms
from batchresize import cli
from batchresize.cpu_backend import CpuEngine, decode_to_rgb
from batchresize.geometry import make_plan, compute_box
from batchresize.gpu_backend import build_weights
from batchresize.imageio_meta import read_source_meta, splice_metadata, has_alpha_hint, is_truncated_jpeg
from batchresize.pipeline import Pipeline, RunOptions, Job, atomic_write
from batchresize.presets import ALL_QUALITY_PRESETS
from batchresize.safety import ResumeState

ROOT = Path(__file__).resolve().parents[1]

def png(mode='RGB', size=(64, 48), **kw):
    b = io.BytesIO(); Image.new(mode, size).save(b, 'PNG', **kw); return b.getvalue()
def source(tmp_path):
    s = tmp_path/'src'; s.mkdir(); (s/'a.png').write_bytes(png()); return s

def run(s, o, **kw):
    opts = RunOptions(s, o, 32, 32, workers=1, **kw)
    return Pipeline(opts, CpuEngine(opts.filter_name), 'cpu', False).run()
def job(p, alpha=False):
    d = png('RGBA' if alpha else 'RGB'); m = read_source_meta(d)
    return Job(p/'a.png', p/'a.jpg', d, m, make_plan(64,48,32,32), alpha)

@pytest.mark.parametrize('relation', ['same','ancestor'])
def test_IR01_unsafe_roots_rejected(tmp_path, relation):
    s = source(tmp_path); o = s if relation=='same' else tmp_path
    with pytest.raises(ValueError): run(s,o)
    assert (s/'a.png').read_bytes() == png()

def test_IR01_output_links_rejected(tmp_path):
    s=source(tmp_path); (s/'sub').mkdir(); (s/'sub/a.jpg').write_bytes(png())
    o=tmp_path/'out';o.mkdir()
    try:
        (o/'sub').symlink_to(s/'sub',target_is_directory=True)
    except OSError as e:
        if getattr(e, 'winerror', None) == 1314:
            pytest.skip('WinError 1314 symlink privilege')
        raise
    with pytest.raises(ValueError):run(s,o)

def test_IR01_hardlinked_output_rejected(tmp_path):
    s=source(tmp_path);o=tmp_path/'out';o.mkdir();os.link(s/'a.png',o/'a.jpg')
    with pytest.raises(ValueError):run(s,o)

def test_IR02_alpha_normalized_before_GPU(tmp_path):
    class G:
        def decode_batch(self,*a):return [np.zeros((48,64,4),dtype='uint8')]
        def alpha_to_rgb(self,a,bg):assert bg==(1,2,3);return a[:,:,:3].copy()
        def resize(self,a,p):assert a.shape[2]==3;return a
        def encode(self,*a):return b'mock'
    p=Pipeline(RunOptions(tmp_path,tmp_path/'out',32,32,bg_color=(1,2,3)),G(),'gpu',False)
    p._process_gpu_batch_inner([job(tmp_path,True)])
    assert p.results.qsize()==1

@pytest.mark.parametrize('mode', ['LA','RGBA','P'])
def test_IR03_host_channel_contract(mode):
    d=png(mode, **({'transparency':0} if mode=='P' else {}));a,alpha=decode_to_rgb(d)
    assert a.dtype==np.uint8 and a.flags.c_contiguous and a.shape[2]==4 and alpha

def test_IR04_processor_failure_accounted(tmp_path):
    class E(CpuEngine):
        def decode(self,d):raise SystemExit('worker died')
    s=source(tmp_path);p=Pipeline(RunOptions(s,tmp_path/'out',32,32,workers=1),E(),'cpu',False)
    r=p.run();assert r['failed']==1 and not r['complete'] and r['fatal_errors']
    assert all(not t.is_alive() for t in p._readers+p._processors+p._writers)

def test_IR04_writer_failure_does_not_deadlock(tmp_path):
    s=source(tmp_path)
    for i in range(20):(s/f'{i}.png').write_bytes(png())
    class P(Pipeline):
        def _writer(self):raise SystemExit('writer died')
    p=P(RunOptions(s,tmp_path/'out',32,32,workers=1,queue_depth=1),CpuEngine(),'cpu',False)
    r=p.run();assert not r['complete'] and r['processed']+r['failed']==r['discovered']
    assert all(not t.is_alive() for t in p._readers+p._processors+p._writers)

def test_writer_drain_is_not_misreported_as_fatal(tmp_path):
    import threading, time
    s=source(tmp_path)
    p=Pipeline(RunOptions(s,tmp_path/'out',32,32,workers=1),CpuEngine(),'cpu',False)
    def slow_writer():
        for _ in range(10):
            time.sleep(0.05)
            with p.stats.lock:p.stats.processed+=1
    t=threading.Thread(target=slow_writer,name='writer-0');t.start()
    p._writers=[t]
    p._join_writers(p._writers);t.join()
    assert not p.fatal.is_set() and not p._fatal_errors

def test_stuck_writer_is_reported_as_fatal(tmp_path):
    import threading, time
    s=source(tmp_path)
    p=Pipeline(RunOptions(s,tmp_path/'out',32,32,workers=1),CpuEngine(),'cpu',False)
    p._writer_stall_timeout=0.1
    t=threading.Thread(target=time.sleep,args=(0.5,),name='writer-0');t.start()
    p._writers=[t]
    started=time.monotonic();p._join_writers(p._writers);elapsed=time.monotonic()-started
    t.join()
    assert p.fatal.is_set() and any('writer-0' in e for e in p._fatal_errors)
    assert elapsed < 0.45

@pytest.mark.parametrize('value', [float('nan'),float('inf'),0,-1])
def test_IR04_IR10_budget_validated(tmp_path,value):
    with pytest.raises(ValueError):RunOptions(tmp_path,tmp_path/'o',32,32,vram_budget_gb=value)

def test_IR05_partial_batch_exactly_once(tmp_path):
    class G:
        def decode_batch(self,*a):return [np.zeros((48,64,3),dtype='uint8'),None]
        def resize(self,a,p):return a
        def encode(self,*a):return b'mock'
    p=Pipeline(RunOptions(tmp_path,tmp_path/'o',32,32,fallback=False),G(),'gpu',False)
    j=job(tmp_path);j2=job(tmp_path/'b')
    p._process_gpu_batch([j,j2]);assert p.results.qsize()==1 and p.stats.failed==1
    p._finish(j,b'duplicate',{},'gpu');assert p.results.qsize()==1

def test_IR06_CPU_filters_change_pixels():
    a=np.random.default_rng(1).integers(0,256,(81,101,3),dtype=np.uint8);p=make_plan(101,81,37,37)
    outputs=[CpuEngine(n).resize(a,p) for n in ('box','bilinear','lanczos3')]
    assert not np.array_equal(outputs[0],outputs[1]);assert not np.array_equal(outputs[1],outputs[2])

def test_IR06_engine_configuration(tmp_path,monkeypatch):
    class G:
        def __init__(self,**kw):self.kw=kw
        def self_test(self):return True,'ok'
    monkeypatch.setattr(cli,'GpuEngine',G)
    o=RunOptions(tmp_path,tmp_path/'o',32,32,filter_name='box',optimize_huffman=True)
    g,k=cli._select_engine('gpu',o)
    assert g.kw=={'filter_name':'box','optimize_huffman':True}

@pytest.mark.parametrize('key,q',[('light',90),('medium',80),('strong',65),('extreme',50)])
def test_IR07_all_GUI_presets(key,q):
    tree=ast.parse((ROOT/'batchresize/gui.py').read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='BatchResizeUI');fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_on_quality_change')
    ns={'ALL_QUALITY_PRESETS':ALL_QUALITY_PRESETS,'QUALITY_CHOICES':[(key,'label')]};exec(compile(ast.Module(body=[fn],type_ignores=[]),'gui','exec'),ns)
    got={};var=types.SimpleNamespace(set=lambda v:got.update(value=v));scale=types.SimpleNamespace(set=lambda v:got.update(scale=v));o=types.SimpleNamespace(quality_combo=types.SimpleNamespace(get=lambda:'label'),quality_var=types.SimpleNamespace(set=lambda v:None),jpeg_quality_var=var,quality_scale=scale)
    ns['_on_quality_change'](o);assert got=={'value':str(q),'scale':q}

def test_IR08_lightweight_CLI_import():
    env=dict(os.environ,PYTHONPATH=str(ROOT));p=subprocess.run([sys.executable,'-S','-c','import batchresize.cli; import sys; assert "numpy" not in sys.modules'],env=env,capture_output=True,text=True)
    assert p.returncode==0,p.stderr

def test_IR09_verified_resume_and_corruption(tmp_path):
    s=source(tmp_path);o=tmp_path/'out';assert run(s,o)['processed']==1
    assert run(s,o,skip_existing=True,overwrite=False)['skipped']==1
    (o/'a.jpg').write_bytes(b'corrupt')
    r=run(s,o,skip_existing=True,overwrite=False);assert r['failed']==1 and r['skipped']==0
    assert (o/'a.jpg').read_bytes()==b'corrupt'

def test_IR09_configuration_change_not_skipped(tmp_path):
    s=source(tmp_path);o=tmp_path/'out';run(s,o)
    assert run(s,o,skip_existing=True,overwrite=False,quality=90)['failed']==1

def test_IR09_mapping_stable_when_source_added(tmp_path):
    s=source(tmp_path);(s/'photo.jpg').write_bytes(png());(s/'photo.png').write_bytes(png());o=tmp_path/'out';run(s,o)
    before=json.loads((o/'.batchresize-state.json').read_text())['entries'];(s/'photo.bmp').write_bytes(png());run(s,o)
    after=json.loads((o/'.batchresize-state.json').read_text())['entries']
    assert all(before[k]['output']==after[k]['output'] for k in before)

@pytest.mark.parametrize('sizing', ['percent','megapixels','longest','shortest'])
@pytest.mark.parametrize('value', [0,-1,float('nan'),float('inf')])
def test_IR10_sizing_rejected(sizing,value):
    with pytest.raises(ValueError):compute_box(64,48,sizing,32,32,value)
@pytest.mark.parametrize('w,h',[(0,10),(-1,10),(100000,100000)])
def test_IR10_invalid_targets(w,h):
    with pytest.raises(ValueError):make_plan(10,10,w,h)

def test_IR11_extreme_cover_nonempty():
    p=make_plan(10,100,100,1,'cover',False);out=CpuEngine().process(png(size=(10,100)),p,90,0)
    with Image.open(io.BytesIO(out)) as im:im.load();assert im.size==(10,1)

def test_IR12_box_weights():
    idx,w=build_weights(4,2,.5,0,'box');out=(np.array([0.,0.,255.,255.])[idx]*w).sum(1)
    assert np.allclose(out,[0,255])
def test_IR12_bilinear_border():
    idx,w=build_weights(4,8,2.,0,'bilinear');out=(np.array([0.,255.,0.,0.])[idx]*w).sum(1)
    assert out[1]==pytest.approx(63.75)

@pytest.mark.parametrize('mode', ['RGB','L','P'])
def test_IR13_trns_content_detection(mode):
    tr=(0,0,0) if mode=='RGB' else 0;d=png(mode,transparency=tr)
    assert has_alpha_hint(d,'.unrelated') and read_source_meta(d).has_alpha

def test_IR14_valid_CMYK_profile(tmp_path):
    fixture=Path('/usr/share/ghostscript/iccprofiles/default_cmyk.icc')
    if not fixture.exists():pytest.skip('valid CMYK ICC fixture unavailable')
    s=source(tmp_path);(s/'a.png').unlink();Image.new('CMYK',(64,48),(40,100,150,20)).save(s/'a.jpg',icc_profile=fixture.read_bytes())
    o=tmp_path/'out';r=run(s,o);assert r['processed']==1
    with Image.open(o/'a.jpg') as im:
        im.load();profile=ImageCms.getOpenProfile(io.BytesIO(im.info['icc_profile']));assert profile.profile.xcolor_space.strip()=='RGB'
        ImageCms.profileToProfile(im,profile,ImageCms.createProfile('sRGB'),outputMode='RGB')

def test_IR15_metadata_failure_visible(tmp_path,monkeypatch):
    import batchresize.pipeline as mod
    def fail(*a,**kw):raise ValueError('metadata overflow')
    monkeypatch.setattr(mod,'splice_metadata',fail);s=source(tmp_path);r=run(s,tmp_path/'out')
    assert r['failed']==1 and r['processed']==0 and 'metadata overflow' in r['errors'][0][1]

def test_IR16_dimensions_updated():
    ex=Image.Exif();ex[256]=64;ex[257]=48;ex[274]=6
    b=io.BytesIO();Image.new('RGB',(64,48)).save(b,'JPEG',exif=ex);m=read_source_meta(b.getvalue());plain=io.BytesIO();Image.new('RGB',(12,16)).save(plain,'JPEG')
    out=splice_metadata(plain.getvalue(),m.exif,None,dimensions=(12,16))
    with Image.open(io.BytesIO(out)) as im:assert (im.getexif()[256],im.getexif()[257],im.getexif()[274])==(12,16,1)

def test_IR17_valid_trailer_and_missing_EOI():
    b=io.BytesIO();Image.new('RGB',(64,48)).save(b,'JPEG');d=b.getvalue();assert not is_truncated_jpeg(d+b'\x00'*5000);assert is_truncated_jpeg(d[:-2])
def test_IR17_mislabeled_PNG_accepted(tmp_path):
    s=source(tmp_path);(s/'a.png').rename(s/'a.jpg');assert run(s,tmp_path/'out')['processed']==1

def test_IR18_CPU_fallback_does_not_require_GPU(tmp_path):
    class G:
        def to_device(self,*a):raise AssertionError('GPU must not be used')
    s=source(tmp_path);p=Pipeline(RunOptions(s,tmp_path/'out',32,32),G(),'gpu',False)
    p._gpu_fallback(job(s),RuntimeError('device lost'));assert p.results.qsize()==1 and p.results.get().backend=='cpu-fallback'

def test_IR18_source_byte_cap(tmp_path):
    s=source(tmp_path);r=run(s,tmp_path/'out',max_source_bytes=10)
    assert r['failed']==1 and r['processed']==0

def test_IR19_fixed_benchmark_population(tmp_path,monkeypatch):
    monkeypatch.setattr(cli,'GpuEngine',lambda **kw:(_ for _ in ()).throw(RuntimeError('no GPU')))
    s=source(tmp_path);o=s/'bench';assert cli._benchmark(RunOptions(s,o,32,32,workers=1),2,True)==0
    r=json.loads((o/'benchmark.json').read_text());assert [x['discovered'] for x in r['cpu']['runs']]==[1,1]

def test_IR19_failed_benchmark_nonzero(tmp_path,monkeypatch):
    monkeypatch.setattr(cli,'GpuEngine',lambda **kw:(_ for _ in ()).throw(RuntimeError('no GPU')))
    s=source(tmp_path);(s/'a.png').write_bytes(b'broken');assert cli._benchmark(RunOptions(s,tmp_path/'out',32,32,workers=1),1,True)!=0

def test_IR20_IR26_honest_report(tmp_path):
    s=source(tmp_path);r=run(s,tmp_path/'out');assert 'not kernel execution time' in r['timing_semantics']
    assert r['backend_counts']=={'cpu':1} and 'normalize' in r['extra_timings_avg_ms'] and 'scan_sec' in r

def test_IR21_IR22_BAT_static():
    s=(ROOT/'run.bat').read_text();assert 'DisableDelayedExpansion' in s and '-X utf8' in s
    assert 'if not "%~1"=="" (' not in s and 'exit /b %errorlevel%' in s

def test_IR23_GUI_UTF8_and_cooperative_cancel():
    s=(ROOT/'batchresize/gui.py').read_text();assert 'PYTHONIOENCODING="utf-8"' in s and 'self._cancel_file.touch()' in s
    assert 'self._proc.terminate()' not in s

def test_IR24_directory_recreated(tmp_path):
    p=tmp_path/'old/a.jpg';atomic_write(p,b'one');p.parent.rename(tmp_path/'moved');atomic_write(p,b'two');assert p.read_bytes()==b'two'

def test_IR25_discovery_error_in_report(tmp_path,monkeypatch):
    import batchresize.pipeline as mod
    s=source(tmp_path)
    def denied(*a):raise PermissionError('no scan access')
    monkeypatch.setattr(mod.os,'scandir',denied);r=run(s,tmp_path/'out');assert r['discovery_errors'] and not r['complete']

def test_resume_lock_and_exclusive_publication(tmp_path):
    o=tmp_path/'out';state=ResumeState(o,{});state.acquire()
    try:
        with pytest.raises(FileExistsError):ResumeState(o,{}).acquire()
    finally:state.release()
    p=o/'a.jpg';atomic_write(p,b'old')
    with pytest.raises(FileExistsError):atomic_write(p,b'new',overwrite=False)
    assert p.read_bytes()==b'old' and not list(o.glob('*.tmp'))

# --- AUD-002: a lock left behind by a dead run must not block the folder ----
from batchresize.safety import (LOCK_NAME, _pid_alive, clear_stale_lock,
                                lock_is_live, _read_lock, _lock_payload)

def _write_lock(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) if isinstance(payload, dict) else str(payload))

def test_stale_lock_with_dead_pid_is_taken_over(tmp_path):
    o=tmp_path/'out'; o.mkdir()
    _write_lock(o/LOCK_NAME, {'pid': 999999, 'token': 12345})
    assert not lock_is_live(_read_lock(o/LOCK_NAME))
    state=ResumeState(o,{}); state.acquire()          # must not raise
    try:
        assert _read_lock(o/LOCK_NAME)['pid']==os.getpid()
    finally: state.release()
    assert not (o/LOCK_NAME).exists()

def test_stale_lock_with_reused_pid_is_taken_over(tmp_path):
    o=tmp_path/'out'; o.mkdir()
    # our own PID but a different process start token: PID was reused
    _write_lock(o/LOCK_NAME, {'pid': os.getpid(), 'token': 1})
    assert not lock_is_live(_read_lock(o/LOCK_NAME))
    state=ResumeState(o,{}); state.acquire()
    try:
        assert _read_lock(o/LOCK_NAME)['pid']==os.getpid()
    finally:
        state.release()

def test_live_lock_from_another_process_is_refused(tmp_path, monkeypatch):
    o=tmp_path/'out'; o.mkdir()
    payload=_lock_payload()
    monkeypatch.setattr('batchresize.safety._lock_payload',
                        lambda: {'pid': payload['pid'], 'token': payload['token']})
    _write_lock(o/LOCK_NAME, {'pid': payload['pid'], 'token': payload['token']})
    assert lock_is_live(_read_lock(o/LOCK_NAME))
    with pytest.raises(FileExistsError):
        ResumeState(o,{}).acquire()
    with pytest.raises(FileExistsError):
        clear_stale_lock(o/LOCK_NAME)
    assert (o/LOCK_NAME).exists()

def test_pid_liveness_detects_a_terminated_process(tmp_path):
    proc=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'])
    try:
        assert _pid_alive(proc.pid)
    finally:
        proc.terminate(); proc.wait(timeout=10)
    assert not _pid_alive(proc.pid), 'OpenProcess succeeds for exited processes; the exit code must be checked'

def test_clear_stale_lock_removes_only_dead_owner_locks(tmp_path):
    o=tmp_path/'out'; o.mkdir()
    assert clear_stale_lock(o/LOCK_NAME) is False      # nothing to do
    _write_lock(o/LOCK_NAME, {'pid': 999999, 'token': 5})
    assert clear_stale_lock(o/LOCK_NAME) is True
    assert not (o/LOCK_NAME).exists()

def test_cli_unlock_flag_clears_a_stale_lock(tmp_path):
    s=source(tmp_path); o=tmp_path/'out'; o.mkdir()
    _write_lock(o/LOCK_NAME, {'pid': 999999, 'token': 7})
    # a stale lock is taken over automatically: the plain run now succeeds
    assert cli.main(['-i',str(s),'-o',str(o),'--size','thumb_256','--backend','cpu'])==0
    assert (o/'a.jpg').exists() and not (o/LOCK_NAME).exists()
    # --unlock also clears a stale lock and reports it
    _write_lock(o/LOCK_NAME, {'pid': 999999, 'token': 8})
    assert cli.main(['-i',str(s),'-o',str(o),'--size','thumb_256','--backend','cpu','--unlock'])==0
    assert not (o/LOCK_NAME).exists()

def test_cancel_file_saves_manifest(tmp_path):
    s=source(tmp_path);cancel=tmp_path/'cancel';cancel.touch()
    r=run(s,tmp_path/'out',cancel_file=cancel)
    assert r['interrupted'] and not (tmp_path/'out/.batchresize.lock').exists()
    assert (tmp_path/'out/.batchresize-state.json').exists()

def test_IR09_deleted_source_target_stays_reserved(tmp_path):
    s=source(tmp_path);o=tmp_path/'out';run(s,o)
    original=(o/'a.jpg').read_bytes();(s/'a.png').unlink();(s/'a.bmp').write_bytes(png())
    assert run(s,o)['processed']==1
    state=json.loads((o/'.batchresize-state.json').read_text())['entries']
    assert state['a.bmp']['output']=='a__2.jpg' and (o/'a.jpg').read_bytes()==original

def test_IR19_benchmark_never_times_skips(tmp_path,monkeypatch):
    monkeypatch.setattr(cli,'GpuEngine',lambda **kw:(_ for _ in ()).throw(RuntimeError('no GPU')))
    s=source(tmp_path);o=tmp_path/'out';opts=RunOptions(s,o,32,32,workers=1,skip_existing=True,overwrite=False)
    assert cli._benchmark(opts,1,True)==0
    first=json.loads((o/'benchmark.json').read_text())['cpu']['run_root']
    assert cli._benchmark(opts,1,True)==0
    rep=json.loads((o/'benchmark.json').read_text())['cpu']
    assert rep['processed']==1 and rep['skipped']==0 and rep['run_root']!=first

def test_IR19_source_mutation_invalidates_benchmark(tmp_path,monkeypatch):
    monkeypatch.setattr(cli,'GpuEngine',lambda **kw:(_ for _ in ()).throw(RuntimeError('no GPU')))
    s=source(tmp_path);o=tmp_path/'out';real=cli._run_once
    def mutating(*args,**kw):
        result=real(*args,**kw);(s/'a.png').write_bytes(png(size=(80,80)));return result
    monkeypatch.setattr(cli,'_run_once',mutating)
    assert cli._benchmark(RunOptions(s,o,32,32,workers=1),1,True)==1
    assert not json.loads((o/'benchmark.json').read_text())['cpu']['valid']

def test_IR26_explicit_strict_fails_closed():
    from batchresize.gpu_backend import GpuEngine,GpuUnavailable
    with pytest.raises(GpuUnavailable,match='attestation is unsupported'):GpuEngine(strict=True)

def test_IR04_live_native_thread_retains_lock(tmp_path,monkeypatch):
    s=source(tmp_path);o=tmp_path/'out';p=Pipeline(RunOptions(s,o,32,32),CpuEngine(),'cpu',False)
    def live():
        p._processors.append(types.SimpleNamespace(is_alive=lambda:True));return {'complete':False}
    monkeypatch.setattr(p,'_run_locked',live)
    assert not p.run()['complete'] and (o/'.batchresize.lock').exists()
    p._resume.release()  # fake thread only; no real live worker

@pytest.mark.parametrize('kw',[{'quality':80.5},{'max_source_bytes':12.5}])
def test_IR10_fractional_integer_contracts(tmp_path,kw):
    with pytest.raises(ValueError):RunOptions(tmp_path,tmp_path/'o',32,32,**kw)

def test_IR15_metadata_failure_is_visible_and_preserves_output(tmp_path,monkeypatch):
    import batchresize.pipeline as mod
    s=source(tmp_path);o=tmp_path/'out';run(s,o);before=(o/'a.jpg').read_bytes()
    monkeypatch.setattr(mod,'splice_metadata',lambda *a,**kw:(_ for _ in ()).throw(ValueError('bad EXIF')))
    rep=run(s,o);assert rep['failed']==1 and rep['processed']==0 and (o/'a.jpg').read_bytes()==before

@pytest.mark.parametrize('mode,transparency',[('RGB',(0,0,0)),('L',0),('P',0)])
def test_IR13_CPU_transparency_composites_actual_pixels(tmp_path,mode,transparency):
    s=tmp_path/'src';s.mkdir();(s/'a.png').write_bytes(png(mode,transparency=transparency))
    rep=run(s,tmp_path/'out',bg_color=(255,255,255),quality=100,subsampling=0)
    assert rep['processed']==1
    with Image.open(tmp_path/'out/a.jpg') as im:
        a=np.asarray(im);assert a.min()>=254

def test_IR15_EXIF_parser_failure_is_not_silent(monkeypatch):
    monkeypatch.setattr(Image.Image,'getexif',lambda self:(_ for _ in ()).throw(ValueError('bad EXIF')))
    with pytest.raises(ValueError,match='EXIF safely'):read_source_meta(png())

def test_IR20_GPU_decode_is_synchronized_in_profile(tmp_path):
    events=[]
    class G:
        def decode_batch(self,*a):events.append('decode');return [None]
        def synchronize(self):events.append('sync')
    p=Pipeline(RunOptions(tmp_path,tmp_path/'out',32,32,profile_cuda=True),G(),'gpu',False)
    p._gpu_fallback=lambda *a:events.append('fallback')
    p._process_gpu_batch_inner([job(tmp_path)])
    assert events==['decode','sync','fallback']

def test_IR20_CPU_fallback_does_not_sync_broken_CUDA(tmp_path):
    class G:
        def synchronize(self):raise RuntimeError('broken CUDA must not break CPU fallback')
    p=Pipeline(RunOptions(tmp_path,tmp_path/'out',32,32,profile_cuda=True),G(),'cpu',False)
    p._gpu_fallback(job(tmp_path),RuntimeError('GPU failed'))
    assert p.results.qsize()==1 and p.stats.failed==0

def test_IR27_CPU_encoder_honors_optimize_huffman():
    """--optimize-huffman must reach the CPU encoder (it was silently ignored)."""
    rng=np.random.default_rng(7)
    img=rng.integers(0,255,(256,256,3),dtype=np.uint8)
    plain=CpuEngine('lanczos3',False).encode(img,90,0)
    optimized=CpuEngine('lanczos3',True).encode(img,90,0)
    assert plain.startswith(b'\xff\xd8') and optimized.startswith(b'\xff\xd8')
    assert len(optimized)<len(plain)

def test_IR27_optimize_huffman_reaches_CPU_engine_from_options(tmp_path):
    s=source(tmp_path);o=tmp_path/'out'
    opts=RunOptions(s,o,32,32,workers=1,optimize_huffman=True)
    engine,kind=cli._select_engine('cpu',opts)
    assert kind=='cpu' and getattr(engine,'optimize_huffman',False) is True
    rep=Pipeline(opts,engine,'cpu',False).run()
    assert rep['processed']==1
