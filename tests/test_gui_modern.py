"""Real Tk regressions. Skip explicitly when Tk/display is unavailable."""
import json
import os
from pathlib import Path
import sys
import time
import pytest

tk=pytest.importorskip('tkinter',reason='Tkinter not installed in this interpreter')
from batchresize.gui import BatchResizeUI, QUALITY_CHOICES

@pytest.fixture
def app():
    try:
        instance=BatchResizeUI();instance.update()
    except tk.TclError as exc:
        pytest.skip(f'Tk display unavailable: {exc}')
    errors=[]
    instance.report_callback_exception=lambda *e:errors.append(e)
    yield instance
    if instance._is_running():
        instance._cancel()
        wait(instance)
    for timer in instance.tk.call('after','info'):
        instance.after_cancel(timer)
    instance.destroy()
    assert not errors,errors


def wait(app,timeout=20):
    deadline=time.monotonic()+timeout
    while app._proc is not None and time.monotonic()<deadline:
        app.update();time.sleep(.02)
    app.update()
    assert app._proc is None,'child failed to finish'


def test_modern_navigation_and_window_resize(app):
    for i in range(4):
        app.view.nav[i].invoke();app.update();assert app.view.current_page==i
        assert app.view.pages[i].winfo_ismapped()
    app.geometry('1060x740');app.update()
    assert app.view.start_button.winfo_rootx()+app.view.start_button.winfo_width()<=app.winfo_rootx()+app.winfo_width()
    assert app.view.start_button.winfo_rooty()+app.view.start_button.winfo_height()<=app.winfo_rooty()+app.winfo_height()
    app.view.toggle_log();app.update()
    assert app.view.log_panel.winfo_ismapped()
    assert app.view.pages[3].canvas.winfo_height()>100


@pytest.mark.parametrize('index',range(len(QUALITY_CHOICES)))
def test_modern_quality_presets_live_summary(app,index):
    app.quality_combo.current(index);app._on_quality_change();app.update()
    assert f'{app.jpeg_quality_var.get()} / 100' in app.view.summary_quality.get()


def test_modern_size_and_live_summary(app):
    app.preset_combo.set('small_512');app._on_preset_change();app.update()
    assert app.width_var.get()=='512' and '512 × 512' in app.view.summary_size.get()
    app.width_var.set('800');app.update()
    assert app.preset_combo.get()=='custom' and '800 × 512' in app.view.summary_size.get()
    app.sizing_combo.current(1);app._on_sizing_change();app.sizing_value_var.set('50');app.update()
    assert app.view.summary_size.get()=='50 %'


def test_modern_controls_disable_and_reenable(app):
    app.view.set_running(True);assert app.view.start_button.instate(['disabled'])
    assert app.view.cancel_button.instate(['!disabled']) and app.quality_combo.instate(['disabled'])
    app.view.set_running(False);assert app.quality_combo.instate(['readonly','!disabled'])
    assert app.view.start_button.instate(['!disabled']) and app.view.cancel_button.instate(['disabled'])


def test_modern_real_CPU_processing_and_resume(app,tmp_path):
    from PIL import Image
    src=tmp_path/'Фотографии !';src.mkdir();out=tmp_path/'Результаты !'
    for i in range(3):Image.new('RGBA' if i==2 else 'RGB',(64,48),(30,90,180,100) if i==2 else (30,90,180)).save(src/f'{i}.png')
    app.input_var.set(str(src));app.output_var.set(str(out));app.backend_var.set('cpu')
    app.workers_var.set('1');app.update();app._start()
    assert app.view.start_button.instate(['disabled'])
    wait(app)
    report=json.loads((out/'report.json').read_text())
    assert report['complete'] and report['processed']==3
    assert 'Готово за' in app.status_var.get()
    app.skip_existing_var.set(True);app.overwrite_var.set(False);app._start();wait(app)
    assert json.loads((out/'report.json').read_text())['skipped']==3
    assert app.view.start_button.instate(['!disabled'])


def test_modern_cooperative_cancel_IPC(app):
    program='import os,time; from pathlib import Path; p=Path(os.environ["BATCHRESIZE_CANCEL_FILE"]);\nwhile not p.exists(): time.sleep(.02)\nprint("Отмена подтверждена")'
    app._run_process([sys.executable,'-u','-c',program],'running');app._cancel();wait(app)
    assert 'Остановлено' in app.status_var.get() and app.view.start_button.instate(['!disabled'])
    assert not Path(app._cancel_dir.name).exists()


def test_modern_failed_worker_opens_log(app):
    app._run_process([sys.executable,'-u','-c','print("Ошибка тестового worker"); raise SystemExit(2)'],'running');wait(app)
    assert 'код 2' in app.status_var.get() and app.view.log_visible
    assert 'Ошибка тестового worker' in app.log.get('1.0','end')


def test_modern_finite_preflight_no_process(app,tmp_path,monkeypatch):
    app.input_var.set(str(tmp_path));app.output_var.set(str(tmp_path/'out'));app.vram_var.set('nan')
    seen=[];monkeypatch.setattr('batchresize.gui.messagebox.showerror',lambda *a:seen.append(a))
    assert app._build_arguments() is None and seen and app._proc is None


def test_generated_panels_are_local_and_import_safe():
    from batchresize import ui_generated
    import ast
    text=Path(ui_generated.__file__).read_text();tree=ast.parse(text)
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='Tk' for n in ast.walk(tree))
    assert '61ce0665ce68e255fd8bead3c14aac6c877bfa05' in text


def test_modern_GUI_import_without_site_packages():
    import subprocess
    root=Path(__file__).resolve().parents[1]
    code='import batchresize.gui,batchresize.ui_view,sys; assert not any(n in sys.modules for n in ("numpy","cupy","PIL","requests","tkdesigner"))'
    result=subprocess.run([sys.executable,'-S','-c',code],env=dict(os.environ,PYTHONPATH=str(root)),capture_output=True,text=True)
    assert result.returncode==0,result.stderr


def test_modern_close_waits_for_child_and_destroys_window():
    try:
        app=BatchResizeUI();app.update()
    except tk.TclError as exc:
        pytest.skip(str(exc))
    program='import os,time; from pathlib import Path; p=Path(os.environ["BATCHRESIZE_CANCEL_FILE"]);\nwhile not p.exists(): time.sleep(.02)'
    app._run_process([sys.executable,'-u','-c',program],'running')
    app._on_close()
    deadline=time.monotonic()+10
    while app._proc is not None and time.monotonic()<deadline:
        app.update();time.sleep(.02)
    assert app._proc is None and app._closing
    try:
        window_exists=app.winfo_exists()
    except tk.TclError:  # Some Tk builds remove the command on final destroy.
        window_exists=False
    assert not window_exists
