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
    if QUALITY_CHOICES[index][0]=='source':
        assert 'Как у исходника' in app.view.summary_quality.get()
        assert app.quality_scale.instate(['disabled'])
        return
    assert f'{app.jpeg_quality_var.get()} / 100' in app.view.summary_quality.get()
    assert app.quality_scale.instate(['!disabled'])


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


def test_progress_regex_matches_pipeline_line_format():
    from batchresize import pipeline as _pl  # noqa: F401  (documents the producer)
    from batchresize.gui import PROGRESS_RE
    exact=(f"{3}/{7} outputs | {12.5:.1f} output/s | {4.5:.1f}s | "
           f"threads: readers={1} processors={2} writers={1}")
    match=PROGRESS_RE.search(exact)
    assert match and match.groups()==('3','7','12.5','4.5','1','2','1')


def test_modern_progress_line_updates_progress_widgets(app):
    app._handle_line('3/7 outputs | 12.5 output/s | 4.5s | threads: readers=1 processors=2 writers=1');app.update()
    assert app.progress['maximum']==7 and app.progress['value']==3
    assert '3 из 7' in app.status_var.get()
    assert '3 / 7' in app.view.metrics.get()
    assert 'чтение=1' in app.threads_var.get()


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


def test_modern_size_limit_caps_output_size(app,tmp_path):
    """The GUI size limit must reach the worker and cap every produced file."""
    import numpy as np
    from PIL import Image
    rng=np.random.default_rng(3)
    src=tmp_path/'src';src.mkdir();out=tmp_path/'out'
    Image.fromarray(rng.integers(0,255,(300,400,3),dtype=np.uint8)).save(src/'noisy.png')
    app.input_var.set(str(src));app.output_var.set(str(out))
    app.backend_var.set('cpu');app.workers_var.set('1')
    app.max_file_size_var.set('60KB');app.min_quality_var.set('30')
    app.update()
    args=app._build_arguments()
    assert '--max-file-size' in args
    assert args[args.index('--max-file-size')+1]=='60KB'
    assert args[args.index('--min-quality')+1]=='30'
    app._start();wait(app)
    files=list(out.glob('*.jpg'));assert files
    for f in files:
        assert f.stat().st_size<=60_000,f'{f.name} is {f.stat().st_size} bytes'


def test_modern_size_limit_rejects_garbage(app,tmp_path):
    import batchresize.gui as gui
    seen=[];gui.messagebox.showerror=lambda *a,**k:seen.append(a[1])
    app.input_var.set(str(tmp_path));app.output_var.set(str(tmp_path/'out'))
    app.max_file_size_var.set('не размер')
    assert app._build_arguments() is None and seen


def test_modern_same_as_source_preset(app,tmp_path):
    """The 'same as source' preset must reach the worker and keep the tables."""
    from PIL import Image
    labels=[text for _key,text in QUALITY_CHOICES]
    app.quality_combo.set(labels[0]);app._on_quality_change();app.update()
    assert app.quality_var.get()=='source'
    assert app.quality_scale.instate(['disabled'])
    assert 'Как у исходника' in app.view.summary_quality.get()
    src=tmp_path/'src';src.mkdir();out=tmp_path/'out'
    Image.new('RGB',(200,150),(30,90,180)).save(src/'a.png')
    app.input_var.set(str(src));app.output_var.set(str(out))
    app.backend_var.set('cpu');app.workers_var.set('1')
    app.sizing_var.set('percent');app.sizing_value_var.set('80')
    app.update()
    args=app._build_arguments()
    assert args[args.index('--quality')+1]=='source'
    app._start();wait(app)
    assert (out/'a.jpg').exists()


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


def test_ux_sizing_mode_disables_conflicting_preset_and_dims(app):
    # Default is resolution: preset and dims active, value disabled
    assert app.sizing_var.get() == "resolution"
    assert app.preset_combo.instate(["!disabled"])
    assert app.view.width_entry.instate(["!disabled"])
    assert app.view.height_entry.instate(["!disabled"])
    assert app.view.sizing_value_entry.instate(["disabled"])

    # Switch to percent: preset and dims MUST be disabled, value entry MUST be enabled
    app.sizing_var.set("percent")
    app.update()
    assert app.preset_combo.instate(["disabled"])
    assert app.view.width_entry.instate(["disabled"])
    assert app.view.height_entry.instate(["disabled"])
    assert app.view.sizing_value_entry.instate(["!disabled"])
    assert "%" in app.view.sizing_mode_hint.cget("text")
    assert "%" in app.view.sizing_value_label.cget("text")

    # Switch back to resolution: preset and dims re-enabled, value entry disabled
    app.sizing_var.set("resolution")
    app.update()
    assert app.preset_combo.instate(["!disabled"])
    assert app.view.width_entry.instate(["!disabled"])
    assert app.view.height_entry.instate(["!disabled"])
    assert app.view.sizing_value_entry.instate(["disabled"])


def test_ux_fit_mode_controls_crop_anchor(app):
    # contain mode: crop anchor not needed -> disabled
    app.mode_var.set("contain")
    app.update()
    assert app.crop_anchor_combo.instate(["disabled"])

    # cover mode: crop anchor needed -> enabled
    app.mode_var.set("cover")
    app.update()
    assert app.crop_anchor_combo.instate(["!disabled"])

    # pad mode: crop anchor needed -> enabled
    app.mode_var.set("pad")
    app.update()
    assert app.crop_anchor_combo.instate(["!disabled"])


def test_ux_size_limit_controls_min_quality(app):
    # Empty limit -> min_quality and downscale check disabled
    app.max_file_size_var.set("")
    app.update()
    assert app.view.min_quality_entry.instate(["disabled"])
    assert app.view.allow_downscale_check.instate(["disabled"])

    # Set limit -> min_quality and downscale check enabled
    app.max_file_size_var.set("500KB")
    app.update()
    assert app.view.min_quality_entry.instate(["!disabled"])
    assert app.view.allow_downscale_check.instate(["!disabled"])


def test_ux_write_mode_mutual_exclusion(app):
    # Default is overwrite
    assert app.write_mode_var.get() == "overwrite"
    assert app.overwrite_var.get() is True
    assert app.skip_existing_var.get() is False

    # Switch to skip
    app.write_mode_var.set("skip")
    app._sync_write_mode()
    app.update()
    assert app.overwrite_var.get() is False
    assert app.skip_existing_var.get() is True

    # Switch back to overwrite
    app.write_mode_var.set("overwrite")
    app._sync_write_mode()
    app.update()
    assert app.overwrite_var.get() is True
    assert app.skip_existing_var.get() is False


def test_ux_wizard_navigation(app):
    assert app.view.current_page == 0
    app.view.show_page(1)
    app.update()
    assert app.view.current_page == 1
    app.view.show_page(2)
    app.update()
    assert app.view.current_page == 2
    app.view.show_page(3)
    app.update()
    assert app.view.current_page == 3


def test_ux_extended_arguments_propagation(app, tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    out = tmp_path / "out"
    app.input_var.set(str(src))
    app.output_var.set(str(out))
    app.mode_var.set("cover")
    app.crop_anchor_var.set("top")
    app.sharpen_preset_var.set("standard")
    app.allow_downscale_var.set(True)
    app.strip_personal_var.set(True)
    app.web_srgb_var.set(True)
    app.linear_light_var.set(True)
    app.update()

    args = app._build_arguments()
    assert "--crop-anchor" in args and args[args.index("--crop-anchor") + 1] == "top"
    assert "--sharpen" in args and args[args.index("--sharpen") + 1] == "standard"
    assert "--allow-downscale-on-size-limit" in args
    assert "--strip-personal-data" in args
    assert "--web-srgb" in args
    assert "--linear-light" in args


def test_ux_concurrency_arguments_propagation(app, tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    out = tmp_path / "out"
    app.input_var.set(str(src))
    app.output_var.set(str(out))
    app.backend_var.set("cpu")
    app.workers_var.set("8")
    app.processors_var.set("16")
    app.readers_var.set("6")
    app.writers_var.set("3")
    app.update()

    args = app._build_arguments()
    assert "--processors" in args and args[args.index("--processors") + 1] == "16"
    assert "--readers" in args and args[args.index("--readers") + 1] == "6"
    assert "--writers" in args and args[args.index("--writers") + 1] == "3"


def test_log_copy_save_clear(app, tmp_path):
    from tkinter import filedialog, messagebox
    app._clear_log()
    app._log_line("Log line 1: started")
    app._log_line("Log line 2: done")
    assert "Log line 1" in app.log.get("1.0", "end-1c")

    # Test copy
    app._copy_log()
    assert "Log line 1" in app.clipboard_get()

    # Test save
    save_file = tmp_path / "saved_log.txt"
    orig_ask = filedialog.asksaveasfilename
    orig_info = messagebox.showinfo
    try:
        filedialog.asksaveasfilename = lambda **kw: str(save_file)
        messagebox.showinfo = lambda *a, **kw: None
        app._save_log()
        assert save_file.exists()
        assert "Log line 2" in save_file.read_text(encoding="utf-8")
    finally:
        filedialog.asksaveasfilename = orig_ask
        messagebox.showinfo = orig_info

    # Test clear
    app._clear_log()
    assert app.log.get("1.0", "end-1c") == ""


def test_modern_size_limit_rejects_bare_byte_integers(app, tmp_path):
    import batchresize.gui as gui
    seen = []
    orig_show = gui.messagebox.showerror
    try:
        gui.messagebox.showerror = lambda *a, **k: seen.append(a[1])
        src = tmp_path / "src"; src.mkdir()
        out = tmp_path / "out"
        app.input_var.set(str(src))
        app.output_var.set(str(out))
        app.max_file_size_var.set("800")
        app.update()
        assert app._build_arguments() is None
        assert seen and "800KB" in seen[0]
    finally:
        gui.messagebox.showerror = orig_show
