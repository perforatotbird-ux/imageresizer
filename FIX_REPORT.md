# Исправления аудита imageresizer — 0.2.0

## Итог и предел утверждений

Исправления подготовлены в локальной ветке `audit-fixes` поверх
`f4f0782de06aaec63d44486e711cbc8dde349e80` репозитория
https://github.com/perforatotbird-ux/imageresizer. На этапе составления этого отчёта изменения были только локальными,
release не публиковался, production-копия пользователя не изменялась.
Актуальная история публикации устанавливается по Git; новый визуальный слой и
его отдельная приёмка описаны в UI_DESIGN_REPORT.md.
Исходный аудит сохранён отдельно в `AUDIT_REPORT.md`; его находки относятся к базе,
а не описывают состояние исправленного дерева.

Реестр охватывает все 26 ID: **23 code/contract исправления и 3 явно ограниченных
меры снижения риска (IR-17, IR-18, IR-26)**. Это не заявление, что все требования
полностью закрыты: строгая entropy-целостность JPEG, hard VRAM/native-memory cap
и GPU-only codec attestation остаются отдельными нерешёнными гарантиями.
Windows/CUDA acceptance также не выполнен. Замена этих гарантий честными сообщениями
и fail-closed не равнозначна их реализации.

## Проверки исправленной версии

- Полная suite: **135 passed, 24 skipped**. Исходные 79 тестовых cases сохранены
  побайтно; добавлено 80 cases в `tests/test_audit_regressions.py`.
  24 skipped — реальные GPU-тесты: CuPy/nvImageCodec/NVIDIA в среде отсутствуют.
  Tests по GUI/BAT/GPU contracts частично статические/AST/mock — не аппаратная приёмка.
- Сквозной acceptance: 97 синтетических исходников (JPEG/PNG/WebP/TIFF/BMP,
  RGBA, EXIF orientations и отдельный JPEG 8000×6000 = 48 МП).
- Stress: workers=8, queue-depth=1, 97 processed, 0 failed. Resume: 97 skipped,
  0 processed после проверки manifest/hash/full JPEG decode.
- Benchmark: 2 CPU repeats по одному и тому же manifest; оба valid.
  Ни одного GPU результата/speedup не заявлено.
- **291 выходной JPEG** полностью открыт повторно, проверены RGB, JPEG,
  размеры <=320×240, Orientation=1. Это проверка формата/геометрии, не независимая
  эталонная оценка цвета каждого из этих изображений; цвет/alpha/CMS отдельно проверяют
  regression tests. Synthetic source SHA256 до/после совпадают.
- Peak child RSS в acceptance: 626752 KiB
  (612.1 MiB). Это peak отдельного дочернего процесса
  Linux, НЕ суммарная память машины и НЕ доказательство hard memory cap.
- `git diff --check`, AST parse всех Python sources/tests; исходные тесты не удалены
  и не редактировались. Окончательный patch дополнительно применяется к чистой базе
  и проверяется там; его отдельный test log приложен в evidence.

### Окружение

Linux, Python 3.13.14; NumPy 2.5.3, Pillow 12.3.0,
OpenCV headless 5.0.0.93, piexif 1.1.3, pytest 9.1.1.
Изолированная `.venv` с доступом к системным библиотекам. Глобальная установка,
admin-права, изменение системных настроек и запуск пользовательских фотографий
не использовались. Зависимости NVIDIA не устанавливались и не тестировались.
Constraint-файл описывает только эту Linux/Python 3.13 CPU-среду.

## Реестр исправлений

Номера строк относятся к исправленному дереву, не к исходным строкам AUDIT_REPORT.
Для исходного механизма и severity смотрите соответствующую карточку аудита.

| ID | Статус | Файл / функция | Изменение и доказательство / ограничение |
|---|---|---|---|
| IR-01 | Исправление | `batchresize/safety.py:44` (`protect_target`) | Preflight путей, предок/equal input, symlink/junction/reparse, inode aliases; повторная проверка перед публикацией. Регрессии небезопасных roots/links/hardlinks. Не защита от враждебной TOCTOU-подмены. |
| IR-02 | Исправление | `batchresize/pipeline.py:624` (`_process_gpu_batch_inner`) | RGBA смешивается с фоном до C3 resize; устройства принимают только packed uint8 C3/C4 согласно операции. Mock подтверждает порядок, CUDA kernels здесь не исполнялись. |
| IR-03 | Исправление | `batchresize/cpu_backend.py:129` (`decode_to_rgb`) | LA/P/RGBA нормализуются в contiguous RGB/RGBA, C2 в C3 kernels не передаётся; проверка ndarray и alpha. |
| IR-04 | Исправление | `batchresize/pipeline.py:360` (`_guarded`); `batchresize/pipeline.py:772` (`_join_bounded`) | Supervision BaseException, fatal+stop, timeout очередей, writer-done event, bounded joins, accounting каждой задачи. Живой native worker удерживает lock, manifest не сохраняется. Нельзя безопасно принудительно остановить native thread; это остаточный риск. |
| IR-05 | Исправление | `batchresize/pipeline.py:605` (`_process_gpu_batch`) | Идемпотентные terminal/pending, partial decode обрабатывается per-file; split retry не повторяет уже принятые jobs; выход из except до рекурсивного retry освобождает traceback. |
| IR-06 | Исправление | `batchresize/cli.py:215` (`_select_engine`); `batchresize/cpu_backend.py:84` (`resize`) | filter/Huffman передаются GPU конструктору; CPU реально использует выбранный фильтр Pillow. Тест параметров — mock, реальный GPU Huffman требует приёмки. |
| IR-07 | Исправление | `batchresize/gui.py:395` (`_on_quality_change`) | Все quality/compression aliases берутся из ALL_QUALITY_PRESETS. Метод GUI проверен без tkinter через AST; настоящий GUI здесь не запускался. |
| IR-08 | Исправление | `batchresize/cli.py` | Lazy imports: python -S импортирует CLI без numpy/CUDA. Bootstrap GUI не требует научного стека от interpreter-кандидата. |
| IR-09 | Исправление | `batchresize/safety.py:81` (`ResumeState`) | Verified resume по hashes source/config/output и полной RGB JPEG decode/size; lock; историческое соответствие имён сохраняется даже для удалённых источников. Чужой output остаётся на месте, failed вместо ложного skipped. |
| IR-10 | Исправление | `batchresize/validation.py`; `batchresize/pipeline.py:33` (`RunOptions`) | Finite/positive/целочисленные контракты, quality, filter/background, worker/batch/queue caps, target/source pixel limit; invalid параметры не доходят до потоков. |
| IR-11 | Исправление | `batchresize/cpu_backend.py:84` (`resize`) | Pillow fractional-box crop вместо округления координат к пустому ROI. Воспроизведение 10×100 → cover 100×1 без upscale сохраняет корректный 10×1 JPEG. |
| IR-12 | Исправление | `batchresize/gpu_backend.py:145` (`build_weights`) | Box — интеграл overlap; расстояния считаются по виртуальным индексам до clamp. Независимая арифметика weights проверяется CPU-тестами; device исполнение не проверено. |
| IR-13 | Исправление | `batchresize/imageio_meta.py:36` (`read_source_meta`); `batchresize/pipeline.py:579` (`_process_cpu_one`) | Alpha определяется содержимым: bands/transparency, а не suffix. CPU alpha декодируется Pillow; GPU RGBA→RGB retry запрещён, при потере alpha применяется полноценный CPU fallback. tRNS проверен по реальным белым выходным пикселям. |
| IR-14 | Исправление | `batchresize/cpu_backend.py:129` (`decode_to_rgb`); `batchresize/pipeline.py:555` (`_finish`) | ICC non-RGB конвертируется ImageCms в sRGB; JPEG получает sRGB ICC вместо CMYK. Регрессия использует реальный Ghostscript CMYK ICC; при отсутствии профиля на иной машине тест явно skip. |
| IR-15 | Исправление | `batchresize/imageio_meta.py:95` (`_strip_orientation`); `batchresize/imageio_meta.py:36` (`read_source_meta`) | EXIF read/normalize/splice errors не маскируются; оригинальный Orientation на уже повернутые пиксели не приклеивается. Fault injection подтверждает failed и сохранность старого output. |
| IR-16 | Исправление | `batchresize/imageio_meta.py:95` (`_strip_orientation`) | Orientation=1; обновлены width/height и Exif PixelX/YDimension; thumbnail удалён. Проверка EXIF после записи. |
| IR-17 | Снижение риска | `batchresize/imageio_meta.py:58` (`is_truncated_jpeg`) | Вместо last-4096/suffix heuristic — magic + marker parser с APP lengths, stuffing/restarts, EOI и trailers. Ложные отказы исправлены. НЕ entropy-integrity verifier: decoder может восстановить повреждённый entropy stream, и это не объявляется успешно проверенной целостностью. |
| IR-18 | Снижение риска | `batchresize/pipeline.py:370` (`_working`); `batchresize/pipeline.py:523` (`_gpu_estimate`) | Byte budget compressed jobs, bounded read по stat size+1, host working estimate, scratch+cache в GPU estimate, проверка oversized первого job, byte-capped table cache, release traceback. Hard RSS/VRAM cap не реализован: opaque codec/runtime/pools и независимые процессы остаются вне контроля. |
| IR-19 | Исправление | `batchresize/cli.py:291` (`_benchmark`) | Один input manifest, exclude всей output-root, уникальная группа результатов, no timed skips, hashes до/после каждого run, nonzero на failures/changes, все repeats+median, speedup только для valid обоих backend. Регрессии вложенного output, input mutation и повторного benchmark. |
| IR-20 | Исправление | `batchresize/pipeline.py:884` (`_report`) | Отдельно scan/normalize/metadata, read_count denominator, timing semantics. --profile-cuda синхронизирует decode/normalize/resize/encode; CPU fallback не синхронизирует сломанный CUDA. Это service wall time, не kernel event profiler; фактическая CUDA timeline не измерена. |
| IR-21 | Исправление | `run.bat` | Код запуска вынесен из IF-блока, exit /b получает фактический errorlevel после Python. Статическая проверка; cmd.exe здесь отсутствует. |
| IR-22 | Исправление | `run.bat`; `install.bat`; `batchresize/launcher.py` | DisableDelayedExpansion, пути заключены в кавычки, интерактивный ввод обрабатывается Python без shell=True. Unicode/space/! CLI проверен на Linux, Windows cmd/special-profile paths требуют приёмки. |
| IR-23 | Исправление | `batchresize/gui.py:550` (`_run_process`) | Worker получает PYTHONIOENCODING/PYTHONUTF8, reader UTF-8; bounded GUI queue/logs/partial-line buffer; cooperative cancellation и корректная очистка IPC. Windows child/GUI smoke не выполнен. |
| IR-24 | Исправление | `batchresize/safety.py:53` (`atomic_write`) | Убран глобальный cache созданных директорий; mkdir на каждой записи. Тест удаления/перемещения output-dir между публикациями проходит. |
| IR-25 | Исправление | `batchresize/pipeline.py:185` (`discover`) | Ошибка scandir/entry не проглатывается: discovery_errors + complete=false, либо исключение без callback; symlinks/reparse пропускаются явно. PermissionError смоделирован тестом. |
| IR-26 | Снижение риска | `batchresize/gpu_backend.py:182` (`GpuEngine`); `batchresize/pipeline.py:884` (`_report`) | Честная automatic/unverified policy, отдельные маршруты приложения и fallback. Раньше игнорируемый strict теперь default False, явный strict=True fail-closed. GPU-only plugin attestation не реализована: необходимы поддерживаемая backend policy nvImageCodec и аппаратная проверка. --no-fallback не обещает отключать codec-internal CPU. |

## Дополнительное укрепление

- Manifest защищён эксклюзивным run lock. Подмена symlink/junction leaf, path traversal
  manifest и публикация поверх hardlink исходника отвергаются. Duplicate manifest
  mappings не принимаются. Concurrent local runs не делят один output-root.
- Хэши config учитывают input-root, параметры, версию приложения и библиотек;
  manifest другой конфигурации не даёт ложный skip.
- Сбой metadata не оставляет новый output вместо старого: подготовка завершается
  до atomic publication. JPEG и manifest не образуют общую FS-транзакцию, поэтому
  сохранённый JPEG при неуспешном save manifest может требовать явного rebuild.
- Native thread, не завершившийся после bounded joins, не превращается в complete;
  lock оставляется для ручного recovery, manifest не сохраняется одновременно с ним.
- GUI ждёт кооперативную отмену вместо немедленного terminate, ограничивает
  очередь/логи, обрабатывает ошибку Popen и очищает cancel-dir/pipe.
- `run.bat` сохраняет menu workflow через stdlib-only `batchresize.launcher`;
  Python subprocess получает argv-list, а не shell-команду из пользовательского ввода.
- `install.bat --cpu`, отдельные runtime/test requirements, ограниченные диапазоны
  CUDA12 зависимостей; штатная установка в venv. Это не supply-chain сертификация:
  wheel-подписи/hashes/Windows-GPU lock отсутствуют.

## Намеренные изменения поведения и риски миграции

1. **CPU resize теперь Pillow**, не прежний OpenCV AREA/LINEAR. Это необходимо для
   реального выбора filter и fractional cover. Возможны отличия резкости, пикселей
   и скорости. Старые benchmark цифры не переносятся на эту реализацию.
2. Существующий JPEG без подходящего manifest больше не считается основанием для
   skip. Он сохраняется с failed; для пересоздания нужен явный `--overwrite`.
   По умолчанию CLI по-прежнему перезаписывает результаты, если нет --skip-existing.
3. Ошибочный EXIF/CMS теперь может приводить к failed вместо silent metadata drop.
   Это преднамеренный отказ от ложного успеха/неправильной ориентации/цвета.
4. GPU fallback теперь полностью CPU, поэтому успешно сохранённый файл не означает
   GPU execution. --no-fallback не управляет внутренними плагинами nvImageCodec.
5. Conservative source/working limits могут отклонять очень большие RGBA. Пример:
   48 МП RGB проходит сквозной тест, что НЕ обещает обработку 48 МП RGBA при том же
   estimate/бюджете. Лимиты RunOptions нельзя воспринимать как точный RSS predictor.
6. `GpuEngine(strict=True)` ранее не влиял на код; теперь явно unsupported.
   Default strict=False честно сохраняет automatic codec routing; существующий
   внешний код с явным strict=True получит диагностическую ошибку.
7. Потенциально чужие/повреждённые outputs не удаляются автоматически, устаревший
   lock не снимается автоматически; это требует осознанного recovery пользователем.

## Воспроизведение

Из чистой копии исходного commit, не поверх неизвестных незакоммиченных правок:

```sh
git checkout -b audit-fixes f4f0782de06aaec63d44486e711cbc8dde349e80
git apply --check /path/to/imageresizer-fixes.patch
git apply /path/to/imageresizer-fixes.patch
python -m venv .venv
# Windows: .venv\Scripts\python.exe вместо .venv/bin/python
.venv/bin/python -m pip install -r requirements-test.txt
.venv/bin/python -m pytest tests -q -ra
.venv/bin/python -m batchresize --help
.venv/bin/python -m batchresize --doctor
```

Для точной Linux/Python 3.13 CPU среды добавить pip `-c constraints-tested-linux-py313.txt`.
Скрипт `evidence/scripts/acceptance.py` содержит исходные абсолютные sandbox paths;
перед повтором на другой машине замените ROOT/BASE/PYTHON или запустите эквивалентные
argv из `evidence/acceptance/acceptance.json`. Это диагностический harness, не installer.
Synthetic data создаются только в BASE; не указывайте папку пользовательских фото.
Тесты используют отдельные tmp directories; production-код не патчится во время тестов.

Evidence включает raw test logs, acceptance JSON и stdout всех пяти команд, версии,
контрольные hashes и scripts. Сами ~100 synthetic файлов не включены для компактности:
они детерминированно генерируются seed=1719. Native версии декодеров/энкодеров могут
менять полученные байты; SHA manifest фиксирует использованный здесь экземпляр.
Старые audit benchmarks после reset не переобъявлены raw-логами новых исправлений.

## Обязательная приёмка перед Windows/GPU release

- Windows cmd.exe: install --cpu и GPU install, exit codes, Unicode/space/! path,
  профиль пользователя с нестандартным именем, запуск из чужого CWD, junctions,
  защита оригиналов, GUI presets/UTF-8/cancel/close; без admin.
- Матрица реальных GPU/driver/CUDA/runtime/wheels: Pascal отдельно; не предполагать,
  что совпадающее семейство CUDA12 гарантирует поддержку SM 6.1.
- Запустить все 24 skipped GPU tests; отдельно RGBA/LA/tRNS, ICC CMYK,
  progressive/grayscale/CMYK JPEG, все 8 Orientation и pixel comparisons с эталоном.
- Long soak, repeated batches, pool/cache plateau, device OOM, codec partial failure,
  stream producer/consumer hazards, исчезновение GPU, recovery без дубликатов/потери jobs.
- Nsight/CUDA events + codec/plugin policy для доказательства реального decode/encode
  пути. `backend_counts` и library name сами этого не доказывают.
- Crash/kill между write/rename/manifest на локальной NTFS, права, занятое имя,
  full disk, FS без hardlink, network shares и power-loss semantics.
- Измерить throughput по замороженным input hashes; раздельно cold/warm cache,
  ordinary и synchronized profiling. Не сравнивать старый OpenCV-resize baseline
  с новым Pillow filter, выдавая это за hardware speedup.

Ни один из этих непроведённых аппаратных сценариев не помечен passed.
