> Исторический аудит исходного commit f4f0782de06aaec63d44486e711cbc8dde349e80.
> Его находки не описывают исправленное дерево: см. FIX_REPORT.md и UI_DESIGN_REPORT.md.

# AUDIT_REPORT — imageresizer

**Дата:** 2026-10-19. **Аудируемый commit:** `f4f0782de06aaec63d44486e711cbc8dde349e80`. Источник: https://github.com/perforatotbird-ux/imageresizer.

## 1. Executive Summary

CPU-путь работоспособен на проверенных обычных форматах, но продукт нельзя считать подтверждённо безопасным для массовой обработки ценных оригиналов. Выявлено **26 проблем: P0 — 0, P1 — 4, P2 — 19, P3 — 3**. Критичные направления: безопасность source/target путей, RGB/RGBA/LA контракты GPU и контроль отказов потоков. При отдельных условиях оригинал может стать target; наличие этой ошибки не означает, что аудит повреждал пользовательские данные — такие записи не выполнялись.

Исходный полный pytest: **55 passed, 24 skipped, 0 failed (79 collected)**. Первоначально 6.90s, повтор после восстановления среды **5.41s**. Пропущены GPU-dependent проверки, не успешно пройдены. Первоначальный независимый набор — **54 passed за0.58s**. Его raw файлы утрачены при reset среды. В восстановленной среде заново выполнен и приложен компактный набор **39 passed за0.55s** плюс subprocess fault/ICC probes. `defect_*`/IR tests проходят, когда ошибка **воспроизведена**, а не исправлена.

Было выполнено 12 CPU benchmark-прогонов на одинаковом синтетическом наборе 528 файлов, 75947333 bytes (72.429MiB), и отдельный инструментированный профиль. GPU benchmark не запускался: нет NVIDIA device/driver/CuPy/nvImageCodec; Windows/Tk GUI тоже недоступны. GPU/Windows выводы помечены как static/mock, не device execution. Заявленные GTX1060 57file/s и speedup2× не подтверждены и не опровергнуты другим CPU corpus.

**Рекомендация до исправлений:** только копии и отдельный output без пересечения/aliases с input; CPU, явная проверка JPEG/цветов/количества; не доверять skip-existing как верификации. Исправления — отдельный этап, production-код не менялся.

### Сохранность аудиторских артефактов

После вопроса пользователя о выдаче отчёта `/data` оказался пустым: ранее созданный draft, venv, исходные logs/corpus исчезли. Это наблюдаемая потеря файлов среды, а не ошибка аудируемой программы; причина reset не установлена. Checkout восстановлен **на том же commit**, исходные tests и ключевые probes повторены. Предыдущие выводы и числовые outputs сохранились в истории инструментов; в таблицах ниже они обозначены **наблюдавшимися до reset**. Сырые исходные 12 benchmark JSON и 54-test log в приложении отсутствуют; восстановленная запись чисел не выдаётся за оригинальный лог. Это ограничение chain-of-custody явно снижает полноту воспроизводимости данного пакета.

## 2. Git, охват, окружение и правила

Первоначально локального repo не было. Выполнен clone указанного URL; branch main, HEAD `f4f0782de06aaec63d44486e711cbc8dde349e80`, status/diff пусты, modified/untracked файлов не было. Первоначальный commit timestamp: 2026-10-19 12:11:06+03. После reset повторный clone и checkout hash дали detached HEAD того же содержимого; актуальный `logs/git-restored.txt`. После аудита допустимое отличие — только этот untracked AUDIT_REPORT.md.

Изучены **все24 tracked файла, 4864 строки**, включая 18 Python файлов:
- package: __init__, __main__, cli, cpu_backend, env, geometry, gpu_backend, gui, imageio_meta, pipeline, presets;
- tests: conftest, test_cli, test_geometry, test_metadata, test_pipeline, test_robustness;
- tools/make_testset.py, requirements.txt, install.bat, run.bat, README.md, REPORT.md, .gitignore.
Нет tracked CI/lock/pyproject и raw авторских benchmark JSON. Все существенные компоненты изучены по содержимому, а не только поиском ключевых слов. Python исходники первоначально независимо AST-parsed/compiled in memory.

Linux x86_64, Python3.13.14, 2logical CPUs, исходный cgroup RAM limit4GiB. NumPy2.5.3, Pillow12.3.0, opencv-python-headless5.0.0.93 (module5.0.0, libjpeg-turbo3.1.2, pthreads/default threads2), pytest9.1.1, piexif1.1.3. Версии отличаются от авторского Windows REPORT. Audit venv использует system-site-packages для уже установленных image libraries; добавлены только pytest/piexif в venv. Это **не** clean-install production requirements.

Никаких global installs, admin elevation, CUDA/driver установки, системных настроек/cache clearing, запуска BAT вслепую, пользовательских фото или destructive tests. Все scripts/logs/corpus вне source checkout. Перед исполнением изучены scripts; pip устанавливал только тестовые зависимости в отдельный venv. Поиск публичной NVIDIA документации передавал только технические запросы, не фото/секреты.

## 3. Архитектура и данные

```text
CLI / Tk GUI → отдельный CLI worker → opts → engine smoke-test
  scan all paths + sort + mapping (RAM)
  N readers: file.read_bytes → EOI heuristic → Pillow metadata → Plan
  bounded Job queue (compressed bytes + EXIF/ICC + geometry)
  CPU: N processors / GPU,hybrid: один processor
    CPU: OpenCV BGR(A) decode → alpha CPU → orientation CPU
      → rounded crop + AREA/LINEAR → JPEG encode CPU
    GPU: nvImageCodec batch decode → cp.asarray
      → [пропущена C4 composition!] → CUDA prefilter+separable resize
      → nvImageCodec JPEG bytes
    Hybrid/fallback: Pillow decode → full RGB(A) upload → CuPy
      → orientation/alpha только для C4 → resize → GPU-dependent encode
  metadata splice on CPU (errors подавляются)
  bounded Result queue (JPEG + Job, source bytes продолжают жить)
  max(2,N//2) writers: temp → flush [optional fsync] → os.replace
  stats → non-atomic JSON report → exit
```

Модули логично разделены; блокирующих circular imports в проверенном CPU запуске не найдено. Но CLI импортирует вычислительные зависимости даже для GUI bootstrap. Нормализация дублируется в pipeline CPU/GPU/hybrid/fallback и GpuEngine.process_array; основной GPU путь обходит последний. CPU integer crop и GPU fractional sampling имеют разные contracts.

| Этап | Устройство/память | Копии, ожидание, lifetime |
|---|---|---|
| Scan/meta | CPU, полный Path list/map, compressed host bytes | O(N files) names; read_bytes целиком; Pillow header/meta не заменяет full decode |
| CPU decode/resize/encode | CPU ndarray, временные alpha float arrays | N processors + OpenCV internal threads конкурируют; full image decode, не reduced decode |
| nvImageCodec decode | Plugin-dependent CPU/HYBRID/GPU/HW; host bitstream | Default plugin policy не ограничена; фактический device trace отсутствует |
| cp.asarray(Image) | Обычно zero-copy CUDA array interface при compatible device image | Поддержка zero-copy реальна, но не доказывает GPU-only previous decode; buffer_kind/strides/dtype не аттестуются |
| Resize | VRAM source/output uint8, float32 scratch | CPU NumPy weights→device tables; kernels asynchronous current stream; tables cache count256, не bytes |
| Encode/result | Encoder plugin, compressed bytes на host | bytes(cs) материализует result и может оплатить ожидание kernels; реальное CPU/GPU split не измерено |
| Fallback | Pillow host decode + полный H2D RGB | Не полный CpuEngine fallback при runtime OOM/device lost; GPU всё ещё обязателен |
| Write | CPU/host JPEG | Result удерживает compressed input; unique pid/tid tmp; os.replace publication |

Один Device(0) выбирается при engine init в main thread, processor один; отдельный выбор current device в worker отсутствует. Для штатного device0 это не доказанная ошибка, multi-GPU/чужой thread-local context — открытая проверка. decoder/encoder в основном живут один engine lifetime; Decoder пересоздаётся при free<30%. Pool retention не равнозначен leak, комментарий о nvImageCodec leak независимо не подтверждён. Собственных двойных buffers/streams/events для перекрытия batches нет; decode_batch делает null.synchronize, resize/encode по файлам идут последовательно.

## 4. Краткий реестр

| ID | Приоритет | Проблема | Последствие | Доказательство | Рекомендация |
|---|---|---|---|---|---|
| IR-01 | P1 | Выходной путь может совпасть с оригиналом | Возможна необратимая замена оригинальной фотографии уменьшенным JPEG и гонка с чтением. | Подробная карточка ниже | Проверить весь output map против source set; запретить опасные ancestor/alias/junction escape. |
| IR-02 | P1 | RGBA поступает в CUDA kernels с жёстким RGB шагом | Неверные цвета/фон могут сохраниться с success. | Подробная карточка ниже | Общий normalization path RGB/RGBA перед resize, assertions dtype/channels/strides. |
| IR-03 | P1 | LA остаётся C2 и нарушает C3 контракт GPU | Неверное изображение; возможна CUDA illegal memory access и сбой последующих задач. | Подробная карточка ниже | Нормализовать все modes к RGB/RGBA до GPU; добавить shape assertions. |
| IR-04 | P1 | Отказ этапа даёт потерю задач без failed или deadlock | Ложный успех с пропуском файлов либо зависание. | Подробная карточка ниже | Supervisor, fatal channel, cancellable queue operations, accounting invariant и bounded shutdown. |
| IR-05 | P2 | Частичный decode batch повторяет уже успешные jobs | Лишняя запись/обработка, processed>discovered, неверные bytes/file/s. | Подробная карточка ниже | Retry только незавершённых индексов, явный terminal state job. |
| IR-06 | P2 | Filter и optimized Huffman не управляют движком | Не тот визуальный результат/размер JPEG; сравнение GPU/CPU с разными kernels. | Подробная карточка ниже | Одна engine factory из validated opts, реализация/явное ограничение CPU filters. |
| IR-07 | P2 | Четыре GUI-пресета дают KeyError | Пользователь получает старое качество и callback error. | Подробная карточка ниже | Использовать ALL_QUALITY_PRESETS/общий resolver, тестировать все GUI choices. |
| IR-08 | P2 | GUI bootstrap требует numpy у якобы лёгкого Python | Заявленный fallback GUI не запускается на обычном чистом Python. | Подробная карточка ниже | Ленивые imports/отдельный GUI entrypoint. |
| IR-09 | P2 | Skip-existing не проверяет целостность/происхождение | Пропущенные или неправильно сопоставленные результаты при resume. | Подробная карточка ниже | Manifest с source identity/hash/config, JPEG verify и stable collision naming. |
| IR-10 | P2 | Не валидируются finite/positive sizing, budget и limit | Неожиданное массовое уменьшение/обработка большего числа файлов/OOM. | Подробная карточка ниже | Общий validator CLI/GUI/RunOptions/geometry; ограничения ресурсов. |
| IR-11 | P2 | Cover rounding создаёт пустой crop | Допустимый файл не обработан. | Подробная карточка ниже | Общий pixel-center convention, fractional resampling или корректные floor/ceil bounds. |
| IR-12 | P2 | Box weights и border weights математически неверны | Реальный blur/edge color bias до JPEG, не lossy rounding. | Подробная карточка ниже | Исправить area/support и порядок clamp; независимый CPU oracle. |
| IR-13 | P2 | Alpha hint и RGB retry могут отбросить прозрачность | Неверный фон, даже после исправления C4→C3 в IR02. | Подробная карточка ниже | Проверять действительные header/layout, сохранять alpha до композиции. |
| IR-14 | P2 | RGB JPEG получает CMYK ICC | Неправильные цвета/невалидное color-management поведение. | Подробная карточка ниже | Явный output space/CMS transform; не копировать несовместимый ICC. |
| IR-15 | P2 | Metadata ошибки проглатываются; Orientation fallback не безопасен | Silent EXIF/ICC loss, в fallback возможен двойной поворот в viewer. | Подробная карточка ниже | Не подавлять; безопасный EXIF fallback, отдельная metadata diagnostics. |
| IR-16 | P3 | Размеры EXIF остаются от исходника | Неверные размеры в downstream каталогах, не порча bitstream. | Подробная карточка ниже | Передавать final dimensions в normalization. |
| IR-17 | P2 | EOI heuristic не проверяет JPEG целостность и зависит от suffix | False rejection или принятие испорченного изображения за правильное. | Подробная карточка ниже | Sniff format, strict/tolerant validation и surfaced decoder warnings. |
| IR-18 | P2 | VRAM-budget — не верхняя граница памяти | OOM несмотря на budget; per-file fallback всё ещё GPU и не лечит device lost. | Подробная карточка ниже | Estimate source+scratch+output+cache, cap single job; отпускать failed frame до retry. |
| IR-19 | P2 | Benchmark повторно сканирует свои результаты и возвращает0 при failures | Неверные speedups/рост работы и ложный automation success. | Подробная карточка ниже | Один source snapshot, whole output exclusion, validate outputs/counts. |
| IR-20 | P3 | Stage timings неполны и не CUDA execution profile | Неправильная оптимизация и проценты bottleneck. | Подробная карточка ниже | Profiling-mode events/NVTX, раздельный startup/cache, корректные denominators. |
| IR-21 | P2 | BAT возвращает старый errorlevel | Failed обработка может считаться успешной. | Подробная карточка ниже | !errorlevel! в этом block либо exit вне block. |
| IR-22 | P2 | Delayed expansion ломает ! в путях | Ошибка запуска/неожиданный cwd установки. | Подробная карточка ниже | Disable expansion для user paths; scoped menu expansion; check cd. |
| IR-23 | P2 | Worker encoding не согласован с UTF-8 reader | Нечитаемые ошибки или отказ допустимых имён. | Подробная карточка ниже | PYTHONIOENCODING/UTF8 только в child env, совместимый reader. |
| IR-24 | P3 | Глобальный cache директорий устаревает | Ошибка повторного embedded/API run; отдельные CLI cache не разделяют. | Подробная карточка ниже | Убрать вечный cache или run-local/retry mkdir. |
| IR-25 | P2 | Недоступный subtree молча исчезает из отчёта | Пропуск коллекции с видимостью успеха. | Подробная карточка ниже | Возвращать traversal diagnostics, согласовать continue/fail-fast. |
| IR-26 | P2 | GPU label не аттестует реальный codec backend | Нельзя доказать GPU-only/no copies/объяснить performance из JSON. | Подробная карточка ниже | Логировать backend/plugin/buffer_kind, определить strict policy и профилировать. |

### Подробные карточки (причина, воспроизведение, влияние, исправление, риск)

#### IR-01: Выходной путь может совпасть с оригиналом
**Приоритет:** P1. **Категория:** security / reliability.
- **Где:** [batchresize/pipeline.py:161–202](https://github.com/perforatotbird-ux/imageresizer/blob/f4f0782de06aaec63d44486e711cbc8dde349e80/batchresize/pipeline.py#L161); функции `build_output_map, atomic_write; cli.main:398–402`.
- **Причина:** Проверяется равенство корней, но не идентичность каждого target с любым source. Резервируются только output names. os.replace безусловно заменяет target.
- **Условия:** input=/X/photos, output=/X; sources /X/photos/a.jpg и /X/photos/photos/a.jpg. Также output/nested может ссылаться на input/nested.
- **Повторить:** `pytest experiments/reproduce_probes.py -k IR01`; если указана команда/статический разбор вместо IR — использовать её. Только scratch fixtures, не личные фото.
- **Факт:** Для вложенного source target=/X/photos/a.jpg; в symlink test target.resolve()==src.resolve(). Запись не выполнялась.
- **Ожидание:** Отказ до первой записи при любом source/target alias.
- **Доказательство/уверенность:** Эксперимент с путями + статический механизм записи, без перезаписи фотографий. Для повторённых probes полный output в `logs/probes-restored.log`; для subprocess/CMS — `logs/faults-restored.json`, `cmyk-restored.json`. Static выводы этими логами не выдаются за аппаратную проверку.
- **Последствия:** Возможна необратимая замена оригинальной фотографии уменьшенным JPEG и гонка с чтением.
- **План исправления:** Проверить весь output map против source set; запретить опасные ancestor/alias/junction escape.
- **Риск/побочные эффекты:** Не запретить полезный output-inside-input целиком; resolve не устраняет TOCTOU.

#### IR-02: RGBA поступает в CUDA kernels с жёстким RGB шагом
**Приоритет:** P1. **Категория:** GPU/CUDA / correctness.
- **Где:** [batchresize/pipeline.py:440–459](https://github.com/perforatotbird-ux/imageresizer/blob/f4f0782de06aaec63d44486e711cbc8dde349e80/batchresize/pipeline.py#L440); функции `_process_gpu_batch_inner; gpu_backend.rs_h/box_reduce:25–93`.
- **Причина:** Основной pipeline вызывает resize напрямую, обходя process_array и alpha_to_rgb. CUDA row/пиксель адресуются как 3 bytes/pixel.
- **Условия:** Decode вернул C4 и план не identity; при identity результат C4 уходит в encoder, возможен fallback.
- **Повторить:** `pytest experiments/reproduce_probes.py -k IR02`; если указана команда/статический разбор вместо IR — использовать её. Только scratch fixtures, не личные фото.
- **Факт:** MockGpu зафиксировал shape[2]==4 в resize и отсутствие composition. По ядру alpha становится цветом/смещает пиксели.
- **Ожидание:** Packed uint8 RGB после композиции на bg_color до любого C3 kernel.
- **Доказательство/уверенность:** Маршрут подтверждён mock; несовместимость C4/C3 статически; device run не выполнен. Для повторённых probes полный output в `logs/probes-restored.log`; для subprocess/CMS — `logs/faults-restored.json`, `cmyk-restored.json`. Static выводы этими логами не выдаются за аппаратную проверку.
- **Последствия:** Неверные цвета/фон могут сохраниться с success.
- **План исправления:** Общий normalization path RGB/RGBA перед resize, assertions dtype/channels/strides.
- **Риск/побочные эффекты:** Не отбрасывать alpha просто convert RGB; проверить identity и downsizing.

#### IR-03: LA остаётся C2 и нарушает C3 контракт GPU
**Приоритет:** P1. **Категория:** GPU/CUDA / correctness.
- **Где:** [batchresize/cpu_backend.py:125–146](https://github.com/perforatotbird-ux/imageresizer/blob/f4f0782de06aaec63d44486e711cbc8dde349e80/batchresize/cpu_backend.py#L125); функции `decode_to_rgb; Pipeline._process_hybrid_one/_gpu_fallback`.
- **Причина:** Pillow LA разрешён без convert; np.asarray даёт H×W×2. Потребитель композитит только C4 и подаёт C2 в C3 resize.
- **Условия:** LA PNG/TIFF через hybrid/fallback, не-identity resize.
- **Повторить:** `pytest experiments/reproduce_probes.py -k IR03`; если указана команда/статический разбор вместо IR — использовать её. Только scratch fixtures, не личные фото.
- **Факт:** Получено (48,64,2), alpha=True; C3 адреса в последней части кадра превышают размер C2 buffer.
- **Ожидание:** LA→RGBA→RGB либо отдельная корректная L/alpha composition.
- **Доказательство/уверенность:** C2 подтверждено экспериментом; OOB механизм статический, CUDA авария не вызывалась. Для повторённых probes полный output в `logs/probes-restored.log`; для subprocess/CMS — `logs/faults-restored.json`, `cmyk-restored.json`. Static выводы этими логами не выдаются за аппаратную проверку.
- **Последствия:** Неверное изображение; возможна CUDA illegal memory access и сбой последующих задач.
- **План исправления:** Нормализовать все modes к RGB/RGBA до GPU; добавить shape assertions.
- **Риск/побочные эффекты:** Учитывать I;16B/P+tRNS и не применять orientation/alpha дважды.

#### IR-04: Отказ этапа даёт потерю задач без failed или deadlock
**Приоритет:** P1. **Категория:** concurrency / reliability.
- **Где:** [batchresize/pipeline.py:257–264,343–346,372–384,400,561–581](https://github.com/perforatotbird-ux/imageresizer/blob/f4f0782de06aaec63d44486e711cbc8dde349e80/batchresize/pipeline.py#L257); функции `_gpu_processor, _writer, _wait, _shutdown, _report`.
- **Причина:** Нет fatal-error channel и проверки discovered=processed+failed+skipped. Блокирующие puts не проверяют живых потребителей. int(nan budget) исполняется вне per-file try.
- **Условия:** Неожиданное исключение processor/writer; nan budget до GPU; прекращение writers при заполненной results queue.
- **Повторить:** extra_probes.py. Для Python diagnostics запускать изолированные scripts; static/Windows шаги не исполнялись здесь. Только scratch fixtures, не личные фото.
- **Факт:** nan и processor_exit: discovered=1, processed=failed=0, errors=[], exit 0. Оба writer выходят: timeout 5 s.
- **Ожидание:** Учитываемый terminal failure, ненулевой exit, корректное завершение очередей без бесконечного join.
- **Доказательство/уверенность:** Повторённые isolated subprocess fault injections; не реальная CUDA авария. Для повторённых probes полный output в `logs/probes-restored.log`; для subprocess/CMS — `logs/faults-restored.json`, `cmyk-restored.json`. Static выводы этими логами не выдаются за аппаратную проверку.
- **Последствия:** Ложный успех с пропуском файлов либо зависание.
- **План исправления:** Supervisor, fatal channel, cancellable queue operations, accounting invariant и bounded shutdown.
- **Риск/побочные эффекты:** Таймаут без drain-протокола может потерять уже готовые результаты.

#### IR-05: Частичный decode batch повторяет уже успешные jobs
**Приоритет:** P2. **Категория:** concurrency / correctness.
- **Где:** [batchresize/pipeline.py:422–461](https://github.com/perforatotbird-ux/imageresizer/blob/f4f0782de06aaec63d44486e711cbc8dde349e80/batchresize/pipeline.py#L422); функции `_process_gpu_batch, _process_gpu_batch_inner`.
- **Причина:** После _finish ранних файлов более поздний arr=None бросает исключение наружу; outer retry повторяет весь batch.
- **Условия:** Batch>1, первый array успешен, следующий None.
- **Повторить:** `pytest experiments/reproduce_probes.py -k IR05`; если указана команда/статический разбор вместо IR — использовать её. Только scratch fixtures, не личные фото.
- **Факт:** На 2 inputs в results queue 3 элемента: первый опубликован дважды.
- **Ожидание:** Exactly once terminal accounting/publication каждого input.
- **Доказательство/уверенность:** Mock с настоящим production pipeline, без CUDA. Для повторённых probes полный output в `logs/probes-restored.log`; для subprocess/CMS — `logs/faults-restored.json`, `cmyk-restored.json`. Static выводы этими логами не выдаются за аппаратную проверку.
- **Последствия:** Лишняя запись/обработка, processed>discovered, неверные bytes/file/s.
- **План исправления:** Retry только незавершённых индексов, явный terminal state job.
- **Риск/побочные эффекты:** Не потерять failed/fallback учёт при OOM split.

#### IR-06: Filter и optimized Huffman не управляют движком
**Приоритет:** P2. **Категория:** correctness / UX.
- **Где:** [batchresize/cli.py:172–218,271–278](https://github.com/perforatotbird-ux/imageresizer/blob/f4f0782de06aaec63d44486e711cbc8dde349e80/batchresize/cli.py#L172); функции `_make_options, _select_engine, _benchmark; CpuEngine.resize:84–96`.
- **Причина:** Opts содержит выбор, но GpuEngine создаётся с defaults и optimize_huffman=False; benchmark тоже игнорирует opts. CPU self.filter вообще не используется.
- **Условия:** --filter box/bilinear или --optimize-huffman; любой CPU filter.
- **Повторить:** `pytest experiments/reproduce_probes.py -k IR06`; если указана команда/статический разбор вместо IR — использовать её. Только scratch fixtures, не личные фото.
- **Факт:** Requested box/True, actual lanczos3/False. CPU три фильтра дали побитно одинаковые resized arrays.
- **Ожидание:** Применять выбор либо явно сообщать, что backend не поддерживает его.
- **Доказательство/уверенность:** Spy constructor и CPU array experiment. Для повторённых probes полный output в `logs/probes-restored.log`; для subprocess/CMS — `logs/faults-restored.json`, `cmyk-restored.json`. Static выводы этими логами не выдаются за аппаратную проверку.
- **Последствия:** Не тот визуальный результат/размер JPEG; сравнение GPU/CPU с разными kernels.
- **План исправления:** Одна engine factory из validated opts, реализация/явное ограничение CPU filters.
- **Риск/побочные эффекты:** Реальное включение CPU Lanczos меняет качество и speed, это не косметика.

#### IR-07: Четыре GUI-пресета дают KeyError
**Приоритет:** P2. **Категория:** UX / correctness.
- **Где:** [batchresize/gui.py:394–402](https://github.com/perforatotbird-ux/imageresizer/blob/f4f0782de06aaec63d44486e711cbc8dde349e80/batchresize/gui.py#L394); функции `BatchResizeUI._on_quality_change; QUALITY_CHOICES:36–45`.
- **Причина:** Callback берёт light/medium/strong/extreme из QUALITY_PRESETS, где этих compression aliases нет.
- **Условия:** Выбор соответствующего quality preset в GUI.
- **Повторить:** `pytest experiments/reproduce_probes.py -k IR07`; если указана команда/статический разбор вместо IR — использовать её. Только scratch fixtures, не личные фото.
- **Факт:** KeyError для 4 aliases; quality_var меняется раньше ошибки, slider может остаться старым.
- **Ожидание:** q90/80/65/50 и согласованные widgets.
- **Доказательство/уверенность:** Реальная callback-функция извлечена AST, вызвана с stubs; окно не запускалось. Для повторённых probes полный output в `logs/probes-restored.log`; для subprocess/CMS — `logs/faults-restored.json`, `cmyk-restored.json`. Static выводы этими логами не выдаются за аппаратную проверку.
- **Последствия:** Пользователь получает старое качество и callback error.
- **План исправления:** Использовать ALL_QUALITY_PRESETS/общий resolver, тестировать все GUI choices.
- **Риск/побочные эффекты:** Сохранить legacy aliases и ручной slider.

#### IR-08: GUI bootstrap требует numpy у якобы лёгкого Python
**Приоритет:** P2. **Категория:** compatibility / UX.
- **Где:** [batchresize/cli.py:13–16,354–391](https://github.com/perforatotbird-ux/imageresizer/blob/f4f0782de06aaec63d44486e711cbc8dde349e80/batchresize/cli.py#L13); функции `main, _launch_gui_elsewhere; __main__.py:1`.
- **Причина:** __main__ импортирует cli→cpu_backend/env→numpy до ветки --gui. Alternative Python проверяется только на import tkinter.
- **Условия:** Python с tkinter без numpy; GPU packages только в .venv, как обещано README.
- **Повторить:** PYTHONPATH=<repo> python -S -m batchresize --gui. Для Python diagnostics запускать изолированные scripts; static/Windows шаги не исполнялись здесь. Только scratch fixtures, не личные фото.
- **Факт:** Ранее выполненная изоляция site-packages дала ModuleNotFoundError numpy до GUI. Static import chain сохранена.
- **Ожидание:** Лёгкий stdlib/tkinter entrypoint и heavy imports только worker.
- **Доказательство/уверенность:** Ранее исполнено; traceback утрачен при reset, механизм подтверждён источником. Для повторённых probes полный output в `logs/probes-restored.log`; для subprocess/CMS — `logs/faults-restored.json`, `cmyk-restored.json`. Static выводы этими логами не выдаются за аппаратную проверку.
- **Последствия:** Заявленный fallback GUI не запускается на обычном чистом Python.
- **План исправления:** Ленивые imports/отдельный GUI entrypoint.
- **Риск/побочные эффекты:** Проверить --doctor/--version, cwd/PYTHONPATH.

#### IR-09: Skip-existing не проверяет целостность/происхождение
**Приоритет:** P2. **Категория:** reliability / correctness.
- **Где:** [batchresize/pipeline.py:161–181,300–305](https://github.com/perforatotbird-ux/imageresizer/blob/f4f0782de06aaec63d44486e711cbc8dde349e80/batchresize/pipeline.py#L161); функции `build_output_map, _reader`.
- **Причина:** Условие только exists(). Collision naming зависит от текущего source set; добавление photo.bmp может передать старый target другому источнику.
- **Условия:** Повреждённый/чужой output или изменённые параметры, skip без overwrite.
- **Повторить:** `pytest experiments/reproduce_probes.py -k IR09`; если указана команда/статический разбор вместо IR — использовать её. Только scratch fixtures, не личные фото.
- **Факт:** b"not a jpeg" пропущен, failed=0; добавление BMP меняет mapping старого JPEG.
- **Ожидание:** Проверенный completed output, совпадающий с source/config.
- **Доказательство/уверенность:** Повторено corrupt-skip и mapping experiment. Для повторённых probes полный output в `logs/probes-restored.log`; для subprocess/CMS — `logs/faults-restored.json`, `cmyk-restored.json`. Static выводы этими логами не выдаются за аппаратную проверку.
- **Последствия:** Пропущенные или неправильно сопоставленные результаты при resume.
- **План исправления:** Manifest с source identity/hash/config, JPEG verify и stable collision naming.
- **Риск/побочные эффекты:** Нужна совместимость старых имён, дополнительное I/O; exists→replace race также требует политики.

#### IR-10: Не валидируются finite/positive sizing, budget и limit
**Приоритет:** P2. **Категория:** correctness / reliability.
- **Где:** [batchresize/cli.py:144–198](https://github.com/perforatotbird-ux/imageresizer/blob/f4f0782de06aaec63d44486e711cbc8dde349e80/batchresize/cli.py#L144); функции `_make_options; geometry.compute_box/make_plan:51–133`.
- **Причина:** Negative sizing маскируется clamp до tiny output; nan/inf budget принят; target0 в API даёт out_w=0; limit0 означает все, negative — slice.
- **Условия:** percent=-10, MP=-1, longest=-1, shortest0, vram nan/inf, огромный upscale.
- **Повторить:** `pytest experiments/reproduce_probes.py -k IR10`; если указана команда/статический разбор вместо IR — использовать её. Только scratch fixtures, не личные фото.
- **Факт:** Невалидные значения приняты; tiny box; nan убивает worker (IR04). Огромные allocations намеренно не выполнялись.
- **Ожидание:** Ранний понятный отказ, positive finite values, input/output pixel caps.
- **Доказательство/уверенность:** Параметризованные tests + static limit/size trace. Для повторённых probes полный output в `logs/probes-restored.log`; для subprocess/CMS — `logs/faults-restored.json`, `cmyk-restored.json`. Static выводы этими логами не выдаются за аппаратную проверку.
- **Последствия:** Неожиданное массовое уменьшение/обработка большего числа файлов/OOM.
- **План исправления:** Общий validator CLI/GUI/RunOptions/geometry; ограничения ресурсов.
- **Риск/побочные эффекты:** Изменить default sizing-value=0 согласованно с режимом.

#### IR-11: Cover rounding создаёт пустой crop
**Приоритет:** P2. **Категория:** correctness.
- **Где:** [batchresize/cpu_backend.py:84–96](https://github.com/perforatotbird-ux/imageresizer/blob/f4f0782de06aaec63d44486e711cbc8dde349e80/batchresize/cpu_backend.py#L84); функции `CpuEngine.resize`.
- **Причина:** Независимый banker round start/end: y49.5 и50.5 оба→50.
- **Условия:** Source10×100, box100×1, cover, upscale=False.
- **Повторить:** `pytest experiments/reproduce_probes.py -k IR11`; если указана команда/статический разбор вместо IR — использовать её. Только scratch fixtures, не личные фото.
- **Факт:** Plan обещает10×1, cv2.resize падает !ssize.empty().
- **Ожидание:** Непустой геометрически корректный region.
- **Доказательство/уверенность:** Повторено на реальном OpenCV. Для повторённых probes полный output в `logs/probes-restored.log`; для subprocess/CMS — `logs/faults-restored.json`, `cmyk-restored.json`. Static выводы этими логами не выдаются за аппаратную проверку.
- **Последствия:** Допустимый файл не обработан.
- **План исправления:** Общий pixel-center convention, fractional resampling или корректные floor/ceil bounds.
- **Риск/побочные эффекты:** Простой max(end,start+1) меняет crop и CPU/GPU parity.

#### IR-12: Box weights и border weights математически неверны
**Приоритет:** P2. **Категория:** GPU/CUDA / correctness.
- **Где:** [batchresize/gpu_backend.py:130–172](https://github.com/perforatotbird-ux/imageresizer/blob/f4f0782de06aaec63d44486e711cbc8dde349e80/batchresize/gpu_backend.py#L130); функции `_kernel_value, build_weights`.
- **Причина:** Box возвращает ones на всех taps без area/support. idx clamp выполняется до distance, повторные edge indices меняют веса bilinear/Lanczos.
- **Условия:** Resize box через API; edges/crop/upscale с bilinear/Lanczos. CLI filter дополнительно сломан IR06.
- **Повторить:** `pytest experiments/reproduce_probes.py -k IR12`; если указана команда/статический разбор вместо IR — использовать её. Только scratch fixtures, не личные фото.
- **Факт:** [0,0,255,255],4→2 box:85 вместо0. Bilinear4→8 в .25:36.42857 вместо63.75.
- **Ожидание:** Area box; distance по виртуальным координатам и затем border map.
- **Доказательство/уверенность:** Production NumPy weight builder исполнен; GPU не нужен для математического контрпримера. Для повторённых probes полный output в `logs/probes-restored.log`; для subprocess/CMS — `logs/faults-restored.json`, `cmyk-restored.json`. Static выводы этими логами не выдаются за аппаратную проверку.
- **Последствия:** Реальный blur/edge color bias до JPEG, не lossy rounding.
- **План исправления:** Исправить area/support и порядок clamp; независимый CPU oracle.
- **Риск/побочные эффекты:** Пересмотреть cache/test PSNR, tiny/identity axes/prefilter.

#### IR-13: Alpha hint и RGB retry могут отбросить прозрачность
**Приоритет:** P2. **Категория:** GPU/CUDA / correctness.
- **Где:** [batchresize/imageio_meta.py:155–185](https://github.com/perforatotbird-ux/imageresizer/blob/f4f0782de06aaec63d44486e711cbc8dde349e80/batchresize/imageio_meta.py#L155); функции `png_has_alpha/has_alpha_hint; GpuEngine.decode_batch:420–445`.
- **Причина:** TIFF всегда False; RGB/grayscale PNG+tRNS не распознаются; P поиск ограничен4KB. Failed RGBA повторяется как RGB без composite.
- **Условия:** RGB PNG+tRNS, alpha TIFF, plugin без RGBA support.
- **Повторить:** `pytest experiments/reproduce_probes.py -k IR13`; если указана команда/статический разбор вместо IR — использовать её. Только scratch fixtures, не личные фото.
- **Факт:** HintFalse для PNG, где OpenCV вернул C4. После RGB decode alpha уже невозможно восстановить.
- **Ожидание:** Alpha-aware detection и host RGBA fallback/composite вместо silent drop.
- **Доказательство/уверенность:** Hint mismatch повторён; plugin retry hardware outcome статический. Для повторённых probes полный output в `logs/probes-restored.log`; для subprocess/CMS — `logs/faults-restored.json`, `cmyk-restored.json`. Static выводы этими логами не выдаются за аппаратную проверку.
- **Последствия:** Неверный фон, даже после исправления C4→C3 в IR02.
- **План исправления:** Проверять действительные header/layout, сохранять alpha до композиции.
- **Риск/побочные эффекты:** Дополнительный host decode меняет performance части formats.

#### IR-14: RGB JPEG получает CMYK ICC
**Приоритет:** P2. **Категория:** correctness.
- **Где:** [batchresize/cpu_backend.py:33–51,134–135](https://github.com/perforatotbird-ux/imageresizer/blob/f4f0782de06aaec63d44486e711cbc8dde349e80/batchresize/cpu_backend.py#L33); функции `CpuEngine.decode, decode_to_rgb; Pipeline._finish:389–397`.
- **Причина:** Декодер конвертирует CMYK→RGB без CMS, pipeline копирует исходный profile независимо от его color space.
- **Условия:** CMYK JPEG с валидным ICC, keep_icc=True.
- **Повторить:** python experiments/extra_probes.py. Для Python diagnostics запускать изолированные scripts; static/Windows шаги не исполнялись здесь. Только scratch fixtures, не личные фото.
- **Факт:** Output RGB + profile color space CMYK; CMS→sRGB: cannot build transform; MAE23.045/255 к managed reference на fixture.
- **Ожидание:** Соответствующий RGB ICC после правильного CMYK→output-space transform.
- **Доказательство/уверенность:** Выполнено и повторено на валидном Ghostscript default_cmyk.icc. Для повторённых probes полный output в `logs/probes-restored.log`; для subprocess/CMS — `logs/faults-restored.json`, `cmyk-restored.json`. Static выводы этими логами не выдаются за аппаратную проверку.
- **Последствия:** Неправильные цвета/невалидное color-management поведение.
- **План исправления:** Явный output space/CMS transform; не копировать несовместимый ICC.
- **Риск/побочные эффекты:** Не удалять wide-gamut RGB profile бездумно; визуальный результат изменится.

#### IR-15: Metadata ошибки проглатываются; Orientation fallback не безопасен
**Приоритет:** P2. **Категория:** reliability / correctness.
- **Где:** [batchresize/pipeline.py:389–397](https://github.com/perforatotbird-ux/imageresizer/blob/f4f0782de06aaec63d44486e711cbc8dde349e80/batchresize/pipeline.py#L389); функции `_finish; imageio_meta._strip_orientation:62–84`.
- **Причина:** except Exception:pass пишет plain JPEG как success. _strip_orientation при piexif error возвращает старый EXIF, включая старый Orientation.
- **Условия:** APP1 overflow, unsupported EXIF, piexif error/неполная установка.
- **Повторить:** `pytest experiments/reproduce_probes.py -k IR15`; если указана команда/статический разбор вместо IR — использовать её. Только scratch fixtures, не личные фото.
- **Факт:** Инъекция metadata overflow не увеличивает failed, JPEG ставится в queue; reset Orientation при fallback не гарантирован.
- **Ожидание:** Явный warning/error policy и нормализованный Orientation или удаление несогласованного тега.
- **Доказательство/уверенность:** Failure injection повторён; старый Orientation fallback статический. Для повторённых probes полный output в `logs/probes-restored.log`; для subprocess/CMS — `logs/faults-restored.json`, `cmyk-restored.json`. Static выводы этими логами не выдаются за аппаратную проверку.
- **Последствия:** Silent EXIF/ICC loss, в fallback возможен двойной поворот в viewer.
- **План исправления:** Не подавлять; безопасный EXIF fallback, отдельная metadata diagnostics.
- **Риск/побочные эффекты:** Не терять уже корректный image без согласованного fail/warn/strip решения.

#### IR-16: Размеры EXIF остаются от исходника
**Приоритет:** P3. **Категория:** correctness.
- **Где:** [batchresize/imageio_meta.py:71–80](https://github.com/perforatotbird-ux/imageresizer/blob/f4f0782de06aaec63d44486e711cbc8dde349e80/batchresize/imageio_meta.py#L71); функции `_strip_orientation/splice_metadata`.
- **Причина:** Обновляются Orientation/thumbnail, но не dimension tags.
- **Условия:** ImageWidth/Length или PixelX/YDimension в resized source.
- **Повторить:** `pytest experiments/reproduce_probes.py -k IR16`; если указана команда/статический разбор вместо IR — использовать её. Только scratch fixtures, не личные фото.
- **Факт:** JPEG12×16, EXIF width64.
- **Ожидание:** Обновлённые или удалённые stale dimension tags.
- **Доказательство/уверенность:** Повторено. Для повторённых probes полный output в `logs/probes-restored.log`; для subprocess/CMS — `logs/faults-restored.json`, `cmyk-restored.json`. Static выводы этими логами не выдаются за аппаратную проверку.
- **Последствия:** Неверные размеры в downstream каталогах, не порча bitstream.
- **План исправления:** Передавать final dimensions в normalization.
- **Риск/побочные эффекты:** Не менять произвольно vendor MakerNote.

#### IR-17: EOI heuristic не проверяет JPEG целостность и зависит от suffix
**Приоритет:** P2. **Категория:** reliability / correctness.
- **Где:** [batchresize/imageio_meta.py:55–59](https://github.com/perforatotbird-ux/imageresizer/blob/f4f0782de06aaec63d44486e711cbc8dde349e80/batchresize/imageio_meta.py#L55); функции `is_truncated_jpeg; Pipeline._reader:310–315`.
- **Причина:** Поиск FF D9 только в последних4096B: valid trailer ошибочно rejected; missing entropy с EOI accepted; PNG с .jpg отклоняется до decoder.
- **Условия:** Valid JPEG+5000 zero bytes; entropy-cut+EOI; mislabel.
- **Повторить:** `pytest experiments/reproduce_probes.py -k IR17; ранее image-probe matrix`; если указана команда/статический разбор вместо IR — использовать её. Только scratch fixtures, не личные фото.
- **Факт:** Valid trailer Pillow decodes, checker=True. Ранее entropy-cut success с premature end warning; PNG.jpg rejected.
- **Ожидание:** Различать heuristic/format mismatch/corruption; recovery отмечать.
- **Доказательство/уверенность:** Trailer повторён; entropy/mislabel наблюдались до reset, raw images/log утрачены. Для повторённых probes полный output в `logs/probes-restored.log`; для subprocess/CMS — `logs/faults-restored.json`, `cmyk-restored.json`. Static выводы этими логами не выдаются за аппаратную проверку.
- **Последствия:** False rejection или принятие испорченного изображения за правильное.
- **План исправления:** Sniff format, strict/tolerant validation и surfaced decoder warnings.
- **Риск/побочные эффекты:** Строгая проверка дороже; не считать любой trailer corruption.

#### IR-18: VRAM-budget — не верхняя граница памяти
**Приоритет:** P2. **Категория:** GPU/CUDA / reliability.
- **Где:** [batchresize/pipeline.py:343–365,428–435](https://github.com/perforatotbird-ux/imageresizer/blob/f4f0782de06aaec63d44486e711cbc8dde349e80/batchresize/pipeline.py#L343); функции `_gpu_processor/_process_gpu_batch; GpuEngine.resize/_table`.
- **Причина:** 8bytes/source pixel не учитывает output, tmp=12*src_h*out_w, cache/decoder/pools; первый oversized job всегда принимается. Retry внутри except может удерживать failed frame/arrays traceback.
- **Условия:** Oversized first job, upscale, уникальные geometries или pressure.
- **Повторить:** Статический расчёт allocation форм. Для Python diagnostics запускать изолированные scripts; static/Windows шаги не исполнялись здесь. Только scratch fixtures, не личные фото.
- **Факт:** 1000² source ~3MB →10000² output ~300MB +tmp120MB при source estimate8MB. Это расчёт, не device measurement.
- **Ожидание:** Честная soft heuristic либо memory reservation/output limits.
- **Доказательство/уверенность:** Подтверждено статически; реальный VRAM/OOM не проверен. Для повторённых probes полный output в `logs/probes-restored.log`; для subprocess/CMS — `logs/faults-restored.json`, `cmyk-restored.json`. Static выводы этими логами не выдаются за аппаратную проверку.
- **Последствия:** OOM несмотря на budget; per-file fallback всё ещё GPU и не лечит device lost.
- **План исправления:** Estimate source+scratch+output+cache, cap single job; отпускать failed frame до retry.
- **Риск/побочные эффекты:** memGetInfo моментален; агрессивная очистка ухудшает latency.

#### IR-19: Benchmark повторно сканирует свои результаты и возвращает0 при failures
**Приоритет:** P2. **Категория:** testing / performance.
- **Где:** [batchresize/cli.py:271–311](https://github.com/perforatotbird-ux/imageresizer/blob/f4f0782de06aaec63d44486e711cbc8dde349e80/batchresize/cli.py#L271); функции `_benchmark; Pipeline.discover/run`.
- **Причина:** Exclude только текущий _bench_kind_i, не общий output root; previous repeats внутри input видны. return0 и best selection не проверяют complete success.
- **Условия:** Output-inside-input либо все inputs corrupted.
- **Повторить:** `pytest experiments/reproduce_probes.py -k IR19`; если указана команда/статический разбор вместо IR — использовать её. Только scratch fixtures, не личные фото.
- **Факт:** На1input repeats discovered=[1,2]; all-failed benchmark return0.
- **Ожидание:** Frozen input manifest и invalid-run nonzero, одинаковое population.
- **Доказательство/уверенность:** Повторено настоящим CPU benchmark. Для повторённых probes полный output в `logs/probes-restored.log`; для subprocess/CMS — `logs/faults-restored.json`, `cmyk-restored.json`. Static выводы этими логами не выдаются за аппаратную проверку.
- **Последствия:** Неверные speedups/рост работы и ложный automation success.
- **План исправления:** Один source snapshot, whole output exclusion, validate outputs/counts.
- **Риск/побочные эффекты:** Сохранить все repeats/причины invalid, не выбирать best незаметно.

#### IR-20: Stage timings неполны и не CUDA execution profile
**Приоритет:** P3. **Категория:** performance / testing.
- **Где:** [batchresize/pipeline.py:402–418,440–458,469–487,625–656](https://github.com/perforatotbird-ux/imageresizer/blob/f4f0782de06aaec63d44486e711cbc8dde349e80/batchresize/pipeline.py#L402); функции `processing methods, _report`.
- **Причина:** perf_counter вокруг async launches не меряет GPU work; encode может оплатить resize wait. CPU alpha/orientation, H2D, metadata/scan/report не полностью попадают в published stages.
- **Условия:** Использование timings_avg_ms для bottleneck выводов.
- **Повторить:** Сопоставить t0…t4 и null.synchronize в decode_batch:446. Для Python diagnostics запускать изолированные scripts; static/Windows шаги не исполнялись здесь. Только scratch fixtures, не личные фото.
- **Факт:** Host launch latency/неполный coverage; read time включает failures, denominator только processed.
- **Ожидание:** Separate end-to-end wall, service/queue/transfer и CUDA events.
- **Доказательство/уверенность:** Статически; CPU profiling отдельно выполнен, GPU execution не измерен. Для повторённых probes полный output в `logs/probes-restored.log`; для subprocess/CMS — `logs/faults-restored.json`, `cmyk-restored.json`. Static выводы этими логами не выдаются за аппаратную проверку.
- **Последствия:** Неправильная оптимизация и проценты bottleneck.
- **План исправления:** Profiling-mode events/NVTX, раздельный startup/cache, корректные denominators.
- **Риск/побочные эффекты:** Synchronize в production меняет throughput.

#### IR-21: BAT возвращает старый errorlevel
**Приоритет:** P2. **Категория:** compatibility / reliability.
- **Где:** [run.bat:19–22](https://github.com/perforatotbird-ux/imageresizer/blob/f4f0782de06aaec63d44486e711cbc8dde349e80/run.bat#L19); функции `argument branch`.
- **Причина:** %errorlevel% раскрывается до выполнения Python внутри родительского IF block.
- **Условия:** Запуск run.bat с arguments, CLI возвращает failure.
- **Повторить:** Windows: run.bat --invalid-option; проверить RC. Для Python diagnostics запускать изолированные scripts; static/Windows шаги не исполнялись здесь. Только scratch fixtures, не личные фото.
- **Факт:** Статически exit /b использует предыдущее значение, не RC Python.
- **Ожидание:** Точный CLI status1/2/130.
- **Доказательство/уверенность:** Статически по cmd expansion; Windows test не выполнялся. Для повторённых probes полный output в `logs/probes-restored.log`; для subprocess/CMS — `logs/faults-restored.json`, `cmyk-restored.json`. Static выводы этими логами не выдаются за аппаратную проверку.
- **Последствия:** Failed обработка может считаться успешной.
- **План исправления:** !errorlevel! в этом block либо exit вне block.
- **Риск/побочные эффекты:** Проверить cmd interpreter-not-found и другие branches.

#### IR-22: Delayed expansion ломает ! в путях
**Приоритет:** P2. **Категория:** compatibility.
- **Где:** [run.bat:2–3,19–20](https://github.com/perforatotbird-ux/imageresizer/blob/f4f0782de06aaec63d44486e711cbc8dde349e80/run.bat#L2); функции `initial cd / argument forwarding; install.bat:2–3`.
- **Причина:** EnableDelayedExpansion включён до cd "%~dp0" и parsing %*. Кавычки не запрещают ! expansion.
- **Условия:** Checkout/username/input path содержит !.
- **Повторить:** Windows fresh scratch checkout C:\Audit!\imageresizer. Для Python diagnostics запускать изолированные scripts; static/Windows шаги не исполнялись здесь. Только scratch fixtures, не личные фото.
- **Факт:** Unpaired ! удаляется/текст воспринимается как variable; cd error не проверен.
- **Ожидание:** Буквальные пути и остановка при failed cd.
- **Доказательство/уверенность:** Статически, Windows отсутствует. Для повторённых probes полный output в `logs/probes-restored.log`; для subprocess/CMS — `logs/faults-restored.json`, `cmyk-restored.json`. Static выводы этими логами не выдаются за аппаратную проверку.
- **Последствия:** Ошибка запуска/неожиданный cwd установки.
- **План исправления:** Disable expansion для user paths; scoped menu expansion; check cd.
- **Риск/побочные эффекты:** Menu !CMD! зависит от delayed expansion, нужна regression.

#### IR-23: Worker encoding не согласован с UTF-8 reader
**Приоритет:** P2. **Категория:** compatibility / UX.
- **Где:** [batchresize/gui.py:549–590](https://github.com/perforatotbird-ux/imageresizer/blob/f4f0782de06aaec63d44486e711cbc8dde349e80/batchresize/gui.py#L549); функции `_run_process, _pump_output`.
- **Причина:** -u лишь unbuffered. GUI всегда decode utf8/replace, но не задаёт UTF-8 child env.
- **Условия:** Legacy Windows codepage, без UTF8 mode, Unicode paths.
- **Повторить:** Windows cp1251/cp1252 worker pipe + Unicode path. Для Python diagnostics запускать изолированные scripts; static/Windows шаги не исполнялись здесь. Только scratch fixtures, не личные фото.
- **Факт:** Mojibake; при непредставимом символе возможен UnicodeEncodeError до processing. На UTF-8 Windows этого условия нет.
- **Ожидание:** Явный согласованный UTF-8 IPC.
- **Доказательство/уверенность:** Статически, Windows integration не выполнялась. Для повторённых probes полный output в `logs/probes-restored.log`; для subprocess/CMS — `logs/faults-restored.json`, `cmyk-restored.json`. Static выводы этими логами не выдаются за аппаратную проверку.
- **Последствия:** Нечитаемые ошибки или отказ допустимых имён.
- **План исправления:** PYTHONIOENCODING/UTF8 только в child env, совместимый reader.
- **Риск/побочные эффекты:** Не менять global console/system codepage; native stderr может иметь иной encoding.

#### IR-24: Глобальный cache директорий устаревает
**Приоритет:** P3. **Категория:** reliability.
- **Где:** [batchresize/pipeline.py:184–195](https://github.com/perforatotbird-ux/imageresizer/blob/f4f0782de06aaec63d44486e711cbc8dde349e80/batchresize/pipeline.py#L184); функции `atomic_write`.
- **Причина:** _MADE_DIRS исключает mkdir навсегда после первого создания.
- **Условия:** Directory перемещена между двумя вызовами API в одном процессе.
- **Повторить:** `pytest experiments/reproduce_probes.py -k IR24`; если указана команда/статический разбор вместо IR — использовать её. Только scratch fixtures, не личные фото.
- **Факт:** atomic_write old/a; rename old→moved; второй atomic_write old/a →FileNotFoundError.
- **Ожидание:** Создать directory снова/обработать ENOENT.
- **Доказательство/уверенность:** Повторено на синтетических файлах через rename. Для повторённых probes полный output в `logs/probes-restored.log`; для subprocess/CMS — `logs/faults-restored.json`, `cmyk-restored.json`. Static выводы этими логами не выдаются за аппаратную проверку.
- **Последствия:** Ошибка повторного embedded/API run; отдельные CLI cache не разделяют.
- **План исправления:** Убрать вечный cache или run-local/retry mkdir.
- **Риск/побочные эффекты:** Не добавить check-then-act race вместо исправления.

#### IR-25: Недоступный subtree молча исчезает из отчёта
**Приоритет:** P2. **Категория:** reliability.
- **Где:** [batchresize/pipeline.py:137–154](https://github.com/perforatotbird-ux/imageresizer/blob/f4f0782de06aaec63d44486e711cbc8dde349e80/batchresize/pipeline.py#L137); функции `discover, Pipeline.run`.
- **Причина:** Traversal OSError подавлены без Stats errors/warnings.
- **Условия:** Permission/network scan failure при существующем input.
- **Повторить:** `pytest experiments/reproduce_probes.py -k IR25`; если указана команда/статический разбор вместо IR — использовать её. Только scratch fixtures, не личные фото.
- **Факт:** Injected PermissionError: discovered=processed=failed=0, errors=[].
- **Ожидание:** Отличать inaccessible от empty, partial/error status.
- **Доказательство/уверенность:** Повторён fault-injection. Для повторённых probes полный output в `logs/probes-restored.log`; для subprocess/CMS — `logs/faults-restored.json`, `cmyk-restored.json`. Static выводы этими логами не выдаются за аппаратную проверку.
- **Последствия:** Пропуск коллекции с видимостью успеха.
- **План исправления:** Возвращать traversal diagnostics, согласовать continue/fail-fast.
- **Риск/побочные эффекты:** Один недоступный subtree не обязан отменять все хорошие файлы.

#### IR-26: GPU label не аттестует реальный codec backend
**Приоритет:** P2. **Категория:** GPU/CUDA / testing.
- **Где:** [batchresize/gpu_backend.py:180–207,409–447](https://github.com/perforatotbird-ux/imageresizer/blob/f4f0782de06aaec63d44486e711cbc8dde349e80/batchresize/gpu_backend.py#L180); функции `GpuEngine.__init__/decode_batch; Pipeline._report`.
- **Причина:** Decoder/Encoder default plugins не ограничены; strict лишь сохранён. nvImageCodec поддерживает CPU/HYBRID/GPU/HW и внутренний fallback; stats всегда gpu.
- **Условия:** Формат обслужен CPU/hybrid plugin внутри nvImageCodec.
- **Повторить:** Inspect constructor и actual extension/backend logs на GPU. Для Python diagnostics запускать изолированные scripts; static/Windows шаги не исполнялись здесь. Только scratch fixtures, не личные фото.
- **Факт:** fallback_files считает только explicit fallback, не внутренний. Buffer в VRAM не доказывает весь decode на GPU.
- **Ожидание:** Requested/actual backend и buffer placement различаются в telemetry.
- **Доказательство/уверенность:** Static absence of enforcement; конкретный runtime выбор не проверялся. Для повторённых probes полный output в `logs/probes-restored.log`; для subprocess/CMS — `logs/faults-restored.json`, `cmyk-restored.json`. Static выводы этими логами не выдаются за аппаратную проверку.
- **Последствия:** Нельзя доказать GPU-only/no copies/объяснить performance из JSON.
- **План исправления:** Логировать backend/plugin/buffer_kind, определить strict policy и профилировать.
- **Риск/побочные эффекты:** Жёсткий запрет CPU plugins уменьшит форматы; не менять fallback молча.


## 5. Изображения и результаты тестов

| Проверка | Фактический результат | Ограничение |
|---|---|---|
| Исходный pytest | 79 collected, 55 passed/24 skipped/0 failed; повтор 5.41s | GPU tests пропущены, логи повтора приложены |
| Независимый initial pytest | 54 passed,0failed,0skipped,0.58s | Наблюдался до reset, original script/log утрачен |
| Восстановленный compact pytest | 39 passed,0failed,0skipped,0.55s | Приложен исполняемый reproduce_probes.py и лог; это не тот же 54-case набор |
| Faults | Lost task без failed при nan/processor exit; writers deadlock timeout5s | Safe subprocess exceptions, не real GPU reset/OS crash |
| CPU JPEG/progressive/gray/CMYK/PNG/WebP/TIFF/BMP | Decode/resize/encode/redecode работали на fixtures до reset | CMYK ICC ошибка отдельно IR14; это не все варианты native codecs |
| EXIF1–8 JPEG | Upstream comparisons и independent pre-JPEG pixel-exact orientation прошли | Independent 8 orientation cases повторены; GPU не исполнялся |
| TIFF orientation1–8 | До reset shape/ориентация совпали Pillow, PSNR51.60–51.86dB | Не найден предполагаемый double TIFF rotation в этом OpenCV; raw matrix утрачен |
| CPU alpha | Цветной фон RGBA, channel error≤2 при q100/444; PNG/WebP matrix | До reset; не device alpha test |
| Metadata | APP2 150000B roundtrip прошёл; CMYK valid profile несовместимость повторена | Byte preservation не равна цветовой корректности; нулевые ICC upstream не CMS-тест |
| 48MP CPU | До reset8000×6000→1024×768, full redecode,0.416s,peakRSS326.37MiB | Solid-colour JPEG750630B, не worst-case память; raw output утрачен |
| Filesystem | Linux Unicode/spaces/hash originals, symlink cycles, injected OSError/replace lock | Unicode/hash и writer OSError повторены; Windows ACL/junction/SMB не проверялись |
| Actual CLI без GPU | До reset auto→CPU exit0, explicit gpu/hybrid exit2 | GPU initialization failure не доказывает runtime fallback |

Пресеты256/512/1024/1600 и custom resolving корректны на исходных tests. Percent/MP/longest/shortest вычисляются общей geometry: scale=value/100, sqrt(target pixels/source pixels) или отношение стороны; проверенные позитивные формулы разумны. Invalid values и extreme crop нарушают контракт — IR10/11. Integer округление до1px и JPEG loss допустимы при известном эталоне; mathematical weights IR12 проверены **до JPEG**, не являются noise compression.

`plan_cover()` default allow_upscale=True, но `make_plan()` и CLI явно передают False. Поэтому README «cover — исключение и всегда увеличивает до рамки» не соответствует CLI: small source может дать меньшее output. Это документальный/API-default конфликт, исправление требует согласовать семантику, не молча менять настройку.

До reset image_probe обработал28inputs:27outputs,1rejected PNG с suffix.jpg. Отдельный entropy-cut+EOI JPEG прошёл как success с native warning `premature end of data segment`. Высокий PSNR к повторно декодированному **тому же повреждённому** источнику не доказывает его целостность; нужен intact reference. Визуально contact sheet CPU/эталон был просмотрен: обычные RGB,alpha,EXIF6JPEG/TIFF совпадали, это ограниченное наблюдение, не замена pixel metrics. Файлы этого sheet утрачены.

## 6. Производительность

### Набор и методика (наблюдались до reset)

Генератор seed20261019. **512 небольших JPEG** (~256–288×192–208,q60/85/98), **6JPEG1600×1200**, **6JPEG8.64–24MP**, **2PNG RGBA+2WebP RGBA**; EXIF1/3/6/8 и progressive JPEG. Всего528 unique generated images,75947333bytes=72.429MiB. Большие изображения наполовину structured/наполовину random, small — шумовые: это не репрезентативная выборка натуральных фотографий. Source SHA256 проверялись до/после12runs, изменений не было, но сам исходный manifest утрачен.

Все runs CPU,contain1024²,q80,4:2:0,no-upscale,whitebg,keep EXIF/ICC,fsync=False,batch_size8 (не влияет на CPU). Fresh subprocess на повтор, outputs вне input. Число reader и processor threads=workers, writers=max(2,workers//2). CPU subclass собирал timing без правки production. Elapsed Pipeline включает scan/read/processing/splice/write/wait, но **не** Python startup/imports, JSON-report и post-timing validation. Output JPEG после timing полностью декодировались: **в каждом run528 processed,0failed,0fallback**.

Первый measured run **не cold**: corpus только что создан, manifest уже читал bytes. Cache не очищался. Warm reads/write-back sandbox не равны NTFS/SMB/power-loss persistence. RSS=process high-water включая imports/processing, не только image arrays и не VRAM. Таблицы транскрибированы из действительно полученных результатов в истории; original full JSON не приложены, повторный timed experiment после reset не выполнялся.

### Все12повторов

| Workers | Queue | Repeat | Elapsed s | file/s | MiB/s input | PeakRSSMiB | Processing mean/max ms |
|---|---|---|---|---|---|---|---|
| 1 | 32 | 1 | 2.225 | 237.26 | 32.55 | 258.22 | 4.064 / 356.809 |
| 1 | 32 | 2 | 2.229 | 236.86 | 32.49 | 258.35 | 4.011 / 340.376 |
| 1 | 32 | 3 | 2.442 | 216.21 | 29.66 | 256.40 | 4.367 / 358.545 |
| 4 | 32 | 1 | 1.474 | 358.11 | 49.12 | 329.30 | 10.096 / 830.277 |
| 4 | 32 | 2 | 1.543 | 342.22 | 46.94 | 346.61 | 10.852 / 782.108 |
| 4 | 32 | 3 | 1.453 | 363.33 | 49.84 | 385.09 | 10.277 / 902.038 |
| 8 | 32 | 1 | 1.793 | 294.46 | 40.39 | 414.47 | 18.300 / 1399.063 |
| 8 | 32 | 2 | 1.587 | 332.71 | 45.64 | 439.26 | 20.708 / 1417.428 |
| 8 | 32 | 3 | 1.564 | 337.62 | 46.31 | 428.56 | 17.842 / 1188.372 |
| 8 | 1 | 1 | 1.569 | 336.61 | 46.17 | 376.13 | 15.053 / 851.262 |
| 8 | 1 | 2 | 1.487 | 355.07 | 48.71 | 444.59 | 15.143 / 1257.632 |
| 8 | 1 | 3 | 1.573 | 335.69 | 46.05 | 448.30 | 14.951 / 1218.960 |

Processing mean/max: decode begin→encode end одного файла, включая CPU alpha/orientation, исключая read/write/queue. Max — не end-to-end latency файла. Это ограничение явно учитывается.

### Средние времена стадий, ms/successfully written file

| Workers/queue/repeat | Read | Decode | Resize | Encode | Write |
|---|---|---|---|---|---|
| 1/32/1 | 0.101 | 2.609 | 0.534 | 0.489 | 0.127 |
| 1/32/2 | 0.113 | 2.560 | 0.532 | 0.489 | 0.136 |
| 1/32/3 | 0.109 | 2.786 | 0.592 | 0.556 | 0.156 |
| 4/32/1 | 0.602 | 6.439 | 1.472 | 1.121 | 0.410 |
| 4/32/2 | 0.877 | 6.710 | 1.605 | 1.235 | 0.566 |
| 4/32/3 | 0.650 | 6.364 | 1.749 | 1.054 | 0.464 |
| 8/32/1 | 1.119 | 11.937 | 2.878 | 1.707 | 0.845 |
| 8/32/2 | 2.183 | 13.763 | 2.699 | 2.195 | 1.628 |
| 8/32/3 | 1.586 | 12.198 | 2.200 | 1.686 | 1.238 |
| 8/1/1 | 0.772 | 10.015 | 2.229 | 1.150 | 0.397 |
| 8/1/2 | 0.710 | 10.001 | 2.385 | 1.124 | 0.489 |
| 8/1/3 | 0.520 | 9.874 | 2.298 | 1.228 | 0.597 |

**Стадии перекрываются. Их сумма не выдается за wall time.** Average decode растёт с workers из-за contention/scheduling и смешанных размеров; это не замедление алгоритма само по себе. Отсутствующие GPU execution/H2D/queue residence метрики не придумываются.

Read-only тех же528files после обработки, один поток, discovery заранее: **0.024449/0.019971/0.020184s**, **2962.5/3626.8/3588.5MiB/s**. Это warm-cache time floor для input-only API, **не физическая пропускная способность диска**; нет write/decode. Нельзя сравнивать с авторским K: как cold-disk result.

Отдельный exploratory profile, workers4,q32, до reset:
- Discovery528files:6.600/10.201/6.736ms.
- Header/meta mean0.178ms,max27.581ms,sum93.795ms; первый дорогой header включает lazy initialization. Geometry mean0.0043ms,max0.0238ms.
- JSON serialize0.057ms,write0.174ms,без fsync.

| Queue operation (blocking+overhead) | Samples | Mean ms | Max ms |
|---|---|---|---|
| jobs.put | 542 | 8.910 | 103.997 |
| jobs.get | 528 | 0.052 | 9.366 |
| results.put | 530 | 0.205 | 5.708 |
| results.get | 530 | 4.467 | 265.824 |

Put samples включают timeout retries/sentinels; суммировать ожидания разных потоков с elapsed нельзя. Readers испытывали backpressure, writers ждали results — «всё упирается только в запись» данным не подтверждено. Исходные extended per-stage max/raw JSON profile утрачены; приведены только реально напечатанные показатели.

### Выводы и пределы

Медианы file/s: w1/q32=236.86,w4/q32=358.11,w8/q32=332.71,w8/q1=336.61. На2CPUs8workers не лучший default; это **локальный** tuning, не универсальный совет Windows Xeon. Increasing workers повышает per-file latency/RSS; очередь ограничивает количество jobs, не bytes. Host RAM включает полные compressed source в обеих очередях/Result, одновременно decoded frames каждого processor и float alpha temporaries. Discovery/map полностью O(N).

CPU decode доминирует measured service, но whole throughput зависит от mixed sizes, scheduler, OpenCV internal threads и чтения/записи. 100ms polling `_wait` создаёт floor маленьким runs. JSON/profile formatting здесь не bottleneck. GPU decode batched, но resize/encode per-file последовательно; hybrid Pillow decode тоже один processor, workers добавляет readers, не hybrid decoders. Table reuse полезен, но count-limit не byte-budget. Глубина очереди сама по себе не защищает память.

**GPU/CPU ratio не рассчитан**, GPU warmup/batch-size/VRAM/cold-cache scaling не проведены. Бенчмарки тысяч больших реальных фото не запускались; subset выбран по ресурсам/безопасности. Сравнение со57/28file/s авторского100JPEG/4unique photographs недопустимо из-за corpus/hardware/version differences.

## 7. GPU: совместимость и ресурсы

1. Код действительно использует nvImageCodec API, CuPy RawKernel resize/alpha/box; прямого nvJPEG C wrapper,NPP,cv2.cuda,NVENC нет. nvJPEG внутри framework — не любой image decoder.
2. Pascal GTX1060 не имеет специализированного nvJPEG hardware JPEG decoder. Но утверждение REPORT/README «он появился в Turing/RTX20xx+, любой новый RTX автоматически быстрее» неправильно: NVIDIA описывала hardware decode с GA100 в CUDA11, современные docs перечисляют Ampere A100/A30 и определённые более новые architectures. Не все Ampere GeForce равны A100; не путать CUDA/hybrid algorithms с dedicated JPEG/NVDEC. [^https://docs.nvidia.com/cuda/archive/11.6.2/nvjpeg/index.html] [^https://docs.nvidia.com/cuda/nvjpeg/index.html]
3. BackendKind различает CPU_ONLY,GPU_ONLY,HYBRID_CPU_GPU,HW_GPU_ONLY. Default Decoder/Encoder без policy и `strict` unused не аттестуют execution; fallback_files отражает только явный pipeline recovery. Buffer в VRAM не означает zero CPU decode. [^https://docs.nvidia.com/cuda/nvimagecodec/py_api.html] [^https://developer.nvidia.com/nvimagecodec]
4. NVIDIA пример действительно поддерживает cp.asarray(nv_img) zero-copy CUDA array interface. Требуются реальные pointer/buffer_kind/dtype/strides checks конкретной установки, особенно packed C3 kernels. Default allow_any_depth=False масштабирует decode к8bit по current docs; код не проверяет результат. [^https://docs.nvidia.com/cuda/nvimagecodec/samples/cupy_sample.html] [^https://docs.nvidia.com/cuda/nvimagecodec/py_api.html]
5. Default stream0 и null.synchronize после decode дают blocking barrier. Отсутствие явного sync перед encode само по себе **не доказывает race**: межбиблиотечный stream contract может быть достаточен. Нужен CUDA trace/events на exact versions. Current docs не заменяют тест отсутствующего0.9wheel.
6. arrs,local arrays и cache держат buffers до конца use; interoperability обычно удерживает owner. Но async producer/consumer lifetime, stream changes, cache eviction и pool reuse аппаратно не проверялись. Подтверждённая самостоятельная use-after-free/context leak **не заявлена**. Pools retaining freed blocks — не обязательно утечка.
7. nvJPEG state/handles привязаны к current device; не все states thread-safe. Один processor снижает concurrency риска, но main-thread creation→worker use на device0 и teardown требуют hardware campaign. Документация требует корректный current device и отдельные state для concurrent threads. [^https://docs.nvidia.com/cuda/nvjpeg/index.html]
8. Recreate Decoder при free<30% и free_all_blocks — эвристика, не доказанное лечение leak/OOM. Активные arrays и library buffers не обязаны освобождаться pool-clear; traceback retry может удерживать failed allocations. Нет измеренного long-run VRAM graph. [^https://docs.nvidia.com/cuda/nvimagecodec/release_notes_v0.9.0.html]

| GPU проверка | Статус |
|---|---|
| No-stack auto fallback/explicit gpu refusal | Первоначально реальный CLI проверен; upstream CPU/mock tests повторены |
| RGBA/LA/partial-batch | Production route/CPU mock/static, **не NVIDIA execution** |
| GTX1060 SM6.1, driver/CUDA12 | Не проверены: нет hardware/driver |
| Real progressive/CMYK/PNG/WebP/TIFF GPU plugins | Не запускались, extension-dependent |
| VRAM peak/OOM/device lost/streams/lifetime | Только estimates/fault injection, не реальные измерения |
| Nsight/compute-sanitizer/multi-GPU | Не выполнены; device selector не реализован, device0 hardcoded |

## 8. Файловая система и установка

**Проверенные полезные свойства:** обычный output-inside-input исключается, in-run collisions различаются, temp в той же папке имеет pid/tid, обычный write/replace error учитывается/сохраняет старый target. Linux symlink directories не обходятся (follow_symlinks=False). Оригиналы в тестах непересекающегося режима сохраняли hashes. Это не устраняет source alias IR01, resume IR09 и benchmark-rescan IR19.

**Атомарность не равна durability:** optional fsync flushes файл до rename, directory fsync отсутствует. Kill/TerminateProcess не исполняет cleanup: может остаться temp, JSON report отсутствовать/быть частично записан. GUI Cancel/Close использует terminate, не graceful Ctrl+C; upstream synthetic `_wait` interrupt не имитирует OS kill. NTFS/SMB locking/power loss/junction/ACL не проверены.

**Установка:** install.bat создаёт local.venv и использует pip; нет admin elevation/registry/global PATH/driver modifications. В рассмотренном source нет upload/exfiltration фото, diagnostics запускает local nvidia-smi. Это не CVE-сертификат native wheels. Все requirements lower bounds, нет lock/hashes/CPU-GPU extras; default ставит весь CUDA stack на CPU-only машине, хотя ручная CPU-команда документирована. Clean Windows install не выполнена. Pinned trusted packages/SBOM/native codec review остаются задачами.

**Ресурсы:** Pillow default decompression-bomb guard на header path есть, но единых caps source bytes/output pixels/CPU queue bytes нет; native hostile files и output-upscale могут исчерпать RAM. Неограниченный workers способен создать множество threads. Не выполнялись намеренный OOM-killer, driver reset или malicious fuzz native decoders.

**Пути/команды:** Spaces в основных BAT calls quoted; ! expansion ошибочна (IR22). Unicode Linux прошёл, Windows legacy stdout IPC риск IR23. GUI Popen args=list,shell=True нет. BAT `:num` делает raw echo %ANS% до numeric validation: metacharacters может выполнить вводящий сам пользователь; это hardening, **не доказанная remote exploit/privilege escalation**. Статус doctor игнорируется install и печатается Done; --doctor exit0 при доступном CPU сам по себе соответствует CPU-only policy.

**Отчёты/GUI:** JSON write_text не atomic; произвольный --report может заменить указанный важный файл — не указывать его на originals. GUI Count files делает синхронный rglob в UI thread, не исключает output как Pipeline: потенциальное зависание/несовпадение count на big/network tree. UI `_events` и log text не bounded, но runtime flood на Windows не проверен.

## 9. Документация: что подтверждено и что нет

| Заявление | Вердикт |
|---|---|
| 79 tests | Collected79 подтверждено; CPU55 passed,24skipped, не79 GPU passes |
| GTX1060≈57file/s/1.8–2.2× | Нет карты/авторских raw JSON/fixtures; не независимо подтверждено |
| Hardware decode=Turing/RTX20xx+ | Противоречит NVIDIA hardware list; см. §7 |
| GPU PNG alpha composition | Main route обходится без composition IR02; RGB retry теряет alpha IR13 |
| GPU-only decode/no full image PCIe copies | Не аттестовано default plugin policy; hybrid явно full RGB upload |
| Filter/Huffman controls | Часть только UI/opts, engine wiring нарушен IR06 |
| Cover always enlarges | CLI False против direct plan_cover True; tiny source не обязательно target-size |
| ICC сохранён → правильный цвет | Byte splice не CMS; реальный CMYK→RGB mismatch IR14 |
| Bounded queues → memory safe | Ограничены objects, не bytes; IR18/resource model |
| Self-test доказывает pipeline | Только JPEG smoke/size SOF, не alpha/GUI/queues/ICC; CPU doctor проверяет SOI |
| Resume safe | Существующий повреждённый/чужой output пропускается IR09 |
| Нет tmp после любого interrupt | Только обычные exceptions/synthetic Ctrl+C; terminate/power-loss не гарантированы |
| CPU fallback | Auto initialization — CpuEngine; per-file — CPU decode+GPU resize/encode, не лечение broken GPU |

Авторский REPORT честно описывает100JPEG как копии4real images и3repeats с лучшим временем. Это мало уникального контента, cache/table locality и best-only selection ограничивают выводы о тысячах разных фото. Bench_out JSON указаны, но отсутствуют в tracked checkout/.gitignore их исключает. Прямое сравнение CPU213.5ms/file при8workers с GPU12.1ms/file (batch wall/size) не доказывает15–18× к изолированному1CPU thread. PSNR50.9–52 и cold-cache claims авторскогоREPORT без fixtures/measurement scripts не воспроизведены. Не нужно объявлять их ложными, но нельзя принять за доказательство текущего checkout.

## 10. Риски и недостающее покрытие

| Контракт | Уже проверено | Следующая проверка |
|---|---|---|
| Original safety | Aliases/mapping without overwrite | Windows junction/hardlink/rename race + reject before write |
| GPU C3 memory safety | Real LA shape, mock RGBA route | Device/sanitizer, contain/cover/identity, all alpha formats |
| Exactly-once | Partial decode duplicate | OOM after partial success, mixed corrupt batches, counters invariant |
| Worker lifecycle | Lost tasks/deadlock injection | Each stage death, queue-full cancel, repeated Ctrl+C/GUI terminate |
| Color/metadata | CMYK ICC/EXIF suppression | AdobeRGB/DisplayP3/CMYK real profiles, vendor EXIF, all8 orientations paths |
| Resume | Corrupt skip/map shift | Manifest source/config, interrupted publication, concurrent processes |
| Memory | CPU RSS/48MP initial; VRAM static | ≥1000unique images soak, used/allocated/free pools and context teardown |
| Storage | Linux local/warm write-back | NTFSSSD/HDD/SMB, cold cache controlled safely, ACL/locks/power durability |
| Install/GUI | Source inspection/callbacks/bootstrap | Fresh nonadmin Windows, tkinter-only interpreter, Unicode/!/ANSI paths |
| Dependencies | Inspected requirements/scripts | Lock/hashes, supported driver matrix, SBOM/CVE native libraries |

GPU fixture `gpu_available` skips important tests on CPU-only CI; add CPU contract mocks unconditionally. Upstream VRAM test only asserts success, не actual peak/split. Large image test doesn't prove all memory paths. Existing tests behavioral and useful, но они не покрывают границы модулей, где найдены ошибки.

## 11. План исправлений (не выполнен)

1. **Данные/статус:** IR01/04/25 и Windows RC21. Source-target identity до записи, discovery diagnostics, supervisor/fatal channel, accounting invariant, partial failure nonzero.
2. **GPU normalization:** IR02/03/13. Один uint8 packed RGB/RGBA path, LA and alpha fallbacks, channel/stride assertions. Не лечить бездумным drop-alpha.
3. **Exactly-once/memory:** IR05/18. Retry только unfinished, release failed frames до retry, source+scratch+output/cache estimate, single oversized job policy. Определить отдельный полный CPU recovery при device errors.
4. **Качество:** IR11/12/14/15/16/17. Sampling convention/weights, ICC output-space/CMS, EXIF orientation/dimensions, strict/tolerant validation с предупреждениями.
5. **Настройки/UI:** IR06/07/08/10/22/23. Общий validated config→engine, реальные filters/Huffman, quality dictionary, лёгкий bootstrap, UTF-8 IPC, безопасная cmd expansion.
6. **Resume/publication:** IR09/24. Stable manifest+fingerprint/verify, stale-dir correction, atomic JSON; не удалять старые результаты автоматически.
7. **Бенчмарк/телеметрия:** IR19/20/26. Fixed input manifest, invalid-run errors, all repeats+median, actual plugin/buffer reporting, profiling events отдельно от throughput.
8. **Setup/docs:** Tested constraints/lock/hashes, CPU extras, driver/GPU support matrix, correction hardware/cover/UI/performance claims.

Каждый шаг — узкий патч с regression и оценкой API/качества/скорости; массовое переписывание, смена default ради warnings и удаление существующих tests не нужны.

## 12. Порядок повторной проверки после исправлений

1. Зафиксировать fresh Git commit/status/diff, исходные fixtures не удалять.
2. Исходный pytest без «исправления ожиданий ради зелёного». Defect probes после согласования контракта превратить в desired-behaviour regression.
3. Source hashes и all-path aliases/no-overwrite, traversal failures, write/MemoryError injection; exactly-once count при каждом partial batch.
4. Pixel-level weights/crop/channel tests **до JPEG**; full decode/EXIF/ICC/alpha затем; impulses/checkerboard/thin colored lines/edge crop и визуальные differences.
5. Clean nonadmin Windows+full Tk GUI и cmd,8presets, pure tkinter interpreter, ANSI/UTF-8/Unicode/!/spaces, cancel and reports.
6. GTX1060 SM6.1 exact CUDA12 stack и отдельная подтверждённо hardware-JPEG GPU; actual plugin logs, pointer/stride checks, sanitizer/Nsight.
7. ≥1000unique разрешённых/synthetic images soak, batches1/2/4/8, budgets, repeated engine runs, stable VRAM/RSS, no missing/duplicates.
8. Frozen одинаковый corpus/quality/subsampling/filter CPU/GPU; warmup отдельно, ≥3repeats все результаты, complete-image validation; cold disk только безопасной документированной процедурой.
9. Финальная повторная проверка storage failures/resume/manifest/report; только затем публиковать доказанные speedups.

## 13. Ограничения

Конкретный commit, не все ветки/history/машина автора. Нет NVIDIA/Windows/tkinter; actual CUDA races,lifetime,VRAM,SM6.1 и NTFS/SMB/GUI не исполнялись. Mock failures не равны реальным hardware reset/segfault. CPU wheels отличаются от авторских, audit venv не clean production install. Synthetic528 mixed images и low-entropy48MP — не всё разнообразие фотографий. Native codec pentest/fuzz/CVE/SBOM/supply-chain verification не выполнены. Статические findings не выдаются за device experiments. **Reset среды уничтожил часть raw evidence**: в пакете есть только повторённые logs и явно помеченные наблюдения предыдущих outputs. Это не полная восстановленная цепочка первоначальных raw артефактов.

## 14. Артефакты и завершение

`repo/AUDIT_REPORT.md` — этот отчёт; `logs/pytest-restored.log`, `probes-restored.log`, `faults-restored.json`, `cmyk-restored.json`, `git-restored.txt`, `findings.json`, `benchmark-observations-transcribed.json`; `experiments/reproduce_probes.py`, `extra_probes.py` — воспроизводимые probes. Все scripts вне производственного дерева. Архив включает24исходных tracked файла точного checkout, но не.git/venv/личные данные. Instructions в README_REPRODUCE.md.

Весь существенный tracked tree изучен; доступные безопасные CPU tests/probes/ограниченные benchmarks были выполнены; код не исправлялся. Отчёт создан в корне восстановленного checkout. Аппаратная NVIDIA/Windows acceptance, long-run soak и полные первоначальные raw benchmark logs остаются открытыми/недоступными, а не «успешными».
