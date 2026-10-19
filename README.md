# batchresize — пакетный ресайзер изображений

CLI, настольный GUI и CPU/CUDA-конвейер для преобразования JPEG, PNG, WebP,
TIFF и BMP в JPEG. Фактическая поддержка вариантов форматов зависит от декодера.
Анимация и многостраничные документы не сохраняются: обрабатывается первый кадр.

**Версия 0.2.0: исправления независимого аудита.** Подробности и ограничения —
в [FIX_REPORT.md](FIX_REPORT.md). [AUDIT_REPORT.md](AUDIT_REPORT.md) описывает
исходный commit, а [REPORT.md](REPORT.md) — исторические заявления автора.
Старые скорости и результаты Windows/GTX 1060 не являются проверкой этой версии.
Исправления протестированы на Linux/CPU; Windows и CUDA требуют отдельной приёмки.

## Новый интерфейс · Desktop UI 01

Русскоязычная боковая навигация «Папки / Размер / JPEG / Движок», живая сводка
настроек, отдельный журнал, доступная панель запуска и кооперативная отмена.
Компоненты действительно сгенерированы Tkinter Designer из локального макета;
адаптивная рабочая компоновка отделена от генерации. Figma-токен для запуска не нужен.
Подробности использования генератора, проверки и ограничения: [UI_DESIGN_REPORT.md](UI_DESIGN_REPORT.md).

```bat
run.bat --gui
```

## Установка без прав администратора

Windows, только CPU:

```bat
install.bat --cpu
run.bat --doctor
```

Для NVIDIA/CUDA 12: `install.bat` без `--cpu`. Скрипт создаёт `.venv`, ставит
пакеты только туда, сохраняет коды ошибок. Зависимости загружаются из pip-индекса;
проверяйте окружение и индекс до запуска. Наличие NVIDIA не означает совместимость
конкретного GPU с выбранным драйвером, runtime и wheel. Особенно это важно для
старых архитектур (Pascal/SM 6.1). Непроверенный GPU не объявляется поддерживаемым.

Ручная CPU-установка, в том числе Linux:

```sh
python -m venv .venv
# Windows: .venv\Scripts\python.exe вместо .venv/bin/python
.venv/bin/python -m pip install -r requirements-cpu.txt
.venv/bin/python -m batchresize --doctor
```

`--doctor` проверяет доступное окружение и пытается выполнить GPU smoke-test.
Успешная CPU-диагностика не является сертификатом CUDA, качества изображений или
чисто GPU-декодирования. GUI требует Python с tkinter; bootstrap CLI не импортирует
NumPy/CUDA до выбора рабочего движка.

## Запуск

Без параметров `run.bat` показывает меню, пункт 9 запускает GUI. CLI по умолчанию
читает `<проект>/input` и пишет `<проект>/output`, независимо от текущего каталога.

```bat
run.bat -i "D:\Photos" -o "D:\Resized" --backend cpu --size web_1024 --quality balanced
run.bat -i "D:\Photos" -o "D:\Thumbs" --size thumb_256 --mode cover --allow-upscale
run.bat --size custom --width 800 --height 600 --jpeg-quality 88 --filter lanczos3
run.bat --sizing percent --sizing-value 50 --backend cpu
run.bat --sizing megapixels --sizing-value 2 --backend auto
run.bat --sizing longest --sizing-value 1600
run.bat -i "D:\Photos" -o "D:\Out" --variant thumb:128x128:cover:80:420 --variant web:1024:contain:85
run.bat --gui
run.bat --help
```

### Несколько размеров за один запуск (`--variant`)

Флаг `--variant ID:SIZE[:MODE[:QUALITY[:SUBSAMPLING]]]` можно указывать несколько раз.
При наличии вариантов результаты записываются в подкаталоги `<output>/<variant_id>/<относительный_путь>.jpg`.
Оригинал декодируется один раз на CPU/GPU и переиспользуется для всех вариантов без накопления потерь сжатия.
В текущей версии генерация нескольких вариантов поддерживается через CLI и профили задач; визуальный редактор вариантов в Desktop UI запланирован на следующий этап.

Размеры: `thumb_256`, `small_512`, `web_1024`, `large_1600` (квадратная граница),
либо `custom` с шириной и высотой. Дополнительные режимы: `percent`, `megapixels`,
`longest`, `shortest` с положительным конечным `--sizing-value`.
`contain` вписывает изображение; `cover` обрезает по центру с дробной геометрией.
**Без `--allow-upscale` увеличение запрещено и для cover**: маленький исходник
может дать результат меньше целевого прямоугольника. Измерения ведутся после EXIF
Orientation. Выход имеет минимум один пиксель по каждой оси; округление допустимо.

Качество: `maximum=95`, `high=90`, `balanced=80`, `small=60`, `very_high=95`,
`compact=65`, `light=90`, `medium=80`, `strong=65`, `extreme=50`, либо явное
`--jpeg-quality 1..100`. Это lossy JPEG, даже q=100 не гарантирует идентичность.
`--subsampling auto`: 4:2:0 до q85, 4:2:2 до q94, затем 4:4:4;
явные значения: `420`, `422`, `444`.

### Фильтры масштабирования и выходная резкость

- **Фильтры (`--filter`)**:
  - `lanczos3` (по умолчанию): 3-лепестковый фильтр Ланцоша для фотографий высокой детализации.
  - `mitchell`: кубический фильтр Митчелла-Нетравали ($B=1/3, C=1/3$) — плавное уменьшение без ringing-артефактов.
  - `area`: точное усреднение площади пикселей для чистого уменьшения (downscale). Увеличение (upscale) запрещено.
  - `bilinear`: быстрая билинейная интерполяция с anti-aliasing.
  - `nearest`: выбор ближайшего соседа для пиксель-арта и схем.
  - `box`: блочный фильтр с единичным носителем.
- **Выходная резкость (`--sharpen`)**:
  - Unsharp Mask выполняется после ресайза и до кодирования JPEG на канале яркости ($Y$ в $YCrCb$), исключая цветные ореолы.
  - Пресеты: `--sharpen off` (по умолчанию), `--sharpen weak` («Слабая»), `--sharpen standard` («Стандартная»).
  - Тонкая настройка: `--sharpen-amount`, `--sharpen-radius`, `--sharpen-threshold`.
  - Порог отсечки шума защищает ровные тона от усиления сенсорного шума, а аппаратный клиппинг [0, 255] защищает от переполнений.

## Реальный путь данных

- Чтение/метаданные: CPU, Pillow headers, ограниченные очереди и byte-reservations.
- `cpu`: OpenCV decode/encode; Pillow resampling с выбранным `box`, `bilinear`
  или `lanczos3`. Прозрачность и ICC non-RGB декодируются на CPU через Pillow.
- `gpu`: nvImageCodec automatic decode → CuPy device array → alpha/orientation
  normalization → CUDA separable resize → nvImageCodec JPEG encode → host bytes.
  Декодер nvImageCodec сам применяет EXIF; повторно поворачивать нельзя.
- `hybrid`: Pillow host decode → upload → orientation/alpha/CUDA resize → JPEG.
- При отказе GPU-файла включается **полный CPU fallback**, не повторная попытка
  CUDA-resize. `--no-fallback` отключает именно fallback приложения.

nvImageCodec сам выбирает плагины и может использовать CPU/гибридный кодек.
`--backend gpu` означает обязательный доступный CUDA-движок, **не GPU-only codec**.
`codec_backend` прямо сообщает, что реальный plugin не аттестован; `backend_counts`
отделяет маршруты приложения и не доказывает исполнение JPEG целиком на GPU.
API `GpuEngine(strict=True)` теперь явно отклоняется, вместо молчаливого игнорирования.
Для GPU-only аттестации нужны отдельная политика библиотеки и аппаратный trace.

Прозрачность смешивается с `--bg-color '#FFFFFF'` (также `R,G,B`) до resize.
CMYK/gray с подходящим ICC преобразуются в sRGB; выход получает sRGB ICC, а не CMYK
профиль на RGB-пикселях. RGB ICC сохраняется без универсальной конверсии в sRGB.
EXIF Orientation нормализуется до 1, размеры обновляются, thumbnail удаляется.
Ошибки EXIF/CMS видимы как ошибки файла; `--no-exif`/`--no-icc` управляют переносом,
но не отменяют необходимую геометрию/цветовую интерпретацию исходника.
Высокобитные данные переводятся в 8 бит: это не HDR-процессор.
CPU/GPU фильтры и JPEG могут различаться численно; byte-identical результат не обещается.

## Сохранность файлов и повторный запуск

Выход не может совпадать с входом или быть его предком. Вложенный output исключается
из обнаружения. Выходные symlink/junction и совпадения inode с оригиналом отвергаются.
Распознавание Windows reparse points присутствует, но не проверено здесь на Windows.
Это защита локального рабочего процесса, **не sandbox для враждебной файловой системы**:
TOCTOU-атаки при одновременной подмене каталогов не исключены. Не обрабатывайте
изменяемые посторонними пользователями деревья и не запускайте с админскими правами.

Публикация: случайный временный файл в каталоге назначения → flush → atomic replace.
`--fsync` дополнительно синхронизирует файл и на POSIX каталог; power-loss guarantees
зависят от файловой системы и Windows. При запрете overwrite применяется exclusive
hard-link publication: на FS без hard links операция безопасно откажет, не перезапишет.

**По умолчанию результаты перезаписываются** (оригиналы защищены отдельно).
`--skip-existing` не доверяет одному имени. Для пропуска нужны:

1. `.batchresize-state.json` с совпадающими source SHA256, настройками, версией
   приложения/библиотек и соответствием output-name;
2. совпадающий output SHA256 и полное повторное декодирование RGB JPEG нужного размера.

Старый, повреждённый или чужой output без доказанного происхождения даёт ошибку и
сохраняется. Проверьте его и используйте явный `--overwrite`, если хотите пересоздать.
Коллизии `photo.png`/`photo.jpg` разрешаются стабильными суффиксами; исторические
имена, в том числе удалённых исходников, резервируются manifest-ом.

Каталог защищён `.batchresize.lock` от двух экземпляров приложения. При обычной
отмене manifest сохраняется, lock снимается. При жёстком убийстве процесса либо
неостановившемся native worker lock остаётся: **сначала убедитесь, что процесс
полностью завершён, сохраните/проверьте manifest и результаты**, только затем
удалите stale lock вручную. Автоматического «разблокирования» нет.
Нет полной транзакции между JPEG и manifest: файл, записанный перед сбоем manifest,
не будет ошибочно пропущен; его нужно проверить/пересоздать явно.
GUI отменяет работу кооперативно. Python не может безопасно прервать зависший native
codec/драйвер; принудительное завершение процесса остаётся внешней аварийной мерой.

## Память, ошибки и отчётность

Валидация отвергает NaN/Inf, нулевые/отрицательные размеры, слишком большие targets.
Текущие защитные границы: до 100 млн пикселей на изображение, до 128 MiB compressed
на исходник, 256 MiB resident compressed jobs, оценочный 1 GiB host working budget.
Последние три доступны через `RunOptions`, не отдельными CLI-флагами. Alpha working
estimate консервативен: большие RGBA могут быть отклонены вместо риска OOM.
`workers<=64`, `batch_size<=128`, `queue_depth<=256`; увеличение потоков не означает
ускорения и увеличивает ресурсы очередей и кодеков.

`--vram-budget` — **мягкая оценка массивов приложения**, учитывает первый oversized
job и scratch/tables. Это НЕ cap всей VRAM: native codecs, runtime, CuPy pools,
другие процессы и некоторые временные allocations непрозрачны. Превышающий оценку
job уходит в CPU fallback либо failed. Для строгой изоляции нужны отдельный процесс,
аппаратные испытания и контроль внешних ресурсов; текущие лимиты не обещают этого.

JPEG проверяется по magic и marker structure, а не по расширению/последним 4096 байтам.
Допустимы trailers и APP payloads. `--allow-truncated` снимает структурный запрет,
но не гарантирует восстановление. Структура + успешный decode **не доказывают
целостность всех entropy-данных**: декодеры могут восстанавливать повреждения.

По умолчанию пишется `<output>/report.json`, либо отдельный `--report ...json`.
Он содержит complete/failed/skipped, fatal/discovery errors, fallback и маршруты,
read/decode/normalize/resize/encode/metadata/write timing. Ошибки потока и недоступные
поддеревья не считаются успехом. Неустранённый отказ даёт ненулевой exit code.
Обычные GPU timings — host service envelopes, **не CUDA kernel execution time**.
`--profile-cuda` ставит синхронизации между стадиями: это service wall time и оно
меняет throughput; это не заменяет Nsight/CUDA-events профилирование.

## Воспроизводимые проверки

```sh
.venv/bin/python -m pip install -r requirements-test.txt
.venv/bin/python -m pytest tests -q -ra
# Точные CPU-версии проверенной Linux/Python 3.13 среды:
.venv/bin/python -m pip install -r requirements-test.txt -c constraints-tested-linux-py313.txt
.venv/bin/python -m batchresize -i /path/to/synthetic -o /path/to/bench --benchmark --benchmark-repeats 3
```

Benchmark фиксирует один список входов, SHA256 до/после повторов, исключает весь
output-root, создаёт уникальную группу результатов, запрещает измерять skip-existing.
`benchmark.json` хранит все повторы, median и best valid run. Невалидные/изменённые
входы дают ненулевой код, speedup не публикуется. Hash-проверки читают входы до замера:
**это warm-cache сценарий**, а не cold-disk benchmark. При недоступном GPU измеряется
только CPU, без вывода о GPU speedup.

Границы зависимостей — не Windows/CUDA lockfile. Linux constraint-файл не переносит
аппаратную совместимость на другую машину. Приёмочный checklist и реальные логи
исправлений находятся в FIX_REPORT и сопроводительном evidence-архиве.
