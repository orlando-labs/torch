# LibTorch и macOS/MPS

## 1. Подтверждённый baseline

macOS 26.6.2 arm64, AppleClang 21.0.0, Command Line Tools, CMake 4.4.3,
Ninja 1.13.2, Homebrew pytorch 2.14.0_1. CPU и MPS C++ forward/backward
и обучение прошли; два Gloo процесса выполнили SUM, broadcast и barrier.
Полный Xcode для потребления этой binary-сборки не потребовался.

## 2. Готовый LibTorch вместо собственного source build

Используется `brew --prefix pytorch` либо явно указанный `LIBTORCH_PREFIX`.
Сборка sputniktorch использует shell/CMake/C++ и никогда не запускает Python,
`torchgen`, pip или upstream скрипты сборки. Upstream build dependencies
не становятся зависимостями самого пакета.

Homebrew-формула отдельно устанавливает Python и другие компоненты своего
пакета; sputniktorch потребляет исключительно C++ headers и shared libraries.
Для полностью независимой от Homebrew установки можно передать готовый
LibTorch prefix той же версии, прошедший наши probes.

## 3. Gloo headers

В bottle есть c10d и Gloo symbols, но отсутствует каталог `gloo/` headers.
`tools/build --fetch-deps` получает headers submodule PyTorch v2.14.0/v2.14.1:

- commit: `74cc005ae13f69c11d8a41e50b42025b6730e796`;
- archive SHA256: `1d7260c64008e5ad430a01f63dfd94a3e04ea663f2694a3834f3877b4da66833`.

Никакой дополнительной Gloo library не строится. Оба тега используют один
Gloo commit; bootstrap принимает только `2.14.0` и `2.14.1`. Другие версии
требуют проверки pin и повторения тестов. Реальный prefix, версия,
compiler и CMake config hash записываются в `build/native-lock.json`.

### Patch-релиз 2.14.1

[Release notes](https://github.com/pytorch/pytorch/releases/tag/v2.14.1)
перечисляют MPS исправления SVD/svdvals/lstsq: Metal pipeline compilation,
complex conjugation/layout, ортогональность U и точность малых singular
values. Они относятся к ATen/Metal implementation; эти операции пока не
входят в публичный Sputnik allowlist. Текущие Tensor/NN/optim Sputnik API менять
из-за этих fixes не требуется. CUDA binary update на Mac неприменим.

Для получения fixes нужны новые LibTorch libraries и Metal resources,
затем rebuild adapter/executable и MPS/Gloo tests. Проверка tag metadata
подтвердила неизменный Gloo pin. Это разрешённый bootstrap patch version,
а не утверждение, что 2.14.1 уже проверен запуском: текущий machine baseline
по-прежнему Homebrew `2.14.0_1`. При портировании linalg MPS tests должны
включать большие, rank-deficient и ill-conditioned inputs.

## 4. Текущий build path

`tools/build` вызывает `cmake/BuildSputnik.cmake`: он создаёт manifest для
каждого `.s` модуля, задаёт C++20 для adapter, `torch`, `torch_cpu`, `c10`,
явные `library_dirs` и `runtime_library_dirs`. Sputnik runtime остаётся C++17.
`--probes` дополнительно собирает CMake consumer через `find_package(Torch)`.
MPS capability проверяется реальным выполнением, а не именем bottle или flags.

Остальная часть документа описывает целевой packaging; не всё из неё уже
реализовано. Текущая сборка привязана к установленному Homebrew prefix.

## 5. Сборка adapter и линковка Sputnik

Adapter собирается отдельным CMake project через `find_package(Torch CONFIG
REQUIRED)` с конкретным `Torch_DIR`/prefix. Используются exported targets,
include dirs, flags и libraries этого package, а не вручную угаданная
пара `-ltorch -lc10`. Это рекомендованный upstream способ consumption.
[LibTorch C++ distribution](https://docs.pytorch.org/cppdocs/installing.html)

Основной supported integration path:

1. CMake собирает статический adapter archive из C++20 translation units;
   он ссылается на shared LibTorch.
2. Adapter публикует только C ABI thunks; native registration сохраняет
   присутствие этих symbols при dead-strip.
3. Sputnik manifest с новым structured external-artifact contract подключает
   archive, include metadata и transitive link/runtime dependency closure.
4. Final executable линкуется compiler driver-ом, совместимым с libc++
   adapter-а. Sputnik source/runtime не обязаны компилироваться как C++20.

Для spike допустимо использовать existing `cxxflags` с explicit C++20 и
ручной consumer build. В текущем driver уже добавлены library search directories
и runtime search paths с digest/cache invalidation для executable и VM.
Полный production packaging ещё требует versioned schema для prebuilt artifacts,
frameworks и language standard, а также relocatable dependency bundle.

Абсолютный путь к `.dylib` нельзя записать в `link_libraries` и ожидать,
что текущий driver воспримет его как файл: сейчас он добавляет префикс `-l`.
Likewise, linker flags в `cxxflags` не должны служить обходом структуры manifest.

## 6. Runtime package на macOS

LibTorch и его dependencies размещаются рядом с executable/app bundle.
Loader paths используют `@rpath`/`@loader_path`/`@executable_path` согласно
конкретному layout. Developer prefix разрешён только в dev profile.

Проверяются `otool -L`, install names, LC_RPATH, architecture и отсутствие
случайной зависимости от build directory/Homebrew prefix. После relocation
на чистый путь приложение должно запускаться без `DYLD_LIBRARY_PATH`.

Обязательны Metal resources: `.metallib` и любые paths/resources, ожидаемые
данной сборкой. Перенос только `.dylib` недостаточен без проверки resource
lookup. Имя `libtorch_cpu.dylib` само по себе не доказывает отсутствие MPS:
на macOS backend code может находиться в этой dependency graph.

Минимальный macOS version включает ограничения SDK и скомпилированных Metal
kernels. Понизить `MACOSX_DEPLOYMENT_TARGET` только для C++ недостаточно.
Поддержка более старой macOS появляется только после запуска на этой версии.
Signing выполняется после окончательного rpath/install-name изменения;
для app distribution проверяется полная dylib resource closure.

## 7. MPS API и policy

Read API: `backends.mps.built?`, `available?`, device information и versioned
capability report. Built и available различаются: первое — свойство artifact,
второе — host/device. Runtime smoke check выполняет kernel и backward,
а не ограничивается `available?`.

Operational API: `mps.synchronize` (ожидание, без mutation tensor data),
`mps.empty_cache!` (allocator state), memory stats и profiling hooks там,
где pinned C++ API это предоставляет. Private upstream functions помещаются
в один version-specific adapter, без их типов в публичном ABI.

Default compute dtype — float32. Float16/bfloat16 — по per-operation
capability matrix и численной проверке. Float64 MPS отвергает; upstream
обрабатывает его отдельно как unsupported dtype.
[MPS dtype implementation](https://raw.githubusercontent.com/pytorch/pytorch/v2.14.0/aten/src/ATen/native/mps/OperationUtils.mm)

Не поддержанный MPS op по умолчанию даёт ошибку с именем операции и
возможными явными действиями. `PYTORCH_ENABLE_MPS_FALLBACK=1` допускается
в отдельном opt-in run profile до startup, отражается в report; это
process-wide upstream policy, не флаг отдельного вызова. Она не гарантирует
успешный fallback любой операции.
[MPS environment policy](https://docs.pytorch.org/docs/2.14/mps_environment_variables.html)

В acceptance fallback выключен, кроме заранее объявленного CPU-generated
RNG/mask transfer path. Нельзя выдавать CPU execution за MPS coverage.
`item`, `to_a`, CPU serialization и explicit synchronize — известные sync
points. Shape/dtype metadata и default inspect не читают весь GPU buffer.
Timing выполняется после warmup с synchronization до/после измерения;
отдельно измеряется end-to-end throughput с data/transfer costs.

## 8. Acceptance source build

Нужно доказать все пункты на одном install prefix:

1. C++ consumer запускает CPU tensor op + backward без Python.
2. MPS built и available, tensor реально на MPS; matmul/activation/backward
   дают правильные значения при отключённом fallback.
3. Маленький training loop на MPS обновляет параметры и снижает loss.
4. Два fresh C++/Sputnik worker processes создают Gloo group и выполняют
   all-reduce/broadcast/barrier с timeout и чистым teardown.
5. MPS + Gloo присутствуют в одной сборке; headers и symbols согласованы.
6. Native Sputnik executable проходит те же probes через C ABI и после
   переноса в другой каталог без build environment.

Build report публикует версии, effective flags, install/resource paths,
backend probes и checksum артефактов. До его появления нельзя писать
«MPS поддержан» только на основании успешной линковки.
