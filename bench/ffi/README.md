# Стоимость вызова Sputnik → LibTorch

Диагностический benchmark отделяет loop/block, Sputnik free/method call,
минимальный C ABI с возвратом существующего value, scalar decoding/reboxing,
tensor handle validation, numel, strings, List[Int] conversion и relu.
`native_calls.cpp` и `raw_calls.cpp` подключаются только для `calls.s`;
обычный адаптер и notebook library не содержат этих entry points.
Python нужен только для опционального сравнения и анализа результатов.

```sh
./tools/build bench/ffi/calls.s > build/ffi-calls-build.log 2>&1
/opt/homebrew/opt/pytorch/libexec/bin/python bench/ffi/run.py \
  --device cpu --processes 3 --output build/bench/ffi/cpu
/opt/homebrew/opt/pytorch/libexec/bin/python bench/ffi/run.py \
  --device mps --processes 3 --output build/bench/ffi/mps
```

Все процессы запускаются последовательно, с вращением VM/native/Python.
Каждый case имеет прогрев и пять внутренних повторов; таблица использует
медиану медиан отдельных процессов. Checksum проверяется после таймера.
Размер relu по умолчанию 25088 (32×1×28×28); `--size 1` усиливает цену
обвязки. На MPS синхронизация находится на границах kernel-циклов, CPU
fallback запрещён. Compilation/cache preparation исключены из измерений.
Нельзя одновременно запускать сборки, тесты или другие benchmarks.

Все обычные rows включают одинаковую схему loop + вызов captured block;
`loop_scalar`/`loop_handle` показывают её стоимость. Разности rows —
контролируемые сравнения, их нельзя складывать в независимые составляющие.
Разность ABI/Sputnik call также включает разные пути dispatch/frame setup.
Чистая стоимость инструкции C-call отдельно этим сравнением не определяется.

`cpp_*` выполняет цикл внутри одного native-вызова, без упаковки аргументов
на каждой итерации. Вызываемая функция находится в отдельном translation
unit без LTO; реальный вызов в цикле дополнительно проверяется дизассемблером.
Эти строки включают C++ loop/function call и служат нижней границей.
У них другой harness, поэтому напрямую вычитать `loop_scalar` нельзя.

Python rows повторяют свободные/методные вызовы и PyTorch numel/relu.
Все rows используют lambda, включая numel: метод ищется внутри измеряемого
block, как в Sputnik. У custom Sputnik ABI controls нет Python counterparts;
runner показывает `—`, а не подставляет ctypes или другой ABI.

Полное native-покрытие, совпадение исходников Sputnik с build log, hashes
исходников/бинарников/LibTorch и версия PyTorch проверяются или сохраняются
в `results.json`; CMake fingerprint дополнительно отклоняет изменения
C++/headers/Sputnik/manifest/compiler после сборки. Исходники и бинарники
не должны меняться в ходе замера.

Для A/B разрешено указать сохранённую VM через `--vm-sputnik PATH`.
Runner отдельно проверяет compiler, которым собран native, и записывает
hash обоих compiler binaries. Native executable и C ABI controls остаются
общими для двух VM; native fingerprint продолжает проверяться полностью.

`--before-native PATH --before-build-log PATH` добавляет сохранённый native
executable в тот же A/B. Нужен его исходный build log с полным native-покрытием,
отсутствием bytecode fallback и совпадающим binary hash. Эти два аргумента
обязательны вместе; можно одновременно добавить `--before-vm`. Старый native
содержит собственный адаптер, поэтому исходники адаптера между сборками должны
совпадать, если нужно изолировать именно runtime/compiler.

`--before-vm PATH` добавляет сохранённую VM четвёртой lane в тот же вращающийся
порядок процессов. Это позволяет сравнивать две VM внутри одного прогона,
с общими manifest/adapter и native/Python controls; оба executable hashes
проверяются до и после. Для перенесённого compiler runner задаёт runtime root.
`--before-vm-manifest PATH` задаёт ей сохранённые module/adapter inputs, когда
меняется и C++ boundary. Их hashes проверяются до и после; без этой опции обе
VM используют текущий адаптер и изолируют только изменение VM runtime.
`--before-vm-runtime-root PATH` задаёт сохранённый SDK для компиляции старого
адаптера. Так при изменении typed ABI старая VM получает и прежний C header;
его hash проверяется до и после измерения.

Измерения, конкретные функции в samples и следующие оптимизации описаны в
[профилировании 2026-10-05](../../docs/ffi-call-profile.md).
Оптимизация самих вызовов и проверенные A/B:
[обвязка 2026-10-06](../../docs/native-call-overhead-profile.md).
Native cache race, attr/default paths, list/arena ablations и оставшийся
разрыв с Python: [native dispatch 2026-10-07](../../docs/native-dispatch-profile.md).
Причины медленного numel, общий native frame/metadata fast path и приоритеты
native/VM: [numel 2026-10-08](../../docs/native-numel-profile.md).
Первый optional typed ABI, выигрыш native numel и отсутствие выигрыша VM:
[typed leaf 2026-10-08](../../docs/typed-leaf-abi-profile.md).
Оптимизация TaskLocal и autograd carrier state, CPU/MPS A/B и оставшийся разрыв:
[execution mode 2026-10-08](../../docs/execution-mode-profile.md).
Direct method entry, trivial bodies и typed forwarding в native/VM:
[call fast paths 2026-10-08](../../docs/call-fast-profile.md).
Coallocation результата, empty constructor и typed resource ABI для public relu:
[typed results 2026-10-09](../../docs/typed-resource-result-profile.md).

Stack samples выполняются отдельно от timing results:

```sh
/opt/homebrew/opt/pytorch/libexec/bin/python bench/ffi/sample_calls.py \
  --lane sputnik-vm --case abi_identity_int --iterations 20000000 \
  --output build/bench/ffi/sample-vm-identity
```

Sample содержит все threads. Ожидание главного потока и работа FFI worker
не складываются в один процент CPU; анализировать нужно активный поток.
