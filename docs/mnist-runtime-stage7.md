# MNIST: heap registry and returned values, 2026-10-12

Продолжение [stage 6](mnist-runtime-stage6.md). Эксперименты находятся в
`build/bench/mnist-stage7`; контроль собран заново из снимка рабочего
`sputnik-lang`. Исходники, binaries и профили имеют сохранённые SHA-256 в
[provenance](../build/bench/mnist-stage7/provenance.json).

## Условия

Homebrew Clang 22.1.8, O2 (`-DNDEBUG -gline-tables-only`), системный allocator.
Для PGO: тот же Clang, O3/ThinLTO, отдельные instrumented profiles контроля
и кандидата. Обучение профиля: MNIST seed=11, batch=64, 4096/1024, три эпохи;
`task_local_get.s` и `calls_collections.s`. Численные результаты обучения
профилей совпали. Измерения не используют training inputs.

Проверочный MNIST: SmallResNet, 19706 parameters, seed=7, batch=32,
2048/512, MPS, четыре threads, десять эпох, execution scope `run`.
Python использует packed optimizer; MPS CPU fallback выключен.
Отдельный process warmup, вращение порядка lanes, проверка loss/accuracy
каждой эпохи и checkpoint round trip. Наши сборки и тесты не запускаются
одновременно с throughput measurements. Фоновая системная нагрузка остаётся.

## Ограниченный кеш служебных узлов heap

В профиле O2 присутствовало выделение `ObjectRecord` внутри unordered_map.
Кандидат сохраняет до 32 извлечённых C++17 node handles на shard (64 shards),
сбрасывая старую запись и повторно используя её память. Узлы отсутствуют в
таблице во время хранения в кеше, не удерживают объекты и защищены прежними
shard locks. Правила GC, pin, remote free и счётчики live objects сохраняются.

`vm_tests`, `stdlib_task_tests`, `sputnik_ext_tests` прошли. Однако девять пар
MNIST по десять эпох дали median candidate/control **1.004439**, 2/9 побед.
Медианы loop: control **2.699605 s**, candidate **2.720216 s**. Изменение
не принято; источник сохранён в `heap-node-cache.cpp`.

[Полные результаты](../build/bench/mnist-stage7/node-cache-mps32/results.json).

## Передача результата вызова

Первый вариант заменяет четыре ненужных копии локального `Value` на передачу
владения: оба RETURN paths, запись в caller register и запись final result.
Исходный регистр кадра остаётся неизменным.

Второй вариант дополнительно передаёт владение самим возвращаемым регистром
обычного завершающегося кадра. Обычное чтение сохраняется для watched registers,
handler merges, completed-frame snapshots, scope exits и pending unwind state.
Integer sidecars читаются как integers. Native emitter уже передаёт владение
в аналогичном обычном RETURN path; в нём дополнительное изменение не сделано.

Микротест: 300000 проходов через два нетривиальных метода с возвращаемым
объектом; checksum 18. В отдельной серии из 11 пар первая версия дала
median candidate/control **0.961350**. Совместная серия из 11 троек дала
**0.958252** для первой и **0.935950** для второй версии; обе выиграли все пары.
Это измерение накладных расходов вызова, не оценка ускорения обучения.

Первый вариант на полном MNIST, семь пар по десять эпох: median
candidate/control **1.001576**, 3/7 побед. Медианы loop: **2.622797 s** и
**2.626930 s**. Сквозной выигрыш не подтверждён.

Отдельная диагностика с `AMT_BOUNDARY_PROFILE=1`: восемь эпох, warmup и пять
вращающихся троек. Exclusive CPU `amt_execution_batch` исключает вложенные
kernel boundaries, охватывая главным образом Sputnik callback и его обвязку;
включает setup и checkpoint, поэтому не равен времени training loop.

| Вариант | Median exclusive CPU, ms | Median paired / control | Побед |
| --- | ---: | ---: | ---: |
| Control O2 | 259.003 | 1.000000 | — |
| Передача промежуточных Value | 256.593 | 0.985211 | 4/5 |
| Передача регистра кадра | 254.205 | 0.972705 | 4/5 |

Количество `execution_batch` одинаково: 530. Loss/accuracy и checkpoint
совпадают. Counters добавляют расходы; это диагностический результат,
не throughput gate.

Оба варианта прошли VM/task/FFI suites. Второй дополнительно проверен на
идентичность возвращаемых объектов при повторном использовании кадров,
escaping closures, return через ensure и сохранность completed root locals.
PGO control и candidate также прошли VM/FFI suites.

[Микротест](../build/bench/mnist-stage7/return-take-micro.json),
[O2 MNIST](../build/bench/mnist-stage7/return-move-mps32/results.json),
[диагностика CPU](../build/bench/mnist-stage7/return-boundary/results.json).

## Фоновая нагрузка при PGO-проверке

Во время `return-pgo-mps32` все lanes, включая Python и native reference,
одновременно замедлились примерно с 2.4–2.6 до 3.5–3.8 секунд.
`ps` зафиксировал два внешних Apple clang с CPU около 97% каждый.
Эта серия сохранена целиком и не служит доказательством ускорения.

После завершения внешних компиляций проведена отдельная PGO-диагностика CPU:
семь пар по восемь эпох. Вариант с передачей регистра дал median paired
exclusive CPU **1.012284**, 2/7 побед. Улучшение O2 не сохранилось после PGO;
добавление ветвлений в `take_return_register` не оправдано. В рабочий runtime
этот вариант не перенесён.

[PGO CPU](../build/bench/mnist-stage7/return-pgo-boundary/results.json),
[серия с внешними компиляциями](../build/bench/mnist-stage7/return-pgo-mps32/results.json).

## Минимальная правка и итог

Отдельно обучена и собрана PGO-версия только с четырьмя `std::move`.
VM/FFI tests прошли, включая экспериментальные проверки владения возвращаемым
объектом. PGO-микротест: median move/control **0.999245**, 7/11 побед;
вариант с передачей регистра **1.021276**, 1/11. Локальное улучшение O2
нельзя переносить на PGO.

После проверки отсутствия внешних clang проведена новая серия без counters:
девять пар, десять эпох. Все проверки численных результатов прошли.

| Вариант | Median train + evaluation, s |
| --- | ---: |
| PGO control | 2.575230 |
| PGO, четыре перемещения Value | 2.549567 |
| Built O2 reference | 2.423781 |
| Python | 2.462365 |

Median paired move/control **0.986920**, только **5/9** побед, диапазон
**0.977504–1.015781**. Это недостаточно для утверждения устойчивого ускорения
MNIST. Move/Python **1.035365**, 0/9 побед: цель «VM быстрее Python»
по-прежнему не достигнута.

В рабочий `sputnik-lang/runtime/vm.cpp` перенесена только минимальная правка
из четырёх перемещений: она убирает ненужные операции reference counting
без дополнительных ветвлений, изменения ABI, регистров, GC roots или правил
возврата. Её подтверждённая польза — O2-микротест вызовов; ускорение полного
MNIST не заявляется. Добавлен `bench/vm/return_values.s` с тем же checksum.
Более сложные heap-cache и take-return-register остались экспериментами.

Тестирование проведено на изолированной копии с тем же содержимым runtime;
перенесённый файл отличается от проверенного только переносом одной строки.
Основной `build/sputnik` не перезаписывался. Проверенная минимальная PGO-сборка:
`build/bench/mnist-stage7/sputnik-lang/build/pgo-return-move/sputnik`.

[Итоговый MNIST](../build/bench/mnist-stage7/return-pgo-move-mps32/results.json),
[PGO-микротест](../build/bench/mnist-stage7/return-pgo-micro.json),
[минимальный patch](../build/bench/mnist-stage7/return-move.patch).
