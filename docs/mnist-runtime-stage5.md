# MNIST: runtime investigation, 2026-10-11

Продолжение [предыдущего этапа](mnist-runtime-stage4.md). Benchmark:
`bench/mnist/benchmark.s`, Python reference: `bench/mnist/python_mnist.py`.
Рабочие копии обоих репозиториев содержали посторонние изменения. Эксперименты
проводятся на снимке в `build/bench/mnist-stage5`; исходные patches и binaries
сохранены там же. Python используется только как optional benchmark/reference.

## Исходная точка

Apple M4, macOS 26.6.2, LibTorch/PyTorch 2.14.0, четыре threads.
SmallResNet: 19706 parameters, MNIST 2048 train / 512 test, три эпохи,
seed=7, batch=32, run execution scope, Python packed optimizer.
MPS CPU fallback выключен. Один process warmup, пять measured repeats,
порядок запуска lanes вращается. Медианы в секундах:

| Фаза | VM | Built target | Python |
| --- | ---: | ---: | ---: |
| Train | 0.820116 | 0.750860 | 0.767647 |
| Evaluation | 0.090775 | 0.083692 | 0.079246 |
| Train + evaluation | 0.915114 | 0.845641 | 0.846868 |

Медиана суммы не обязана совпадать с суммой медиан фаз. VM проиграла Python
во всех пяти парах, built target выиграл три из пяти. Checkpoint round trip,
loss и accuracy каждой эпохи совпали. Меньшее process wall Sputnik не
означает выигрыша в скорости обучения.

[Baseline results](../build/bench/mnist-stage5/torch/build/baseline-mps32-metal/results.json),
[report](../build/bench/mnist-stage5/torch/build/baseline-mps32-metal/report.md).

## Динамический состав работы VM

Отдельный диагностический binary считал инструкции и вызовы `acquire_frame`.
Счётчики охватывают **весь процесс**, включая setup и checkpoint, а не только
training loop. Их timings не используются для оценки ускорения. Всего:
780256 выполненных инструкций и 83768 получений кадров из pool/allocator.
Повторные вызовы с direct block frame reuse в последнюю величину не входят.

| Opcode | Выполнений |
| --- | ---: |
| SEND | 115714 |
| RETURN | 90890 |
| CLOSE_UPVALUES | 90889 |
| LOAD_SELF | 79882 |
| CALL | 59073 |
| MOVE | 55244 |
| LOAD_UPVAL | 55196 |
| LOAD_IVAR | 55040 |
| SEND_SPREAD | 9216 |

Частые источники кадров: `autograd/context.s:11` (`execution_mode`, 13546),
`nn/module.s:37` (`stateful?`, 5184), `nn/functional.s:49` (4608),
`nn/sequential.s:15` (4416), `nn/batch_norm.s:43` (3328).
Это количество, не attribution времени: оно помогает выбрать кандидатов,
но само по себе не доказывает, какой из них определяет wall time.

[Raw counts](../build/bench/mnist-stage5/counts.log),
[instrumented VM source](../build/bench/mnist-stage5/vm-counts.cpp).

Предыдущий stage profiler локализовал excess VM в forward, optimizer и batch
selection; backward был близок к Python. Новые counts согласуются с большим
числом коротких вызовов и операций владения вокруг одинаковых LibTorch kernels.
TaskLocal не заменяется глобальным состоянием: нужны изоляция логических задач
и корректное восстановление execution mode.

## Эксперименты с циклом интерпретатора

VM-only A/B использует сохранённый baseline native binary и тот же Python.
Один process warmup на lane, парный rotating order, проверка всех epoch
metrics и checkpoint, hashes исходников и executables до/после. Между
throughput runs не выполняются сборки, тесты или profiler.

| Вариант | Пар | Медиана after/before | Побед |
| --- | ---: | ---: | ---: |
| Отдельная noinline функция fallback switch | 5 | 1.004506 | 2 |
| До 32 простых инструкций внутри `step`, cancellation check на каждой | 7 | 0.989834 | 6 |
| То же, cancellation check один раз на группу | 7 | 0.997911 | 4 |

Вынос fallback не дал выигрыша. Группировка простых инструкций дала около 1%
в одной серии, но разрыв с Python сохранился. Уменьшение частоты cancellation
checks не улучшило результат и не принято. Эти варианты пока не перенесены
в рабочий runtime.

Ограничение ранних серий: baseline использовала готовые object files из
рабочей сборки. После эксперимента с общими headers весь runtime был
пересобран, поэтому старый executable больше нельзя использовать для
точной attribution маленьких изменений. Начиная с кэша signatures, control
полностью пересобран через `make -B` с явными одинаковыми flags
`-O2 -DNDEBUG -gline-tables-only`; кандидат отличается от него только
проверяемым patch. Ранние серии сохранены как exploratory observations,
они не служат основанием для переноса оптимизаций.

[Split A/B](../build/bench/mnist-stage5/split-mps32/results.json),
[batch A/B](../build/bench/mnist-stage5/batch-mps32/results.json),
[cancellation slice A/B](../build/bench/mnist-stage5/slice-mps32/results.json),
[diagnostic runner](../build/bench/mnist-stage5/vm_ab.py).

## Built target: таблица code entries — отклонена

Проверена замена большого `switch (code_id)` таблицей указателей на native
функции. Для разреженных IDs сохранялся switch; исключения и trace PC
оставались на прежнем пути. Замысел — ограничить разрастание dispatcher из-за
inlining. Это гипотеза, не установленная причина проигрыша.

VM tests, full-native call buffers, trivial methods/forwarding, property/error
traces и dispatch-cache fixture прошли. Полная MNIST сборка сохранила native
coverage без bytecode fallback. Generated C++ уменьшился с 23948660 до
23826228 bytes, но executable вырос с 11437600 до 11470624 bytes.

В семи MPS batch=32 парах медиана native after/before **1.002991**, 3/7 побед:
исходный native 0.786516 s, кандидат 0.787790 s, Python 0.803209 s.
Изменение исключено из compiler. Меньший generated source не оказался
доказательством ускорения.

[A/B](../build/bench/mnist-stage5/torch/build/native-table-ab-mps32/results.json),
[fixture log](../build/bench/mnist-stage5/native-table-fixture-tests.log),
[full-native build](../build/bench/mnist-stage5/torch/build/native-table-ready/build.log).

## Value retain/release inline — отклонено

Эксперимент переносил обычные атомарные increment/decrement в `value.h`;
последнее освобождение вызывало прежнюю heap/tail destruction логику отдельной
функцией. `ObjHeader` вынесен в самостоятельный header без изменения layout,
memory ordering остался relaxed для retain и acq_rel для release.
VM/extension tests прошли, включая alias assignment и coallocated handles.

Контроль коротких методов с heap arguments/results: median paired ratio
0.974681, 4/5 побед, по пять внутренних measured repeats после warmup.
Но две независимые MNIST серии показали противоположный результат:

| MPS batch=32 | Пар | After/before | Побед |
| --- | ---: | ---: | ---: |
| Первая серия | 7 | 0.986107 | 4 |
| Подтверждающая серия | 7 | 1.019856 | 1 |

Первое наблюдение не воспроизвелось. Кроме того, затронутые общие headers
вызвали более широкую пересборку, чем в control (см. ограничение выше).
Поэтому эти серии не изолируют эффект inlining. Вариант исключён из runtime,
включая вынос заголовка объекта; регрессия именно retain/release не доказана.

[Micro](../build/bench/mnist-stage5/refcount-micro.json),
[первая серия](../build/bench/mnist-stage5/refcount-mps32/results.json),
[повтор](../build/bench/mnist-stage5/refcount-confirm-mps32/results.json),
[experimental sources](../build/bench/mnist-stage5/refcount-src/runtime/value.h).

## Signature lookup и повторная упаковка аргументов

Для следующих двух экспериментов контроль полностью пересобран с теми же
flags, что и кандидат. Первый вариант добавлял локальный для VM кэш поиска
`BcMethod` по `entry_code_id`: module metadata неизменяемы, live method
dispatch остаётся на прежнем пути. На искусственном наборе из 2000 signatures
поиск заметно ускорился, но это не перенеслось на MNIST.

Второй вариант устранял повторное создание пустых rest/kwrest containers.
Общий binder уже подготовил значения, однако `push_frame` сначала создавал
пустые Tuple/Map, которые `materialize_defaults` сразу заменял. Кандидат
инициализировал такой кадр без промежуточной упаковки; default evaluation,
captures, self/block и обработка ошибок оставались прежними.

| MPS batch=32 | Пар | Median after/before | Побед |
| --- | ---: | ---: | ---: |
| Кэш signature lookup | 7 | 1.005631 | 2 |
| Удаление повторной упаковки | 7 | 1.001125 | 3 |

VM tests прошли. Ни один вариант не дал выигрыша полной модели; оба исключены
из экспериментального runtime перед следующим этапом. Уменьшение конкретной
операции само по себе не доказывает ускорение обучения.

[Signature A/B](../build/bench/mnist-stage5/method-entry-mps32/results.json),
[bound-args A/B](../build/bench/mnist-stage5/bound-args-mps32/results.json),
[full control build](../build/bench/mnist-stage5/rebuilt-control-build.log).

## Профиль компиляции VM

Apple clang 21.0.0. Инструментированная VM собрана в отдельном `BUILD_DIR`
с `-O2 -DNDEBUG -gline-tables-only -fprofile-instr-generate`.
Обучающий профиль: MNIST MPS, seed=11, batch=64, 4096 train / 1024 test,
три эпохи; дополнительно `bench/vm/task_local_get.s` и
`bench/polyglot/sputnik/src/calls_collections.s`. Проверочный workload
остался seed=7, batch=32, 2048/512. Profile generation не является timing run.

| Проверка MPS batch=32 | Пар | Median after/before | Побед |
| --- | ---: | ---: | ---: |
| PGO, O2 | 7 | 0.959748 | 7 |
| PGO, O3, ThinLTO | 7 | 0.952990 | 7 |
| PGO с весом MNIST ×16, O3, ThinLTO | 7 | 0.986817 | 5 |

Первые две серии дали устойчивое улучшение относительно полностью
пересобранного O2 control, но ещё не победу над Python. Это сравнение
конфигураций сборки; отдельный вклад O3/ThinLTO проверен в следующей серии. Увеличение
веса MNIST ухудшило результат относительно смешанного профиля. Выбор
workloads и весов влияет на компиляторные решения и требует независимой
проверки; профиль не является универсальным ускорением любой программы.

Во всех вариантах прошли `vm_tests`, `sputnik_ext_tests`, epoch metrics и
checkpoint checks. Отсутствующий профиль для неиспользованных translation
units отмечен компилятором; предупреждений о несовпадении CFG не было.

[PGO O2](../build/bench/mnist-stage5/pgo-mps32/results.json),
[PGO O3/ThinLTO](../build/bench/mnist-stage5/pgo-lto-mps32/results.json),
[MNIST-weighted](../build/bench/mnist-stage5/pgo-ml-mps32/results.json),
[training log](../build/bench/mnist-stage5/pgo/training-mnist.log),
[PGO build](../build/bench/mnist-stage5/pgo/use-build.log),
[ThinLTO build](../build/bench/mnist-stage5/pgo/lto-build.log).

## Повтор и проверка других workloads

В повторной MPS batch=32 серии добавлены O3/ThinLTO без PGO и отдельный
PGO built target. Native profile обучен только на MNIST seed=11, batch=64,
4096/1024, три эпохи; исходники и generated C++ filename между generation/use
не менялись. Его runtime archive, generated code и extension units используют
O3/ThinLTO/PGO. Полная native coverage сохранилась: 1973 из 1973 code bodies,
без bytecode fallback и runtime bridge. Source/artifact hashes всех modules
и native-extension source hash совпали с исходной сборкой.

Медианы train + evaluation, секунды:

| Workload | VM O2 | VM PGO | Built O2 | Built PGO | Python |
| --- | ---: | ---: | ---: | ---: | ---: |
| MPS, batch=32, 7 пар | 0.890797 | 0.856567 | 0.783390 | 0.776330 | 0.799715 |
| CPU, batch=32, 5 пар | 3.027183 | 3.011478 | 3.000057 | 3.002339 | 2.957727 |
| MPS, batch=128, 5 пар | 0.592282 | 0.585246 | 0.579841 | 0.574629 | 0.570818 |

Медианы парных отношений; в скобках число побед:

| Workload | VM after/before | Built after/before | VM / Python | Built / Python |
| --- | ---: | ---: | ---: | ---: |
| MPS 32 | 0.956761 (7/7) | 0.988683 (6/7) | 1.070811 (0/7) | 0.970758 (6/7) |
| CPU 32 | 0.994068 (5/5) | 0.971710 (3/5) | 1.018173 (0/5) | 1.014782 (2/5) |
| MPS 128 | 0.993097 (3/5) | 0.990760 (4/5) | 1.025277 (1/5) | 1.008946 (2/5) |

В той же MPS 32 серии O3/ThinLTO **без** PGO дал лишь 0.995645 after/before,
5/7 побед, против 0.956761 у PGO. Это отделяет основной эффект профиля от
одной лишь смены optimization level/LTO. Для VM улучшение MPS 32 воспроизвелось
в двух независимых сериях, но в CPU и batch=128 оно существенно меньше.
У native CPU большой разброс: отношение медиан и медиана парных отношений
дают разные впечатления, поэтому устойчивое ускорение CPU не установлено.

PGO принят как **опциональный способ сборки**, не как новый default или
доказательство достижения speed gate. Обе Sputnik lanes всё ещё не быстрее
Python во всех проверенных парах. Метрики каждой эпохи и checkpoint round trip
совпали с Python во всех измерениях. Profile generation, компиляция, unit tests
и timing runs разделены; MPS fallback выключен.

[MPS 32](../build/bench/mnist-stage5/pgo-complete-mps32-valid/results.json),
[CPU 32](../build/bench/mnist-stage5/pgo-complete-cpu32/results.json),
[MPS 128](../build/bench/mnist-stage5/pgo-complete-mps128/results.json),
[native build](../build/bench/mnist-stage5/pgo/native-use-build.log),
[source/profile hashes](../build/bench/mnist-stage5/pgo/provenance.json).

## Воспроизводимая сборка

Добавлен [`sputnik-lang/tools/profile-build`](../../sputnik-lang/tools/profile-build),
без зависимости от Python: instrument/optimize для VM и compiler wrappers для
built target. Каждый режим требует новый directory; snapshot профиля и native
wrapper read-only. Это устраняет обнаруженное смешивание CXXFLAGS/object files,
а новый путь wrapper также разделяет native runtime archive cache.
Optimized VM build запускает VM/FFI tests до выдачи executable.
Ошибки несовпадения CFG и профиля не подавляются.

[Инструкция](../../sputnik-lang/docs/engineering/profile-builds.md) описывает
сборку, representative training и независимую проверку. Regression test
`tests/profile_build_tests.sh` проверяет настоящий Clang generate/merge/use,
имена с пробелами и апострофом, snapshot профиля, отказ от reuse directory и
некорректных inputs. Смена defaults глобальной сборки не выполнялась.

Инструмент применён к рабочему `sputnik-lang`: 142 C++/header/include files,
Makefile и mk/config.mk совпали с изолированным control checkout. Получен
[`build/pgo-mnist-vm/sputnik`](../../sputnik-lang/build/pgo-mnist-vm/sputnik),
`vm_tests` и `sputnik_ext_tests` прошли. Дополнительная MPS 32 серия проверила
именно этот binary, пять пар после process warmup:

| Train + evaluation | Median, s | Paired / Python | Побед над Python |
| --- | ---: | ---: | ---: |
| VM O2 control | 0.911239 | — | — |
| VM из profile-build | 0.863339 | 1.066215 | 0/5 |
| Built PGO | 0.783146 | 0.971553 | 5/5 |
| Python | 0.809822 | 1.000000 | — |

VM after/before: 0.941889, 5/5 побед. Built after/before: 0.994545,
4/5 побед. Выигрыш VM относительно O2 в подтверждающих MPS 32 сериях
составил примерно 4–6%; оставшееся отставание от Python — около 7%.
Маленький выигрыш native не следует экстраполировать на CPU/другие workloads.

[Workspace artifact A/B](../build/bench/mnist-stage5/pgo-workspace-mps32/results.json),
[workspace build](../../sputnik-lang/build/pgo-mnist-vm/build.log),
[VM tests](../../sputnik-lang/build/pgo-mnist-vm/vm-tests.log),
[FFI tests](../../sputnik-lang/build/pgo-mnist-vm/ffi-tests.log).

## Где остаётся работа VM после PGO

Отдельная диагностическая серия использует `AMT_BOUNDARY_PROFILE=1`, восемь
эпох, seed=7, batch=32, 2048/512, process warmup и три measured repeats с
вращением порядка. Это **не throughput benchmark**: counters добавляют расходы.
Счётчики охватывают весь процесс, включая setup и checkpoint evaluation.
Вложенные native scopes вычитаются из `amt_execution_batch`, поэтому его
exclusive CPU total в основном отражает Sputnik callback и обвязку между
вызовами kernels. Это не время одного kernel и не полная стоимость VM.

Медианы exclusive CPU time, миллисекунды; число вызовов совпало во всех lanes:

| Boundary | Вызовов | VM O2 | VM PGO | Built PGO |
| --- | ---: | ---: | ---: | ---: |
| `amt_execution_batch` | 530 | 276.496 | 221.892 | 83.862 |
| `amt_batch_norm` | 3072 | 254.882 | 251.857 | 247.574 |
| `amt_nn` | 6112 | 228.257 | 225.078 | 221.546 |
| `amt_binary` | 2788 | 78.288 | 78.219 | 75.561 |
| `amt_foreach_binary` | 5632 | 18.418 | 17.551 | 15.332 |
| `amt_backward` | 512 | 7.424 | 7.407 | 6.978 |

PGO сократил CPU work между native boundaries примерно на 55 мс в этой серии.
Разрыв с built target в той же области остаётся около 138 мс. Backward wall
time — около 493/495/482 мс соответственно при CPU time около 7 мс: большую
часть времени вызывающий поток ждёт. Поэтому wall time операции нельзя
автоматически приписать интерпретатору или считать равным CPU стоимости.
Loss/accuracy каждой эпохи и checkpoint совпали между Sputnik lanes; Python
в этой дополнительной диагностической серии не запускался.

Проверка с `--profile` после PGO дала заметный разброс forward/backward у VM.
Она вводит дополнительные synchronizations после стадий и меняет перекрытие
host/MPS работы. Эти результаты сохранены, но не используются как доказательство
регрессии backward или слагаемые обычного throughput run.

Дополнительно сняты 12-секундные macOS `sample` trees с интервалом 1 мс,
после трёх секунд прогрева на 80 эпохах. В стеке PGO VM на главном потоке
829 из 9793 samples классифицированы как active Sputnik runtime. Среди них
252 exclusive samples находятся в `Vm::step`; остальные распределены между
вызовами, возвратами, frame recycling и allocation/free. Это статистические
samples, не per-opcode timings: ThinLTO включает многие helpers в `step`.
Числа разных потоков и inclusive ancestors не складываются в wall time.

При разборе исправлена ошибка `bench/mnist/sample_report.py`: имя
`sputnik_manifest_extensions.dylib` проверялось после префикса `sputnik`.
Из-за этого C++ helpers расширения ошибочно попадали в runtime. Теперь exact
library match имеет приоритет; summaries пересчитаны. Классификация стеков
остаётся эвристикой по symbols/libraries, особенно у статически связанного
built target; boundary counters дают отдельное измерение этой границы.

[Boundary raw results](../build/bench/mnist-stage5/pgo-boundary-valid/results.json),
[runner](../build/bench/mnist-stage5/boundary_compare.rb),
[stage timings](../build/bench/mnist-stage5/pgo-stages-mps32/results.json),
[PGO stack summary](../build/bench/mnist-stage5/pgo-stacks/sputnik-pgo.sample.summary.json),
[sampling metadata](../build/bench/mnist-stage5/pgo-stacks/metadata.json).

## Специализация TaskLocal.get — не перенесена

Экспериментальный quick opcode для `get(default: value)` исключал общий
разбор keywords и повторный stdlib dispatch. Он проверял тип receiver и
действующий builtin handler перед каждым чтением; embedding overrides,
другие receivers, watches и нестандартная форма вызова сохраняли общий путь.
Чтение оставалось привязано к текущему logical task context.

Прошли VM, task и FFI suites, включая добавленные проверки bound null,
scope restoration, обычного `get` у другого класса и замены/восстановления
handler во время работы VM. Однако семь пар полной MPS 32 модели дали
**0.998399 after/before**, только 4/7 побед, диапазон 0.970788–1.117661.
Устойчивое улучшение не установлено. Изменение сохранено как эксперимент,
но не перенесено в рабочий runtime и исключено из следующих PGO builds.

[A/B](../build/bench/mnist-stage5/task-local-get-mps32/results.json),
[experimental source](../build/bench/mnist-stage5/task-local-get-src/runtime/vm.cpp),
[VM tests](../build/bench/mnist-stage5/task-local-get-vm-tests.log),
[task tests](../build/bench/mnist-stage5/task-local-get-task-tests.log),
[FFI tests](../build/bench/mnist-stage5/task-local-get-ffi-tests.log).

## Длинное обучение PGO и финальная перепроверка

Ещё один профиль получен тем же instrumented binary на 80 эпохах MNIST,
seed=11, batch=64, 4096/1024, MPS fallback=0. В него не добавлялись microbench
profiles. Цель — увеличить долю training loop относительно однократных
startup/verification costs, а не умножать весь короткий профиль вместе с setup.
Исходники совпали с сохранёнными control hashes. `tools/profile-build` создал
отдельную `build/pgo-mnist-loop-vm`; VM/FFI tests прошли.

Независимая MPS 32 проверка: seed=7, три эпохи, 2048/512, семь пар после
process warmup. В одной серии присутствовали оба PGO profiles, O2 control,
оба native binaries и Python:

| Lane | Median train + eval, s | Paired / Python | Побед над Python |
| --- | ---: | ---: | ---: |
| VM O2 | 0.889047 | — | — |
| VM PGO, прежний смешанный профиль | 0.855935 | 1.068729 | 0/7 |
| VM PGO, 80 эпох | 0.851370 | 1.066190 | 0/7 |
| Built O2 | 0.773194 | — | — |
| Built PGO | 0.771398 | 0.958662 | 7/7 |
| Python | 0.802415 | 1.000000 | — |

Медиана long/previous = 0.992987, 5/7 побед, диапазон 0.977058–1.052150.
Этого недостаточно, чтобы объявить длинный профиль новым предпочтительным
вариантом: разрыв с Python сохранился, дополнительный эффект мал и нестабилен.
Прежняя рабочая PGO VM снова улучшила O2 во всех семи парах, median ratio
0.961853. Loss/accuracy всех эпох и checkpoint round trip совпали с Python.

Практический результат этапа: воспроизводимый optional PGO build и измеренное
ускорение VM примерно на 4–6% для MPS 32. Built target в повторных сериях
быстрее Python примерно на 3–4%. **Speed gate для VM не достигнут**, и общего
выигрыша на CPU/batch=128 также нет. Оставшаяся работа локализована главным
образом в выполнении Sputnik между kernels: кадры коротких методов,
dispatch/argument shaping и создание/освобождение промежуточных значений.
Stack samples не выделяют одну функцию, исправление которой заведомо устранит
весь разрыв. Следующая архитектурная гипотеза — сокращать полные активации
и упаковку аргументов у проверенных forwarding bodies, сохраняя live dispatch,
exception traces, task scopes и cancellation; она здесь ещё не реализована.

[Длинный профиль и provenance](../build/bench/mnist-stage5/pgo/long/provenance.json),
[финальный A/B](../build/bench/mnist-stage5/pgo-long-mps32/results.json),
[build/test logs](../../sputnik-lang/build/pgo-mnist-loop-vm/build.log).
