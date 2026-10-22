# MNIST: Value и dispatch VM, 2026-10-06

Sputnik VM ускорена, но устойчивое преимущество над Python пока не достигнуто.
На Apple M4, macOS 26.6.2 и PyTorch/LibTorch 2.14.0 новая VM выигрывает у
исходной во всех пяти MPS-парах с batch 32. Медиана отношения времени новой VM
к старой — 0,96554: выигрыш 3,4%. Отставание от Python сократилось с 14,5% до
9,6% по медианам пар. При batch 128 и на CPU эффект всего обучения мал и
смешан; побед над Python у VM нет ни в одной из четырёх финальных серий.

Этот этап продолжает [исследование обвязки вызовов](native-call-overhead-profile.md).
Defaults, LOOKUP_CONST, polymorphic caches, `attr … from`, быстрый Tensor → Array,
packed AdamW и execution scope из предыдущих этапов уже присутствуют в baseline.
Модель, Sputnik train/eval-алгоритмы, bridge и число операций LibTorch не менялись.

## Найденные расходы и изменения

**Поиск отсутствующего constructor hook.** После каждого `Tensor(core)` VM
искала `after_init!` в большой таблице методов Tensor и заново создавала три
`vector<bool>` для обхода классов/mixins. У Tensor hook отсутствует. В прогретом
CPU sample публичного relu этот путь занял 343 из 4580 samples активного потока
(7,5%). Первый sample попал в компиляцию адаптера и исключён; использован
повтор после прогрева кэша.

Кэш сохраняет найденный метод или доказанное отсутствие hook. Guards —
`world_epoch` и receiver `method_version`; глобальный epoch учитывает изменение
родителя и mixin. Hook разрешается после завершения init, а не до него.
Записи immutable и удерживаются через shared ownership, включая время
вложенного вызова. Scheduler VMs имеют собственный кэш, host-сессия — общий.
Проверка допустимой нулевой арности и запрет property вместо метода сохранены.

**Подготовка CALL.** Прогретый простой позиционный constructor теперь использует
тот же ограниченный stack buffer на восемь аргументов, что closure CALL,
без повторного разбора `CallPacket` и временного argument vector. Guards
сохраняют актуальный init; экземпляр создаётся заново. Defaults, rest,
keywords, blocks, clauses, native handles, native errors и overflow идут через
общий binder. Stats учитывают hit/miss один раз. Дополнительно `push_frame`
принимает captures по const reference: раньше они копировались в параметр,
затем ещё раз в frame. Ownership frame по-прежнему материализуется.

**Пустой cancellation guard.** Каждая обычная VM-инструкция вызывала constructor
и destructor `RuntimeRunCancellationMask(false)`, хотя оба ничего не делали.
Теперь guard создаётся только при cancellation unwinding. Checkpoint на каждой
инструкции и маскирование отмены в ensure остаются.

**Межмодульные вызовы для простого Value.** В tagged representation отдельными
C++-вызовами были даже `is_null`, `is_integer`, прочие tag predicates, factories
immediate values, scalar accessors, move, reset и destructor. Эти операции
повторяются в decode, guards, регистрах, arguments и frame cleanup. Их небольшие
реализации перенесены в header. Retain/release вызываются только для managed
tags; их atomic ordering, alias-safe copy assignment и final deletion сохранены.
Value остаётся 16-байтным, legacy variant — 24-байтным. Tail-box layout виден
предикатам, но typed shared ownership и reclaim продолжают работать как раньше.

Отдельно проверен более агрессивный inline самого non-final retain/release.
Он отклонён: пять вращающихся процессов CPU public relu дали отношения
managed-inline/первый-inline 1,0482 / 1,0618 / 1,0274 / 1,0391 / 1,0471.
Медиана — замедление 4,7%, все пять пар хуже. Эта версия не входит в итоговый
runtime. [Её A/B и hashes](../build/bench/vm-stage3/managed-vs-inline-relu/results.json)
сохранены, чтобы не выдавать более широкий inline за гарантированное ускорение.

## Контролируемые атомарные замеры

Три последовательных процесса каждой lane с вращением порядка; внутри каждого
case один прогрев и пять timed repeats. CPU: size=1, 300000 host / 10000 kernel
iterations; MPS: size=25088, 300000 host / 3000 kernel iterations. Все 31 Sputnik
и 15 Python checksums проверены. Native freshly built с full coverage, без
fallback. Две VM используют общий manifest и adapter; throughput, samples,
тесты и сборки выполняются раздельно. Таблица — медианы process medians, мкс.

| Случай | VM до | VM после | Native | Python |
| --- | ---: | ---: | ---: | ---: |
| CPU loop + captured block | 0,33113 | 0,24283 | 0,02877 | 0,01449 |
| CPU free constant | 0,53198 | 0,37999 | 0,04471 | 0,02628 |
| CPU Tensor numel | 0,68004 | 0,52918 | 0,10573 | 0,04151 |
| CPU execution mode | 0,73861 | 0,54030 | 0,06976 | 0,04568 |
| CPU specialized relu → core | 0,96468 | 0,87443 | 0,47123 | 0,48073 |
| CPU generic relu → Tensor | 2,14920 | 1,61004 | 0,57375 | 0,47782 |
| CPU public relu | 3,12460 | 2,33658 | 0,68960 | 0,41203 |
| MPS specialized relu → core | 2,886 | 2,827 | 2,410 | 2,652 |
| MPS generic relu → Tensor | 4,066 | 3,561 | 2,498 | 2,378 |
| MPS public relu | 5,226 | 4,368 | 2,634 | 2,387 |

Public relu выигрывает 25,2% на CPU и 16,4% на MPS. CPU loop — 26,7%.
Это сокращение host overhead: CPU specialized core минус loop меняется
примерно с 0,634 до 0,632 мкс. Kernel implementation не изменялась.
Разности rows — контролируемые contrasts разных call paths, а не независимые
слагаемые, которые можно суммировать. MPS queue/kernel variability особенно
ограничивает трактовку небольших соседних разностей.

[CPU полный отчёт](../build/bench/vm-stage3/selected-atoms-cpu/report.md),
[MPS полный отчёт](../build/bench/vm-stage3/selected-atoms-mps/report.md).
Сырые repeats, executable/source/LibTorch hashes находятся рядом в results.json.
Промежуточные `ffi-cpu-1`, `ffi-final-*`, `inline-atoms-*` и `managed-atoms-cpu`
принадлежат отдельным итерациям и не смешиваются с выбранной финальной версией.

## Полный MNIST

SmallResNet, 19706 parameters, train/test 2048/512, три эпохи, seed=7,
четыре intra-op threads, packed Python optimizer, Sputnik execution-scope=run.
Один process warmup и пять измерений на lane с вращением порядка. CPU fallback
на настоящем MPS отключён. Train + evaluation исключает imports, data loading,
setup и checkpoint; его секунды приведены ниже.

| Устройство / batch | VM до | VM после | Native | Python |
| --- | ---: | ---: | ---: | ---: |
| CPU / 32 | 2,950801 | 2,921283 | 2,857617 | 2,875673 |
| CPU / 128 | 2,917569 | 2,929202 | 2,881748 | 2,898384 |
| MPS / 32 | 0,905198 | 0,865180 | 0,771198 | 0,790652 |
| MPS / 128 | 0,574592 | 0,577756 | 0,563171 | 0,566234 |

Медианы отдельных lanes выше отличаются от медиан отношений внутри пар:

| Устройство / batch | Изменение VM, paired | Отставание VM от Python, paired | VM wins / 5 | Native wins / 5 |
| --- | ---: | ---: | ---: | ---: |
| CPU / 32 | −0,73% | +1,83% | 0 | 4 |
| CPU / 128 | +0,40% | +1,05% | 0 | 3 |
| MPS / 32 | −3,45% | +9,63% | 0 | 4 |
| MPS / 128 | −0,25% | +1,65% | 0 | 4 |

Новая VM быстрее старой в 4/5, 1/5, 5/5 и 3/5 пар соответственно.
Пять побед над старой VM на MPS/32 подтверждают эффект этого workload;
они не доказывают превосходство над Python. При CPU и batch 128 изменение
всего обучения близко к разбросу. Первый этап без inline Value случайно дал
3/5 побед VM над Python на MPS/128; финальная серия этого не повторила.

Все epoch train/test losses совпадают с Python в пределах atol=1e-5,
accuracy совпадает точно, восстановленные checkpoints сохраняют метрики.
Свежий native: 1404/1404 code objects, fallback=0, VM-independent=true;
SHA256 `a145f40557845b601d15148ba348fef5221aa46a58ecd495fbe38c09c3f54b55`.

[Все четыре серии и пары](../build/bench/vm-stage3/selected-throughput-summary.json).
Полные results.json, logs и checkpoints: `selected-mnist-cpu-32/`,
`selected-mnist-cpu-128/`, `selected-mnist-mps-32/`, `selected-mnist-mps-128/`
в `build/bench/vm-stage3/`. Единственный запуск `selected-mnist-build-control`
использован для rebuild/provenance и correctness, а не доказательства скорости.

## Почему микроускорение не равно ускорению обучения

Отдельный VM-only forward A/B: пять вращающихся процессов, 1000 операций на
case, десять прогревов внутри каждой программы; checksums совпадают. Мкс/операцию:

| Call path | VM до | VM после | Изменение |
| --- | ---: | ---: | ---: |
| Immutable eval model(x) | 782,710 | 698,152 | −10,8% |
| Eval model.call!(x) | 673,985 | 657,845 | −2,4% |
| Root stateful? | 23,629 | 19,655 | −16,8% |

Обучение использует mutable call path. Сильный выигрыш immutable forward
нельзя переносить на train forward. Разница двух forward paths включает
повторные stateful-проверки и разные adapters/dispatch: 108,7 мкс до против
40,3 после. Она не является чистой ценой одного root `stateful?`.
Алгоритм проверок не сокращался; независимый child.train! и пользовательские
overrides stateful? продолжают учитываться.
[Forward A/B](../build/bench/vm-stage3/selected-forward-mps/results.json).

Отдельный sample полного MPS forward после правок CALL, до inline Value:
2796 samples main thread, 1131 (40,5%) runtime, 1267 Metal/MPS и 355 LibTorch.
После выбранного inline Value: 2320 samples, 875 (37,7%) runtime,
1084 Metal/MPS и 327 LibTorch. Это разные diagnostic runs, не throughput A/B;
проценты CPU samples не являются процентами ускорения или GPU wall time.
Работа других threads и их waits не суммируется с main thread.

В финальном runtime hot list остаются `Vm::step`, heap release, frame recycling,
register materialization/read, allocator и method/name dispatch. Plain Value
предикаты/copy/move больше не обязаны пересекать translation-unit boundary,
но интерпретация инструкций, refcounts и allocation не устранены.
[Финальные стеки](../build/bench/vm-stage3/sample-forward-selected/forward.sample.txt),
[exclusive summary](../build/bench/vm-stage3/sample-forward-selected/forward.sample.summary.json).

Следующая работа должна измерять прежде всего mutable forward, optimizer и
полный train step. Кандидаты: guarded specialization цепочки
`Module.call! → forward! → forward`, повторная упаковка rest/keyword parameters,
декодирование instance CALL и количество frames/Value ownership операций.
Signature plans и superinstructions требуют сохранения watch, cancellation,
defaults, dynamic method replacement и lifetime; сам по себе перенос горячего
NN-алгоритма в C++ не проверяет скорость VM. Это направления, а не уже доказанное
объяснение всего оставшегося разрыва.

## Проверки и воспроизведение

Прошли `vm_tests` на tagged и legacy variant, `sputnik_ext_tests`,
`stdlib_collections_tests`, `stdlib_task_tests`, VM/full-native
`test-native-call-buffers` и 14 optional runner tests. Новые регрессии проверяют
negative/positive hook caches, inherited hook replacement, dynamic mixin,
invalid hook arity, fresh construction, 8/9 arguments, defaults, blocks,
init/hook exceptions, alias/self assignment, move и final heap/tail release.
Task suite запускался с разрешённым loopback; первый sandbox run остановился
на запрете listen, а не runtime-регрессии. Legacy moved-from Value сохраняет
tag при пустом pointer; tagged становится Null. Проверки ownership учитывают оба.

Исходная VM сохранена до правок; все сборки и измерения использовали
`-O2 -DNDEBUG`, VM также `-gline-tables-only` для samples.
[Stage provenance и hashes](../build/bench/vm-stage3/provenance.json).
Исходники VM — соседний `sputnik-lang/runtime/{vm.cpp,vm_internal.h,value.cpp,value.h}`,
регрессии — `sputnik-lang/tests/vm_tests.cpp`.

```sh
# Сохранить baseline до следующего изменения VM.
cp ../sputnik-lang/build/sputnik /tmp/sputnik-before
# После правки VM пересобрать её с теми же flags, затем собирать drivers.
make -C ../sputnik-lang -j3 build/sputnik \
  CXXFLAGS='-std=c++17 -Wall -Wextra -Wpedantic -O2 -DNDEBUG -gline-tables-only'
./tools/build bench/ffi/calls.s > build/ffi-calls-build.log 2>&1
/opt/homebrew/opt/pytorch/libexec/bin/python bench/ffi/run.py \
  --device cpu --size 1 --kernel-iterations 10000 --processes 3 \
  --before-vm /tmp/sputnik-before --output build/bench/ffi/my-vm-ab

# run.py подготавливает full-native stamp и caches вне измерений.
/opt/homebrew/opt/pytorch/libexec/bin/python bench/mnist/run.py \
  --device mps --epochs 3 --batch-size 32 --train-limit 2048 --test-limit 512 \
  --repeats 1 --warmups 0 --python-optimizer packed \
  --output build/bench/mnist/my-build-control
/opt/homebrew/opt/pytorch/libexec/bin/python bench/mnist/compare_calls.py \
  --device mps --python /opt/homebrew/opt/pytorch/libexec/bin/python \
  --before-vm /tmp/sputnik-before --batch-size 32 --processes 5 \
  --output build/bench/mnist/my-vm-throughput
```

`compare_calls.py` теперь принимает VM-only baseline без фиктивной native lane
и `--batch-size` для одинаковой нагрузки во всех lanes. Saved native и его
build log должны передаваться вместе. Full-native stamp, hashes, численная
проверка и checkpoint validation сохранены. Python остаётся optional benchmark
tooling и не требуется пакету, сборке или основным тестам Torch.
