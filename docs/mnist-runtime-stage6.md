# MNIST: call arguments and frames, 2026-10-12

Продолжение [stage 5](mnist-runtime-stage5.md). Все runtime-эксперименты
выполнены в `build/bench/mnist-stage6/sputnik-lang`; чужие изменения в рабочем
`sputnik-lang` сохранены. Runtime-кандидаты не приняты: выигрыш полной модели
не подтвердился в повторных проверках.

## Методика и повтор после фоновой нагрузки

SmallResNet, 19706 parameters, MNIST 2048 train / 512 test, seed=7,
batch=32, три эпохи, MPS, четыре threads, run execution scope. Python использует
packed optimizer. MPS CPU fallback выключен. Один отдельный warmup process
на вариант, затем вращающийся порядок измерений. Проверяются loss и accuracy
каждой эпохи и checkpoint round trip; source, executable и dataset hashes
фиксируются до/после. В throughput runs не запускаются наши сборки или тесты.

Новый O2 control полностью собран из текущего снимка с Homebrew Clang 22.1.8,
`-O2 -DNDEBUG -gline-tables-only`. O2-кандидаты собраны тем же компилятором
с теми же flags. Старые PGO VM/native из stage 5 используют Apple Clang 21:
сравнение новой и старой PGO — сравнение готовых конфигураций, оно **не**
изолирует вклад runtime patch.

Серии `pool-mps32`, `confirm-mps32` и `recheck-mps32` содержат внешнюю нагрузку.
В последней через `ps` подтверждён активный Apple clang с CPU около 100%.
Они сохранены, но не используются как доказательство ускорения. После просьбы
пользователя о повторе проверено отсутствие активного clang и выполнена новая
серия `quiet-mps32`: 9 measured processes на вариант. Обычная системная
нагрузка, включая Spotlight, оставалась; это не выделенная benchmark-машина.

## Передача rest-аргументов без Tuple/Map

Кандидат распознаёт точное тело `def relay(*args, **options):
target(*args, **options)`. Вместо создания rest Tuple и frozen keyword Map
с немедленной распаковкой кадр хранит исходные значения в повторно используемых
векторах. Все четыре инструкции и сам кадр сохраняются: обычный dispatch,
стек ошибок, cancellation и cooperative parking продолжают работать.
Живая сигнатура проверяется при каждом вызове; дополнительные действия,
handlers, captures, defaults, auto-assigns и другие формы идут обычным путём.

Два переадресатора на 100000 вызовов ускорились примерно в 2,5 раза. Однако
полная MNIST-модель не ускорилась. Это локальное улучшение не принимается
как оптимизация MNIST.

## Пул кадров рядом с quickened code

Второй кандидат дополняет первый: заменяет hash lookups quick code и frame pool
на индекс по code ID, а пул связывает с записью quick code. Память записей
выделяется лениво. Подготовленные кадры при переходе во вложенную VM должны
сбросить ссылку на quick code прежнего владельца; это проверяется VM suite.

Первое измерение показало около 8% выигрыша, но повтор без фоновой компиляции
его не подтвердил. В рабочий runtime вариант не переносится.

## Повторные результаты MPS batch=32

Медианы train + evaluation, секунды, `quiet-mps32`:

| Вариант | Секунды |
| --- | ---: |
| VM O2 control | 0.875062 |
| VM без упаковки rest | 0.885947 |
| VM без упаковки rest + новый пул | 0.876736 |
| VM PGO stage 5 | 0.845033 |
| VM rest + пул + новый PGO | 0.842177 |
| Built O2 stage 5 | 0.768761 |
| Built PGO stage 5 | 0.765998 |
| Python | 0.795045 |

Медианы парных отношений: rest/control **1.014707** (2/9 побед),
rest+пул/control **0.997530** (6/9), новая PGO/старая PGO **0.990276** (5/9).
Последнее сравнение дополнительно смешивает версии компилятора. Новая VM PGO
проиграла Python во всех 9 парах: медиана отношения **1.056010**.
Built PGO выиграл 8/9 пар у Python, медиана отношения **0.962608**.
Все численные проверки прошли. Общая цель «VM быстрее Python» ещё не достигнута.

[Results](../build/bench/mnist-stage6/quiet-mps32/results.json),
[provenance](../build/bench/mnist-stage6/provenance.json),
[forwarding micro](../build/bench/mnist-stage6/forwarding-micro.json),
[forwarding/pool micro](../build/bench/mnist-stage6/pool-micro.json).

## Повторное использование буферов обычного SEND

Третий кандидат проверен отдельно, на исходном O2 control без двух предыдущих
patches. Позиционные и keyword векторы берутся из пула VM по глубине вложенных
синхронных вызовов. RAII освобождает значения при завершении вызова, сохраняя
ёмкость; активные аргументы входят в GC roots. Буфер живёт независимо от
кадра, который может быть уничтожен при раскрутке исключения.

На коротком MNIST-прогоне: 9 пар, median after/before **0.986365**, 7/9 побед.
Медианы loop: control 0.874721, candidate 0.871770 секунд. Медиана парных
отношений отличается от отношения медиан; это не оценка гарантированного
ускорения каждого запуска.

Для подтверждения проведён прогон на **10 эпохах**, остальные параметры
не менялись: 7 пар, median after/before **1.003954**, только 3/7 побед.
Медианы loop: control 2.668094, candidate 2.648827, прежняя PGO VM 2.535640,
native O2 2.294546, Python 2.375586 секунд. Численные результаты совпали
во всех 10 эпохах; checkpoint проверен. Устойчивого выигрыша не установлено,
patch также не принят.

В отдельном micro из 100000 обычных вызовов с positional и keyword args
буферы ускорились примерно на 8,2% (median after/before 0.917850, 9/9 побед).
Это подтверждает локальное устранение расходов, но не его значимость для
полного training loop. Оба micro сохранены в рабочем `sputnik-lang/bench/vm`:
`forwarded_calls.s` и `send_arguments.s`.

[Короткая серия](../build/bench/mnist-stage6/send-buffers-mps32/results.json),
[10 эпох](../build/bench/mnist-stage6/send-buffers-long-mps32/results.json),
[patch](../build/bench/mnist-stage6/send-buffers.patch),
[micro](../build/bench/mnist-stage6/send-buffers-micro.json).

## Проверки и исправления PGO tooling

O2-кандидаты прошли `vm_tests`, `sputnik_ext_tests`, `stdlib_task_tests`.
Добавлены экспериментальные проверки rest forwarding: значения/defaults,
identity heap objects, отсутствие неявной передачи блока, polymorphic inherited
dispatch, diagnostics/trace и сохранение аргументов после cooperative park.
PGO runtime прошёл VM и FFI suites.
Третий кандидат дополнительно проверен на вложенном вызове из `kwargs`
property, исключении/rescue и повторном использовании буфера; VM/FFI/task
suites прошли. Экспериментальные тесты и исходники сохранены в stage6 snapshot.

В рабочий `tools/profile-build` перенесены две независимо подтверждённые
поправки. Test harness translation units компилируются без профиля CLI `main`,
но линкуются с тем же профилированным runtime. Для VM ThinLTO выбирается
`llvm-ar` рядом с заданным Clang, если доступен: системный Apple ar некорректно
собирал archive из Homebrew LLVM 22 bitcode. `AR` позволяет задать явный выбор.
`tests/profile_build_tests.sh` теперь проверяет настоящий generate/merge/use
цикл с разными CLI/test entry points и runtime archive. Тест прошёл.
