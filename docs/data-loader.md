# DataLoader и no-GIL Sputnik tasks

```sputnik
from torch.data import DataLoader

batches = DataLoader(dataset, batch_size: 64, shuffle: true, seed: 7,
  num_workers: 4, prefetch_factor: 2)
batches.each |features, labels|:
  train(features, labels)
```

`num_workers: 0` сохраняет синхронный путь. Положительное значение задаёт
число параллельных рабочих очередей: Sputnik `task.spawn` исполняет загрузку
и collation на нативных тредах VM/native backend. Каждый worker обрабатывает
свои задания последовательно; одновременно работают не более `num_workers`
загрузчиков. Свободная очередь завершает свою task и запускает новую при
следующем задании, чтобы не занимать scheduler thread ожиданием потребителя.

Dataset, collate callable, samples и готовые batches передаются по ссылке
в одном адресном пространстве. Очереди ссылок защищены `Mutex`; `Channel`
передаёт только сигнал о готовности. Проверка shareability канала не требует
freeze, pickle, копирования tensor storage или IPC. Обычная операция collate
по-прежнему может выделять новый batch: например, stack samples или
vectorized `TensorDataset.batch` через index_select.

`prefetch_factor: 2` ограничивает число отправленных, но ещё не выданных
батчей: не больше `num_workers * prefetch_factor`. Это общий предел для
заданий в очереди, вычисляемых и уже готовых результатов. Батч, выданный
потребителю, в этот предел не входит. `batch_size` и `prefetch_factor` —
положительные Int, `num_workers` — неотрицательный Int.

Выдача сохраняет порядок sampler, даже если поздний batch готов раньше.
`drop_last`, explicit sampler epoch и custom collate работают в обоих
режимах. Каждая итерация создаёт свои очереди и tasks, повторный `each`
не меняет sampler epoch. Worker наследует task-local контекст при spawn,
включая текущий grad/inference mode.

Для indexed Dataset в worker выполняются index access и collate; fast path
TensorDataset сохраняет векторную выборку. `IterableDataset.each` читается
один раз в вызывающей task, которая собирает sample batches и передаёт
их workers для параллельного collate. Stream не дублируется по workers,
length остаётся неизвестной. Помимо отправленных batches вызывающая task
может держать один собираемый sample batch.

Исключение Dataset/collate доходит до потребителя с исходным типом и
сообщением. `ensure` отменяет и join-ит workers при завершении, ошибке
потребителя и nonlocal return из блока. Отмена cooperative: уже начатый
монолитный native kernel завершается до join.

Shared Dataset и collate должны допускать параллельные вызовы: read-only
данные можно читать совместно, изменяемые caches/RNG нужно синхронизировать
или держать раздельно. После публикации готового batch worker больше его
не меняет. Если custom collate возвращает alias к Dataset, потребитель
должен учитывать продолжающееся чтение Dataset другими workers. Это
обычный контракт разделяемых ссылок, без неявного snapshot или freeze.

Для дешёвой выборки из уже загруженного TensorDataset полезность нескольких
workers зависит от стоимости batch и обучения. Синхронный режим остаётся
доступен без накладных расходов tasks. Pin-memory и weighted sampler пока
не реализованы.
