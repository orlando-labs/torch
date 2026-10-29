package torch.data.data_loader

from torch.data.sequential_sampler import SequentialSampler
from torch.data.random_sampler import RandomSampler
from torch.data.collate import default_collate
from torch.data.iterable_dataset import IterableDataset
from torch.data.tensor_dataset import TensorDataset
from torch.data.loader_worker import LoaderWorker

export DataLoader

class DataLoader:
  attr sampler

  def init(@dataset, @batch_size: 1, shuffle: false, seed: 0, sampler: null,
      @drop_last: false, collate: null, @num_workers: 0, @prefetch_factor: 2):
    raise TypeError("batch_size must be an integer") unless Int === batch_size
    raise ValueError("batch_size must be positive") if batch_size <= 0
    raise TypeError("num_workers must be an integer") unless Int === num_workers
    raise ValueError("num_workers must be non-negative") if num_workers < 0
    raise TypeError("prefetch_factor must be an integer") unless Int === prefetch_factor
    raise ValueError("prefetch_factor must be positive") if prefetch_factor <= 0
    @streaming = IterableDataset === dataset
    if @streaming and (sampler != null or shuffle):
      raise ValueError("streaming datasets cannot use a sampler or shuffle")
    raise ValueError("provide sampler or shuffle") if sampler != null and shuffle
    @sampler = if @streaming:
      null
    else if sampler != null:
      sampler
    else if shuffle:
      RandomSampler(dataset.count, seed:)
    else:
      SequentialSampler(dataset.count)
    @collate = if &collate == null then &default_collate else &collate
    @tensor_batches = &collate == null and TensorDataset === dataset

  def count():
    raise TypeError("streaming DataLoader has no known length") if @streaming
    _batch_count(@sampler.indices.count)

  def _batch_count(size):
    if @drop_last then size // @batch_size else (size + @batch_size - 1) // @batch_size

  def each(&block):
    raise ValueError("DataLoader.each requires a block") if &block == null
    return _each_prefetched((&block)) if @num_workers > 0
    if @streaming:
      self._stream_batches |samples|: block(@collate(samples))
      return self

    order = @sampler.indices
    _batch_count(order.count).times |index|:
      block(_fetch(_indices(order, index)))
    self

  def _indices(order, index):
    start = index * @batch_size
    finish = Math.min(start + @batch_size, order.count)
    order[start...finish]

  def _fetch(job):
    if @streaming:
      @collate(job)
    else if @tensor_batches:
      @dataset.batch(job)
    else:
      @collate(job.map: @dataset[$it])

  def _stream_batches(&block):
    batch = []
    @dataset.each |sample|:
      batch.push!(sample)
      if batch.count == @batch_size:
        block(batch)
        batch = []
    block(batch) if not @drop_last and not batch.empty?

  def _each_prefetched(block):
    order = if @streaming then null else @sampler.indices
    worker_count = if @streaming:
      @num_workers
    else:
      Math.min(@num_workers, _batch_count(order.count))
    return self if worker_count == 0
    workers = []
    try:
      worker_count.times:
        workers.push!(LoaderWorker(@prefetch_factor, |job|: _fetch(job)))
      submitted = 0
      received = 0
      consume = ||:
        batch = workers[received % worker_count].next!
        received += 1
        block(batch)
      submit = |job|:
        consume() if submitted - received == worker_count * @prefetch_factor
        workers[submitted % worker_count].submit!(job)
        submitted += 1
      if @streaming:
        self._stream_batches |samples|: submit(samples)
      else:
        _batch_count(order.count).times |index|: submit(_indices(order, index))
      while received < submitted:
        consume()
    ensure:
      workers.each: $it.cancel!
      workers.each: $it.join!
    self

  def map(&block):
    values = []
    self.each |batch|: values.push!(block(batch))
    values
