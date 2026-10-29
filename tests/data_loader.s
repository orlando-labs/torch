package torch.data_loader_test

import task
import torch
from sync import Atomic, Channel
from torch.data import Dataset, IterableDataset, DataLoader, RandomSampler
from torch.autograd import no_grad, grad_enabled?
from torch.checks import check

export main

class ItemsDataset < Dataset:
  def init(@items)
  def count(): @items.count
  def [](index): @items[index]

class ItemsStream < IterableDataset:
  def init(@items)

  def each(&block):
    @items.each |item|: block(item)
    self

class Sample:
  def init(@value)
  attr value
  def value!(value): @value = value

class FailingDataset < Dataset:
  def count(): 2
  def [](index): raise TypeError("dataset worker failed")

def _first(loader):
  loader.each |batch|: return batch
  null

def _bounded_first(loader, count, fetched):
  loader.each |batch|:
    4.times: fetched.recv(timeout: 2.0)
    check(count.get == 4, "prefetch bounds submitted batches to workers times factor")
    return batch
  null

def _ordering():
  dataset = ItemsDataset([0, 1, 2, 3, 4])
  identity = |samples|: samples
  loader = DataLoader(dataset, batch_size: 2, collate: &identity, num_workers: 2)
  check(loader.count == 3 and (loader.map: $it) == [[0, 1], [2, 3], [4]],
    "ordered parallel batches and partial tail")
  check((loader.map: $it) == [[0, 1], [2, 3], [4]], "fresh iteration state")
  dropped = DataLoader(dataset, batch_size: 2, collate: &identity,
    num_workers: 2, prefetch_factor: 1, drop_last: true)
  check(dropped.count == 2 and (dropped.map: $it) == [[0, 1], [2, 3]], "parallel drop_last")
  sampler = RandomSampler(5, seed: 7)
  shuffled = DataLoader(dataset, batch_size: 2, sampler:, collate: &identity, num_workers: 2)
  synchronous = DataLoader(dataset, batch_size: 2, sampler:, collate: &identity)
  check((shuffled.map: $it) == (synchronous.map: $it), "sampler order survives parallel collation")
  sampler.set_epoch!(1)
  check((shuffled.map: $it) == (synchronous.map: $it), "explicit sampler epoch")
  check(DataLoader(ItemsDataset([]), num_workers: 20).count == 0, "empty indexed loader")
  check((DataLoader(ItemsDataset([]), num_workers: 20).map: $it) == [], "empty iteration")
  check((DataLoader(dataset, collate: &identity, num_workers: 32).map: $it) == [[0], [1], [2], [3], [4]],
    "more worker lanes than batches")
  many = ItemsDataset(Array.of(64): $it)
  expected = Array.of(64): $it
  check((DataLoader(many, collate: &identity, num_workers: 32).map: $it[0]) == expected,
    "worker lanes exceeding scheduler threads do not park idle workers")
  check((DataLoader(dataset, collate: |samples|: null, num_workers: 2).map: $it) == [null, null, null, null, null],
    "null is a valid custom batch")

def _references():
  sample = Sample(7)
  loader = DataLoader(ItemsDataset([sample]), collate: |samples|: samples[0], num_workers: 1)
  batch = _first(loader)
  batch.value!(42)
  check(sample.value == 42, "custom samples cross worker boundary by reference")
  original = torch.tensor([1.0, 2.0])
  tensors = DataLoader(ItemsDataset([original]), collate: |samples|: samples[0], num_workers: 1)
  transferred = _first(tensors)
  check(transferred._identity == original._identity, "tensor storage is never serialized or copied")
  transferred.fill!(9)
  check(original.to_a == [9.0, 9.0], "consumer retains a writable tensor reference after join")
  tracked = torch.tensor([2.0], requires_grad: true)
  contextual = DataLoader(ItemsDataset([tracked, tracked]), num_workers: 2,
    collate: |samples|: samples[0].square)
  detached = no_grad():
    contextual.map: $it.requires_grad?
  check(detached == [false, false] and grad_enabled?(), "workers inherit task-local grad mode")

def _parallel_and_bounded():
  later = Channel.new(capacity: 1)
  collate = |samples|:
    if samples[0] == 0:
      later.recv(timeout: 2.0)
    else:
      later.send(true)
    samples
  loader = DataLoader(ItemsDataset([0, 1]), collate: &collate, num_workers: 2, prefetch_factor: 1)
  check((loader.map: $it) == [[0], [1]], "two workers execute concurrently and deliver sampler order")
  fetched = Channel.new(capacity: 4)
  count = Atomic.new(0)
  counted = |samples|:
    count.update: $it + 1
    fetched.send(true)
    samples
  bounded = DataLoader(ItemsDataset(Array.of(20): $it), collate: &counted,
    num_workers: 2, prefetch_factor: 2)
  first = _bounded_first(bounded, count, fetched)
  check(first == [0] and count.get == 4, "early return joins workers without extra dataset reads")
  check(_first(DataLoader(ItemsDataset([3, 4]), collate: |samples|: samples, num_workers: 2)) == [3],
    "nonlocal return unwinds and joins the loader")

def _streaming():
  stream = ItemsStream([0, 1, 2, 3, 4])
  loader = DataLoader(stream, batch_size: 2, num_workers: 2, prefetch_factor: 1)
  check((loader.map: $it.to_a) == [[0, 1], [2, 3], [4]], "stream is read once with parallel collation")
  dropped = DataLoader(stream, batch_size: 2, num_workers: 2, drop_last: true)
  check((dropped.map: $it.to_a) == [[0, 1], [2, 3]], "stream tail is dropped once")
  check((DataLoader(ItemsStream([]), num_workers: 2).map: $it) == [], "empty stream joins workers")
  unknown = false
  try:
    loader.count
  rescue TypeError:
    unknown = true
  check(unknown, "parallel streaming length remains unknown")

def _failures_and_cleanup():
  failed = false
  try:
    DataLoader(FailingDataset(), num_workers: 2).map: $it
  rescue TypeError |error|:
    failed = error.message == "dataset worker failed"
  check(failed, "dataset worker exception reaches consumer unchanged")
  [:worker, :consumer, :early].each |mode|:
    started = Channel.new(capacity: 1)
    active = Atomic.new(0)
    collate = |samples|:
      if samples[0] == 0:
        started.recv(timeout: 2.0)
        raise ValueError("collate worker failed") if mode == :worker
        samples
      else:
        active.update: $it + 1
        try:
          started.send(true)
          task.sleep(30.0)
          samples
        ensure:
          active.update: $it - 1
    loader = DataLoader(ItemsDataset([0, 1]), collate: &collate, num_workers: 2, prefetch_factor: 1)
    caught = false
    try:
      if mode == :early:
        check(_first(loader) == [0], "early return produces first batch")
      else:
        loader.each |batch|: raise ValueError("consumer failed")
    rescue ValueError |error|:
      expected = if (mode == :worker) then "collate worker failed" else "consumer failed"
      caught = error.message == expected
    check(caught or mode == :early, "worker and consumer errors survive cleanup")
    check(active.get == 0, "sleeping sibling is cancelled and joined before iteration exits")

def _validation():
  dataset = ItemsDataset([0])
  [{num_workers: -1}, {num_workers: 1.5}, {prefetch_factor: 0}, {prefetch_factor: 1.5}].each |options|:
    refused = false
    try:
      DataLoader(dataset, **options)
    rescue ValueError:
      refused = true
    rescue TypeError:
      refused = true
    check(refused, "worker and prefetch configuration validation")

def main():
  _ordering()
  _references()
  _parallel_and_bounded()
  _streaming()
  _failures_and_cleanup()
  _validation()
  print "PASS threaded DataLoader: references, ordering, bounded prefetch, cleanup"
  0
