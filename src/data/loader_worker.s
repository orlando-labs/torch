package sputniktorch.data.loader_worker

import task
from sync import Channel, Mutex

export LoaderWorker

# Channels carry notifications; mutable jobs and batches cross by reference
# through queues protected by the same mutex on both sides.
class LoaderWorker:
  def init(capacity, @fetch):
    @ready = Channel.new(capacity:)
    @mutex = Mutex.new()
    @inputs = []
    @outputs = []
    @handle = null
    @running = false

  def submit!(job):
    @handle.wait if @ready.closed?
    start = @mutex.synchronize:
      @inputs.push!(job)
      idle = not @running
      @running = true
      idle
    if start:
      # Idle lanes retire their task instead of occupying a scheduler thread
      # while the consumer trains. Rejoin before restarting the lane.
      @handle.wait if @handle != null
      @handle = task.spawn: _run

  def next!():
    try:
      @ready.recv
    rescue ChannelClosedError:
      @handle.wait
    @mutex.synchronize: @outputs.shift!

  def cancel!():
    @handle.cancel if @handle != null
    @ready.close

  def join!():
    try:
      @handle.wait if @handle != null
    rescue:
      # Cleanup must preserve an exception or nonlocal exit from the consumer.
      null

  def _run():
    try:
      job = _take_job!
      while job != null:
        batch = @fetch(job)
        @mutex.synchronize: @outputs.push!(batch)
        @ready.send(true)
        job = _take_job!
    rescue |error|:
      @ready.close
      raise error

  def _take_job!():
    @mutex.synchronize:
      if @inputs.empty?:
        @running = false
        null
      else:
        @inputs.shift!
