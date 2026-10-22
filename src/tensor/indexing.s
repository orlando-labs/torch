package torch.tensor.indexing

from torch.tensor.slice import Slice

export Slice, index_select, gather, take, take_along_dim, put, index_put
export scatter, scatter_add, scatter_reduce, index_add, index_copy, index_fill, index_reduce
export masked_select, masked_fill, masked_scatter, nonzero, argwhere
export select_scatter, slice_scatter, diagonal_scatter

def index_select(input, dim, index): input.index_select(dim, index)
def gather(input, dim, index, sparse_grad: false): input.gather(dim, index, sparse_grad:)
def take(input, index): input.take(index)
def take_along_dim(input, indices, dim: null): input.take_along_dim(indices, dim:)
def put(input, index, source, accumulate: false): input.put(index, source, accumulate:)
def index_put(input, indices, values, accumulate: false): input.index_put(indices, values, accumulate:)
def scatter(input, dim, index, source, reduce: null): input.scatter(dim, index, source, reduce:)
def scatter_add(input, dim, index, source): input.scatter_add(dim, index, source)

def scatter_reduce(input, dim, index, source, reduce, include_self: true):
  input.scatter_reduce(dim, index, source, reduce, include_self:)

def index_add(input, dim, index, source, alpha: 1): input.index_add(dim, index, source, alpha:)
def index_copy(input, dim, index, source): input.index_copy(dim, index, source)
def index_fill(input, dim, index, value): input.index_fill(dim, index, value)

def index_reduce(input, dim, index, source, reduce, include_self: true):
  input.index_reduce(dim, index, source, reduce, include_self:)

def masked_select(input, mask): input.masked_select(mask)
def masked_fill(input, mask, value): input.masked_fill(mask, value)
def masked_scatter(input, mask, source): input.masked_scatter(mask, source)
def nonzero(input, as_tuple: false): input.nonzero(as_tuple:)
def argwhere(input): input.nonzero
def select_scatter(input, source, dim, index): input.select_scatter(source, dim, index)

def slice_scatter(input, source, dim: 0, start: null, end: null, step: 1):
  input.slice_scatter(source, dim:, start:, end:, step:)

def diagonal_scatter(input, source, offset: 0, dim1: 0, dim2: 1):
  input.diagonal_scatter(source, offset:, dim1:, dim2:)
