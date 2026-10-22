package torch.optim.adam

from torch.optim.optimizer import Optimizer, validate_coefficients
from torch.tensor.factories import zeros_like
from torch.tensor.value import Tensor
from torch.tensor.foreach import binary, unary, mutate!, gradients, packable?, independent?, pack, unpack

export Adam

class Adam < Optimizer:
  def init(parameters, lr: 0.001, betas: (0.9, 0.999), eps: 1e-8,
      weight_decay: 0.0, decoupled_weight_decay: false):
    _initialize_adam!(parameters, lr, betas, eps, weight_decay, decoupled_weight_decay)

  def _initialize_adam!(parameters, lr, betas, eps, weight_decay, decoupled):
    if betas.count != 2 or betas[0] < 0 or betas[0] >= 1 or betas[1] < 0 or betas[1] >= 1:
      raise ValueError("Adam betas must lie in [0, 1)")
    raise ValueError("invalid Adam epsilon or weight decay") if eps <= 0 or weight_decay < 0
    validate_coefficients([*betas, eps, weight_decay])
    @betas = betas.tuple
    @eps = eps
    @weight_decay = weight_decay
    @decoupled_weight_decay = decoupled
    _initialize_optimizer!(parameters, lr)
    _initialize_moments!

  def _initialize_moments!():
    @first = @parameters.map: zeros_like($it)._core
    @second = @parameters.map: zeros_like($it)._core
    @steps = Array.filled(@parameters.count, 0)
    @uniform_step = 0
    @packable = packable?(@cores)
    @independent = independent?(@cores)
    @packed_first = null
    @packed_second = null
    self

  def state_dict():
    first_cores = if null === @packed_first then @first else unpack(@packed_first, @cores)
    second_cores = if null === @packed_second then @second else unpack(@packed_second, @cores)
    first = first_cores.map: Tensor($it).snapshot
    second = second_cores.map: Tensor($it).snapshot
    {kind: "Adam", lr: @lr, betas: @betas.tuple, eps: @eps,
      weight_decay: @weight_decay, decoupled: @decoupled_weight_decay,
      first:, second:, steps: @steps.copy}

  def load_state_dict!(state):
    if state[:kind] != "Adam" or state[:decoupled] != @decoupled_weight_decay:
      raise ValueError("optimizer kind/decay policy mismatch")
    betas = state[:betas]
    if betas.count != 2 or betas[0] < 0 or betas[0] >= 1 or betas[1] < 0 or betas[1] >= 1:
      raise ValueError("invalid Adam betas")
    raise ValueError("invalid Adam coefficients") if state[:eps] <= 0 or state[:weight_decay] < 0
    validate_coefficients([*betas, state[:eps], state[:weight_decay]])
    _validate_slots(state[:first])
    _validate_slots(state[:second])
    raise ValueError("optimizer step count mismatch") if state[:steps].count != @parameters.count
    state[:steps].each |step|:
      raise ValueError("invalid optimizer step") if not (Int === step) or step < 0
    first = Array.of(@parameters.count) |index|:
      state[:first][index].to(device: @parameters[index].device).copy._core
    second = Array.of(@parameters.count) |index|:
      state[:second][index].to(device: @parameters[index].device).copy._core
    lr!(state[:lr])
    @betas = betas.tuple
    @eps = state[:eps]
    @weight_decay = state[:weight_decay]
    @first = first
    @second = second
    @steps = state[:steps].copy
    _refresh_uniform_step!
    @packed_first = null
    @packed_second = null
    self

  def step!():
    _run_step!():
      all_gradients = gradients(@cores)
      if not @independent:
        _step_serial!(all_gradients)
        _refresh_uniform_step!
      else if not all_gradients.empty? and not all_gradients.contains?(null) and not (null === @uniform_step):
        step = @uniform_step + 1
        first_bias = 1 - @betas[0] ** step
        second_bias = 1 - @betas[1] ** step
        if @packable:
          _step_packed!(all_gradients, first_bias, second_bias)
        else:
          moments = _update!(@cores, all_gradients, @first, @second, first_bias, second_bias)
          @first = moments[0]
          @second = moments[1]
        @steps = Array.filled(@steps.count, step)
        @uniform_step = step
      else:
        _materialize_moments!
        _step_selected!(all_gradients)
        _refresh_uniform_step!
    self

  def _refresh_uniform_step!():
    if @steps.empty?:
      @uniform_step = 0
    else:
      first = @steps[0]
      uniform = @steps.all?: $it == first
      @uniform_step = if uniform then first else null

  def _materialize_moments!():
    unless null === @packed_first:
      @first = unpack(@packed_first, @cores)
      @second = unpack(@packed_second, @cores)
      @packed_first = null
      @packed_second = null

  def _step_packed!(all_gradients, first_bias, second_bias):
    if null === @packed_first:
      @packed_first = pack(@first)
      @packed_second = pack(@second)
      @first = []
      @second = []
    all_gradients = _decay!(@cores, all_gradients)
    moments = _moments!([pack(all_gradients)], [@packed_first], [@packed_second], first_bias, second_bias)
    @packed_first = moments[0][0]
    @packed_second = moments[1][0]
    mutate!(@cores, unpack(moments[2][0], @cores), "sub")

  def _step_serial!(all_gradients):
    # Distinct leaf tensors can alias the same storage. Preserve update order
    # for this case rather than applying all decay operations before updates.
    all_gradients.each_with_index |gradient, index|:
      if null === gradient:
        next null
      step = @steps[index] + 1
      moments = _update!([@cores[index]], [gradient], [@first[index]], [@second[index]],
        1 - @betas[0] ** step, 1 - @betas[1] ** step)
      @first[index] = moments[0][0]
      @second[index] = moments[1][0]
      @steps[index] = step

  def _step_selected!(all_gradients):
    indices = []
    parameters = []
    active_gradients = []
    first = []
    second = []
    first_bias = []
    second_bias = []
    corrections = {}
    all_gradients.each_with_index |gradient, index|:
      if null === gradient:
        next null
      indices.push!(index)
      parameters.push!(@cores[index])
      active_gradients.push!(gradient)
      first.push!(@first[index])
      second.push!(@second[index])
      step = @steps[index] + 1
      @steps[index] = step
      unless corrections.has_key?(step):
        corrections[step] = (1 - @betas[0] ** step, 1 - @betas[1] ** step)
      first_bias.push!(corrections[step][0])
      second_bias.push!(corrections[step][1])
    unless indices.empty?:
      moments = _update!(parameters, active_gradients, first, second, first_bias, second_bias)
      indices.each_with_index |index, position|:
        @first[index] = moments[0][position]
        @second[index] = moments[1][position]

  def _update!(parameters, active_gradients, first, second, first_bias, second_bias):
    active_gradients = _decay!(parameters, active_gradients)
    moments = _moments!(active_gradients, first, second, first_bias, second_bias)
    mutate!(parameters, moments[2], "sub")
    moments

  def _decay!(parameters, active_gradients):
    if @decoupled_weight_decay:
      mutate!(parameters, 1 - @lr * @weight_decay, "mul")
    else if @weight_decay != 0:
      active_gradients = binary(active_gradients, binary(parameters, @weight_decay, "mul"), "add")
    active_gradients

  def _moments!(active_gradients, first, second, first_bias, second_bias):
    first = binary(binary(first, @betas[0], "mul"), binary(active_gradients, 1 - @betas[0], "mul"), "add")
    second = binary(binary(second, @betas[1], "mul"),
      binary(unary(active_gradients, "square"), 1 - @betas[1], "mul"), "add")
    corrected_first = binary(first, first_bias, "div")
    corrected_second = binary(second, second_bias, "div")
    denominator = binary(unary(corrected_second, "sqrt"), @eps, "add")
    update = binary(binary(corrected_first, denominator, "div"), @lr, "mul")
    moments = (first, second, update)
    moments
