package torch.nn.ctc_loss

from torch.nn.module import Module
from torch.nn.functional import ctc_loss

export CTCLoss

class CTCLoss < Module:
  def init(@blank: 0, @reduction: :mean, @zero_infinity: false): _initialize_registry!

  def forward(log_probs, targets, input_lengths, target_lengths):
    ctc_loss(log_probs, targets, input_lengths, target_lengths,
      blank: @blank, reduction: @reduction, zero_infinity: @zero_infinity)
