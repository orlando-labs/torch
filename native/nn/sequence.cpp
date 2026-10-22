#include "tensor/kernel_args.h"

extern "C" SputnikStatus amt_sequence(SputnikCtx *cx, const SputnikValue *a,
                                      size_t n, SputnikValue *out) {
  return sputnik_torch::boundary(cx, [&] {
    using namespace sputnik_torch;
    argc_check(n, 4);
    ExecutionGuard guard(integer(cx, a[3]));
    KernelArgs args(cx, a[1], a[2]);
    const auto op = string(cx, a[0]);
    if (op == "pack") {
      *out = args.result(at::_pack_padded_sequence(args.tensor(0), args.tensor(1), args.boolean(0)));
    } else if (op == "unpack") {
      *out = args.result(at::_pad_packed_sequence(args.tensor(0), args.tensor(1), args.boolean(0),
        numeric_scalar(cx, args.arg(1)), args.integer(2)));
    } else if (op == "pad") {
      std::vector<at::Tensor> sequences;
      for (size_t i = 0; i < sputnik_list_len(cx, a[1]); ++i) sequences.push_back(args.tensor(i));
      *out = args.result(at::pad_sequence(sequences, args.boolean(0), args.number(1), args.string(2)));
    } else {
      throw Error("ArgumentError", "unknown sequence primitive");
    }
  });
}
