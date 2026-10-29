# torch

Use Sputnik for model composition, optimizer/scheduler algorithms and data
pipelines. Keep C++ at the LibTorch kernel, lifetime and serialization boundary.
Python must never be a package, build, code generation or test dependency.

## Sputnik style

- Keep each public Torch layer/container/value class in its own `.s` file.
  Facades import and reexport; native fault declarations keep their ABI hierarchy.
- Separate sibling functions/methods with a blank line whenever either is
  multiline. Only genuine one-line definitions may be adjacent.
- Forward same-named locals with keyword shorthand: `layer(x, mask:, causal:)`.
  Keep explicit values for renamed arguments, expressions and instance fields.
- Use postfix `if`/`unless` for simple guards and conditional assignments.
  Use a multiline suite when control flow or a block would make the modifier
  ambiguous. Preserve conditional execution and exception behavior.
- Use bare calls for nullary methods: `x.shape`, `x.square`,
  `optimizer.zero_grad!`, `batches.each |batch|:`. This includes mutating
  methods and calls accepting a block. Keep `!` for mutation. Sputnik
  permits this form whenever a method accepts zero arguments, including
  default-argument methods and empty rest parameters. Bare function-valued
  identifiers also invoke, including imports, parameters and callback locals.
  Use `&f` to read/pass/store the function without invoking; use `apply((&f))`
  for a positional argument, distinct from block-pass `apply(&f)`. Constructors and
  declarations retain `()`.
- Prefer blocks, `times`, `Array.of`, tuple patterns and collection operations
  over manual counters when they make the algorithm clearer. Do not add
  indirection solely to shorten a line. Break long argument lists naturally.

## Verification

Native builds require `--require-full-native`. Do not replace unsupported
Sputnik syntax with opaque C++ algorithms just to make the build pass.
Run affected VM/native tests and actual MPS tests for backend-sensitive work.
MPS tests disable automatic CPU fallback. Keep the verified LibTorch version
distinct from a version merely accepted by the bootstrap.

The neighboring `sputnik-lang` may have unrelated working-tree changes;
preserve them. Builds using `build/sputnik` must run sequentially.
