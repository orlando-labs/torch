# Notebook Package Updates

Notebook preparation publishes one versioned generation under
`build/notebook/<fingerprint>/`: the complete production module map, source
snapshots (including nested `nn.utils` modules and MNIST helpers), native
library and input/artifact hashes. The current descriptor is
`build/notebook/current.json`. Test oracle sources are never included.

Prepare and register an existing notebook in one step:

```sh
./tools/build --prepare-notebook --notebook-project /path/to/My.sputnikbook
```

The option is repeatable. Preparation updates only the matching existing
native dependency's manifest, library and SHA-256. It preserves cells,
formatting, inputs, boards, other dependencies and unknown project metadata,
using Sputnik's locked atomic project save. Source-only links are never
silently converted to native dependencies. Preparation does not execute cells
or load a library, and it does not bypass execution-time native trust.

Registered projects are refreshed after subsequent successful `tools/build`
runs, including VM/native tests. An unchanged package reuses its bundle and
does not rewrite projects. MNIST's notebook generator uses the current
descriptor and automatically registers newly created projects. Other notebook
generators can use the shared `prepared_native_dependency` and
`register_project_native_updates` helpers.

Before rebuilding, save edits and leave notebook execution idle. After a
dependency update, reopen the existing project so its worker and project-save
baseline adopt the new dependency metadata. No new notebook is needed, and
no SHA-256 needs manual editing. Already running workers are not restarted
or replayed automatically. Their old library/source generation is retained;
preparation never replaces the library they have mapped.

Apply reloads a prepared source snapshot, not an arbitrary recursive directory
scan or an automatic compiler. Preparing a changed package produces its new
snapshot. Source edits/additions during compilation reject publication;
modified cached manifests, source snapshots or libraries are rejected. A
failed build keeps the published descriptor and previous project link.

`tests/notebook_bundle.cmake` covers nested additions, immutable old generations,
automatic registered-project updates, idempotence and failure preservation.
Sputnik's C++ notebook dependency tests cover metadata/number preservation,
path containment, malformed descriptors, hash checks and non-execution.
