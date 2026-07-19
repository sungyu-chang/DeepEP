# Repository guidance

This file applies to the entire repository. DeepEP is a performance-sensitive PyTorch/CUDA extension for MoE dispatch and combine. Treat kernel ownership, synchronization, memory ordering, buffer layouts, and launch geometry as API contracts; a change that produces correct values once can still be wrong because it races, hangs under skew, changes layout metadata, or regresses latency.

## Repository map

- `deep_ep/`: Python API and buffer/event wrappers.
- `csrc/deep_ep.cpp` and `csrc/deep_ep.hpp`: PyTorch bindings, tensor validation, stream handling, and low-latency hook launch sequencing.
- `csrc/kernels/internode_ll.cu`: pure-RDMA low-latency dispatch/combine kernels, including the warp-balanced work division described below.
- `csrc/kernels/internode.cu` and `intranode.cu`: normal internode and intranode kernels.
- `csrc/kernels/launch.cuh`: cooperative launch and cluster configuration. `cg::this_grid().sync()` is valid only because these launches are cooperative.
- `csrc/kernels/configs.cuh`, `utils.cuh`, and `ibgda_device.cuh`: phase flags, CUDA/PTX utilities, and IBGDA primitives.
- `tests/test_low_latency.py`: low-latency correctness, hook/non-hook, format, zero-copy, repeatability, and performance coverage.
- `tests/test_internode.py` and `tests/test_intranode.py`: normal-kernel distributed tests.
- `tests/utils.py`: distributed initialization and benchmark helpers; adapt its cluster settings when necessary.

## Build and run

The low-latency path requires CUDA, PyTorch, RDMA-capable GPUs, and NVSHMEM. A build without NVSHMEM disables internode and low-latency sources, so it does not validate changes to `internode_ll.cu`.

```bash
NVSHMEM_DIR=/path/to/nvshmem TORCH_CUDA_ARCH_LIST=9.0 python setup.py build
```

`DISABLE_SM90_FEATURES=1` is for SM80/CUDA 11-style builds and requires NVSHMEM to be disabled. `DISABLE_AGGRESSIVE_PTX_INSTRS=1` is required for architectures that do not support the aggressive load/store PTX variants. The CMake files under `csrc/` are for debugging; `setup.py` is the primary build.

Tests are executable distributed scripts rather than a conventional CPU-only unit suite:

```bash
python tests/test_intranode.py
python tests/test_internode.py
python tests/test_low_latency.py --num-processes 8
```

For multi-node runs, `tests/utils.py` reads `MASTER_ADDR`, `MASTER_PORT`, `WORLD_SIZE` (node count), and `RANK` (node rank), then spawns `--num-processes` local workers. Report the GPU model, CUDA/PyTorch/NVSHMEM versions, rank count, expert count, and relevant flags with performance or hang results.

## Warp-balanced low-latency work division

The baseline low-latency mapping assigns one expert slot to each warp group:

```cpp
responsible_expert_idx = sm_id * num_warp_groups + warp_group_id;
```

Preserve that mapping whenever spread mode is inactive. In particular, the fused send-and-receive launch used when `return_recv_hook=False` must retain the original path.

`return_recv_hook=True` splits each operation in `csrc/deep_ep.cpp`: the initial launch runs `LOW_LATENCY_SEND_PHASE`, and the returned hook later runs `LOW_LATENCY_RECV_PHASE`. Warp balancing is intentionally enabled only for the two imbalanced halves of those split launches:

- Dispatch receive-only: `num_warp_groups == 1`, no send phase, and `num_device_sms > num_experts`.
- Combine send-only: `num_warp_groups == 1`, `phases == LOW_LATENCY_SEND_PHASE`, and `num_device_sms > num_experts`.
- Dispatch send, combine receive, and all fused launches remain on the baseline mapping.

The host computes `spread = num_device_sms / num_experts` and launches `num_experts * spread` blocks. The device must use the matching mapping:

```cpp
expert_idx = sm_id / spread;
sub_block = sm_id % spread;
group_warp_rank = sub_block * num_warps_per_group + sub_warp_id;
group_warp_count = spread * num_warps_per_group;
```

Keep the host and device activation predicates and integer mapping in lockstep. The floor division is deliberate: it keeps the grid within the available SM count and leaves only the remainder unused. Do not silently extend spread mode to phase combinations that have not been designed and tested for it.

### Dispatch receive invariants

- `rdma_recv_count` uses zero as “not ready” and encodes an arrived count as `-num_tokens - 1`. Read it with system-scope acquire semantics.
- Cooperating blocks cannot use an atomic allocation order for packed offsets. Every block computes the same deterministic prefix over source ranks for its `(local_expert, src_rank)` slot.
- Only `sub_block == 0` writes `packed_recv_layout_range` and updates diagnostic statistics. Statistics must be counted once, not once per cooperating block.
- Only the `src_rank == 0`, `sub_block == 0` owner writes the total `packed_recv_count` for a local expert, after observing all source-rank counts.
- Token data, source indices, and FP8/UE8M0 scales use the aggregate warp rank/count so each token is copied by exactly one warp.
- Preserve the non-spread `atomicAdd` layout path unless a change explicitly intends to alter baseline ordering and has corresponding tests.

### Combine send invariants

- Cooperating blocks stride the expert's token range by the aggregate warp rank/count; each token must be sent exactly once.
- A receiver completion flag means all sends for that expert are complete. In spread mode, every thread must reach the cooperative grid barrier before signaling.
- After the grid barrier, only the `sub_block == 0` leader signals the expert and decrements `atomic_clean_flag`. There must be exactly one signal/decrement per expert, including experts with zero tokens.
- Do not place an early return or divergent control flow around a grid-wide or block-wide barrier. `bar.sync` IDs and participant counts are local to a block and must continue to match the warp-group layout.
- Preserve both the IBGDA remote atomic path and the NVLink/P2P system-release store path.

## Change and test expectations

When changing low-latency ownership or synchronization, test all affected launch modes, not only the new fast path:

1. Run with `return_recv_hook=False` to cover the fused baseline.
2. Run with `return_recv_hook=True` to cover split send/receive launches.
3. Use an expert count divisible by the rank count and smaller than the device SM count so `spread > 1`; for an 8-rank setup, `--num-experts 16 --num-topk 8` is a useful starting point.
4. Also use a configuration where spread mode is inactive (`num_experts >= num_device_sms` or `num_warp_groups > 1`).
5. Exercise skewed routing in which one or a few expert slots receive most tokens. Balanced random routing alone does not reproduce the target bottleneck.
6. Check BF16 and FP8 dispatch, rounded scales and UE8M0 when supported, combine LogFMT, and zero-copy for changes that touch their shared loops or layouts.
7. Verify received counts, per-rank layout ranges, source metadata, cumulative stats, numerical output, repeat hashes, and absence of hangs. Performance work should include warmups and before/after latency from the same hardware and topology.

`tests/test_low_latency.py --pressure-test` repeats seeds and hashes outputs; use it for race-sensitive changes when hardware time permits. If the required multi-GPU/RDMA environment is unavailable, perform static/build checks that are possible and state clearly that the low-latency CUDA path was not executed.

## Coding conventions

- Follow nearby C++/CUDA style: four-space indentation, braces on the same line, C++ alternative tokens (`and`, `or`, `not`), and `EP_HOST_ASSERT`, `EP_DEVICE_ASSERT`, or `EP_STATIC_ASSERT` for invariants.
- Reuse the established vectorized copy, acquire/release, NVSHMEM/IBGDA, TMA, and launch helpers. Do not replace them with ordinary loads/stores or generic CUDA launches without proving equivalent scope and ordering.
- Keep comments focused on ownership, synchronization, layout, and why a single writer/signaler is safe.
- Avoid broad formatting or unrelated cleanup in kernel patches. Small diffs make instruction scheduling and baseline-path changes reviewable.
- Python follows the existing typed, direct-script test style. Add assertions that expose counts, offsets, duplicated work, and determinism rather than checking only approximate output values.
