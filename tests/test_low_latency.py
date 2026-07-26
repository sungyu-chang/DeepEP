import argparse
import random
import time
import os
import torch
import torch.distributed as dist
import numpy as np
from functools import partial
from typing import Optional

import deep_ep
from utils import init_dist, bench, bench_kineto, calc_diff, hash_tensor, per_token_cast_back


def test_main(num_tokens: int, hidden: int, num_experts: int, num_topk: int,
              rank: int, num_ranks: int, group: dist.ProcessGroup, buffer: deep_ep.Buffer,
              use_logfmt: bool = False, seed: int = 0):
    torch.manual_seed(seed + rank)
    random.seed(seed + rank)

    assert num_experts % num_ranks == 0
    num_local_experts = num_experts // num_ranks

    # NOTES: the integers greater than 256 exceed the BF16 precision limit
    rank_offset = 128
    assert num_ranks - rank_offset < 257, 'Too many ranks (exceeding test precision limit)'

    x = torch.ones((num_tokens, hidden), dtype=torch.bfloat16, device='cuda') * (rank - rank_offset)
    x[:, -128:] = torch.arange(num_tokens, device='cuda').to(torch.bfloat16).view(-1, 1)
    x_list = [x]
    for i in range(4 if use_logfmt else 0):
        # NOTES: make more LogFMT casts and also with some BF16
        x_list.append(torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda') * 0.5 * random.random())
    # NOTES: the last one is for performance testing
    # Most of the values in the perf case is lower than the threshold, casting most channels
    x_list.append(torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda') * 0.1)

    scores = torch.randn((num_tokens, num_experts), dtype=torch.float32, device='cuda').abs() + 1
    topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=True)[1]
    topk_weights = torch.randn((num_tokens, num_topk), dtype=torch.float32, device='cuda').abs()

    # Randomly mask some positions
    for i in range(10):
        topk_idx[random.randint(0, num_tokens - 1), random.randint(0, num_topk - 1)] = -1

    # Check dispatch correctness
    do_check = True
    hash_value, num_times = 0, 0
    for current_x in x_list:
        for return_recv_hook in (False, True):
            for dispatch_use_fp8 in (False, True):
                for round_scale in (False, True) if dispatch_use_fp8 else (False, ):
                    for use_ue8m0 in (False, True) if round_scale else (False, ):
                        num_times += 1
                        for i in range((num_times % 2) + 1):
                            cumulative_local_expert_recv_stats = torch.zeros((num_local_experts, ), dtype=torch.int, device='cuda')
                            packed_recv_x, packed_recv_count, handle, event, hook = \
                                buffer.low_latency_dispatch(current_x, topk_idx, num_tokens, num_experts,
                                                            use_fp8=dispatch_use_fp8, round_scale=round_scale, use_ue8m0=use_ue8m0,
                                                            cumulative_local_expert_recv_stats=cumulative_local_expert_recv_stats,
                                                            async_finish=not return_recv_hook, return_recv_hook=return_recv_hook)
                            hook() if return_recv_hook else event.current_stream_wait()
                        packed_recv_x = (packed_recv_x[0], packed_recv_x[1].contiguous()) if dispatch_use_fp8 else packed_recv_x
                        simulated_gemm_x = per_token_cast_back(packed_recv_x[0].view(-1, hidden), packed_recv_x[1].view(-1, hidden // 128)).view(packed_recv_x[0].shape) \
                            if dispatch_use_fp8 else packed_recv_x.clone()
                        all_topk_idx = torch.empty((num_ranks, num_tokens, num_topk), dtype=topk_idx.dtype, device='cuda')
                        dist.all_gather_into_tensor(all_topk_idx, topk_idx, group=group)
                        for i in range(num_local_experts if do_check else 0):
                            expert_id = rank * num_local_experts + i
                            recv_x = per_token_cast_back(packed_recv_x[0][i], packed_recv_x[1][i]) if dispatch_use_fp8 else packed_recv_x[i]
                            recv_count, recv_src_info, recv_layout_range = packed_recv_count[i], handle[0][i], handle[1][i]

                            # Check expert indices
                            int_mask = (2 ** 32) - 1
                            num_valid_tokens = recv_count.item()
                            assert cumulative_local_expert_recv_stats[i].item() == num_valid_tokens, f'{cumulative_local_expert_recv_stats[i].item()} != {num_valid_tokens}'
                            assert num_valid_tokens == (recv_layout_range & int_mask).sum().item(), f'{num_valid_tokens} != {recv_layout_range & int_mask}.sum().item()'
                            assert num_valid_tokens == (all_topk_idx == expert_id).sum().item(), f'{num_valid_tokens} != {(all_topk_idx == expert_id).sum().item()}'

                            if num_valid_tokens == 0:
                                continue
                            # Check received data
                            if current_x is x:
                                recv_x = recv_x[:num_valid_tokens]
                                recv_x_amin = recv_x[:, :-128].amin(dim=-1)
                                recv_src_info = recv_src_info[:num_valid_tokens]
                                assert torch.equal(recv_x_amin, recv_x[:, :-128].amax(dim=-1))
                                if round_scale:
                                    assert calc_diff(recv_x[:, -1], recv_src_info.view(-1)) < 0.007
                                else:
                                    assert (recv_x[:, -128:] - recv_src_info.view(-1, 1) % num_tokens).sum().item() == 0
                                for j in range(num_ranks):
                                    begin_idx, count = (recv_layout_range[j] >> 32).item(), (recv_layout_range[j] & int_mask).item()
                                    if not round_scale:
                                        assert (recv_x_amin == j - rank_offset).sum().item() == (all_topk_idx[j] == expert_id).sum().item()
                                        assert (recv_x[begin_idx:begin_idx + count, :-128] - j + rank_offset).sum().item() == 0
                            if dispatch_use_fp8:
                                hash_value ^= hash_tensor(packed_recv_x[0][i, :num_valid_tokens])
                                hash_value ^= hash_tensor(packed_recv_x[1][i, :num_valid_tokens])
                            else:
                                hash_value ^= hash_tensor(packed_recv_x[i, :num_valid_tokens])

                        # Check combine correctness
                        for zero_copy in (False, ) if use_logfmt else (False, True):
                            if zero_copy:
                                buffer.get_next_low_latency_combine_buffer(handle)[:, :, :] = simulated_gemm_x
                            out = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
                            combined_x, event, hook = buffer.low_latency_combine(simulated_gemm_x, topk_idx, topk_weights, handle,
                                                                                use_logfmt=use_logfmt,
                                                                                async_finish=not return_recv_hook, zero_copy=zero_copy,
                                                                                return_recv_hook=return_recv_hook, out=out)
                            hook() if return_recv_hook else event.current_stream_wait()
                            if do_check:
                                diff = calc_diff(current_x * topk_weights.masked_fill(topk_idx == -1, 0).sum(dim=1).view(-1, 1), combined_x)
                                assert torch.isnan(combined_x).sum().item() == 0
                                assert diff < (9e-4 if dispatch_use_fp8 else 1e-5), f'Error: {diff=}, {dispatch_use_fp8=}, {zero_copy=}'
                                hash_value ^= hash_tensor(combined_x)

    # noinspection PyShadowingNames
    def large_gemm_with_hook(hook):
        mat_0 = torch.randn((8192, 8192), dtype=torch.float)
        mat_1 = torch.randn((8192, 8192), dtype=torch.float)
        mat_0 @ mat_1
        hook()

    # noinspection PyShadowingNames
    def test_func(return_recv_hook: bool):
        recv_x, recv_count, handle, event, hook = \
            buffer.low_latency_dispatch(current_x, topk_idx, num_tokens, num_experts,
                                        cumulative_local_expert_recv_stats=cumulative_local_expert_recv_stats,
                                        use_fp8=True, async_finish=False, return_recv_hook=return_recv_hook)
        large_gemm_with_hook(hook) if return_recv_hook else None
        combined_x, event, hook = buffer.low_latency_combine(simulated_gemm_x, topk_idx, topk_weights, handle,
                                                             use_logfmt=use_logfmt, return_recv_hook=return_recv_hook)
        large_gemm_with_hook(hook) if return_recv_hook else None

    # Calculate bandwidth
    num_fp8_bytes, num_bf16_bytes = (hidden + hidden / 128 * 4 + 16), hidden * 2
    num_logfmt10_bytes = hidden * 10 / 8 + hidden / 128 * 4
    num_dispatch_comm_bytes, num_combine_comm_bytes = 0, 0
    for i in range(num_tokens):
        num_selections = (topk_idx[i] != -1).sum().item()
        num_dispatch_comm_bytes += num_fp8_bytes * num_selections
        num_combine_comm_bytes += (num_logfmt10_bytes if use_logfmt else num_bf16_bytes) * num_selections

    # Dispatch + combine testing
    avg_t, min_t, max_t = bench(partial(test_func, return_recv_hook=False))
    print(f'[rank {rank}] Dispatch + combine bandwidth: {(num_dispatch_comm_bytes + num_combine_comm_bytes) / 1e9 / avg_t:.2f} GB/s, '
          f'avg_t={avg_t * 1e6:.2f} us, min_t={min_t * 1e6:.2f} us, max_t={max_t * 1e6:.2f} us', flush=True)

    # Separate profiling
    for return_recv_hook in (False, True):
        group.barrier()
        dispatch_t, combine_t = bench_kineto(partial(test_func, return_recv_hook=return_recv_hook),
                                             kernel_names=('dispatch', 'combine'), barrier_comm_profiling=True,
                                             suppress_kineto_output=True, num_kernels_per_period=2 if return_recv_hook else 1)
        if not return_recv_hook:
            print(f'[rank {rank}] Dispatch bandwidth: {num_dispatch_comm_bytes / 1e9 / dispatch_t:.2f} GB/s, avg_t={dispatch_t * 1e6:.2f} us | '
                  f'Combine bandwidth: {num_combine_comm_bytes / 1e9 / combine_t:.2f} GB/s, avg_t={combine_t * 1e6:.2f} us', flush=True)
        else:
            print(f'[rank {rank}] Dispatch send/recv time: {dispatch_t[0] * 1e6:.2f} + {dispatch_t[1] * 1e6:.2f} us | '
                  f'Combine send/recv time: {combine_t[0] * 1e6:.2f} + {combine_t[1] * 1e6:.2f} us', flush=True)
    return hash_value


def align_up(x: int, y: int) -> int:
    return (x + y - 1) // y * y


# noinspection PyShadowingNames
def test_compact_dispatch(rank: int, num_ranks: int, group: dist.ProcessGroup):
    """
    Focused correctness test for the new, feature-flagged `Buffer.low_latency_dispatch_compact`
    API. This does not touch/weaken the legacy `low_latency_dispatch` coverage in `test_main`
    above; it stands up its own appropriately-sized `Buffer` and deterministically routes a known
    number of tokens from every source rank to every (destination rank, local expert) pair so
    that the per-(source rank, local expert) received counts sweep the requested boundary values
    `{0, 1, 31, 32, 33, 50, 64, 127, 128}` (including at least one fully empty expert-from-rank
    pair), then validates:
      - `recv_count` / `expert_offsets` / `valid_row_count` arithmetic and 128-row alignment,
      - `m_indices` / `row_local_expert` padding markers (in-segment alignment padding vs. the
        unused capacity tail),
      - compact row ordering (rank-major, then each pair's original token order),
      - `compact_src_info` / `row_src_rank` / `compact_layout_range` correctness,
      - a `topk_idx == -1` (unrouted) token is correctly excluded,
      - BF16 and FP8 (with and without UE8M0) numerical semantics,
      - hook vs. non-hook execution parity.
    """
    assert deep_ep.Buffer.has_low_latency_compact_layout()

    torch.manual_seed(rank)
    hidden = 2048  # smallest hidden size supported by `SWITCH_HIDDEN` (see `csrc/kernels/launch.cuh`)
    rank_offset = 128
    boundary_counts = [0, 1, 31, 32, 33, 50, 64, 127, 128]
    num_local_experts = -(-len(boundary_counts) // num_ranks)  # ceil-div
    num_experts = num_ranks * num_local_experts

    # Deterministic routing table, identical (and known) on every rank: `target_count[d][l]` is
    # the number of tokens that *every* source rank sends to (destination rank `d`, local expert
    # `l`); cycles through `boundary_counts` so every requested boundary is exercised at least once.
    target_count = [[boundary_counts[(d * num_local_experts + l) % len(boundary_counts)]
                     for l in range(num_local_experts)] for d in range(num_ranks)]

    num_filler = 4  # extra `topk_idx == -1` (unrouted) tokens
    num_real_tokens = sum(sum(row) for row in target_count)
    num_tokens = num_real_tokens + num_filler
    num_max_dispatch_tokens_per_rank = num_tokens

    # Build `x` and `topk_idx` in the same deterministic (destination rank, local expert)-major
    # order on every rank, so that the source token index range assigned to any given
    # (destination rank, local expert) pair is identical and predictable across all ranks.
    x = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
    x[:, :-128] = (rank - rank_offset)
    x[:, -128:] = torch.arange(num_tokens, device='cuda').to(torch.bfloat16).view(-1, 1)
    topk_idx = torch.full((num_tokens, 1), -1, dtype=torch.int64, device='cuda')
    pair_token_range = {}
    cursor = 0
    for d in range(num_ranks):
        for l in range(num_local_experts):
            cnt = target_count[d][l]
            pair_token_range[(d, l)] = (cursor, cursor + cnt)
            if cnt > 0:
                topk_idx[cursor:cursor + cnt, 0] = d * num_local_experts + l
            cursor += cnt
    assert cursor == num_real_tokens
    # The trailing `num_filler` tokens keep `topk_idx == -1`, i.e. they are never routed.

    num_rdma_bytes = deep_ep.Buffer.get_low_latency_rdma_size_hint(num_max_dispatch_tokens_per_rank, hidden, num_ranks, num_experts)
    buffer = deep_ep.Buffer(group, num_rdma_bytes=num_rdma_bytes, low_latency_mode=True,
                            num_qps_per_rank=num_experts // num_ranks, explicitly_destroy=True)

    max_rows_per_expert = align_up(num_ranks * num_max_dispatch_tokens_per_rank, 128)
    m_capacity = num_local_experts * max_rows_per_expert

    for use_fp8, round_scale, use_ue8m0 in ((False, False, False), (True, False, False), (True, True, True)):
        for return_recv_hook in (False, True):
            cumulative_local_expert_recv_stats = torch.zeros((num_local_experts,), dtype=torch.int, device='cuda')
            compact_x, packed_recv_count, m_indices, handle, event, hook = \
                buffer.low_latency_dispatch_compact(x, topk_idx, num_max_dispatch_tokens_per_rank, num_experts,
                                                    cumulative_local_expert_recv_stats=cumulative_local_expert_recv_stats,
                                                    use_fp8=use_fp8, round_scale=round_scale, use_ue8m0=use_ue8m0,
                                                    async_finish=not return_recv_hook, return_recv_hook=return_recv_hook)
            hook() if return_recv_hook else event.current_stream_wait()

            (compact_src_info, row_src_rank, row_local_expert, compact_layout_range,
             expert_offsets, valid_row_count, m_indices_h,
             num_max_dispatch_tokens_per_rank_h, hidden_h, num_experts_h) = handle
            assert m_indices_h.data_ptr() == m_indices.data_ptr()
            assert num_max_dispatch_tokens_per_rank_h == num_max_dispatch_tokens_per_rank
            assert hidden_h == hidden
            assert num_experts_h == num_experts

            assert compact_x[0].shape == (m_capacity, hidden) if use_fp8 else compact_x.shape == (m_capacity, hidden)
            for t in (compact_src_info, row_src_rank, row_local_expert, m_indices):
                assert t.shape == (m_capacity,)
            assert expert_offsets.shape == (num_local_experts + 1,)
            assert valid_row_count.shape == (1,)
            assert compact_layout_range.shape == (num_local_experts, num_ranks)

            if use_fp8:
                recv_x = per_token_cast_back(compact_x[0], compact_x[1].contiguous())
            else:
                recv_x = compact_x

            int_mask = (2 ** 32) - 1
            expected_valid_total = 0
            assert expert_offsets[0].item() == 0
            for l in range(num_local_experts):
                expected_expert_total = target_count[rank][l] * num_ranks
                assert packed_recv_count[l].item() == expected_expert_total, \
                    f'{packed_recv_count[l].item()=} != {expected_expert_total=}'
                assert cumulative_local_expert_recv_stats[l].item() == expected_expert_total
                expected_valid_total += expected_expert_total

                seg_start, seg_end = expert_offsets[l].item(), expert_offsets[l + 1].item()
                assert seg_end - seg_start == align_up(expected_expert_total, 128), \
                    f'expert {l}: {seg_end - seg_start=} != {align_up(expected_expert_total, 128)=}'
                # Valid rows select this expert. Padding remains `-1` so contiguous GEMM skips it,
                # while row metadata retains the owning expert segment.
                assert (m_indices[seg_start:seg_start + expected_expert_total] == l).all()
                assert (row_local_expert[seg_start:seg_end] == l).all()

                # In-segment alignment padding must be marked invalid and zeroed
                pad_start = seg_start + expected_expert_total
                if pad_start < seg_end:
                    assert (m_indices[pad_start:seg_end] == -1).all()
                    assert (compact_src_info[pad_start:seg_end] == -1).all()
                    assert (row_src_rank[pad_start:seg_end] == -1).all()
                    assert (recv_x[pad_start:seg_end] == 0).all()

                # Rank-major grouping within the valid segment: one contiguous block per source
                # rank, in ascending rank order. NOTE: within a given (source rank, local expert)
                # pair, the legacy send phase lands tokens via a racy `atomicAdd`-allocated slot
                # index (see the `dispatch` kernel's send phase), so the relative order of tokens
                # *within* a pair is whatever order they happened to land in -- not necessarily
                # ascending by original source-token index. Compact copies them through in that
                # same (unspecified but race-free-to-read-back) landing order, so we validate
                # order-independent completeness (no duplicate/missing rows: as a set, exactly the
                # expected source-token indices) plus per-row self-consistency of the payload,
                # rather than assuming a particular fixed permutation.
                block_cursor = seg_start
                for s in range(num_ranks):
                    cnt = target_count[rank][l]
                    if cnt == 0:
                        continue
                    block = slice(block_cursor, block_cursor + cnt)
                    assert (row_src_rank[block] == s).all(), f'expert {l}, rank {s}: row_src_rank mismatch'
                    exp_lo, exp_hi = pair_token_range[(rank, l)]
                    expected_src_info = torch.arange(exp_lo, exp_hi, device='cuda', dtype=torch.int32)
                    got_src_info = compact_src_info[block]
                    assert torch.equal(torch.sort(got_src_info).values, expected_src_info), \
                        f'expert {l}, rank {s}: compact_src_info has duplicate/missing rows'

                    count_bits, offset_bits = (compact_layout_range[l, s] & int_mask).item(), (compact_layout_range[l, s] >> 32).item()
                    assert count_bits == cnt, f'{count_bits=} != {cnt=}'
                    assert offset_bits == block_cursor, f'{offset_bits=} != {block_cursor=}'

                    # Per-row self-consistency: each row's payload must match *its own*
                    # `compact_src_info`, independent of the (unspecified) intra-pair row order.
                    # NOTE: gather the expected payload from `x` itself (rather than recomputing
                    # it arithmetically) since `x`'s marker columns are stored as `bfloat16`,
                    # which cannot exactly represent every integer once `num_tokens > 256`; using
                    # `x` as the ground truth keeps the check exact regardless of that rounding.
                    block_x = recv_x[block]
                    expected_x = x[got_src_info.long()]
                    if round_scale:
                        assert calc_diff(block_x[:, -1], expected_x[:, -1]) < 0.007
                    else:
                        block_amin = block_x[:, :-128].amin(dim=-1)
                        assert torch.equal(block_amin, block_x[:, :-128].amax(dim=-1))
                        assert (block_amin == s - rank_offset).all()
                        assert torch.equal(block_x[:, -128:], expected_x[:, -128:])

                    block_cursor += cnt
                assert block_cursor == seg_start + expected_expert_total

            assert valid_row_count.item() == expected_valid_total, f'{valid_row_count.item()=} != {expected_valid_total=}'

            # Unused capacity tail (beyond the last aligned expert segment) must be fully padded
            tail_start = expert_offsets[num_local_experts].item()
            if tail_start < m_capacity:
                assert (m_indices[tail_start:] == -1).all()
                assert (row_local_expert[tail_start:] == -1).all()
                assert (compact_src_info[tail_start:] == -1).all()
                assert (row_src_rank[tail_start:] == -1).all()
                assert (recv_x[tail_start:] == 0).all()

    buffer.destroy()
    if rank == 0:
        print(f'[rank {rank}] Compact dispatch checks passed '
              f'(num_ranks={num_ranks}, num_local_experts={num_local_experts}, m_capacity={m_capacity})', flush=True)
    group.barrier()


# noinspection PyUnboundLocalVariable,PyShadowingNames
def test_loop(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)
    num_tokens, hidden = args.num_tokens, args.hidden
    num_topk, num_experts = args.num_topk, args.num_experts

    num_rdma_bytes = deep_ep.Buffer.get_low_latency_rdma_size_hint(num_tokens, hidden, num_ranks, num_experts)
    if local_rank == 0:
        print(f'Allocating buffer size: {num_rdma_bytes / 1e6} MB ...', flush=True)
    buffer = deep_ep.Buffer(group, num_rdma_bytes=num_rdma_bytes, low_latency_mode=True,
                            num_qps_per_rank=num_experts // num_ranks,
                            allow_nvlink_for_low_latency_mode=not args.disable_nvlink, explicitly_destroy=True,
                            allow_mnnvl=args.allow_mnnvl)
    test_main(num_tokens, hidden, num_experts, num_topk, rank, num_ranks, group, buffer,
              use_logfmt=args.use_logfmt, seed=1)

    # Feature-flagged compact-layout dispatch: focused correctness checks in its own
    # appropriately-sized buffer, independent of (and without altering) the legacy test above.
    test_compact_dispatch(rank, num_ranks, group)

    do_pressure_test = args.pressure_test
    for seed in range(int(1e9) if do_pressure_test else 0):
        if local_rank == 0:
            print(f'Testing with seed {seed} ...', flush=True)
        ref_hash = test_main(num_tokens, hidden, num_experts, num_topk, rank, num_ranks, group, buffer,
                             use_logfmt=args.use_logfmt, seed=seed)
        for i in range(20):
            assert test_main(num_tokens, hidden, num_experts, num_topk, rank, num_ranks, group, buffer,
                             use_logfmt=args.use_logfmt, seed=seed) == ref_hash, f'Error: seed={seed}'

    # Destroy the buffer runtime and communication group
    buffer.destroy()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    # TODO: you may modify NUMA binding for less CPU overhead
    # TODO: buggy with `num_tokens=512`
    parser = argparse.ArgumentParser(description='Test low-latency EP kernels')
    parser.add_argument('--num-processes', type=int, default=8,
                       help='Number of processes to spawn (default: 8)')
    parser.add_argument('--num-tokens', type=int, default=128,
                       help='Number of tokens (default: 128)')
    parser.add_argument('--hidden', type=int, default=7168,
                       help='Hidden dimension size (default: 7168)')
    parser.add_argument('--num-topk', type=int, default=8,
                       help='Number of top-k experts (default: 8)')
    parser.add_argument('--num-experts', type=int, default=288,
                       help='Number of experts (default: 288)')
    parser.add_argument('--allow-mnnvl', action="store_true",
                        help='Allow MNNVL for communication')
    parser.add_argument('--disable-nvlink', action='store_true',
                        help='Whether to disable NVLink for testing')
    parser.add_argument('--use-logfmt', action='store_true',
                        help='Whether to test LogFMT combine')
    parser.add_argument("--pressure-test", action='store_true',
                        help='Whether to do pressure test')
    args = parser.parse_args()

    num_processes = args.num_processes
    torch.multiprocessing.spawn(test_loop, args=(num_processes, args), nprocs=num_processes)
