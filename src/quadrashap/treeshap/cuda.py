"""CUDA implementation of the edge-telescoping Quadrature TreeSHAP solver.

The implementation follows the O(m_q L) formulation used by the native CPU
kernel, reorganised into GPU-friendly, sample-contiguous primitives:

1. large scalar batches partition trees into connected 15-node treelets and
   keep only component-root ``K/G`` checkpoints in global memory;
2. each treelet reconstructs local ``K`` in shared memory, then a top-down
   pass contracts sibling edges while overwriting ``K`` with ``G``;
3. q-collapsed parent contributions are reduced by feature without atomics.

Internal nodes are stored in level/feature execution order.  Warps stream row
banks, each tree uses its own exact quadrature order, and the sibling
recurrences use two cached parent factors instead of separate factors for
both edges.  Small batches retain the lower-overhead compact-node path and
multi-output models retain the full-state fallback.  Model metadata stays
resident on the device; no leaf transpose or global prefix scan is required.

CuPy is an optional dependency.  Importing quadrashap does not require it;
the module is loaded only when ``device="cuda"`` is requested.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .quadrature_tree import PreparedQuadratureTreeModel


_CUDA_SOURCE = r"""
extern "C" __global__
void init_roots(
    double* G, const int* roots, int n_roots, int n_nodes, int m_q,
    int n_samples)
{
    long long tid = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    long long count = (long long)n_samples * m_q * n_roots;
    if (tid >= count) return;
    int root_i = tid % n_roots;
    long long z = tid / n_roots;
    int qi = z % m_q;
    int sample = z / m_q;
    G[((long long)qi * n_nodes + roots[root_i]) * n_samples + sample] = 1.0;
}

extern "C" __global__
void init_compact_roots(
    double* state, const int* roots, int n_roots, int n_internal, int m_q,
    int n_samples)
{
    long long tid = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    long long count = (long long)n_samples * m_q * n_roots;
    if (tid >= count) return;
    int sample = tid % n_samples;
    long long z = tid / n_samples;
    int qi = z % m_q;
    int root_i = z / m_q;
    state[((long long)roots[root_i] * m_q + qi)
          * n_samples + sample] = 1.0;
}

// Connected treelets trade one local K reconstruction for a much smaller
// global checkpoint set.  The fixed sizes make all sample-dependent local
// state live in shared memory; only component-root K/G values cross kernels.
#define TREELET_SIZE 15
#define TREELET_MAX_PORTALS 10
#define TREELET_MAX_Q 16
#define TREELET_WARPS 4
#define TREELET_ROWS_PER_WARP 128
#define TREELET_CACHE_PORTALS 1
#define TREELET_CACHE_LEAVES 1
#define TREELET_VALUE_SLOTS (TREELET_SIZE + TREELET_MAX_PORTALS)
#if TREELET_CACHE_LEAVES
#define TREELET_LEAF_VALUE(local, side) sh_leaf[(local) * 2 + (side)]
#else
#define TREELET_LEAF_VALUE(local, side) \
    treelet_leaf_record[((long long)tile * TREELET_SIZE + (local)) * 2 \
                        + (side)]
#endif

extern "C" __global__
void build_treelet_level(
    const double* __restrict__ X, double* __restrict__ state,
    const int* __restrict__ level_treelets, int n_level,
    const int* __restrict__ treelet_headers,
    const int* __restrict__ treelet_q_offsets,
    const int* __restrict__ treelet_int_record,
    const double* __restrict__ treelet_float_record,
    const double* __restrict__ treelet_leaf_record,
    const int* __restrict__ treelet_portals,
    const double* __restrict__ sibling_prop_s,
    int m_q, int n_samples)
{
    int tile_i = blockIdx.x % n_level;
    int bank_group = blockIdx.x / n_level;
    int tile = level_treelets[tile_i];
    int warp = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    int bank = bank_group * TREELET_WARPS + warp;
    int n_banks =
        (n_samples + TREELET_ROWS_PER_WARP - 1)
        / TREELET_ROWS_PER_WARP;

    __shared__ int sh_int[TREELET_SIZE * 5];
    __shared__ int sh_portals[TREELET_MAX_PORTALS];
    __shared__ double sh_float[TREELET_SIZE * 5];
#if TREELET_CACHE_LEAVES
    __shared__ double sh_leaf[TREELET_SIZE * 2];
#endif
    __shared__ double sh_s[TREELET_SIZE * TREELET_MAX_Q];
    __shared__ double sh_value[
        TREELET_WARPS * TREELET_VALUE_SLOTS * 32];

    int count = treelet_headers[tile * 3];
    int q_count = treelet_headers[tile * 3 + 1];
    int portal_count = treelet_headers[tile * 3 + 2];
    for (int i = threadIdx.x; i < TREELET_SIZE * 5; i += blockDim.x) {
        sh_int[i] = treelet_int_record[(long long)tile
                                      * TREELET_SIZE * 5 + i];
        sh_float[i] = treelet_float_record[(long long)tile
                                           * TREELET_SIZE * 5 + i];
    }
#if TREELET_CACHE_LEAVES
    for (int i = threadIdx.x; i < TREELET_SIZE * 2; i += blockDim.x) {
        sh_leaf[i] = treelet_leaf_record[(long long)tile
                                        * TREELET_SIZE * 2 + i];
    }
#endif
    for (int i = threadIdx.x; i < TREELET_MAX_PORTALS;
         i += blockDim.x) {
        sh_portals[i] =
            treelet_portals[tile * TREELET_MAX_PORTALS + i];
    }
    __syncthreads();
    for (int i = threadIdx.x; i < TREELET_SIZE * TREELET_MAX_Q;
         i += blockDim.x) {
        int local = i / TREELET_MAX_Q;
        int qi = i - local * TREELET_MAX_Q;
        if (local < count && qi < q_count) {
            int parent = sh_int[local * 5 + 2];
            sh_s[i] = sibling_prop_s[(long long)parent * m_q + qi];
        } else {
            sh_s[i] = 0.0;
        }
    }
    __syncthreads();
    if (bank >= n_banks) return;

    int begin = bank * TREELET_ROWS_PER_WARP + lane;
    int end = min(
        (bank + 1) * TREELET_ROWS_PER_WARP, n_samples);
    int q_offset = treelet_q_offsets[tile];
    int warp_value_base = warp * TREELET_VALUE_SLOTS * 32;
    for (int sample = begin; sample < end; sample += 32) {
        unsigned int inside_mask = 0;
        unsigned int left_hot_mask = 0;
        for (int local = 0; local < count; ++local) {
            int f = sh_int[local * 5 + 3];
            double x = X[(long long)f * n_samples + sample];
            const double* rec = sh_float + local * 5;
            bool inside = x > rec[2] && x <= rec[3];
            inside_mask |= ((unsigned int)inside << local);
            left_hot_mask |= (
                (unsigned int)(inside && x <= rec[4]) << local);
        }

        #pragma unroll 16
        for (int qi = 0; qi < q_count; ++qi) {
#if TREELET_CACHE_PORTALS
            for (int portal = 0; portal < portal_count; ++portal) {
                int child_tile = sh_portals[portal];
                int child_offset = treelet_q_offsets[child_tile];
                sh_value[
                    warp_value_base
                    + (TREELET_SIZE + portal) * 32 + lane] =
                    state[((long long)child_offset + qi)
                          * n_samples + sample];
            }
#endif
            for (int local = count - 1; local >= 0; --local) {
                int left_ref = sh_int[local * 5];
                int right_ref = sh_int[local * 5 + 1];
#if TREELET_CACHE_PORTALS
                double left_k = left_ref >= 0
                    ? sh_value[warp_value_base + left_ref * 32 + lane]
                    : TREELET_LEAF_VALUE(local, 0);
                double right_k = right_ref >= 0
                    ? sh_value[warp_value_base + right_ref * 32 + lane]
                    : TREELET_LEAF_VALUE(local, 1);
#else
                double left_k;
                if (left_ref < 0) {
                    left_k = TREELET_LEAF_VALUE(local, 0);
                } else if (left_ref < TREELET_SIZE) {
                    left_k = sh_value[
                        warp_value_base + left_ref * 32 + lane];
                } else {
                    int child_tile =
                        sh_portals[left_ref - TREELET_SIZE];
                    int child_offset = treelet_q_offsets[child_tile];
                    left_k = state[((long long)child_offset + qi)
                                   * n_samples + sample];
                }
                double right_k;
                if (right_ref < 0) {
                    right_k = TREELET_LEAF_VALUE(local, 1);
                } else if (right_ref < TREELET_SIZE) {
                    right_k = sh_value[
                        warp_value_base + right_ref * 32 + lane];
                } else {
                    int child_tile =
                        sh_portals[right_ref - TREELET_SIZE];
                    int child_offset = treelet_q_offsets[child_tile];
                    right_k = state[((long long)child_offset + qi)
                                    * n_samples + sample];
                }
#endif
                const double* rec = sh_float + local * 5;
                unsigned int bit = 1u << local;
                double parent_k;
                if (!(inside_mask & bit)) {
                    parent_k = rec[0] * left_k + rec[1] * right_k;
                } else {
                    double s = sh_s[local * TREELET_MAX_Q + qi];
                    if (left_hot_mask & bit) {
                        double cold = rec[1] * s;
                        parent_k = left_k + cold * (right_k - left_k);
                    } else {
                        double cold = rec[0] * s;
                        parent_k = right_k + cold * (left_k - right_k);
                    }
                }
                sh_value[warp_value_base
                         + (long long)local * 32 + lane] = parent_k;
            }
            state[((long long)q_offset + qi) * n_samples + sample] =
                sh_value[warp_value_base + lane];
        }
    }
}

extern "C" __global__
void propagate_contract_treelet_level(
    const double* __restrict__ X, double* __restrict__ state,
    double* __restrict__ contribution,
    const int* __restrict__ level_treelets, int n_level,
    int root_level,
    const int* __restrict__ treelet_headers,
    const int* __restrict__ treelet_q_offsets,
    const int* __restrict__ treelet_int_record,
    const double* __restrict__ treelet_float_record,
    const double* __restrict__ treelet_leaf_record,
    const int* __restrict__ treelet_portals,
    const double* __restrict__ sibling_prop_s,
    const double* __restrict__ sibling_contract_c,
    int m_q, int n_samples)
{
    int tile_i = blockIdx.x % n_level;
    int bank_group = blockIdx.x / n_level;
    int tile = level_treelets[tile_i];
    int warp = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    int bank = bank_group * TREELET_WARPS + warp;
    int n_banks =
        (n_samples + TREELET_ROWS_PER_WARP - 1)
        / TREELET_ROWS_PER_WARP;

    __shared__ int sh_int[TREELET_SIZE * 5];
    __shared__ int sh_portals[TREELET_MAX_PORTALS];
    __shared__ double sh_float[TREELET_SIZE * 5];
#if TREELET_CACHE_LEAVES
    __shared__ double sh_leaf[TREELET_SIZE * 2];
#endif
    __shared__ double sh_s[TREELET_SIZE * TREELET_MAX_Q];
    __shared__ double sh_c[TREELET_SIZE * TREELET_MAX_Q];
    __shared__ double sh_value[
        TREELET_WARPS * TREELET_VALUE_SLOTS * 32];
    __shared__ double sh_acc[
        TREELET_WARPS * TREELET_SIZE * 32];

    int count = treelet_headers[tile * 3];
    int q_count = treelet_headers[tile * 3 + 1];
    int portal_count = treelet_headers[tile * 3 + 2];
    for (int i = threadIdx.x; i < TREELET_SIZE * 5; i += blockDim.x) {
        sh_int[i] = treelet_int_record[(long long)tile
                                      * TREELET_SIZE * 5 + i];
        sh_float[i] = treelet_float_record[(long long)tile
                                           * TREELET_SIZE * 5 + i];
    }
#if TREELET_CACHE_LEAVES
    for (int i = threadIdx.x; i < TREELET_SIZE * 2; i += blockDim.x) {
        sh_leaf[i] = treelet_leaf_record[(long long)tile
                                        * TREELET_SIZE * 2 + i];
    }
#endif
    for (int i = threadIdx.x; i < TREELET_MAX_PORTALS;
         i += blockDim.x) {
        sh_portals[i] =
            treelet_portals[tile * TREELET_MAX_PORTALS + i];
    }
    __syncthreads();
    for (int i = threadIdx.x; i < TREELET_SIZE * TREELET_MAX_Q;
         i += blockDim.x) {
        int local = i / TREELET_MAX_Q;
        int qi = i - local * TREELET_MAX_Q;
        if (local < count && qi < q_count) {
            int parent = sh_int[local * 5 + 2];
            long long factor_idx = (long long)parent * m_q + qi;
            sh_s[i] = sibling_prop_s[factor_idx];
            sh_c[i] = sibling_contract_c[factor_idx];
        } else {
            sh_s[i] = 0.0;
            sh_c[i] = 0.0;
        }
    }
    __syncthreads();
    if (bank >= n_banks) return;

    int begin = bank * TREELET_ROWS_PER_WARP + lane;
    int end = min(
        (bank + 1) * TREELET_ROWS_PER_WARP, n_samples);
    int q_offset = treelet_q_offsets[tile];
    int warp_value_base = warp * TREELET_VALUE_SLOTS * 32;
    int warp_acc_base = warp * TREELET_SIZE * 32;
    for (int sample = begin; sample < end; sample += 32) {
        unsigned int inside_mask = 0;
        unsigned int left_hot_mask = 0;
        for (int local = 0; local < count; ++local) {
            int f = sh_int[local * 5 + 3];
            double x = X[(long long)f * n_samples + sample];
            const double* rec = sh_float + local * 5;
            bool inside = x > rec[2] && x <= rec[3];
            inside_mask |= ((unsigned int)inside << local);
            left_hot_mask |= (
                (unsigned int)(inside && x <= rec[4]) << local);
            sh_acc[warp_acc_base + (long long)local * 32 + lane] =
                0.0;
        }

        #pragma unroll 16
        for (int qi = 0; qi < q_count; ++qi) {
#if TREELET_CACHE_PORTALS
            // Cache portal K before this q-plane is overwritten with G.
            for (int portal = 0; portal < portal_count; ++portal) {
                int child_tile = sh_portals[portal];
                int child_offset = treelet_q_offsets[child_tile];
                sh_value[
                    warp_value_base
                    + (TREELET_SIZE + portal) * 32 + lane] =
                    state[((long long)child_offset + qi)
                          * n_samples + sample];
            }
#endif
            // Reconstruct every local K from leaves and portal checkpoints.
            for (int local = count - 1; local >= 0; --local) {
                int left_ref = sh_int[local * 5];
                int right_ref = sh_int[local * 5 + 1];
#if TREELET_CACHE_PORTALS
                double left_k = left_ref >= 0
                    ? sh_value[warp_value_base + left_ref * 32 + lane]
                    : TREELET_LEAF_VALUE(local, 0);
                double right_k = right_ref >= 0
                    ? sh_value[warp_value_base + right_ref * 32 + lane]
                    : TREELET_LEAF_VALUE(local, 1);
#else
                double left_k;
                if (left_ref < 0) {
                    left_k = TREELET_LEAF_VALUE(local, 0);
                } else if (left_ref < TREELET_SIZE) {
                    left_k = sh_value[
                        warp_value_base + left_ref * 32 + lane];
                } else {
                    int child_tile =
                        sh_portals[left_ref - TREELET_SIZE];
                    int child_offset = treelet_q_offsets[child_tile];
                    left_k = state[((long long)child_offset + qi)
                                   * n_samples + sample];
                }
                double right_k;
                if (right_ref < 0) {
                    right_k = TREELET_LEAF_VALUE(local, 1);
                } else if (right_ref < TREELET_SIZE) {
                    right_k = sh_value[
                        warp_value_base + right_ref * 32 + lane];
                } else {
                    int child_tile =
                        sh_portals[right_ref - TREELET_SIZE];
                    int child_offset = treelet_q_offsets[child_tile];
                    right_k = state[((long long)child_offset + qi)
                                    * n_samples + sample];
                }
#endif
                const double* rec = sh_float + local * 5;
                unsigned int bit = 1u << local;
                double parent_k;
                if (!(inside_mask & bit)) {
                    parent_k = rec[0] * left_k + rec[1] * right_k;
                } else {
                    double s = sh_s[local * TREELET_MAX_Q + qi];
                    if (left_hot_mask & bit) {
                        double cold = rec[1] * s;
                        parent_k = left_k + cold * (right_k - left_k);
                    } else {
                        double cold = rec[0] * s;
                        parent_k = right_k + cold * (left_k - right_k);
                    }
                }
                sh_value[warp_value_base
                         + (long long)local * 32 + lane] = parent_k;
            }

            sh_value[warp_value_base + lane] = root_level
                ? 1.0
                : state[((long long)q_offset + qi)
                        * n_samples + sample];
            // Preorder turns K into G.  A local child's K is consumed before
            // its slot is overwritten, matching the compact solver invariant.
            for (int local = 0; local < count; ++local) {
                int left_ref = sh_int[local * 5];
                int right_ref = sh_int[local * 5 + 1];
#if TREELET_CACHE_PORTALS
                double left_k = left_ref >= 0
                    ? sh_value[warp_value_base + left_ref * 32 + lane]
                    : TREELET_LEAF_VALUE(local, 0);
                double right_k = right_ref >= 0
                    ? sh_value[warp_value_base + right_ref * 32 + lane]
                    : TREELET_LEAF_VALUE(local, 1);
#else
                double left_k;
                if (left_ref < 0) {
                    left_k = TREELET_LEAF_VALUE(local, 0);
                } else if (left_ref < TREELET_SIZE) {
                    left_k = sh_value[
                        warp_value_base + left_ref * 32 + lane];
                } else {
                    int child_tile =
                        sh_portals[left_ref - TREELET_SIZE];
                    int child_offset = treelet_q_offsets[child_tile];
                    left_k = state[((long long)child_offset + qi)
                                   * n_samples + sample];
                }
                double right_k;
                if (right_ref < 0) {
                    right_k = TREELET_LEAF_VALUE(local, 1);
                } else if (right_ref < TREELET_SIZE) {
                    right_k = sh_value[
                        warp_value_base + right_ref * 32 + lane];
                } else {
                    int child_tile =
                        sh_portals[right_ref - TREELET_SIZE];
                    int child_offset = treelet_q_offsets[child_tile];
                    right_k = state[((long long)child_offset + qi)
                                    * n_samples + sample];
                }
#endif
                double parent_g =
                    sh_value[warp_value_base
                             + (long long)local * 32 + lane];
                const double* rec = sh_float + local * 5;
                unsigned int bit = 1u << local;
                double left_g;
                double right_g;
                if (inside_mask & bit) {
                    double s = sh_s[local * TREELET_MAX_Q + qi];
                    double c = sh_c[local * TREELET_MAX_Q + qi];
                    if (left_hot_mask & bit) {
                        right_g = parent_g * rec[1] * s;
                        left_g = parent_g - right_g;
                        sh_acc[warp_acc_base
                               + (long long)local * 32 + lane] +=
                            parent_g * c * rec[1]
                            * (left_k - right_k);
                    } else {
                        left_g = parent_g * rec[0] * s;
                        right_g = parent_g - left_g;
                        sh_acc[warp_acc_base
                               + (long long)local * 32 + lane] -=
                            parent_g * c * rec[0]
                            * (left_k - right_k);
                    }
                } else {
                    left_g = parent_g * rec[0];
                    right_g = parent_g * rec[1];
                }
                if (left_ref >= 0) {
                    if (left_ref < TREELET_SIZE) {
                        sh_value[warp_value_base
                                 + (long long)left_ref * 32 + lane] =
                            left_g;
                    } else {
                        int child_tile =
                            sh_portals[left_ref - TREELET_SIZE];
                        int child_offset =
                            treelet_q_offsets[child_tile];
                        state[((long long)child_offset + qi)
                              * n_samples + sample] = left_g;
                    }
                }
                if (right_ref >= 0) {
                    if (right_ref < TREELET_SIZE) {
                        sh_value[warp_value_base
                                 + (long long)right_ref * 32 + lane] =
                            right_g;
                    } else {
                        int child_tile =
                            sh_portals[right_ref - TREELET_SIZE];
                        int child_offset =
                            treelet_q_offsets[child_tile];
                        state[((long long)child_offset + qi)
                              * n_samples + sample] = right_g;
                    }
                }
            }
        }
        for (int local = 0; local < count; ++local) {
            int slot = sh_int[local * 5 + 4];
            contribution[(long long)slot * n_samples + sample] =
                sh_acc[warp_acc_base + (long long)local * 32 + lane];
        }
    }
}

extern "C" __global__
void reduce_treelet_level(
    const double* __restrict__ contribution, double* __restrict__ out,
    const int* __restrict__ used_features,
    const int* __restrict__ feature_offsets,
    int n_used_features, int n_features, int n_samples)
{
    const int warp_size = 32;
    const int rows_per_warp = 32;
    long long tid = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    long long warp = tid / warp_size;
    int lane = threadIdx.x % warp_size;
    int n_banks = (n_samples + rows_per_warp - 1) / rows_per_warp;
    long long n_warps = (long long)n_used_features * n_banks;
    if (warp >= n_warps) return;

    int feature_i = warp % n_used_features;
    int bank = warp / n_used_features;
    int f = used_features[feature_i];
    int first = feature_offsets[feature_i];
    int last = feature_offsets[feature_i + 1];
    int sample = bank * rows_per_warp + lane;
    if (sample < n_samples) {
        double total = 0.0;
        for (int slot = first; slot < last; ++slot) {
            total += contribution[(long long)slot * n_samples + sample];
        }
        out[(long long)sample * n_features + f] += total;
    }
}

extern "C" __global__
void build_compact_subtrees_level(
    const double* __restrict__ X, double* __restrict__ state,
    const int* __restrict__ level_parents, int n_level,
    const int* __restrict__ compact_int_record,
    const double* __restrict__ leaf_value,
    const double* __restrict__ compact_float_record,
    const double* __restrict__ sibling_prop_s,
    int n_internal, int m_q, int n_samples)
{
    const int warp_size = 32;
    const int rows_per_warp = 1024;
    long long tid = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    long long warp = tid / warp_size;
    int lane = threadIdx.x % warp_size;
    int warp_in_block = threadIdx.x / warp_size;
    int n_banks = (n_samples + rows_per_warp - 1) / rows_per_warp;
    long long n_warps = (long long)n_level * n_banks;
    bool active = warp < n_warps;

    int level_i = active ? warp % n_level : 0;
    int bank = active ? warp / n_level : 0;
    int p = active ? level_parents[level_i] : 0;
    const int* int_record = compact_int_record + (long long)p * 4;
    const double* float_record = compact_float_record + (long long)p * 5;
    int left_ref = active ? int_record[0] : -1;
    int right_ref = active ? int_record[1] : -1;
    int f = active ? int_record[2] : 0;
    int q_count = active ? int_record[3] : 0;
    double left_weight = active ? float_record[0] : 0.0;
    double right_weight = active ? float_record[1] : 0.0;
    extern __shared__ double cached_prop_s[];
    for (int qi = lane; qi < m_q; qi += warp_size) {
        cached_prop_s[warp_in_block * m_q + qi] =
            active && qi < q_count
            ? sibling_prop_s[(long long)p * m_q + qi] : 0.0;
    }
    __syncthreads();
    if (!active) return;

    int begin = bank * rows_per_warp + lane;
    int end = min((bank + 1) * rows_per_warp, n_samples);
    for (int sample = begin; sample < end; sample += warp_size) {
        double x = X[(long long)f * n_samples + sample];
        bool inside_old =
            x > float_record[2] && x <= float_record[3];
        bool left_is_hot = inside_old && x <= float_record[4];
        for (int qi = 0; qi < q_count; ++qi) {
            double left_k = left_ref >= 0
                ? state[((long long)left_ref * m_q + qi)
                        * n_samples + sample]
                : leaf_value[-left_ref - 1];
            double right_k = right_ref >= 0
                ? state[((long long)right_ref * m_q + qi)
                        * n_samples + sample]
                : leaf_value[-right_ref - 1];
            double parent_k;
            if (!inside_old) {
                parent_k = left_weight * left_k + right_weight * right_k;
            } else {
                double s = cached_prop_s[warp_in_block * m_q + qi];
                if (left_is_hot) {
                    double cold = right_weight * s;
                    parent_k = left_k + cold * (right_k - left_k);
                } else {
                    double cold = left_weight * s;
                    parent_k = right_k + cold * (left_k - right_k);
                }
            }
            state[((long long)p * m_q + qi)
                  * n_samples + sample] = parent_k;
        }
    }
}

extern "C" __global__
void propagate_contract_compact_segments(
    const double* __restrict__ X, double* __restrict__ state,
    double* __restrict__ partial,
    const int* __restrict__ segment_features,
    const int* __restrict__ segment_parent_offsets,
    const int* __restrict__ segment_parents,
    const int* __restrict__ segment_ids,
    int n_segments_level, int cache_factors,
    const int* __restrict__ compact_int_record,
    const double* __restrict__ leaf_value,
    const double* __restrict__ compact_float_record,
    const double* __restrict__ sibling_prop_s,
    const double* __restrict__ sibling_contract_c, int n_internal,
    int n_features, int m_q, int n_samples)
{
    const int warp_size = 32;
    const int rows_per_warp = 64;
    long long tid = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    long long warp = tid / warp_size;
    int lane = threadIdx.x % warp_size;
    int warp_in_block = threadIdx.x / warp_size;
    int n_banks = (n_samples + rows_per_warp - 1) / rows_per_warp;
    long long n_warps = (long long)n_segments_level * n_banks;
    bool active = warp < n_warps;

    int segment_i = active ? warp % n_segments_level : 0;
    int bank = active ? warp / n_segments_level : 0;
    int f = active ? segment_features[segment_i] : 0;
    int first = active ? segment_parent_offsets[segment_i] : 0;
    int last = active ? segment_parent_offsets[segment_i + 1] : 0;
    int n_parents = last - first;
    int first_parent =
        active && n_parents > 0 ? segment_parents[first] : 0;
    const int max_parents_per_segment = 8;
    extern __shared__ double cached_factors_sc[];
    if (cache_factors) {
        int factor_count = n_parents * m_q;
        for (int i = lane; i < factor_count; i += warp_size) {
            int parent_offset = i / m_q;
            int qi = i - parent_offset * m_q;
            int p = first_parent + parent_offset;
            int q_count = compact_int_record[(long long)p * 4 + 3];
            long long dst =
                ((long long)warp_in_block * max_parents_per_segment
                 + parent_offset) * m_q * 2 + qi * 2;
            if (qi < q_count) {
                cached_factors_sc[dst] =
                    sibling_prop_s[(long long)p * m_q + qi];
                cached_factors_sc[dst + 1] =
                    sibling_contract_c[(long long)p * m_q + qi];
            } else {
                cached_factors_sc[dst] = 0.0;
                cached_factors_sc[dst + 1] = 0.0;
            }
        }
    }
    __syncthreads();
    if (!active) return;

    int begin = bank * rows_per_warp + lane;
    int end = min((bank + 1) * rows_per_warp, n_samples);
    for (int sample = begin; sample < end; sample += warp_size) {
        double acc = 0.0;
        double x = X[(long long)f * n_samples + sample];
        for (int parent_i = first; parent_i < last; ++parent_i) {
            int parent_offset = parent_i - first;
            int p = first_parent + parent_offset;
            const int* int_record =
                compact_int_record + (long long)p * 4;
            const double* float_record =
                compact_float_record + (long long)p * 5;
            int left_ref = int_record[0];
            int right_ref = int_record[1];
            int q_count = int_record[3];
            double left_weight = float_record[0];
            double right_weight = float_record[1];
            bool inside_old =
                x > float_record[2] && x <= float_record[3];
            bool left_is_hot = inside_old && x <= float_record[4];
            for (int qi = 0; qi < q_count; ++qi) {
                long long parent_idx =
                    ((long long)p * m_q + qi)
                    * n_samples + sample;
                double parent_g = state[parent_idx];
                double left_k = left_ref >= 0
                    ? state[((long long)left_ref * m_q + qi)
                            * n_samples + sample]
                    : leaf_value[-left_ref - 1];
                double right_k = right_ref >= 0
                    ? state[((long long)right_ref * m_q + qi)
                            * n_samples + sample]
                    : leaf_value[-right_ref - 1];
                double left_g;
                double right_g;
                if (inside_old) {
                    long long cache_idx =
                        ((long long)warp_in_block
                         * max_parents_per_segment + parent_offset)
                        * m_q * 2 + qi * 2;
                    double s = cache_factors
                        ? cached_factors_sc[cache_idx]
                        : sibling_prop_s[(long long)p * m_q + qi];
                    double c = cache_factors
                        ? cached_factors_sc[cache_idx + 1]
                        : sibling_contract_c[(long long)p * m_q + qi];
                    if (left_is_hot) {
                        right_g = parent_g * right_weight * s;
                        left_g = parent_g - right_g;
                        acc += parent_g * c * right_weight
                               * (left_k - right_k);
                    } else {
                        left_g = parent_g * left_weight * s;
                        right_g = parent_g - left_g;
                        acc -= parent_g * c * left_weight
                               * (left_k - right_k);
                    }
                } else {
                    left_g = parent_g * left_weight;
                    right_g = parent_g * right_weight;
                }
                if (left_ref >= 0) {
                    state[((long long)left_ref * m_q + qi)
                          * n_samples + sample] = left_g;
                }
                if (right_ref >= 0) {
                    state[((long long)right_ref * m_q + qi)
                          * n_samples + sample] = right_g;
                }
            }
        }
        partial[(long long)segment_ids[segment_i] * n_samples + sample] =
            acc;
    }
}

extern "C" __global__
void reduce_compact_segments(
    const double* partial, double* out, const int* used_features,
    const int* feature_segment_offsets, const int* feature_segment_ids,
    int n_used_features, int n_features, int n_samples)
{
    const int warp_size = 32;
    // There are only O(depth * features) segment rows.  Giving a warp 1024
    // samples left the final reduction with as few as two resident blocks on
    // the 10-feature benchmark.  One sample per lane exposes enough parallel
    // work while preserving each sample's summation order exactly.
    const int rows_per_warp = 32;
    long long tid = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    long long warp = tid / warp_size;
    int lane = threadIdx.x % warp_size;
    int n_banks = (n_samples + rows_per_warp - 1) / rows_per_warp;
    long long n_warps = (long long)n_used_features * n_banks;
    if (warp >= n_warps) return;

    int feature_i = warp % n_used_features;
    int bank = warp / n_used_features;
    int f = used_features[feature_i];
    int first = feature_segment_offsets[feature_i];
    int last = feature_segment_offsets[feature_i + 1];
    int begin = bank * rows_per_warp + lane;
    int end = min((bank + 1) * rows_per_warp, n_samples);
    for (int sample = begin; sample < end; sample += warp_size) {
        double total = 0.0;
        for (int i = first; i < last; ++i) {
            int segment = feature_segment_ids[i];
            total += partial[(long long)segment * n_samples + sample];
        }
        out[(long long)sample * n_features + f] = total;
    }
}

extern "C" __global__
void propagate_children_level(
    const double* X, double* G, const int* level_parents, int n_level,
    const int* left_child, const int* right_child, const int* feature,
    const int* node_to_leaf, const double* leaf_value, int scale_scalar_leaves,
    const double* edge_weight, const double* old_lower,
    const double* old_upper, const double* new_upper,
    const double* prop_inside_new, const double* prop_inside_old,
    int n_nodes, int n_features, int m_q, int n_samples)
{
    const int warp_size = 32;
    const int rows_per_warp = 1024;
    long long tid = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    long long warp = tid / warp_size;
    int lane = threadIdx.x % warp_size;
    int n_banks = (n_samples + rows_per_warp - 1) / rows_per_warp;
    long long n_warps = (long long)n_level * n_banks;
    if (warp >= n_warps) return;

    int level_i = warp % n_level;
    int bank = warp / n_level;
    int p = level_parents[level_i];
    int left = left_child[p];
    int right = right_child[p];
    int f = feature[left];
    int begin = bank * rows_per_warp + lane;
    int end = min((bank + 1) * rows_per_warp, n_samples);
    for (int sample = begin; sample < end; sample += warp_size) {
        double x = X[(long long)f * n_samples + sample];
        bool inside_old =
            x > old_lower[left] && x <= old_upper[left];
        bool left_is_hot = inside_old && x <= new_upper[left];
        bool right_is_hot = inside_old && !left_is_hot;
        for (int qi = 0; qi < m_q; ++qi) {
            long long left_factor_idx = (long long)left * m_q + qi;
            long long right_factor_idx = (long long)right * m_q + qi;
            double left_factor = left_is_hot
                ? prop_inside_new[left_factor_idx]
                : (inside_old
                    ? prop_inside_old[left_factor_idx] : edge_weight[left]);
            double right_factor = right_is_hot
                ? prop_inside_new[right_factor_idx]
                : (inside_old
                    ? prop_inside_old[right_factor_idx] : edge_weight[right]);
            long long parent_idx =
                ((long long)qi * n_nodes + p) * n_samples + sample;
            double parent_g = G[parent_idx];
            double left_g = parent_g * left_factor;
            double right_g = parent_g * right_factor;
            if (scale_scalar_leaves && left_child[left] < 0) {
                left_g *= leaf_value[node_to_leaf[left]];
            }
            if (scale_scalar_leaves && left_child[right] < 0) {
                right_g *= leaf_value[node_to_leaf[right]];
            }
            G[((long long)qi * n_nodes + left) * n_samples + sample] = left_g;
            G[((long long)qi * n_nodes + right) * n_samples + sample] = right_g;
        }
    }
}

extern "C" __global__
void gather_leaf_outputs(
    const double* G, double* H, const int* leaf_node,
    const double* leaf_value, int n_leaves, int n_nodes,
    int n_outputs, int m_q, int n_samples)
{
    long long tid = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    long long count = (long long)n_leaves * n_samples;
    if (tid >= count) return;
    int sample = tid % n_samples;
    int leaf = tid / n_samples;
    int node = leaf_node[leaf];
    for (int qi = 0; qi < m_q; ++qi) {
        double g = G[((long long)qi * n_nodes + node) * n_samples + sample];
        for (int out_i = 0; out_i < n_outputs; ++out_i) {
            long long dst =
                (((long long)qi * n_outputs + out_i) * n_nodes + node)
                * n_samples + sample;
            H[dst] = g * leaf_value[(long long)leaf * n_outputs + out_i];
        }
    }
}

extern "C" __global__
void sum_children_level(
    double* H, const int* level_nodes, int n_level,
    const int* left_child, const int* right_child,
    int n_nodes, int n_outputs, int m_q, int n_samples)
{
    const int warp_size = 32;
    const int rows_per_warp = 1024;
    long long tid = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    long long warp = tid / warp_size;
    int lane = threadIdx.x % warp_size;
    int n_banks = (n_samples + rows_per_warp - 1) / rows_per_warp;
    long long n_warps = (long long)n_level * n_banks;
    if (warp >= n_warps) return;

    int level_i = warp % n_level;
    int bank = warp / n_level;
    int node = level_nodes[level_i];
    int left = left_child[node];
    int right = right_child[node];
    int begin = bank * rows_per_warp + lane;
    int end = min((bank + 1) * rows_per_warp, n_samples);
    for (int sample = begin; sample < end; sample += warp_size) {
        for (int qi = 0; qi < m_q; ++qi) {
            for (int out_i = 0; out_i < n_outputs; ++out_i) {
                long long base =
                    ((long long)qi * n_outputs + out_i) * n_nodes;
                H[(base + node) * n_samples + sample] =
                    H[(base + left) * n_samples + sample]
                    + H[(base + right) * n_samples + sample];
            }
        }
    }
}

extern "C" __global__
void contract_parents_direct(
    const double* X, const double* H, double* out,
    const int* used_features, const int* feature_parent_offsets,
    const int* feature_parents, int n_used_features,
    const int* left_child, const int* right_child,
    const double* old_lower, const double* old_upper,
    const double* new_upper, const double* contract_inside_new,
    const double* contract_inside_old, int n_nodes, int n_features,
    int n_outputs, int m_q, int n_samples)
{
    const int warp_size = 32;
    const int warps_per_block = 8;
    int feature_i = blockIdx.x;
    int sample = blockIdx.y * warp_size + (threadIdx.x % warp_size);
    int out_i = blockIdx.z;
    int warp = threadIdx.x / warp_size;
    int lane = threadIdx.x % warp_size;
    if (feature_i >= n_used_features || out_i >= n_outputs) return;

    double acc = 0.0;
    int f = used_features[feature_i];
    int first = feature_parent_offsets[feature_i];
    int last = feature_parent_offsets[feature_i + 1];
    if (sample < n_samples) {
        double x = X[(long long)f * n_samples + sample];
        for (int parent_i = first + warp;
             parent_i < last; parent_i += warps_per_block) {
            int p = feature_parents[parent_i];
            int left = left_child[p];
            int right = right_child[p];
            bool inside_old =
                x > old_lower[left] && x <= old_upper[left];
            if (!inside_old) continue;
            bool left_is_hot = x <= new_upper[left];
            const double* left_coefficient = left_is_hot
                ? contract_inside_new : contract_inside_old;
            const double* right_coefficient = left_is_hot
                ? contract_inside_old : contract_inside_new;
            for (int qi = 0; qi < m_q; ++qi) {
                long long base =
                    ((long long)qi * n_outputs + out_i) * n_nodes;
                double left_h =
                    H[(base + left) * n_samples + sample];
                double right_h =
                    H[(base + right) * n_samples + sample];
                acc +=
                    left_coefficient[(long long)left * m_q + qi] * left_h
                    + right_coefficient[(long long)right * m_q + qi]
                    * right_h;
            }
        }
    }

    __shared__ double partial[warps_per_block * warp_size];
    partial[warp * warp_size + lane] = acc;
    __syncthreads();
    if (warp == 0 && sample < n_samples) {
        double total = 0.0;
        #pragma unroll
        for (int w = 0; w < warps_per_block; ++w) {
            total += partial[w * warp_size + lane];
        }
        out[((long long)sample * n_features + f) * n_outputs + out_i] =
            total;
    }
}
"""

def _positive_env_int(name: str, default: int) -> int:
    """Read a positive integer tuning override, retaining stable defaults."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer, got {raw!r}") from exc
    if value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value}")
    return value


_TREELET_MIN_ROWS = _positive_env_int(
    "QUADRASHAP_CUDA_TREELET_MIN_ROWS", 1536
)
_TREELET_ROWS_PER_WARP = _positive_env_int(
    "QUADRASHAP_CUDA_TREELET_ROWS_PER_WARP", 128
)
_TREELET_WARPS = 4
_TREELET_SIZE_OVERRIDE = os.environ.get("QUADRASHAP_CUDA_TREELET_SIZE")
if _TREELET_SIZE_OVERRIDE is not None:
    try:
        _TREELET_SIZE_OVERRIDE = int(_TREELET_SIZE_OVERRIDE)
    except ValueError as exc:
        raise ValueError(
            "QUADRASHAP_CUDA_TREELET_SIZE must be 7 or 15"
        ) from exc
    if _TREELET_SIZE_OVERRIDE not in (7, 15):
        raise ValueError("QUADRASHAP_CUDA_TREELET_SIZE must be 7 or 15")


@dataclass
class _HostPlan:
    n_nodes: int
    n_internal: int
    n_leaves: int
    n_features: int
    n_outputs: int
    m_q: int
    roots: np.ndarray
    internal_roots: np.ndarray
    levels: list[np.ndarray]
    internal_levels: list[np.ndarray]
    compact_internal_levels: list[np.ndarray]
    n_segments: int
    level_segment_features: list[np.ndarray]
    level_segment_offsets: list[np.ndarray]
    level_segment_parents: list[np.ndarray]
    level_segment_ids: list[np.ndarray]
    segment_used_features: np.ndarray
    feature_segment_offsets: np.ndarray
    feature_segment_ids: np.ndarray
    parent: np.ndarray
    left_child: np.ndarray
    right_child: np.ndarray
    feature: np.ndarray
    edge_weight: np.ndarray
    old_lower: np.ndarray
    old_upper: np.ndarray
    old_invw: np.ndarray
    new_lower: np.ndarray
    new_upper: np.ndarray
    new_invw: np.ndarray
    leaf_begin: np.ndarray
    leaf_end: np.ndarray
    leaf_node: np.ndarray
    node_to_leaf: np.ndarray
    node_to_internal: np.ndarray
    compact_left_ref: np.ndarray
    compact_right_ref: np.ndarray
    compact_feature: np.ndarray
    compact_m_q: np.ndarray
    compact_left_weight: np.ndarray
    compact_right_weight: np.ndarray
    compact_old_lower: np.ndarray
    compact_old_upper: np.ndarray
    compact_threshold: np.ndarray
    compact_int_record: np.ndarray
    compact_float_record: np.ndarray
    compact_siblings_complementary: bool
    leaf_value: np.ndarray
    parent_used_features: np.ndarray
    feature_parent_offsets: np.ndarray
    feature_parents: np.ndarray
    prop_inside_new: np.ndarray
    prop_inside_old: np.ndarray
    contract_inside_new: np.ndarray
    contract_inside_old: np.ndarray
    sibling_prop_s: np.ndarray
    sibling_contract_c: np.ndarray
    quad_x: np.ndarray
    quad_w: np.ndarray


def _build_host_plan(prepared: "PreparedQuadratureTreeModel") -> _HostPlan:
    ensemble = prepared.ensemble
    n_nodes = sum(len(pt.tree.children_left) for pt in prepared.trees)
    n_features = int(ensemble.n_features)
    n_outputs = int(ensemble.n_outputs)
    # A user-supplied m_q is reflected in every prepared tree.  With the
    # default exact mode, taking the maximum per-tree order yields one common
    # quadrature rule that is exact for every tree in the ensemble.
    m_q = max((len(pt.quad_x) for pt in prepared.trees), default=1)
    leg_x, leg_w = np.polynomial.legendre.leggauss(m_q)
    quad_x = np.ascontiguousarray(0.5 * (leg_x + 1.0), dtype=np.float64)
    quad_w = np.ascontiguousarray(0.5 * leg_w, dtype=np.float64)

    parent = np.full(n_nodes, -1, dtype=np.int32)
    left_child = np.full(n_nodes, -1, dtype=np.int32)
    right_child = np.full(n_nodes, -1, dtype=np.int32)
    feature = np.full(n_nodes, -1, dtype=np.int32)
    edge_weight = np.ones(n_nodes, dtype=np.float64)
    old_lower = np.full(n_nodes, -np.inf, dtype=np.float64)
    old_upper = np.full(n_nodes, np.inf, dtype=np.float64)
    old_invw = np.ones(n_nodes, dtype=np.float64)
    new_lower = old_lower.copy()
    new_upper = old_upper.copy()
    new_invw = old_invw.copy()
    leaf_begin = np.zeros(n_nodes, dtype=np.int32)
    leaf_end = np.zeros(n_nodes, dtype=np.int32)

    roots: list[int] = []
    levels: list[list[int]] = []
    leaf_nodes: list[int] = []
    leaf_values: list[np.ndarray] = []
    tree_slices: list[tuple[int, int, object]] = []

    node_base = 0
    for pt in prepared.trees:
        tree = pt.tree
        tree_begin = node_base
        roots.append(node_base)

        def rec(
            u: int,
            depth: int,
            state: dict[int, tuple[float, float, float]],
        ) -> tuple[int, int]:
            gu = node_base + u
            while len(levels) <= depth:
                levels.append([])
            levels[depth].append(gu)
            left = int(tree.children_left[u])
            if left == -1:
                start = len(leaf_nodes)
                leaf_nodes.append(gu)
                leaf_values.append(
                    np.asarray(tree.values[u], dtype=np.float64)
                    * float(pt.tree_weight)
                )
                leaf_begin[gu] = start
                leaf_end[gu] = start + 1
                return start, start + 1

            f = int(tree.feature[u])
            threshold = float(tree.threshold[u])
            left_child[gu] = node_base + left
            right_child[gu] = node_base + int(tree.children_right[u])
            old_rule = state.get(f, (-np.inf, np.inf, 1.0))
            child_ranges = []
            for child, is_left in (
                (left, True),
                (int(tree.children_right[u]), False),
            ):
                gc = node_base + child
                w = float(pt.edge_weight[child])
                lo, hi, invw = old_rule
                nlo = lo if is_left else max(lo, threshold)
                nhi = min(hi, threshold) if is_left else hi
                ninvw = invw / w
                parent[gc] = gu
                feature[gc] = f
                edge_weight[gc] = w
                old_lower[gc], old_upper[gc], old_invw[gc] = lo, hi, invw
                new_lower[gc], new_upper[gc], new_invw[gc] = nlo, nhi, ninvw
                child_state = dict(state)
                child_state[f] = (nlo, nhi, ninvw)
                child_ranges.append(rec(child, depth + 1, child_state))
            start, end = child_ranges[0][0], child_ranges[-1][1]
            leaf_begin[gu], leaf_end[gu] = start, end
            return start, end

        rec(0, 0, {})
        node_base += len(tree.children_left)
        tree_slices.append((tree_begin, node_base, pt))

    parent_nodes = np.flatnonzero(left_child >= 0).astype(np.int32)
    parent_features = feature[left_child[parent_nodes]]
    parent_order = np.lexsort((leaf_begin[parent_nodes], parent_features))
    feature_parents = np.ascontiguousarray(
        parent_nodes[parent_order], dtype=np.int32
    )
    sorted_parent_features = feature[left_child[feature_parents]]
    parent_used_features, parent_counts = np.unique(
        sorted_parent_features, return_counts=True
    )
    feature_parent_offsets = np.empty(
        len(parent_used_features) + 1, dtype=np.int32
    )
    feature_parent_offsets[0] = 0
    np.cumsum(
        parent_counts, dtype=np.int32, out=feature_parent_offsets[1:]
    )

    # For a fixed edge and quadrature node there are only three possible
    # states for x: inside the new interval, inside only the old interval, or
    # outside the old interval.  Precompute the expensive rational factors
    # once instead of evaluating several FP64 divisions for every sample.
    t = quad_x.reshape(1, -1)
    wq = quad_w.reshape(1, -1)
    q_old = old_invw.reshape(-1, 1)
    q_new = new_invw.reshape(-1, 1)
    a_old = (1.0 - t) + t * q_old
    a_new = (1.0 - t) + t * q_new
    prop_inside_new = edge_weight.reshape(-1, 1) * (a_new / a_old)
    prop_inside_old = edge_weight.reshape(-1, 1) * ((1.0 - t) / a_old)
    contract_inside_new = wq * (
        (q_new - 1.0) / a_new - (q_old - 1.0) / a_old
    )
    contract_inside_old = wq * (
        -1.0 / (1.0 - t) - (q_old - 1.0) / a_old
    )
    level_arrays = [np.asarray(x, dtype=np.int32) for x in levels]
    internal_levels = [
        np.ascontiguousarray(level[left_child[level] >= 0], dtype=np.int32)
        for level in level_arrays
    ]
    ordered_internal_levels: list[np.ndarray] = []
    for level in internal_levels:
        if len(level) == 0:
            ordered_internal_levels.append(level)
            continue
        level_features = feature[left_child[level]]
        order = np.argsort(level_features, kind="stable")
        ordered_internal_levels.append(
            np.ascontiguousarray(level[order], dtype=np.int32)
        )
    internal_nodes = (
        np.concatenate(ordered_internal_levels).astype(np.int32, copy=False)
        if ordered_internal_levels
        else np.empty(0, dtype=np.int32)
    )
    node_to_internal = np.full(n_nodes, -1, dtype=np.int32)
    node_to_internal[internal_nodes] = np.arange(
        len(internal_nodes), dtype=np.int32
    )
    compact_internal_levels = [
        np.ascontiguousarray(node_to_internal[level], dtype=np.int32)
        for level in ordered_internal_levels
    ]
    internal_roots = node_to_internal[np.asarray(roots, dtype=np.int32)]
    internal_roots = np.ascontiguousarray(
        internal_roots[internal_roots >= 0], dtype=np.int32
    )
    level_segment_features: list[np.ndarray] = []
    level_segment_offsets: list[np.ndarray] = []
    level_segment_parents: list[np.ndarray] = []
    level_segment_ids: list[np.ndarray] = []
    all_segment_features: list[int] = []
    segment_size = 8
    for level in internal_levels:
        if len(level) == 0:
            level_segment_features.append(np.empty(0, dtype=np.int32))
            level_segment_offsets.append(np.zeros(1, dtype=np.int32))
            level_segment_parents.append(np.empty(0, dtype=np.int32))
            level_segment_ids.append(np.empty(0, dtype=np.int32))
            continue
        level_features = feature[left_child[level]]
        order = np.argsort(level_features, kind="stable")
        sorted_parents = np.ascontiguousarray(level[order], dtype=np.int32)
        sorted_features = level_features[order]
        used, starts, counts = np.unique(
            sorted_features, return_index=True, return_counts=True
        )
        segment_features: list[int] = []
        segment_offsets = [0]
        segment_parents: list[int] = []
        segment_ids: list[int] = []
        for f, start, count in zip(used, starts, counts):
            stop = int(start + count)
            for begin in range(int(start), stop, segment_size):
                end = min(begin + segment_size, stop)
                segment_features.append(int(f))
                segment_parents.extend(
                    node_to_internal[sorted_parents[begin:end]].tolist()
                )
                segment_offsets.append(len(segment_parents))
                segment_ids.append(len(all_segment_features))
                all_segment_features.append(int(f))
        level_segment_features.append(
            np.asarray(segment_features, dtype=np.int32)
        )
        level_segment_offsets.append(
            np.asarray(segment_offsets, dtype=np.int32)
        )
        level_segment_parents.append(
            np.asarray(segment_parents, dtype=np.int32)
        )
        level_segment_ids.append(np.asarray(segment_ids, dtype=np.int32))
    all_segment_features_array = np.asarray(
        all_segment_features, dtype=np.int32
    )
    segment_order = np.argsort(all_segment_features_array, kind="stable")
    feature_segment_ids = np.ascontiguousarray(
        segment_order, dtype=np.int32
    )
    sorted_segment_features = all_segment_features_array[segment_order]
    segment_used_features, segment_counts = np.unique(
        sorted_segment_features, return_counts=True
    )
    feature_segment_offsets = np.empty(
        len(segment_used_features) + 1, dtype=np.int32
    )
    feature_segment_offsets[0] = 0
    np.cumsum(
        segment_counts, dtype=np.int32, out=feature_segment_offsets[1:]
    )
    node_to_leaf = np.full(n_nodes, -1, dtype=np.int32)
    node_to_leaf[np.asarray(leaf_nodes, dtype=np.int32)] = np.arange(
        len(leaf_nodes), dtype=np.int32
    )
    compact_left_node = left_child[internal_nodes]
    compact_right_node = right_child[internal_nodes]
    compact_left_internal = node_to_internal[compact_left_node]
    compact_right_internal = node_to_internal[compact_right_node]
    compact_left_ref = np.where(
        compact_left_internal >= 0,
        compact_left_internal,
        -node_to_leaf[compact_left_node] - 1,
    ).astype(np.int32)
    compact_right_ref = np.where(
        compact_right_internal >= 0,
        compact_right_internal,
        -node_to_leaf[compact_right_node] - 1,
    ).astype(np.int32)
    compact_feature = np.ascontiguousarray(
        feature[compact_left_node], dtype=np.int32
    )
    compact_left_weight = np.ascontiguousarray(
        edge_weight[compact_left_node], dtype=np.float64
    )
    compact_right_weight = np.ascontiguousarray(
        edge_weight[compact_right_node], dtype=np.float64
    )
    compact_old_lower = np.ascontiguousarray(
        old_lower[compact_left_node], dtype=np.float64
    )
    compact_old_upper = np.ascontiguousarray(
        old_upper[compact_left_node], dtype=np.float64
    )
    compact_threshold = np.ascontiguousarray(
        new_upper[compact_left_node], dtype=np.float64
    )
    compact_int_record = np.ascontiguousarray(
        np.column_stack(
            (
                compact_left_ref,
                compact_right_ref,
                compact_feature,
                np.zeros(len(internal_nodes), dtype=np.int32),
            )
        ),
        dtype=np.int32,
    )
    compact_float_record = np.ascontiguousarray(
        np.column_stack(
            (
                compact_left_weight,
                compact_right_weight,
                compact_old_lower,
                compact_old_upper,
                compact_threshold,
            )
        ),
        dtype=np.float64,
    )

    # Scalar compact execution can use each tree's own exact Gauss rule
    # instead of padding every tree to the ensemble-wide maximum order.
    # Siblings share their old path state. With
    #   s = (1 - t) / a_old, c = w_q * q_old / a_old**2,
    # the cold propagation is w*s, the hot propagation is
    # 1-(1-w)*s, and propagation times contraction is +/-c*w.
    # This replaces eight edge/q factor streams with two parent/q streams.
    sibling_prop_s = np.zeros((len(internal_nodes), m_q), dtype=np.float64)
    sibling_contract_c = np.zeros(
        (len(internal_nodes), m_q), dtype=np.float64
    )
    compact_m_q = np.zeros(len(internal_nodes), dtype=np.int32)
    for begin, end, pt in tree_slices:
        tree_parents = parent_nodes[
            (parent_nodes >= begin) & (parent_nodes < end)
        ]
        compact_ids = node_to_internal[tree_parents]
        q_count = len(pt.quad_x)
        compact_m_q[compact_ids] = q_count
        tree_t = np.asarray(pt.quad_x, dtype=np.float64).reshape(1, -1)
        tree_w = np.asarray(pt.quad_w, dtype=np.float64).reshape(1, -1)
        tree_left = left_child[tree_parents]
        tree_q_old = old_invw[tree_left].reshape(-1, 1)
        tree_a_old = (1.0 - tree_t) + tree_t * tree_q_old
        sibling_prop_s[compact_ids, :q_count] = (
            (1.0 - tree_t) / tree_a_old
        )
        sibling_contract_c[compact_ids, :q_count] = (
            tree_w * (tree_q_old / tree_a_old) / tree_a_old
        )
    compact_siblings_complementary = bool(np.all(
        compact_left_weight + compact_right_weight == 1.0
    ))
    compact_int_record[:, 3] = compact_m_q

    return _HostPlan(
        n_nodes=n_nodes,
        n_internal=len(internal_nodes),
        n_leaves=len(leaf_nodes),
        n_features=n_features,
        n_outputs=n_outputs,
        m_q=m_q,
        roots=np.asarray(roots, dtype=np.int32),
        internal_roots=internal_roots,
        levels=level_arrays,
        internal_levels=internal_levels,
        compact_internal_levels=compact_internal_levels,
        n_segments=len(all_segment_features),
        level_segment_features=level_segment_features,
        level_segment_offsets=level_segment_offsets,
        level_segment_parents=level_segment_parents,
        level_segment_ids=level_segment_ids,
        segment_used_features=np.ascontiguousarray(
            segment_used_features, dtype=np.int32
        ),
        feature_segment_offsets=feature_segment_offsets,
        feature_segment_ids=feature_segment_ids,
        parent=parent,
        left_child=left_child,
        right_child=right_child,
        feature=feature,
        edge_weight=edge_weight,
        old_lower=old_lower,
        old_upper=old_upper,
        old_invw=old_invw,
        new_lower=new_lower,
        new_upper=new_upper,
        new_invw=new_invw,
        leaf_begin=leaf_begin,
        leaf_end=leaf_end,
        leaf_node=np.asarray(leaf_nodes, dtype=np.int32),
        node_to_leaf=node_to_leaf,
        node_to_internal=node_to_internal,
        compact_left_ref=np.ascontiguousarray(
            compact_left_ref, dtype=np.int32
        ),
        compact_right_ref=np.ascontiguousarray(
            compact_right_ref, dtype=np.int32
        ),
        compact_feature=compact_feature,
        compact_m_q=compact_m_q,
        compact_left_weight=compact_left_weight,
        compact_right_weight=compact_right_weight,
        compact_old_lower=compact_old_lower,
        compact_old_upper=compact_old_upper,
        compact_threshold=compact_threshold,
        compact_int_record=compact_int_record,
        compact_float_record=compact_float_record,
        compact_siblings_complementary=compact_siblings_complementary,
        leaf_value=np.ascontiguousarray(np.asarray(leaf_values), dtype=np.float64),
        parent_used_features=np.ascontiguousarray(
            parent_used_features, dtype=np.int32
        ),
        feature_parent_offsets=feature_parent_offsets,
        feature_parents=feature_parents,
        prop_inside_new=np.ascontiguousarray(
            prop_inside_new, dtype=np.float64
        ),
        prop_inside_old=np.ascontiguousarray(
            prop_inside_old, dtype=np.float64
        ),
        contract_inside_new=np.ascontiguousarray(
            contract_inside_new, dtype=np.float64
        ),
        contract_inside_old=np.ascontiguousarray(
            contract_inside_old, dtype=np.float64
        ),
        sibling_prop_s=np.ascontiguousarray(
            sibling_prop_s, dtype=np.float64
        ),
        sibling_contract_c=np.ascontiguousarray(
            sibling_contract_c, dtype=np.float64
        ),
        quad_x=quad_x,
        quad_w=quad_w,
    )


class CudaQuadratureTreeSolver:
    """Prepared CUDA solver for one tree ensemble."""

    def __init__(self, prepared: "PreparedQuadratureTreeModel"):
        try:
            import cupy as cp
        except ImportError as exc:
            raise ImportError(
                "CUDA QuadraSHAP requires CuPy. Install `cupy-cuda12x` (or the "
                "CuPy package matching the local CUDA runtime)."
            ) from exc

        if cp.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError(
                "CUDA QuadraSHAP requested but no CUDA device is visible."
            )

        self.cp = cp
        self.host = _build_host_plan(prepared)
        from .cuda_treelets import build_treelet_plan

        # Lower-q models benefit from smaller treelets: their additional
        # checkpoint traffic is modest while twice as many warps fit in shared
        # memory. Higher-q models use larger components to minimize checkpoints.
        self.treelet_size = (
            _TREELET_SIZE_OVERRIDE
            if _TREELET_SIZE_OVERRIDE is not None
            else (7 if self.host.m_q <= 5 else 15)
        )
        self.treelet_max_portals = 8 if self.treelet_size == 7 else 10
        self.treelets = build_treelet_plan(
            self.host,
            treelet_size=self.treelet_size,
            max_portals=self.treelet_max_portals,
        )
        cuda_source = _CUDA_SOURCE
        if self.treelet_size != 15:
            cuda_source = cuda_source.replace(
                "#define TREELET_SIZE 15",
                f"#define TREELET_SIZE {self.treelet_size}",
            ).replace(
                "#define TREELET_MAX_PORTALS 10",
                f"#define TREELET_MAX_PORTALS {self.treelet_max_portals}",
            )
        if _TREELET_ROWS_PER_WARP != 128:
            cuda_source = cuda_source.replace(
                "#define TREELET_ROWS_PER_WARP 128",
                f"#define TREELET_ROWS_PER_WARP {_TREELET_ROWS_PER_WARP}",
            )
        if self.treelets.supported and self.host.m_q != 16:
            cuda_source = cuda_source.replace(
                "#define TREELET_MAX_Q 16",
                f"#define TREELET_MAX_Q {self.host.m_q}",
            )
        module = cp.RawModule(
            code=cuda_source,
            options=("--std=c++11",),
            name_expressions=(
                "init_roots",
                "init_compact_roots",
                "build_treelet_level",
                "propagate_contract_treelet_level",
                "reduce_treelet_level",
                "build_compact_subtrees_level",
                "propagate_contract_compact_segments",
                "reduce_compact_segments",
                "propagate_children_level",
                "gather_leaf_outputs",
                "sum_children_level",
                "contract_parents_direct",
            ),
        )
        self.init_roots = module.get_function("init_roots")
        self.init_compact_roots = module.get_function("init_compact_roots")
        self.build_treelet_level = module.get_function(
            "build_treelet_level"
        )
        self.propagate_contract_treelet_level = module.get_function(
            "propagate_contract_treelet_level"
        )
        self.reduce_treelet_level = module.get_function(
            "reduce_treelet_level"
        )
        self.build_compact_subtrees_level = module.get_function(
            "build_compact_subtrees_level"
        )
        self.propagate_contract_compact_segments = module.get_function(
            "propagate_contract_compact_segments"
        )
        self.reduce_compact_segments = module.get_function(
            "reduce_compact_segments"
        )
        self.propagate_children_level = module.get_function(
            "propagate_children_level"
        )
        self.gather_leaf_outputs = module.get_function("gather_leaf_outputs")
        self.sum_children_level = module.get_function("sum_children_level")
        self.contract_parents_direct = module.get_function(
            "contract_parents_direct"
        )

        # Persistent model data. Keeping it resident follows the prepared
        # explainer API used by the CPU backend.
        h = self.host
        names = (
            "roots", "internal_roots", "left_child", "right_child", "feature",
            "node_to_leaf", "leaf_node", "leaf_value",
            "edge_weight", "old_lower",
            "old_upper", "new_upper", "parent_used_features",
            "feature_parent_offsets", "feature_parents", "prop_inside_new",
            "prop_inside_old", "contract_inside_new", "contract_inside_old",
            "segment_used_features", "feature_segment_offsets",
            "feature_segment_ids", "sibling_prop_s", "sibling_contract_c",
            "compact_int_record", "compact_float_record",
        )
        self.dev = {name: cp.asarray(getattr(h, name)) for name in names}
        self.internal_levels = [
            cp.asarray(level) for level in h.internal_levels
        ]
        self.compact_internal_levels = [
            cp.asarray(level) for level in h.compact_internal_levels
        ]
        self.compact_levels = [
            (
                cp.asarray(features),
                cp.asarray(offsets),
                cp.asarray(parents),
                cp.asarray(segment_ids),
            )
            for features, offsets, parents, segment_ids in zip(
                h.level_segment_features,
                h.level_segment_offsets,
                h.level_segment_parents,
                h.level_segment_ids,
            )
        ]
        if self.treelets.supported:
            t = self.treelets
            self.treelet_dev = {
                "headers": cp.asarray(t.headers),
                "q_offsets": cp.asarray(t.q_offsets),
                "int_record": cp.asarray(t.int_record),
                "float_record": cp.asarray(t.float_record),
                "leaf_record": cp.asarray(t.leaf_record),
                "portals": cp.asarray(t.portals),
            }
            self.treelet_levels = [
                (
                    cp.asarray(level.treelets),
                    cp.asarray(level.used_features),
                    cp.asarray(level.feature_offsets),
                    level.n_parents,
                )
                for level in t.levels
            ]
        else:
            self.treelet_dev = {}
            self.treelet_levels = []

    @staticmethod
    def _blocks(count: int, threads: int = 256) -> tuple[int]:
        return ((int(count) + threads - 1) // threads,)

    def _batch_capacity(self, *, use_treelets: bool = True) -> int:
        cp = self.cp
        h = self.host
        free, _ = cp.cuda.runtime.memGetInfo()
        # CuPy retains freed blocks in its memory pool.  Driver-level `free`
        # does not include those immediately reusable blocks, so omitting them
        # makes the apparent capacity shrink after every warm call.
        free += cp.get_default_memory_pool().free_bytes()
        if (
            h.n_outputs == 1
            and h.compact_siblings_complementary
            and self.treelets.supported
            and use_treelets
        ):
            # Treelet checkpoints are ragged by each tree's exact q.  The
            # contribution buffer is reused after every component depth.
            bytes_per_sample = 8 * max(
                1, self.treelets.workspace_values_per_sample
            )
        elif h.n_outputs == 1 and h.compact_siblings_complementary:
            # Scalar models store sample-dependent state only for internal
            # nodes. Leaves are reconstructed from their resident values, and
            # each eight-parent segment emits one q-collapsed partial.
            bytes_per_sample = 8 * (
                h.m_q * max(1, h.n_internal) + h.n_segments
            )
        elif h.n_outputs == 1:
            bytes_per_sample = 8 * h.m_q * h.n_nodes
        else:
            # Multi-output models briefly need the output-independent G and
            # output-specific H together.
            bytes_per_sample = (
                8 * h.m_q * h.n_nodes * (1 + h.n_outputs)
            )
        capacity = max(
            1, int((free * 0.80) // max(1, bytes_per_sample))
        )
        # Use stable 1024-row allocation quanta. This avoids tiny tail banks in
        # the compact path and leaves headroom for model/output allocations.
        if capacity >= 1024:
            capacity = (capacity // 1024) * 1024
        return capacity

    def explain(self, X: np.ndarray) -> np.ndarray:
        cp = self.cp
        h = self.host
        X = np.ascontiguousarray(X, dtype=np.float64)
        batches: list[np.ndarray] = []
        use_treelets = bool(
            self.treelets.supported
            and h.n_outputs == 1
            and h.compact_siblings_complementary
            and len(X) >= _TREELET_MIN_ROWS
        )
        capacity = self._batch_capacity(use_treelets=use_treelets)
        if use_treelets and len(X) > capacity:
            # Avoid one tiny, under-filled tail after several maximum-size
            # chunks. Equal-sized chunks keep the same launch count and make
            # every treelet row bank useful.
            n_batches = (len(X) + capacity - 1) // capacity
            capacity = (len(X) + n_batches - 1) // n_batches
        start = 0
        while start < len(X):
            batch_size = min(capacity, len(X) - start)
            try:
                batches.append(
                    self._explain_batch(
                        X[start : start + batch_size],
                        use_treelets=use_treelets,
                    )
                )
            except cp.cuda.memory.OutOfMemoryError:
                # Pool fragmentation can make driver-level free-memory
                # accounting optimistic. Release only unused cached blocks
                # and retry with a smaller, recoverable batch.
                cp.get_default_memory_pool().free_all_blocks()
                if batch_size == 1:
                    raise
                capacity = max(1, batch_size // 2)
                continue
            start += batch_size
        return np.concatenate(batches, axis=0) if batches else np.empty(
            (0, h.n_features, h.n_outputs), dtype=np.float64
        )

    def _explain_batch(
        self, X: np.ndarray, *, use_treelets: bool = False
    ) -> np.ndarray:
        """Propagate G, turn it into subtree sums H, then contract directly."""
        if (
            self.host.n_outputs == 1
            and self.host.compact_siblings_complementary
        ):
            if (
                self.treelets.supported
                and (use_treelets or len(X) >= _TREELET_MIN_ROWS)
            ):
                return self._explain_scalar_treelets(X)
            return self._explain_scalar_compact(X)
        # Non-complementary cover weights cannot use the collapsed sibling
        # identity exactly; retain the general full-state recurrence.

        cp = self.cp
        h = self.host
        d = self.dev
        n_samples = int(len(X))
        Xd = cp.asarray(np.ascontiguousarray(X.T))
        G = cp.empty((h.m_q, h.n_nodes, n_samples), dtype=cp.float64)
        count = n_samples * h.m_q * len(h.roots)
        self.init_roots(
            self._blocks(count), (256,),
            (G, d["roots"], len(h.roots), h.n_nodes, h.m_q, n_samples),
        )
        n_banks = (n_samples + 1024 - 1) // 1024
        for level in self.internal_levels:
            n_level = int(level.size)
            if n_level == 0:
                continue
            self.propagate_children_level(
                ((n_level * n_banks + 8 - 1) // 8,),
                (256,),
                (
                    Xd, G, level, n_level, d["left_child"],
                    d["right_child"], d["feature"], d["node_to_leaf"],
                    d["leaf_value"], int(h.n_outputs == 1),
                    d["edge_weight"], d["old_lower"], d["old_upper"],
                    d["new_upper"], d["prop_inside_new"],
                    d["prop_inside_old"], h.n_nodes, h.n_features, h.m_q,
                    n_samples,
                ),
            )

        count = n_samples * h.n_leaves
        if h.n_outputs == 1:
            H = G
        else:
            H = cp.empty(
                (h.m_q, h.n_outputs, h.n_nodes, n_samples),
                dtype=cp.float64,
            )
            self.gather_leaf_outputs(
                self._blocks(count), (256,),
                (
                    G, H, d["leaf_node"], d["leaf_value"], h.n_leaves,
                    h.n_nodes, h.n_outputs, h.m_q, n_samples,
                ),
            )
            del G

        for depth in range(len(self.internal_levels) - 1, -1, -1):
            level = self.internal_levels[depth]
            n_level = int(level.size)
            if n_level == 0:
                continue
            self.sum_children_level(
                ((n_level * n_banks + 8 - 1) // 8,),
                (256,),
                (
                    H, level, n_level, d["left_child"], d["right_child"],
                    h.n_nodes, h.n_outputs, h.m_q, n_samples,
                ),
            )

        out = cp.zeros(
            (n_samples, h.n_features, h.n_outputs), dtype=cp.float64
        )
        self.contract_parents_direct(
            (
                len(h.parent_used_features),
                (n_samples + 32 - 1) // 32,
                h.n_outputs,
            ),
            (256,),
            (
                Xd, H, out, d["parent_used_features"],
                d["feature_parent_offsets"], d["feature_parents"],
                len(h.parent_used_features), d["left_child"],
                d["right_child"], d["old_lower"], d["old_upper"],
                d["new_upper"], d["contract_inside_new"],
                d["contract_inside_old"], h.n_nodes, h.n_features,
                h.n_outputs, h.m_q, n_samples,
            ),
        )
        result = cp.asnumpy(out)
        del Xd, H, out
        return result

    def _explain_scalar_treelets(self, X: np.ndarray) -> np.ndarray:
        """Run the exact checkpointed treelet recurrence for a large batch."""
        cp = self.cp
        h = self.host
        d = self.dev
        t = self.treelets
        td = self.treelet_dev
        n_samples = int(len(X))
        Xd = cp.asarray(np.ascontiguousarray(X.T))
        state = cp.empty((t.total_q_states, n_samples), dtype=cp.float64)
        contribution = cp.empty(
            (t.max_level_parents, n_samples), dtype=cp.float64
        )
        out = cp.zeros((n_samples, h.n_features), dtype=cp.float64)

        n_banks = (
            n_samples + _TREELET_ROWS_PER_WARP - 1
        ) // _TREELET_ROWS_PER_WARP
        n_bank_groups = (
            n_banks + _TREELET_WARPS - 1
        ) // _TREELET_WARPS

        # Child-component root K must be complete before its parent component
        # reconstructs local state. Forest-root K is never consumed.
        for level_i in range(len(self.treelet_levels) - 1, 0, -1):
            level, _, _, _ = self.treelet_levels[level_i]
            n_level = int(level.size)
            if n_level == 0:
                continue
            self.build_treelet_level(
                (n_level * n_bank_groups,),
                (128,),
                (
                    Xd, state, level, n_level, td["headers"],
                    td["q_offsets"], td["int_record"], td["float_record"],
                    td["leaf_record"], td["portals"],
                    d["sibling_prop_s"], h.m_q, n_samples,
                ),
            )

        # Component depth ordering preserves the K->G overwrite invariant.
        # Each level is reduced immediately, allowing one partial buffer to be
        # reused for all component depths.
        for level_i, (
            level,
            used_features,
            feature_offsets,
            _,
        ) in enumerate(self.treelet_levels):
            n_level = int(level.size)
            if n_level == 0:
                continue
            self.propagate_contract_treelet_level(
                (n_level * n_bank_groups,),
                (128,),
                (
                    Xd, state, contribution, level, n_level,
                    int(level_i == 0), td["headers"], td["q_offsets"],
                    td["int_record"], td["float_record"], td["leaf_record"],
                    td["portals"], d["sibling_prop_s"],
                    d["sibling_contract_c"], h.m_q, n_samples,
                ),
            )
            n_used_features = int(used_features.size)
            n_feature_banks = (n_samples + 32 - 1) // 32
            self.reduce_treelet_level(
                (
                    (n_used_features * n_feature_banks + 8 - 1) // 8,
                ),
                (256,),
                (
                    contribution, out, used_features, feature_offsets,
                    n_used_features, h.n_features, n_samples,
                ),
            )

        result = cp.asnumpy(out)[:, :, None]
        del Xd, state, contribution, out
        return result

    def _explain_scalar_compact(self, X: np.ndarray) -> np.ndarray:
        """Use an internal-node-only workspace for scalar tree ensembles."""
        cp = self.cp
        h = self.host
        d = self.dev
        n_samples = int(len(X))
        if h.n_internal == 0:
            return np.zeros((n_samples, h.n_features, 1), dtype=np.float64)

        Xd = cp.asarray(np.ascontiguousarray(X.T))
        state = cp.empty(
            (h.n_internal, h.m_q, n_samples), dtype=cp.float64
        )
        partial = cp.empty((h.n_segments, n_samples), dtype=cp.float64)
        n_banks = (n_samples + 1024 - 1) // 1024

        # Build descendant-only subtree values K. The roots themselves are
        # not needed because the top-down pass starts with G_root = 1.
        for depth in range(len(self.compact_internal_levels) - 1, 0, -1):
            level = self.compact_internal_levels[depth]
            n_level = int(level.size)
            if n_level == 0:
                continue
            self.build_compact_subtrees_level(
                ((n_level * n_banks + 8 - 1) // 8,),
                (256,),
                (
                    Xd, state, level, n_level, d["compact_int_record"],
                    d["leaf_value"], d["compact_float_record"],
                    d["sibling_prop_s"],
                    h.n_internal, h.m_q, n_samples,
                ),
                shared_mem=8 * h.m_q * 8,
            )

        count = n_samples * h.m_q * len(h.internal_roots)
        self.init_compact_roots(
            self._blocks(count), (256,),
            (
                state, d["internal_roots"], len(h.internal_roots),
                h.n_internal, h.m_q, n_samples,
            ),
        )
        # Features unused by every tree have no segment and must remain zero.
        out = cp.zeros((n_samples, h.n_features), dtype=cp.float64)

        # Depth ordering makes every parent's G available while each internal
        # child's K is still intact. Grouping parents by split feature gives
        # one writer per (depth, feature, sample), so no atomics are needed.
        for features, offsets, parents, segment_ids in self.compact_levels:
            n_segments_level = int(features.size)
            if n_segments_level == 0:
                continue
            n_segment_banks = (n_samples + 64 - 1) // 64
            cache_factors = int(h.m_q <= 32)
            self.propagate_contract_compact_segments(
                ((n_segments_level * n_segment_banks + 8 - 1) // 8,),
                (256,),
                (
                    Xd, state, partial, features, offsets, parents,
                    segment_ids, n_segments_level, cache_factors,
                    d["compact_int_record"], d["leaf_value"],
                    d["compact_float_record"], d["sibling_prop_s"],
                    d["sibling_contract_c"], h.n_internal,
                    h.n_features, h.m_q, n_samples,
                ),
                shared_mem=(
                    8 * 8 * h.m_q * 2 * 8 if cache_factors else 0
                ),
            )

        n_feature_banks = (n_samples + 32 - 1) // 32
        n_used_features = len(h.segment_used_features)
        self.reduce_compact_segments(
            ((n_used_features * n_feature_banks + 8 - 1) // 8,),
            (256,),
            (
                partial, out, d["segment_used_features"],
                d["feature_segment_offsets"], d["feature_segment_ids"],
                n_used_features, h.n_features, n_samples,
            ),
        )
        result = cp.asnumpy(out)[:, :, None]
        del Xd, state, partial, out
        return result
