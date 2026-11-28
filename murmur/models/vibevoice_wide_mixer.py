"""Exact causal mixer using one Metal threadgroup per wide feature row."""

from functools import lru_cache

import mlx.core as mx


@lru_cache(maxsize=1)
def wide_mixer_kernel():
    return mx.fast.metal_kernel(
        name="murmur_vibevoice_exact_wide_mixer",
        input_names=["x", "pre_weight", "taps", "bias", "scale", "post_weight", "eps"],
        output_names=["residual", "ffn_input"],
        source="""
            constexpr uint groups = C / 128;
            uint lane = thread_index_in_simdgroup;
            uint sg = simdgroup_index_in_threadgroup;
            uint channel = thread_index_in_threadgroup * 4;
            int time = int(threadgroup_position_in_grid.x);
            size_t base = size_t(threadgroup_position_in_grid.y) * x_shape[1] * C;
            threadgroup float partial[7][groups];
            threadgroup float inverse[7];
            float window[7][4];

            for (uint k = 0; k < 7; ++k) {
                int t = time + int(k) - 6;
                float s = 0.f;
                for (uint j = 0; j < 4; ++j) {
                    float v = t < 0 ? 0.f : x[base + size_t(t) * C + channel + j];
                    window[k][j] = v;
                    s = fma(v, v, s);
                }
                float sum = simd_sum(s);
                if (lane == 0) partial[k][sg] = sum;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Use the native second-level SIMD reduction, zero padding unused lanes.
            for (uint k = sg; k < 7; k += groups) {
                float s = lane < groups ? partial[k][lane] : 0.f;
                s = simd_sum(s);
                if (lane == 0) inverse[k] = metal::precise::rsqrt(s / float(C) + eps[0]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float mixed[4];
            float s = 0.f;
            for (uint j = 0; j < 4; ++j) {
                uint c = channel + j;
                float acc = 0.f;
                for (uint k = 0; k < 7; ++k) {
                    float n = (window[k][j] * inverse[k]) * float(pre_weight[c]);
                    float product = n * float(taps[c * 7 + k]);
                    acc = acc + product;
                }
                volatile float scaled = (acc + float(bias[c])) * float(scale[c]);
                mixed[j] = window[6][j] + scaled;
                s = fma(mixed[j], mixed[j], s);
            }
            s = simd_sum(s);
            if (lane == 0) partial[0][sg] = s;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (sg == 0) {
                s = lane < groups ? partial[0][lane] : 0.f;
                s = simd_sum(s);
                if (lane == 0) inverse[0] = metal::precise::rsqrt(s / float(C) + eps[1]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint j = 0; j < 4; ++j) {
                uint c = channel + j;
                size_t index = base + size_t(time) * C + c;
                residual[index] = mixed[j];
                ffn_input[index] = (mixed[j] * inverse[0]) * float(post_weight[c]);
            }
        """,
        compile_options={"math_mode": "safe"},
    )
