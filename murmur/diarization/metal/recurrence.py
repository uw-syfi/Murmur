"""Batched bidirectional recurrence for pyannote segmentation."""

import mlx.core as mx


# Each threadgroup advances eight independent windows in the same direction.
# Register-resident weight fragments are shared by all eight recurrent rows.
batched_lstm = mx.fast.metal_kernel(
    name="pyannet_batched_recurrence",
    input_names=["projected_f", "recurrent_f", "projected_r", "recurrent_r"],
    output_names=["result"],
    header="#include <metal_simdgroup_matrix>\n",
    source="""
        uint lane = thread_position_in_threadgroup.x;
        uint simd = simdgroup_index_in_threadgroup;
        uint batch = threadgroup_position_in_grid.y * 8;
        uint direction = threadgroup_position_in_grid.z;
        uint steps = projected_f_shape[1];
        device const float* projected = direction ? projected_r : projected_f;
        device const float* weights = direction ? recurrent_r : recurrent_f;
        threadgroup float hidden[8 * 128];
        threadgroup float gates[8 * 512];
        float cell[2] = {0, 0};
        hidden[lane] = 0;
        hidden[lane + 512] = 0;
        simdgroup_float8x8 packed[16][4];
        #pragma clang loop unroll(full)
        for (uint k = 0; k < 16; ++k) {
            #pragma clang loop unroll(full)
            for (uint tile = 0; tile < 4; ++tile) {
                uint column = (tile * 16 + simd) * 8;
                simdgroup_load(packed[k][tile], weights + k * 8 * 512 + column, 512);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = 0; i < steps; ++i) {
            uint t = direction ? steps - 1 - i : i;
            simdgroup_float8x8 acc[4];
            #pragma clang loop unroll(full)
            for (uint tile = 0; tile < 4; ++tile) {
                uint column = (tile * 16 + simd) * 8;
                simdgroup_load(acc[tile], projected + (batch * steps + t) * 512 + column,
                               steps * 512);
            }
            #pragma clang loop unroll(full)
            for (uint k = 0; k < 16; ++k) {
                simdgroup_float8x8 prev;
                simdgroup_load(prev, hidden + k * 8, 128);
                #pragma clang loop unroll(full)
                for (uint tile = 0; tile < 4; ++tile)
                    simdgroup_multiply_accumulate(acc[tile], prev, packed[k][tile], acc[tile]);
            }
            for (uint tile = 0; tile < 4; ++tile) {
                uint column = (tile * 16 + simd) * 8;
                simdgroup_store(acc[tile], gates + column, 512);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint part = 0; part < 2; ++part) {
                uint index = lane + part * 512;
                uint row = index / 128;
                uint col = index % 128;
                threadgroup const float* g = gates + row * 512 + col;
                float input = 1 / (1 + metal::precise::exp(-g[0]));
                float forget = 1 / (1 + metal::precise::exp(-g[128]));
                float output = 1 / (1 + metal::precise::exp(-g[384]));
                cell[part] = forget * cell[part] + input * metal::precise::tanh(g[256]);
                float h = output * metal::precise::tanh(cell[part]);
                hidden[index] = h;
                result[((batch + row) * steps + t) * 256 + direction * 128 + col] = h;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    """,
)
