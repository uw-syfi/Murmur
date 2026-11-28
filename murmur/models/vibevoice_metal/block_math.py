"""Metal arithmetic shared by exact VibeVoice mixer and FFN kernels."""

BLOCK_MATH = """
    // Match rms_single_row<float, 4>: lane-local FP32 fmas followed by
    // lane zero's SIMD reduction and the non-relaxed reciprocal square root.
    inline float vv_square_sum(float partial, float value) {
        return fma(value, value, partial);
    }

    inline float vv_inverse_rms(float lane_sum, uint width, float eps) {
        float total = simd_broadcast_first(simd_sum(lane_sum));
        return metal::precise::rsqrt(total / float(width) + eps);
    }

    // Stock MLX evaluates the channel scale and residual addition separately.
    inline float vv_add_scaled(float residual, float value, float bias, float scale) {
        volatile float scaled = (value + bias) * scale;
        return residual + scaled;
    }
"""
