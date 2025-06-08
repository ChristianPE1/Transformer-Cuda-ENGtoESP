// mask_utils.cuh
#ifndef MASK_UTILS_CUH
#define MASK_UTILS_CUH

#include <cuda_runtime.h>
#include "matrix.cuh"

class MaskUtils
{
public:
    // Create padding mask (1 where there are real tokens, 0 where there is padding)
    static __device__ Matrix createPaddingMask(const int *tokens, size_t seq_len, int pad_token = 0)
    {
        Matrix mask(1, seq_len);
        for (size_t i = 0; i < seq_len; ++i)
        {
            mask(0, i) = (tokens[i] == pad_token) ? 0.0 : 1.0;
        }
        return mask;
    }

    // Create look-ahead mask (lower triangular) for the decoder
    static __device__ Matrix createLookAheadMask(size_t seq_len)
    {
        Matrix mask(seq_len, seq_len, 0.0);
        for (size_t i = 0; i < seq_len; ++i)
        {
            for (size_t j = 0; j <= i; ++j)
            {
                mask(i, j) = 1.0;
            }
        }
        return mask;
    }

    // Combine padding and look-ahead masks for the decoder
    static __device__ Matrix combineDecoderMasks(const int *tokens, size_t seq_len, int pad_token = 0)
    {
        Matrix look_ahead = createLookAheadMask(seq_len);
        Matrix padding_mask = createPaddingMask(tokens, seq_len, pad_token);

        for (size_t i = 0; i < seq_len; ++i)
        {
            for (size_t j = 0; j < seq_len; ++j)
            {
                look_ahead(i, j) *= padding_mask(0, j);
            }
        }

        return look_ahead;
    }
};

#endif // MASK_UTILS_CUH