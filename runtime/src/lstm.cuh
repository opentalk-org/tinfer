#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>
namespace glue {
// Bidirectional single-layer LSTM. x[B,T,I] fp16 -> y[B,T,2H] fp16 (fwd|bwd concat).
// Weights fp16 per direction: Wih[4H,I], Whh[4H,H], bih[4H], bhh[4H]. xpF/xpR are
// [B,T,4H] fp32 scratch. Matches torch.nn.LSTM (gate order i,f,g,o).
void lstm_bidir(const __half* x, const __half* WihF, const __half* WhhF, const __half* bihF, const __half* bhhF,
                const __half* WihR, const __half* WhhR, const __half* bihR, const __half* bhhR,
                __half* y, float* xpF, float* xpR, int B, int T, int I, int H, cudaStream_t stream);
}
