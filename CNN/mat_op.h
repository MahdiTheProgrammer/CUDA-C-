#pragma once

__global__ void matrixmultiplication(float* a, float* b, float* c,
                                     int batch, int m, int n, int k);
