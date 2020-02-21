---
layout: post
title:  "GPU Friendly Fourier Transforms with GLSL: Part 1 - DFT"
date:   2020-02-17 23:17:00 -0600
categories: gpu fft glsl
---
# The Discrete Fourier Transform

The [Fourier transform](https://en.wikipedia.org/wiki/Fourier_transform) is an incredibly useful tool in signal and image processing. It gives us a way to analyze signals or images in terms of frequencies and [allows us to break down computationally expensive convolutions into cheap multiplications](https://en.wikipedia.org/wiki/Convolution_theorem).

As computer scientists and engineers, we are interested in the [Discrete Fourier transform](https://en.wikipedia.org/wiki/Discrete_Fourier_transform). In this blog series, we will take a look at implementing the Discrete Fourier Transform (DFT) in 1D and 2D from the bottom up on both the CPU and GPU, using C++ and Vulkan/GLSL. (As an aside, I will sacrifice performance for readability until later in the series).

The DFT of a signal {x<sub>n</sub>} of length N is given by:

[DFT Equation](https://latex.codecogs.com/svg.latex?X_k%3D%5Csum_%7Bn%3D0%7D%5E%7BN-1%7Dx_ne%5E%7B-2%7B%5Cpi%7Dink/N%7D)

We'll note that the values of {X<sub>k</sub>} are complex, so we'll be doing all of our calculations in the complex domain.

The DFT is simple enough to implement on the CPU:

```c++
using complex = std::complex<float>;

std::vector<complex> dft1D(std::vector<complex> &input)
{
    size_t N = input.size();
    std::vector<complex> output(N);
    float coef = -2*M_PI/N;
    for(size_t k=0; k<N; ++k)
    {
        output[k] = complex(0.0);
        for(size_t n=0; n<N; ++n)
        {
            float angle = coef*(k*n%N);
            output[k] += input[n]*std::polar(1.0f, angle);
        }
    }
    return output;
}
```

Running this with 1024 samples gives us 38.56ms - definitely not a real-time solution. We shouldn't be surprised by this though, given that we can see that this is an O(N<sup>2</sup>) algorithm. The 2D case is even worse, but we can do better than the naiive approach. Rearranging the DFT equation in 2D gives us

[2D DFT Equation](https://latex.codecogs.com/svg.latex?)

So for a signal/image of size (MxN), we simply apply M 1D DFTs of size N then N 1D DFTs of size M, giving us a run time of O(MN<sup>2</sup>+NM<sup>2</sup>) rather than O((NM)<sup>2</sup>). The implementation for 2D looks the same, but with another loop and some index management.

```c++
std::vector<std::vector<complex>> dft2D(std::vector<std::vector<complex>> &input)
{
    size_t N = input.size();
    size_t M = input[0].size();

    // Pass 1
    std::vector<std::vector<complex>> output(M);
    for(size_t i=0; i<M; ++i)
    {
        output[i].resize(N);
        for(size_t k=0; k<N; ++k)
        {
            output[i][k] = complex(0.0);
            for(size_t n=0; n<N; ++n)
            {
                float angle = -2.0*M_PI*k*n/N;
                output[i][k] += input[n][i]*std::polar(1.0f, angle);
            }
        }
    }

    // Pass 2
    std::vector<std::vector<complex>> tmp(output);
    for(size_t j=0; j<N; ++j)
    {
        for(size_t k=0; k<M; ++k)
        {
            output[k][j] = complex(0.0);
            for(size_t m=0; m<M; ++m)
            {
                float angle = -2.0*M_PI*k*m/M;
                output[k][j] += tmp[j][m]*std::polar(1.0f, angle);
            }
        }
    }

    return output;
}
```

Running this on a signal of size 1024x1024 takes a whopping 79 seconds! Fortunately, some very smart people have figured out ways we can improve the naiive DFT and get much better results. Read more in Part 2 - the FFT.
