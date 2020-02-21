---
layout: post
title:  "GPU Friendly Fourier Transforms with GLSL: Part 2 - FFT"
date:   2020-02-17 23:17:00 -0600
categories: gpu fft glsl
---
# The Fast Fourier Transform

In the last post, we looked at the [Discrete Fourier Transform (DFT)](https://en.wikipedia.org/wiki/Discrete_Fourier_transform) and found that it was too slow for real-time work. Luckily we have the [Fast Fourier Transform (FFT)](https://en.wikipedia.org/wiki/Fast_Fourier_transform). FFT is simply a name for any algorithm that improves the runtime complexity of the DFT algorithm from O(N<sup>2</sup>) to O(N log(N)). We will first look at the most well-known algorithm ([Cooley-Tukey](https://en.wikipedia.org/wiki/Cooley-Tukey_FFT_algorithm)) before investigating others.

## The Cooley-Tukey Algorithm

We have actually already seen one of the key ideas for the Cooley-Tukey algorithm in the last post. The first observation is that given an integer N=N<sub>1</sub>xN<sub>2</sub>, we can write any number n between 0 and N-1 as n=N<sub>1</sub>n<sub>2</sub>+n<sub>1</sub> uniquely, where n<sub>1</sub> is between 0 and N<sub>1</sub>-1 and n<sub>2</sub> is between 0 and N<sub>2</sub>-1. Therefore, given a DFT of size N, we can rewrite it as:

[DFT C-T expansion](X_k=\sum_{n=0}^Nx_ne^{-2{\pi}ikn/N}=\sum_{n_1=0}^{N_1}\sum_{n_2=0}^{N_2}x_{N_1n_2+n_1}e^{-2{\pi}i\cdot(N_1n_2+n_1)\cdot(N_2k_1+k_2))

where k=N<sub>2</sub>k<sub>1</sub>+k<sub>2</sub>. This is just a 2D DFT of size N<sub>1</sub> by N<sub>2</sub>! Applying the separation technique from last post iteratively and using the symmetries in the roots of unity, we get O(N log(N)), much better than DFT's O(N^2).

For a given decomposition of N=N<sub>1</sub>xN<sub>2</sub>, we execute the following steps:
1. Perform N<sub>1</sub> DFTs of size N<sub>2</sub>
2. Multiply by the complex roots of unity (twiddle factors)
3. Perform N<sub>2</sub> DFTs of size N<sub>1</sub>

One of the factors (typically the smallest) is labeled as the "radix". If N<sub>1</sub> is the radix, the algorithm is called "decimation in time" (DIT), if N<sub>2</sub> is the radix, the algorithm is called "decimation in frequency" (DIF).

### Radix-2 DIT

The simplest case is when N is a power of 2. To compute the DFT here, we split the sequence into even and odd indices and the twiddle factor is simply e<sup>-2&pi;ik/N</sup>. Here's our first pass at implementing this case:

```c++
using complex = std::complex<double>;

std::vector<complex> ct_radix_2_recursive(std::vector<complex> &x, size_t stride, size_t offset)
{
    size_t N = x.size()/stride;
    std::vector<complex> output(N);

    // Base case
    if(N <= 1)
    {
        output[0] = x[offset];
        return output;
    }

    // Recursively compute DFT
    auto X_even = ct_radix_2_recursive(x, 2*stride, offset);
    auto X_odd  = ct_radix_2_recursive(x, 2*stride, stride+offset);

    // Combine
    for(size_t n=0; n<N/2; ++n)
    {
        double angle = -2.0*M_PI/N;
        complex twiddle_factor = std::polar(1.0, angle*n);

        output[n]     = X_even[n]+twiddle_factor*X_odd[n];
        output[n+N/2] = X_even[n]-twiddle_factor*X_odd[n];
    }
    return output;
}
```

Running this with a sample size of 1024 only takes 0.284ms now! (Recall that the naiive DFT took 38ms). However, we have some room for improvement:
- Precompute the N twiddle factors (roots of unity)
- Stop allocating extra vectors on the recursive calls

#### The Butterfly
