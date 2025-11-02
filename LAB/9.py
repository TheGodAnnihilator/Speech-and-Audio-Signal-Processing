import numpy as np


def mdct(signal, block_size):
    """
    Compute the MDCT of a 1D signal.

    Parameters:
    - signal: 1D numpy array of input signal
    - block_size: size of each MDCT block (should be even)

    Returns:
    - mdct_coeffs: 2D numpy array with MDCT coefficients per block
    """
    # Make sure block size is even
    N = block_size
    if N % 2 != 0:
        raise ValueError("Block size must be even")

    # Pad signal to be multiple of block_size/2
    step = N // 2
    pad_length = (step - (len(signal) % step)) % step
    signal_padded = np.concatenate((signal, np.zeros(pad_length)))

    # Number of blocks
    num_blocks = (len(signal_padded) - N) // step + 1

    # Precompute MDCT basis matrix
    n = np.arange(N)
    k = np.arange(step)
    basis = np.cos(np.pi / step * (n + 0.5 + step / 2)[:, None] * (k + 0.5))

    # Compute MDCT for each block
    mdct_coeffs = np.empty((num_blocks, step))
    for i in range(num_blocks):
        block = signal_padded[i * step: i * step + N]
        mdct_coeffs[i, :] = np.dot(block, basis)

    return mdct_coeffs


# Example usage:
if __name__ == "__main__":
    # Generate example signal: sine wave
    fs = 8000
    t = np.linspace(0, 1, fs, endpoint=False)
    x = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz tone

    coeffs = mdct(x, block_size=512)
    print("MDCT coefficients shape:", coeffs.shape)
