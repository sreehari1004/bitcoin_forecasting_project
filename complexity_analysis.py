import numpy as np
import nolds
import antropy as ant
import matplotlib.pyplot as plt
import zlib

def compute_hurst(ts):
    return nolds.hurst_rs(ts)

def compute_sample_entropy(ts):
    return ant.sample_entropy(ts)

def compute_lempel_ziv_complexity(ts):
    ts_binary = (ts > np.mean(ts)).astype(int)
    return ant.lziv_complexity(ts_binary)

def compute_kolmogorov_complexity(ts):
    ts_bytes = np.array(ts, dtype=np.float32).tobytes()
    compressed = zlib.compress(ts_bytes)
    compression_ratio = len(compressed) / len(ts_bytes)
    return compression_ratio

def plot_time_series(ts, title="Bitcoin Close Price"):
    plt.figure(figsize=(12, 6))
    plt.plot(ts, color='purple')
    plt.title(title)
    plt.xlabel("Time Steps")
    plt.ylabel("Normalized Close Price")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    import pandas as pd
    from src.preprocessing import load_and_preprocess

    # Load the dataset
    file_path = 'data/bitcoin_5min_latest.csv'
    X, y, df, scaler, timestamps = load_and_preprocess(file_path, n_lags=5)

    # Get the Close price (denormalized)
    close_prices = df['Close'].values

    # Complexity metrics
    hurst = compute_hurst(close_prices)
    sampen = compute_sample_entropy(close_prices)
    lziv = compute_lempel_ziv_complexity(close_prices)
    kolmogorov = compute_kolmogorov_complexity(close_prices)

    print("--- Complexity Analysis Results ---")
    print(f"Hurst Exponent: {hurst:.4f}")
    print(f"Sample Entropy: {sampen:.4f}")
    print(f"Lempel-Ziv Complexity: {lziv:.4f}")
    print(f"Kolmogorov Compression Ratio: {kolmogorov:.4f}")

    # Optional: Plot time series
    plot_time_series(close_prices)

if __name__ == "__main__":
    main()
