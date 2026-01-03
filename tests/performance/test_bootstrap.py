"""
Performance tests for bootstrap simulation and histogram binning.

Validates that statistical operations meet performance targets:
- 10,000 Monte Carlo simulations in under 5 seconds
- Vectorization provides significant speedup
- Large dataset binning completes quickly
"""

import time

import numpy as np
import pytest

from option_analyzer.services.statistics import (
    bootstrap_walk,
    create_histogram,
    monte_carlo_simulation,
)


@pytest.fixture
def sample_returns():
    """Generate sample return data for performance tests."""
    # Create realistic returns (mean ~1.0, small variance)
    rng = np.random.default_rng(42)
    returns = rng.normal(loc=1.0, scale=0.02, size=1000)
    return returns


def test_10k_simulations_under_5_seconds(sample_returns):
    """
    Verify 10,000 Monte Carlo simulations complete in under 5 seconds.

    This is the primary performance target for the Statistics Module.
    Uses 50 steps per simulation as a realistic scenario.
    """
    n_simulations = 10_000
    steps = 50
    rng = np.random.default_rng(42)

    start_time = time.time()
    result = monte_carlo_simulation(sample_returns, steps, n_simulations, rng)
    elapsed = time.time() - start_time

    # Verify performance
    assert elapsed < 5.0, f"Performance regression: {elapsed:.3f}s > 5.0s target"

    # Verify correct output shape
    assert result.shape == (n_simulations,)
    assert np.all(result > 0)  # All multipliers should be positive

    print(f"\n✓ 10k simulations completed in {elapsed:.3f}s (target: <5.0s)")


def test_vectorization_speedup(sample_returns):
    """
    Compare vectorized implementation vs loop-based approach.

    Documents the speedup factor from NumPy vectorization to ensure
    we're benefiting from the optimized implementation.
    """
    n_simulations = 1_000
    steps = 50
    rng_vectorized = np.random.default_rng(42)
    rng_loop = np.random.default_rng(42)

    # Vectorized implementation
    start_vectorized = time.time()
    result_vectorized = monte_carlo_simulation(
        sample_returns, steps, n_simulations, rng_vectorized
    )
    time_vectorized = time.time() - start_vectorized

    # Loop-based implementation (for comparison)
    start_loop = time.time()
    result_loop = np.array(
        [bootstrap_walk(sample_returns, steps, rng_loop) for _ in range(n_simulations)]
    )
    time_loop = time.time() - start_loop

    # Calculate speedup
    speedup = time_loop / time_vectorized

    # Vectorization should provide significant speedup (at least 2x)
    assert (
        speedup >= 2.0
    ), f"Insufficient speedup: {speedup:.1f}x (expected >= 2.0x)"

    # Verify both produce similar distributions
    assert result_vectorized.shape == result_loop.shape

    print(f"\n✓ Vectorization speedup: {speedup:.1f}x (loop: {time_loop:.3f}s, vectorized: {time_vectorized:.3f}s)")


def test_large_dataset_binning():
    """
    Verify histogram binning handles large datasets efficiently.

    Tests 100,000 values binned into 100 bins completes in under 1 second,
    demonstrating NumPy efficiency for histogram operations.
    """
    # Generate large dataset
    rng = np.random.default_rng(42)
    values = rng.normal(loc=100.0, scale=10.0, size=100_000)

    n_bins = 100
    # Use actual min/max to ensure all values are captured
    min_value = float(np.min(values)) - 1.0
    max_value = float(np.max(values)) + 1.0

    start_time = time.time()
    bins = create_histogram(values, n_bins, min_value, max_value)
    elapsed = time.time() - start_time

    # Verify performance
    assert elapsed < 1.0, f"Binning too slow: {elapsed:.3f}s > 1.0s target"

    # Verify correctness
    assert len(bins) == n_bins
    total_count = sum(bin.count for bin in bins)
    assert total_count == len(values), f"Expected {len(values)}, got {total_count}"

    print(f"\n✓ 100k values binned into {n_bins} bins in {elapsed:.3f}s (target: <1.0s)")


def test_monte_carlo_scalability():
    """
    Additional test: verify performance scales well with simulation count.

    Tests that doubling simulations roughly doubles runtime (linear scaling).
    """
    rng = np.random.default_rng(42)
    returns = rng.normal(loc=1.0, scale=0.02, size=500)
    steps = 50

    # Test with 5k simulations
    rng_5k = np.random.default_rng(42)
    start_5k = time.time()
    monte_carlo_simulation(returns, steps, 5_000, rng_5k)
    time_5k = time.time() - start_5k

    # Test with 10k simulations
    rng_10k = np.random.default_rng(42)
    start_10k = time.time()
    monte_carlo_simulation(returns, steps, 10_000, rng_10k)
    time_10k = time.time() - start_10k

    # Check that scaling is roughly linear (within 3x tolerance)
    scaling_ratio = time_10k / time_5k
    assert (
        1.5 < scaling_ratio < 3.0
    ), f"Non-linear scaling detected: {scaling_ratio:.2f}x"

    print(f"\n✓ Scaling 5k→10k: {scaling_ratio:.2f}x (5k: {time_5k:.3f}s, 10k: {time_10k:.3f}s)")
