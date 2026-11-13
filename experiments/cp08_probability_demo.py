"""Run a small probability and sampling experiment."""

from nn.probability import (
    empirical_distribution,
    entropy_bits,
    expected_value,
    sample_categorical,
    variance,
)


def main() -> None:
    """Print exact statistics and seeded sample frequencies."""
    values = [0.0, 1.0]
    probabilities = [0.25, 0.75]
    samples = sample_categorical(probabilities, sample_count=20_000, seed=808)
    print(f"Expected value: {expected_value(values, probabilities):.3f}")
    print(f"Variance: {variance(values, probabilities):.3f}")
    print(f"Entropy: {entropy_bits(probabilities):.3f} bits")
    print("Sample frequencies:", empirical_distribution(samples, category_count=2))


if __name__ == "__main__":
    main()
