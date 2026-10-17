from math import factorial
import numpy as np


def exact_shapley(predict, x, background):
    """Enumerate the empirical marginal game, independently of SHAP code."""
    x, background = np.asarray(x, dtype=float), np.asarray(background, dtype=float)
    p, m = len(x), len(background)
    values = np.empty(2**p)
    hybrids = []
    for mask in range(2**p):
        rows = background.copy()
        for j in range(p):
            if mask & (1 << j):
                rows[:, j] = x[j]
        hybrids.append(rows)
    outputs = np.asarray(predict(np.vstack(hybrids))).reshape(2**p, m)
    values[:] = outputs.mean(axis=1)
    phi = np.zeros(p)
    for j in range(p):
        for mask in range(2**p):
            if not mask & (1 << j):
                k = mask.bit_count()
                weight = factorial(k) * factorial(p - k - 1) / factorial(p)
                phi[j] += weight * (values[mask | (1 << j)] - values[mask])
    return {"baseline": float(values[0]), "attributions": phi, "prediction": float(values[-1]), "coalition_values": values, "additivity_error": float(abs(values[0] + phi.sum() - values[-1]))}


def verify_toy():
    def toy(X):
        return .1 + .2 * X[:, 0] + .1 * X[:, 1] + .2 * X[:, 0] * X[:, 1] + .1 * X[:, 2]
    result = exact_shapley(toy, np.ones(3), np.zeros((1, 3)))
    assert np.allclose(result["attributions"], [.3, .2, .1], atol=1e-12)
    alternative = exact_shapley(toy, np.ones(3), np.array([[1, 0, 0]]))
    assert np.allclose(alternative["attributions"], [0, .3, .1], atol=1e-12)
    assert result["additivity_error"] < 1e-12
    return result, alternative
