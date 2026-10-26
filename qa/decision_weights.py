"""Conservative class decision weights, fitted without confirmation answers."""
import numpy as np


def checked_probability(probability, count=None):
    probability = np.asarray(probability, dtype=float)
    if (probability.ndim != 2 or not len(probability) or probability.shape[1] < 2
            or (count is not None and probability.shape[1] != count)
            or not np.isfinite(probability).all() or (probability < 0).any()
            or not np.allclose(probability.sum(axis=1), 1, atol=1e-5)):
        raise ValueError('Expected finite, normalized class probabilities')
    return probability


def checked_labels(y, rows, count):
    y = np.asarray(y)
    if (y.shape != (rows,) or not np.issubdtype(y.dtype, np.integer)
            or (y < 0).any() or (y >= count).any()):
        raise ValueError('Expected aligned integer labels in the global class range')
    return y


def macro_f1(y, prediction, count):
    from sklearn.metrics import f1_score
    return float(f1_score(y, prediction, labels=np.arange(count), average='macro', zero_division=0))


def fit_decision_weights(probability, y):
    """Small fixed search; exact ties retain the less changed decision rule.

    This is decision optimization, not probability calibration. Callers must
    evaluate it on separate, untouched rows before accepting it.
    """
    probability = checked_probability(probability)
    count = probability.shape[1]
    y = checked_labels(y, len(probability), count)
    weights = np.ones(count)
    score = macro_f1(y, probability.argmax(axis=1), count)
    # Limit the search to moderate changes rather than memorizing rare labels.
    grid = (1., .8, 1.25, .64, 1.5625)
    for _ in range(2):
        changed = False
        for label in range(count):
            if np.sum(y == label) < 5:
                continue
            best_weight = weights[label]
            for value in grid:
                candidate = weights.copy()
                candidate[label] = value
                current = macro_f1(y, (probability * candidate).argmax(axis=1), count)
                if current > score + 1e-12:
                    score, best_weight = current, value
            changed |= weights[label] != best_weight
            weights[label] = best_weight
        if not changed:
            break
    return weights


def decision_probability(probability, weights):
    probability = checked_probability(probability)
    weights = np.asarray(weights, dtype=float)
    if (weights.shape != (probability.shape[1],) or not np.isfinite(weights).all()
            or (weights <= 0).any()):
        raise ValueError('Decision weights must be finite, positive and class aligned')
    weighted = probability * weights
    return weighted / weighted.sum(axis=1, keepdims=True)


def confirmed_weights(probability, y, weights, minimum_gain=.002):
    """One confirmation of a frozen proposal; never retune on these answers."""
    probability = checked_probability(probability)
    y = checked_labels(y, len(probability), probability.shape[1])
    if not np.isfinite(minimum_gain) or minimum_gain <= 0:
        raise ValueError('Confirmation requires a positive minimum gain')
    base = macro_f1(y, probability.argmax(axis=1), probability.shape[1])
    challenger = macro_f1(y, decision_probability(probability, weights).argmax(axis=1), probability.shape[1])
    accepted = challenger >= base + minimum_gain
    return (np.asarray(weights).copy() if accepted else np.ones(probability.shape[1])), dict(
        base_score=base, challenger_score=challenger, gain=challenger - base,
        minimum_gain=minimum_gain, accepted=bool(accepted))
