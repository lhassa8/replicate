"""Numerical export checks; these do not grade a model's task performance."""

import math


def require(condition, message):
    if not condition:
        raise ValueError(message)


def log_probabilities(logits):
    import numpy as np
    value = np.asarray(logits, dtype=np.float64)
    require(value.ndim == 2 and all(size > 0 for size in value.shape)
            and np.isfinite(value).all(), "Expected finite complete logit rows")
    shifted = value - value.max(axis=1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))


def _ordered_entries(entries, vocabulary_size):
    import numpy as np
    require(type(vocabulary_size) is int and vocabulary_size > 0
            and len(entries) == vocabulary_size, "Incomplete probability vocabulary")
    result = np.empty(vocabulary_size, dtype=np.float64)
    seen = set()
    for entry in entries:
        identity, value = entry["id"], entry["logprob"]
        require(type(identity) is int and 0 <= identity < vocabulary_size and identity not in seen,
                "Duplicate or invalid probability token ID")
        require(type(value) in (int, float) and math.isfinite(value), "Nonfinite token log probability")
        seen.add(identity)
        result[identity] = value
    return result


def ordered_log_probabilities(entries, vocabulary_size):
    import numpy as np
    result = _ordered_entries(entries, vocabulary_size)
    require(abs(float(np.exp(result).sum()) - 1) <= 1e-5, "Probabilities are not normalized over the vocabulary")
    return result


def native_probability_rounding(log_probabilities):
    """Check the pinned server's sequential F32 softmax accounting.

    This permits only a bounded normalization error. Callers still compare
    the original, uncorrected log probabilities against the model tolerance.
    It is not a model-quality or formal floating-point proof.
    """
    import numpy as np
    values = np.asarray(log_probabilities, dtype=np.float64)
    require(values.ndim == 2 and all(size > 0 for size in values.shape)
            and values.shape[1] <= 1_048_576 and np.isfinite(values).all()
            and (values <= 0).all(), "Expected finite native probability rows")
    require(np.array_equal(values, values.astype(np.float32).astype(np.float64)),
            "Reported native log probabilities were not F32 values")
    unit = float(np.finfo(np.float32).eps / 2)
    product = (values.shape[1] - 1) * unit
    gamma = product / (1 - product)
    # Sequential positive summation, division, and an allowance for the F32
    # log/exp round trip. This is a data-format check, not the output tolerance.
    round_trip = np.expm1(8 * unit * (1 + np.max(np.abs(values), axis=1)))
    allowance = gamma / (1 - gamma) + round_trip
    normalizers = np.exp(values).sum(axis=1)
    require(np.all(np.abs(normalizers - 1) <= allowance),
            "Native probability normalization exceeds its F32 accounting bound")
    return {
        "policy": "sequential-f32-softmax-accounting-v1",
        "raw_normalizers": normalizers.tolist(),
        "normalization_error_allowances": allowance.tolist(),
        "raw_log_probabilities_preserved": True,
    }


def ordered_native_log_probabilities(entries, vocabulary_size):
    result = _ordered_entries(entries, vocabulary_size)
    return result, native_probability_rounding(result[None, :])


def compare_probabilities(reference_logits, observed_log_probabilities):
    import numpy as np
    reference = log_probabilities(reference_logits)
    observed = np.asarray(observed_log_probabilities, dtype=np.float64)
    require(observed.shape == reference.shape and np.isfinite(observed).all(),
            "Probability rows do not match the complete reference")
    normalizers = np.exp(observed).sum(axis=1)
    require(np.all(np.abs(normalizers - 1) <= 1e-5), "Observed probability rows are not normalized")
    normalized = observed - np.log(normalizers)[:, None]
    divergence = (np.exp(reference) * (reference - normalized)).sum(axis=1)
    require(np.all(divergence >= -1e-10), "Invalid negative divergence")
    return {
        "rows": reference.shape[0], "vocabulary_size": reference.shape[1],
        "maximum_absolute_log_probability_error": float(np.max(np.abs(reference - observed))),
        "mean_kl_divergence": float(np.maximum(divergence, 0).mean()),
        "per_row_kl_divergence": np.maximum(divergence, 0).tolist(),
        "top_one_agreement": int(np.sum(reference.argmax(axis=1) == observed.argmax(axis=1))),
        "qualification_passed": False,
    }


def compare_native_probabilities(reference_logits, observed_log_probabilities):
    import numpy as np
    observed = np.asarray(observed_log_probabilities, dtype=np.float64)
    accounting = native_probability_rounding(observed)
    reference = log_probabilities(reference_logits)
    require(observed.shape == reference.shape, "Native probability rows changed shape")
    normalized = observed - np.log(np.exp(observed).sum(axis=1))[:, None]
    result = compare_probabilities(reference_logits, normalized)
    result["maximum_normalized_log_probability_error"] = result["maximum_absolute_log_probability_error"]
    # Preserve the original comparison: normalization cannot hide raw error.
    result["maximum_absolute_log_probability_error"] = float(np.max(np.abs(reference - observed)))
    result["native_probability_accounting"] = accounting
    return result


def check_merge(reference_logits, merged_logits, *, maximum_error):
    import numpy as np
    # Reuse finite/shape checks, while comparing the actual unnormalized logits.
    log_probabilities(reference_logits)
    log_probabilities(merged_logits)
    require(reference_logits.shape == merged_logits.shape, "Merged logit rows changed shape")
    error = float(np.max(np.abs(reference_logits.astype(np.float64) - merged_logits.astype(np.float64))))
    require(error <= maximum_error, "Adapter merge changed logits beyond the declared tolerance")
    return {"passed": True, "maximum_absolute_logit_error": error,
            "maximum_allowed_error": maximum_error, "qualification_passed": False}
