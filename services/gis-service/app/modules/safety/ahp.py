"""
SylvaNexus — Analytic Hierarchy Process (AHP)
================================================
Derives criteria weights from expert pairwise comparison matrices, and checks
whether those judgements are internally consistent.

Used to weight the occupational safety criteria for forest operations, so that
weights are auditable and region-specific rather than hard-coded guesses.

Method (Saaty 1980):
  1. Experts compare criteria pairwise on the 1-9 fundamental scale.
     a[i][j] = how much more important criterion i is than criterion j.
     Matrix is reciprocal: a[j][i] = 1 / a[i][j].
  2. Weights are the principal eigenvector, approximated here by the
     normalised geometric mean of each row (Crawford & Williams 1985).
     This "row geometric mean" estimator is standard practice and agrees
     closely with the exact eigenvector for consistent matrices.
  3. Consistency Ratio CR = CI / RI, where
       CI = (λ_max - n) / (n - 1)
       RI = Saaty's random index for matrix size n
     CR <= 0.10 is the accepted threshold. Above that, the judgements are
     contradictory and must be revised.

References:
  Saaty, T.L. (1980) The Analytic Hierarchy Process. McGraw-Hill.
  Saaty, T.L. (1987) The analytic hierarchy process — what it is and how it is
    used. Mathematical Modelling, 9(3-5):161-176.
  Crawford, G. & Williams, C. (1985) A note on the analysis of subjective
    judgment matrices. J. Mathematical Psychology, 29(4):387-405.
  Rahmawati, Yovi & Setiawan (2025) Advancing occupational safety in forest
    management through a new GIS-AHP integrated framework.
    European Journal of Forest Engineering, 11(2).
"""

from typing import Dict, List, Sequence, Tuple


# Saaty's Random Consistency Index, indexed by matrix size n.
# Average CI of randomly generated reciprocal matrices.
RANDOM_INDEX: Dict[int, float] = {
    1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12,
    6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49,
    11: 1.51, 12: 1.48, 13: 1.56, 14: 1.57, 15: 1.59,
}

# Saaty's accepted upper bound for the consistency ratio.
MAX_CONSISTENCY_RATIO = 0.10


class InconsistentMatrixError(ValueError):
    """Raised when a pairwise comparison matrix is too inconsistent to use."""

    def __init__(self, name: str, consistency_ratio: float):
        self.name = name
        self.consistency_ratio = consistency_ratio
        super().__init__(
            f"Pairwise matrix '{name}' has CR={consistency_ratio:.3f}, "
            f"above the accepted limit of {MAX_CONSISTENCY_RATIO:.2f}. "
            f"The expert judgements contradict each other and must be revised."
        )


def _validate_matrix(matrix: Sequence[Sequence[float]], name: str) -> int:
    """Check the matrix is square, positive and reciprocal. Returns its size."""
    n = len(matrix)
    if n == 0:
        raise ValueError(f"Pairwise matrix '{name}' is empty.")
    for i, row in enumerate(matrix):
        if len(row) != n:
            raise ValueError(
                f"Pairwise matrix '{name}' is not square: "
                f"row {i} has {len(row)} entries, expected {n}."
            )
        for j, value in enumerate(row):
            if value <= 0:
                raise ValueError(
                    f"Pairwise matrix '{name}' has non-positive entry "
                    f"at [{i}][{j}]: {value}. AHP requires positive ratios."
                )
    for i in range(n):
        for j in range(n):
            product = matrix[i][j] * matrix[j][i]
            if abs(product - 1.0) > 1e-6:
                raise ValueError(
                    f"Pairwise matrix '{name}' is not reciprocal at "
                    f"[{i}][{j}]: a[i][j]*a[j][i]={product:.4f}, expected 1.0."
                )
    return n


def derive_weights(matrix: Sequence[Sequence[float]],
                   name: str = "matrix") -> Tuple[List[float], float]:
    """
    Derive criteria weights and the consistency ratio from a pairwise matrix.

    Returns (weights, consistency_ratio) where weights sum to 1.0 and are
    ordered to match the matrix rows.
    """
    n = _validate_matrix(matrix, name)

    # Row geometric means, then normalise to sum to 1
    geometric_means = []
    for row in matrix:
        product = 1.0
        for value in row:
            product *= value
        geometric_means.append(product ** (1.0 / n))

    total = sum(geometric_means)
    weights = [g / total for g in geometric_means]

    # λ_max via the weighted column sums: λ_max = Σᵢ wᵢ × (Σⱼ aᵢⱼ ... )
    # Practically: λ_max = Σⱼ (colsum_j × w_j)
    lambda_max = 0.0
    for j in range(n):
        column_sum = sum(matrix[i][j] for i in range(n))
        lambda_max += column_sum * weights[j]

    if n <= 2:
        # A 2x2 reciprocal matrix is always perfectly consistent
        consistency_ratio = 0.0
    else:
        consistency_index = (lambda_max - n) / (n - 1)
        random_index = RANDOM_INDEX.get(n)
        if random_index is None:
            raise ValueError(
                f"No Saaty random index available for matrix size {n} "
                f"('{name}'). Split the criteria into smaller groups."
            )
        consistency_ratio = consistency_index / random_index

    return weights, max(consistency_ratio, 0.0)


def weights_for(labels: Sequence[str],
                matrix: Sequence[Sequence[float]],
                name: str = "matrix",
                strict: bool = True) -> Tuple[Dict[str, float], float]:
    """
    Derive named weights from a pairwise matrix.

    `labels` must be in the same order as the matrix rows. When `strict`, an
    inconsistent matrix raises rather than silently producing weights that no
    expert would endorse.
    """
    if len(labels) != len(matrix):
        raise ValueError(
            f"Pairwise matrix '{name}' has {len(matrix)} rows but "
            f"{len(labels)} labels were given."
        )

    weights, consistency_ratio = derive_weights(matrix, name)
    if strict and consistency_ratio > MAX_CONSISTENCY_RATIO:
        raise InconsistentMatrixError(name, consistency_ratio)

    return dict(zip(labels, weights)), consistency_ratio
