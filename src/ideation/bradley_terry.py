"""Arc T / T1b - Bradley-Terry MAP fit with a Laplace posterior.

Why a fit and not sequential updates
------------------------------------
Spec sec. 4.1: LLM judges have non-transitive preferences, so a ranking
built by sequential Elo updates or by comparing everything to one
baseline depends on the order the matches happened to arrive in. A
batch fit over the whole match set does not. The whole point of running
both orientations and k samples is to average the noise out, and a
sequential rule re-introduces exactly the ordering dependence we paid
to remove.

The model
---------
For a match between ``i`` and ``j``::

    P(i beats j) = sigmoid(theta_i - theta_j)

with a Gaussian prior ``theta_i ~ N(mu_i, prior_sd**2)``. ``mu_i`` is
the DETERMINISTIC prior offset (spec sec. 4.1):

    mu_i = w_vf * venue_fit_i - w_pen * feasibility_penalty_i

so the deterministic terms enter as a prior on the strengths rather
than as a post-hoc addition to a fitted number. The prior is also what
makes the model identified: without it, only differences of theta are
determined and the Hessian is singular along the all-ones direction.

Posterior: MAP by damped Newton, then a Laplace approximation - the
covariance is the inverse Hessian of the negative log posterior at the
mode. ``sd`` is the square root of its diagonal, and top-k membership
probability is estimated by sampling that Gaussian.

Determinism
-----------
Everything here is deterministic given the same match set:

* candidates are ordered by a stable rule (first-seen, then sorted) and
  that order is recorded on the posterior;
* Newton iterates to a fixed tolerance from a fixed start (the prior
  mean), so the same input produces bit-identical output;
* the only randomness is the Laplace draw in :func:`top_k_membership`,
  which uses a seeded ``random.Random`` and ``normalvariate`` (which,
  unlike ``gauss``, keeps no cached spare value across calls).

Pure standard library: no numpy, no scipy. The fields here are at most
12 candidates, so an O(n^3) Cholesky in Python costs microseconds and
removes a dependency-version source of non-determinism.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "BTPosterior",
    "Match",
    "fit",
    "normalize_matches",
    "strength_order",
    "top_k_membership",
    "win_probability",
]

#: Outcome encoding. ``y`` is the probability mass assigned to ``left``.
WIN = 1.0
LOSS = 0.0
TIE = 0.5

DEFAULT_PRIOR_SD = 1.0
DEFAULT_SEED = 42
DEFAULT_N_DRAWS = 2000
_MAX_ITER = 100
_TOL = 1e-10


# --------------------------------------------------------------------------
# Match normalisation
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Match:
    """One binary (or tied) observation between two candidates.

    ``left``/``right`` are candidate ids. ``y`` is the outcome from
    ``left``'s point of view: 1.0 left won, 0.0 right won, 0.5 tie.
    ``source`` is free text used only to look up a weight (spec sec.
    5.6 path A weights a human pair at 5.0 against 1.0 per judge match).
    """

    left: str
    right: str
    y: float
    weight: float = 1.0
    source: str = "judge"

    def to_dict(self) -> dict:
        return {
            "left": self.left,
            "right": self.right,
            "y": self.y,
            "weight": self.weight,
            "source": self.source,
        }


def _weight_for(
    row: Mapping[str, Any] | None,
    index: int,
    weights: Any,
    source: str,
) -> float:
    """Resolve the weight for one match.

    Precedence, most specific first: an explicit ``weight`` on the row,
    then ``weights`` (a per-match sequence, a source->weight mapping, or
    a scalar), then 1.0.
    """
    if row is not None:
        explicit = row.get("weight")
        if isinstance(explicit, (int, float)) and not isinstance(explicit, bool):
            return float(explicit)
    if weights is None:
        return 1.0
    if isinstance(weights, Mapping):
        value = weights.get(source)
        return float(value) if isinstance(value, (int, float)) else 1.0
    if isinstance(weights, (int, float)) and not isinstance(weights, bool):
        return float(weights)
    if isinstance(weights, Sequence):
        if 0 <= index < len(weights):
            value = weights[index]
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return float(value)
        return 1.0
    return 1.0


def _pair_from_row(row: Mapping[str, Any]) -> tuple[str, str] | None:
    pair = row.get("pair")
    if isinstance(pair, (list, tuple)) and len(pair) == 2:
        return str(pair[0]), str(pair[1])
    for left_key, right_key in (("left", "right"), ("a", "b"), ("A", "B")):
        if row.get(left_key) and row.get(right_key):
            return str(row[left_key]), str(row[right_key])
    winner, loser = row.get("winner"), row.get("loser")
    if winner and loser:
        return str(winner), str(loser)
    return None


#: Sentinel for "this row carries no outcome field at all", kept
#: distinct from "this row names a winner who was not in the pair" so
#: the dropped-row reason says what actually happened (C2).
_NO_OUTCOME = "no-outcome-field"


def _outcome(row: Mapping[str, Any], left: str, right: str) -> Any:
    """Map a record's winner field onto ``y`` for ``left``.

    Returns None when the row names a winner in neither slot, and
    :data:`_NO_OUTCOME` when it carries no outcome field. Either way the
    row is dropped rather than guessed at, and the caller records which
    of the two it was.
    """
    if "y" in row:
        value = row.get("y")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    if "loser" in row and "winner" in row and row.get("winner") and row.get("loser"):
        return WIN if str(row["winner"]) == left else LOSS
    if "winner" not in row:
        return _NO_OUTCOME
    winner = row.get("winner")
    if winner is None:
        return TIE
    text = str(winner).strip()
    if text.lower() in {"tie", "draw", "none", ""}:
        return TIE
    if text == left:
        return WIN
    if text == right:
        return LOSS
    return None


def normalize_matches(
    matches: Iterable[Any],
    weights: Any = None,
) -> tuple[list[Match], list[dict]]:
    """Coerce heterogeneous match records into :class:`Match` objects.

    Accepts, in this order: ``Match`` instances; mappings in the
    ``matches.jsonl`` shape of spec Appendix B (``pair`` + ``winner``);
    mappings with ``left``/``right``/``winner``; ``(winner, loser)`` and
    ``(left, right, winner)`` tuples.

    Returns ``(matches, dropped)``. ``dropped`` rows carry the reason -
    a silently discarded match is a silently changed ranking.
    """
    out: list[Match] = []
    dropped: list[dict] = []
    for index, raw in enumerate(matches):
        if isinstance(raw, Match):
            out.append(raw)
            continue
        if isinstance(raw, Mapping):
            pair = _pair_from_row(raw)
            if pair is None:
                dropped.append(
                    {"index": index, "reason": "no candidate pair on the row",
                     "row": dict(raw)}
                )
                continue
            left, right = pair
            if left == right:
                dropped.append(
                    {"index": index, "reason": "self-match (left == right)",
                     "row": dict(raw)}
                )
                continue
            y = _outcome(raw, left, right)
            if y is _NO_OUTCOME:
                dropped.append(
                    {
                        "index": index,
                        "reason": "row carries no winner/y field",
                        "row": dict(raw),
                    }
                )
                continue
            if y is None:
                dropped.append(
                    {
                        "index": index,
                        "reason": (
                            "winner names neither candidate in the pair "
                            f"({left!r}, {right!r})"
                        ),
                        "row": dict(raw),
                    }
                )
                continue
            source = str(raw.get("source") or raw.get("judge_model") or "judge")
            out.append(
                Match(
                    left=left,
                    right=right,
                    y=y,
                    weight=_weight_for(raw, index, weights, source),
                    source=source,
                )
            )
            continue
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            items = list(raw)
            if len(items) == 2:
                left, right = str(items[0]), str(items[1])
                y = WIN
            elif len(items) >= 3:
                left, right = str(items[0]), str(items[1])
                winner = items[2]
                if winner is None or str(winner).lower() in {"tie", "draw", ""}:
                    y = TIE
                elif str(winner) == left:
                    y = WIN
                elif str(winner) == right:
                    y = LOSS
                else:
                    dropped.append(
                        {"index": index, "reason": "winner not in pair",
                         "row": list(items)}
                    )
                    continue
            else:
                dropped.append(
                    {"index": index, "reason": "tuple too short", "row": list(items)}
                )
                continue
            if left == right:
                dropped.append(
                    {"index": index, "reason": "self-match (left == right)",
                     "row": list(items)}
                )
                continue
            out.append(
                Match(
                    left=left,
                    right=right,
                    y=y,
                    weight=_weight_for(None, index, weights, "judge"),
                    source="judge",
                )
            )
            continue
        dropped.append({"index": index, "reason": f"unsupported type {type(raw).__name__}"})
    return out, dropped


# --------------------------------------------------------------------------
# Linear algebra (small, dense, pure python)
# --------------------------------------------------------------------------


def _cholesky(matrix: list[list[float]]) -> list[list[float]]:
    n = len(matrix)
    lower = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            total = 0.0
            for k in range(j):
                total += lower[i][k] * lower[j][k]
            if i == j:
                value = matrix[i][i] - total
                if value <= 0.0:
                    raise ValueError(
                        "Hessian is not positive definite at row "
                        f"{i} (value {value!r}); the Gaussian prior should "
                        "make this impossible - check prior_sd > 0."
                    )
                lower[i][j] = math.sqrt(value)
            else:
                lower[i][j] = (matrix[i][j] - total) / lower[j][j]
    return lower


def _chol_solve(lower: list[list[float]], rhs: Sequence[float]) -> list[float]:
    n = len(lower)
    y = [0.0] * n
    for i in range(n):
        total = rhs[i]
        for k in range(i):
            total -= lower[i][k] * y[k]
        y[i] = total / lower[i][i]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        total = y[i]
        for k in range(i + 1, n):
            total -= lower[k][i] * x[k]
        x[i] = total / lower[i][i]
    return x


def _chol_inverse(lower: list[list[float]]) -> list[list[float]]:
    n = len(lower)
    columns: list[list[float]] = []
    for j in range(n):
        unit = [1.0 if i == j else 0.0 for i in range(n)]
        columns.append(_chol_solve(lower, unit))
    # Symmetrise so tiny asymmetries from floating point never make the
    # covariance non-symmetric for the sampler's Cholesky.
    return [
        [0.5 * (columns[j][i] + columns[i][j]) for j in range(n)]
        for i in range(n)
    ]


def _log1pexp(x: float) -> float:
    if x > 35.0:
        return x
    if x < -35.0:
        return math.exp(x)
    return math.log1p(math.exp(x))


def _sigmoid(x: float) -> float:
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


# --------------------------------------------------------------------------
# Posterior
# --------------------------------------------------------------------------


@dataclass
class BTPosterior:
    """MAP strengths plus the Laplace covariance around them."""

    candidates: list[str]
    strength: dict[str, float]
    sd: dict[str, float]
    covariance: list[list[float]] = field(default_factory=list)
    prior_means: dict[str, float] = field(default_factory=dict)
    prior_sd: float = DEFAULT_PRIOR_SD
    n_matches: int = 0
    n_observations: float = 0.0
    matches_per_candidate: dict[str, int] = field(default_factory=dict)
    converged: bool = False
    iterations: int = 0
    log_posterior: float = 0.0
    dropped: list[dict] = field(default_factory=list)
    seed: int = DEFAULT_SEED

    def index(self, candidate: str) -> int:
        return self.candidates.index(candidate)

    def to_dict(self, digits: int = 6) -> dict:
        """Rounded, order-stable dict. Rounding is what makes two runs
        with the same match set byte-identical on disk."""
        return {
            "candidates": list(self.candidates),
            "strength": {c: round(self.strength[c], digits) for c in self.candidates},
            "sd": {c: round(self.sd[c], digits) for c in self.candidates},
            "prior_means": {
                c: round(self.prior_means.get(c, 0.0), digits)
                for c in self.candidates
            },
            "prior_sd": self.prior_sd,
            "n_matches": self.n_matches,
            "n_observations": round(self.n_observations, digits),
            "matches_per_candidate": {
                c: self.matches_per_candidate.get(c, 0) for c in self.candidates
            },
            "converged": self.converged,
            "iterations": self.iterations,
            "log_posterior": round(self.log_posterior, digits),
            "dropped_matches": self.dropped,
            "seed": self.seed,
        }


def _collect_candidates(
    matches: Sequence[Match], candidates: Sequence[str] | None
) -> list[str]:
    if candidates is not None:
        # Caller-supplied order wins, deduplicated, first occurrence kept.
        seen: dict[str, None] = {}
        for name in candidates:
            seen.setdefault(str(name), None)
        for match in matches:
            seen.setdefault(match.left, None)
            seen.setdefault(match.right, None)
        return list(seen)
    seen = {}
    for match in matches:
        seen.setdefault(match.left, None)
        seen.setdefault(match.right, None)
    return sorted(seen)


def fit(
    matches: Iterable[Any],
    weights: Any = None,
    *,
    prior_means: Mapping[str, float] | None = None,
    prior_sd: float = DEFAULT_PRIOR_SD,
    candidates: Sequence[str] | None = None,
    seed: int = DEFAULT_SEED,
    max_iter: int = _MAX_ITER,
    tol: float = _TOL,
) -> BTPosterior:
    """MAP fit with a Gaussian prior; Laplace approximation for the covariance.

    ``prior_means`` carries the deterministic terms (spec sec. 4.1). A
    candidate with no matches at all is still returned: its posterior is
    exactly its prior, which is the honest answer, and dropping it would
    silently shorten the ranking.
    """
    if prior_sd <= 0.0:
        raise ValueError("prior_sd must be positive; it is what identifies the fit")

    normalized, dropped = normalize_matches(matches, weights)
    names = _collect_candidates(normalized, candidates)
    n = len(names)
    index = {name: i for i, name in enumerate(names)}
    mu = [float((prior_means or {}).get(name, 0.0)) for name in names]

    if n == 0:
        return BTPosterior(
            candidates=[], strength={}, sd={}, covariance=[],
            prior_means={}, prior_sd=prior_sd, n_matches=0,
            converged=True, iterations=0, dropped=dropped, seed=seed,
        )

    rows = [
        (index[m.left], index[m.right], m.y, max(0.0, m.weight))
        for m in normalized
    ]
    per_candidate: dict[str, int] = {name: 0 for name in names}
    for match in normalized:
        per_candidate[match.left] += 1
        per_candidate[match.right] += 1

    inv_var = 1.0 / (prior_sd * prior_sd)

    def neg_log_post(theta: Sequence[float]) -> float:
        total = 0.0
        for i, j, y, w in rows:
            if w == 0.0:
                continue
            d = theta[i] - theta[j]
            total -= w * (-(1.0 - y) * d - _log1pexp(-d))
        for i in range(n):
            diff = theta[i] - mu[i]
            total += 0.5 * inv_var * diff * diff
        return total

    theta = list(mu)
    objective = neg_log_post(theta)
    converged = False
    iterations = 0

    for iterations in range(1, max_iter + 1):
        grad = [0.0] * n
        hess = [[0.0] * n for _ in range(n)]
        for i, j, y, w in rows:
            if w == 0.0:
                continue
            d = theta[i] - theta[j]
            p = _sigmoid(d)
            resid = w * (p - y)
            grad[i] += resid
            grad[j] -= resid
            curv = w * p * (1.0 - p)
            hess[i][i] += curv
            hess[j][j] += curv
            hess[i][j] -= curv
            hess[j][i] -= curv
        for i in range(n):
            grad[i] += inv_var * (theta[i] - mu[i])
            hess[i][i] += inv_var

        gnorm = max(abs(g) for g in grad)
        if gnorm < tol:
            converged = True
            break

        lower = _cholesky(hess)
        step = _chol_solve(lower, grad)

        # Damped Newton: halve the step until the objective actually
        # improves. Guarantees monotone progress without a line-search
        # library, and keeps the iteration deterministic.
        scale = 1.0
        accepted = False
        for _ in range(40):
            trial = [theta[i] - scale * step[i] for i in range(n)]
            trial_objective = neg_log_post(trial)
            if trial_objective <= objective + 1e-12:
                theta = trial
                objective = trial_objective
                accepted = True
                break
            scale *= 0.5
        if not accepted:
            converged = True
            break

    # Final Hessian at the mode -> Laplace covariance.
    hess = [[0.0] * n for _ in range(n)]
    for i, j, y, w in rows:
        if w == 0.0:
            continue
        d = theta[i] - theta[j]
        p = _sigmoid(d)
        curv = w * p * (1.0 - p)
        hess[i][i] += curv
        hess[j][j] += curv
        hess[i][j] -= curv
        hess[j][i] -= curv
    for i in range(n):
        hess[i][i] += inv_var
    lower = _cholesky(hess)
    covariance = _chol_inverse(lower)

    return BTPosterior(
        candidates=names,
        strength={name: theta[index[name]] for name in names},
        sd={
            name: math.sqrt(max(covariance[index[name]][index[name]], 0.0))
            for name in names
        },
        covariance=covariance,
        prior_means={name: mu[index[name]] for name in names},
        prior_sd=prior_sd,
        n_matches=len(normalized),
        n_observations=sum(w for _, _, _, w in rows),
        matches_per_candidate=per_candidate,
        converged=converged,
        iterations=iterations,
        log_posterior=-objective,
        dropped=dropped,
        seed=seed,
    )


def strength_order(posterior: BTPosterior) -> list[str]:
    """Candidates best first. Ties broken lexicographically, so the
    order is a pure function of the fitted strengths."""
    return sorted(
        posterior.candidates,
        key=lambda name: (-posterior.strength[name], name),
    )


def win_probability(posterior: BTPosterior, left: str, right: str) -> float:
    """P(left beats right) at the posterior mode."""
    return _sigmoid(posterior.strength[left] - posterior.strength[right])


def top_k_membership(
    posterior: BTPosterior,
    k: int = 2,
    n_draws: int = DEFAULT_N_DRAWS,
    seed: int = DEFAULT_SEED,
) -> dict[str, float]:
    """P(candidate in top-k) by sampling the Laplace posterior.

    Deterministic given ``seed``: ``random.Random.normalvariate`` keeps
    no cached spare value between calls (unlike ``gauss``), so the draw
    sequence depends only on the seed and the number of draws.

    A single-candidate field, k >= n, or a degenerate covariance all
    return 1.0 for everyone rather than raising - the caller is asking
    "how uncertain is membership", and "not uncertain at all" is a
    legitimate answer.
    """
    names = posterior.candidates
    n = len(names)
    if n == 0:
        return {}
    if k >= n:
        return {name: 1.0 for name in names}
    if k <= 0:
        return {name: 0.0 for name in names}
    if n_draws <= 0:
        top = set(strength_order(posterior)[:k])
        return {name: (1.0 if name in top else 0.0) for name in names}

    try:
        lower = _cholesky(posterior.covariance)
    except ValueError:
        top = set(strength_order(posterior)[:k])
        return {name: (1.0 if name in top else 0.0) for name in names}

    mean = [posterior.strength[name] for name in names]
    rng = random.Random(seed)
    counts = [0] * n
    for _ in range(n_draws):
        z = [rng.normalvariate(0.0, 1.0) for _ in range(n)]
        draw = []
        for i in range(n):
            total = mean[i]
            for j in range(i + 1):
                total += lower[i][j] * z[j]
            draw.append(total)
        order = sorted(range(n), key=lambda i: (-draw[i], names[i]))
        for i in order[:k]:
            counts[i] += 1
    return {names[i]: counts[i] / n_draws for i in range(n)}
