"""PAC-Bayes objectives, conservative binomial endpoint, and fresh final MC."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import (
    MAX_EMAX,
    MIN_EMIN,
    Context,
    Decimal,
    ROUND_CEILING,
    ROUND_FLOOR,
    getcontext,
)
import math
import operator
import struct

from scipy.special import betaincinv
import torch
from torch import Tensor

from .canonical import SymmetricIndependentScoreLift


@dataclass(frozen=True)
class Certificate:
    empirical: float
    budget: float
    upper: float


@dataclass(frozen=True)
class MonteCarloResult:
    seed: int
    trials: int
    errors: int
    observed_risk: float
    clopper_pearson_upper: float


def binary_kl(empirical: float, candidate: float) -> float:
    _unit_interval(empirical, "empirical")
    _unit_interval(candidate, "candidate")
    if empirical == candidate:
        return 0.0
    if empirical == 0.0:
        return math.inf if candidate == 1.0 else -math.log1p(-candidate)
    if empirical == 1.0:
        return math.inf if candidate == 0.0 else -math.log(candidate)
    if candidate in {0.0, 1.0}:
        return math.inf
    return empirical * math.log(empirical / candidate) + (
        1.0 - empirical
    ) * math.log((1.0 - empirical) / (1.0 - candidate))


def kl_upper_inverse(empirical: float, budget: float, *, iterations: int = 80) -> float:
    _unit_interval(empirical, "empirical")
    if budget < 0.0 or math.isnan(budget) or iterations <= 0:
        raise ValueError("invalid KL inverse request")
    if empirical == 1.0 or math.isinf(budget):
        return 1.0
    if budget == 0.0:
        return empirical
    if empirical == 0.0:
        return min(1.0, -math.expm1(-budget))
    lower, upper = empirical, 1.0
    for _ in range(iterations):
        midpoint = 0.5 * (lower + upper)
        if binary_kl(empirical, midpoint) > budget:
            upper = midpoint
        else:
            lower = midpoint
    return upper


def pac_bayes_certificate(
    empirical_risk: float,
    output_kl: float,
    sample_size: int,
    delta: float,
) -> Certificate:
    _confidence(sample_size, delta)
    _unit_interval(empirical_risk, "empirical_risk")
    if output_kl < -1e-10 or math.isnan(output_kl):
        raise ValueError("output KL must be nonnegative")
    constant = math.log(2.0) + 0.5 * math.log(sample_size) - math.log(delta)
    budget = (max(0.0, output_kl) + constant) / sample_size
    return Certificate(empirical_risk, budget, kl_upper_inverse(empirical_risk, budget))


def classic_objective(
    empirical_risk: Tensor,
    output_kl: Tensor,
    sample_size: int,
    delta: float,
) -> Tensor:
    _confidence(sample_size, delta)
    constant = output_kl.new_tensor(
        math.log(2.0) + 0.5 * math.log(sample_size) - math.log(delta)
    )
    return empirical_risk + torch.sqrt(
        (output_kl.clamp_min(0.0) + constant) / (2.0 * sample_size)
    )


def quadratic_objective(
    empirical_risk: Tensor,
    output_kl: Tensor,
    sample_size: int,
    delta: float,
) -> Tensor:
    _confidence(sample_size, delta)
    constant = output_kl.new_tensor(
        math.log(2.0) + 0.5 * math.log(sample_size) - math.log(delta)
    )
    complexity = (output_kl.clamp_min(0.0) + constant) / (2.0 * sample_size)
    return (torch.sqrt(empirical_risk + complexity) + torch.sqrt(complexity)).square()


class _DifferentiableKLUpperInverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx: object, empirical: Tensor, budget: Tensor) -> Tensor:
        if empirical.numel() != 1 or budget.numel() != 1:
            raise ValueError("empirical risk and budget must be scalars")
        upper = empirical.new_tensor(
            kl_upper_inverse(float(empirical.detach()), float(budget.detach()))
        )
        ctx.save_for_backward(
            empirical.detach().to(torch.float64), upper.detach().to(torch.float64)
        )
        return upper

    @staticmethod
    def backward(ctx: object, grad_output: Tensor) -> tuple[Tensor, Tensor]:
        empirical, upper = ctx.saved_tensors
        epsilon = torch.finfo(torch.float64).eps
        p = empirical.clamp(epsilon, 1.0 - epsilon)
        q = upper.clamp(epsilon, 1.0 - epsilon)
        derivative_q = (q - p) / (q * (1.0 - q))
        derivative_p = torch.log(p) - torch.log1p(-p) - torch.log(q) + torch.log1p(-q)
        return (
            grad_output * (-derivative_p / derivative_q).to(grad_output),
            grad_output * (1.0 / derivative_q).to(grad_output),
        )


def exact_objective(
    empirical_risk: Tensor,
    output_kl: Tensor,
    sample_size: int,
    delta: float,
) -> Tensor:
    _confidence(sample_size, delta)
    constant = output_kl.new_tensor(
        math.log(2.0) + 0.5 * math.log(sample_size) - math.log(delta)
    )
    budget = (output_kl.clamp_min(0.0) + constant) / sample_size
    return _DifferentiableKLUpperInverse.apply(empirical_risk, budget)


def training_objective(
    name: str,
    empirical_risk: Tensor,
    output_kl: Tensor,
    sample_size: int,
    delta: float,
) -> Tensor:
    if name == "classic":
        return classic_objective(empirical_risk, output_kl, sample_size, delta)
    if name == "quad":
        return quadratic_objective(empirical_risk, output_kl, sample_size, delta)
    if name == "exact":
        return exact_objective(empirical_risk, output_kl, sample_size, delta)
    raise ValueError(f"unknown objective: {name}")


@torch.inference_mode()
def gauss_hermite_risk(
    posterior: SymmetricIndependentScoreLift,
    features: Tensor,
    base_scores: Tensor,
    labels: Tensor,
    *,
    order: int,
    chunk_size: int,
) -> float:
    if labels.numel() == 0 or chunk_size <= 0:
        raise ValueError("risk evaluation needs a nonempty block and positive chunk")
    total = 0.0
    for start in range(0, labels.numel(), chunk_size):
        stop = min(start + chunk_size, labels.numel())
        total += float(
            posterior.errors_gauss_hermite(
                features[start:stop],
                base_scores[start:stop],
                labels[start:stop],
                order=order,
            ).sum()
        )
    return total / labels.numel()


@torch.inference_mode()
def fresh_iid_monte_carlo(
    posterior: SymmetricIndependentScoreLift,
    features: Tensor,
    base_scores: Tensor,
    labels: Tensor,
    *,
    trials: int,
    chunk_size: int,
    seed: int,
    delta: float,
) -> MonteCarloResult:
    """Sample fresh IID (uniform B index, posterior class scores) trials."""

    if min(trials, chunk_size) <= 0 or seed < 0:
        raise ValueError("invalid Monte Carlo request")
    if features.device.type != "cpu" or features.dtype != torch.float64:
        raise ValueError("final Monte Carlo requires CPU float64 inputs")
    means, variances = posterior.score_statistics(features, base_scores)
    if means.device.type != "cpu" or means.dtype != torch.float64:
        raise ValueError("final posterior must be CPU float64")
    stds = torch.sqrt(variances)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    errors = 0
    completed = 0
    while completed < trials:
        count = min(chunk_size, trials - completed)
        indices = torch.randint(
            0, labels.numel(), (count,), generator=generator, dtype=torch.long
        )
        noise = torch.randn(
            count,
            posterior.number_classes,
            generator=generator,
            dtype=torch.float64,
        )
        scores = means.index_select(0, indices) + stds.index_select(0, indices) * noise
        predictions = scores.argmax(dim=1)
        targets = labels.index_select(0, indices)
        errors += int(torch.count_nonzero(predictions != targets))
        completed += count
    observed = errors / trials
    upper = clopper_pearson_upper(errors, trials, delta)
    if upper + 1e-15 < observed:
        raise RuntimeError("Clopper-Pearson endpoint fell below the observation")
    return MonteCarloResult(seed, trials, errors, observed, upper)


def clopper_pearson_upper(errors: int, trials: int, delta: float) -> float:
    """Conservative one-sided endpoint validated with directed Decimal arithmetic."""

    error_count = _integer(errors, "errors")
    trial_count = _integer(trials, "trials")
    if trial_count <= 0 or error_count < 0 or error_count > trial_count:
        raise ValueError("invalid binomial counts")
    confidence_error = float(delta)
    if not math.isfinite(confidence_error) or not 0.0 < confidence_error < 1.0:
        raise ValueError("delta must lie in (0,1)")
    if error_count == trial_count:
        return 1.0
    if error_count == 0:
        endpoint = -math.expm1(math.log(confidence_error) / trial_count)
    else:
        endpoint = float(
            betaincinv(
                error_count + 1,
                trial_count - error_count,
                1.0 - confidence_error,
            )
        )
    if not math.isfinite(endpoint) or not 0.0 <= endpoint <= 1.0:
        raise RuntimeError("beta quantile returned an invalid endpoint")
    if endpoint == 1.0:
        return 1.0
    candidate = math.nextafter(endpoint, 1.0)
    delta_decimal = Decimal.from_float(confidence_error)
    if _binomial_cdf_upper_bound(error_count, trial_count, candidate) <= delta_decimal:
        return candidate
    return _first_conservative_float(
        error_count,
        trial_count,
        delta_decimal,
        invalid_candidate=candidate,
    )


def _first_conservative_float(
    errors: int,
    trials: int,
    delta: Decimal,
    *,
    invalid_candidate: float,
) -> float:
    low_bits = _float_bits(invalid_candidate)
    one_bits = _float_bits(1.0)
    stride = 1
    while True:
        high_bits = min(one_bits, low_bits + stride)
        if _binomial_cdf_upper_bound(errors, trials, _bits_float(high_bits)) <= delta:
            break
        if high_bits == one_bits:  # pragma: no cover
            raise RuntimeError("failed to bracket binomial endpoint")
        low_bits = high_bits
        stride *= 2
    while high_bits - low_bits > 1:
        midpoint = (low_bits + high_bits) // 2
        if _binomial_cdf_upper_bound(errors, trials, _bits_float(midpoint)) <= delta:
            high_bits = midpoint
        else:
            low_bits = midpoint
    return _bits_float(high_bits)


def _binomial_cdf_upper_bound(errors: int, trials: int, probability: float) -> Decimal:
    if probability <= 0.0:
        return Decimal(1)
    if probability >= 1.0:
        return Decimal(0) if errors < trials else Decimal(1)
    p = Decimal.from_float(probability)
    floor = _decimal_context(ROUND_FLOOR)
    ceiling = _decimal_context(ROUND_CEILING)
    one = Decimal(1)
    q_lower = floor.subtract(one, p)
    q_upper = ceiling.subtract(one, p)
    if errors + 1 <= trials - errors:
        term = _positive_integer_power(q_upper, trials, ceiling)
        total = term
        odds_upper = ceiling.divide(p, q_lower)
        for count in range(errors):
            ratio = ceiling.divide(Decimal(trials - count), Decimal(count + 1))
            term = ceiling.multiply(ceiling.multiply(term, ratio), odds_upper)
            total = ceiling.add(total, term)
        return min(one, total)
    term = _positive_integer_power(p, trials, floor)
    upper_tail_lower = term
    reverse_odds = floor.divide(q_lower, p)
    for count in range(trials, errors + 1, -1):
        ratio = floor.divide(Decimal(count), Decimal(trials - count + 1))
        term = floor.multiply(floor.multiply(term, ratio), reverse_odds)
        upper_tail_lower = floor.add(upper_tail_lower, term)
    upper_tail_lower = max(Decimal(0), min(one, upper_tail_lower))
    return ceiling.subtract(one, upper_tail_lower)


def _positive_integer_power(value: Decimal, exponent: int, context: Context) -> Decimal:
    result = Decimal(1)
    base = value
    remaining = exponent
    while remaining:
        if remaining & 1:
            result = context.multiply(result, base)
        remaining >>= 1
        if remaining:
            base = context.multiply(base, base)
    return result


def _decimal_context(rounding: str) -> Context:
    context = getcontext().copy()
    context.prec = 80
    context.rounding = rounding
    context.Emin = MIN_EMIN
    context.Emax = MAX_EMAX
    return context


def _float_bits(value: float) -> int:
    return struct.unpack(">Q", struct.pack(">d", value))[0]


def _bits_float(bits: int) -> float:
    return struct.unpack(">d", struct.pack(">Q", bits))[0]


def _integer(value: int, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        return operator.index(value)
    except TypeError as error:
        raise ValueError(f"{name} must be an integer") from error


def _confidence(sample_size: int, delta: float) -> None:
    if sample_size <= 0 or not 0.0 < delta < 1.0:
        raise ValueError("sample size must be positive and delta in (0,1)")


def _unit_interval(value: float, name: str) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must lie in [0,1]")


__all__ = [
    "Certificate",
    "MonteCarloResult",
    "clopper_pearson_upper",
    "fresh_iid_monte_carlo",
    "gauss_hermite_risk",
    "pac_bayes_certificate",
    "training_objective",
]
