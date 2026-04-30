"""Custom exceptions for the AdaptivePortfolio system."""


class MoexConnectionError(Exception):
    """Raised when MOEX ISS API is unreachable after retries."""


class InsufficientDataError(Exception):
    """Raised when there are too few observations to fit a model."""

    def __init__(self, n_available: int = 0, required: int = 0, message: str = ""):
        if message:
            super().__init__(message)
        else:
            super().__init__(
                f"Insufficient data: have {n_available} observations, need at least {required}."
            )
        self.n_available = n_available
        self.required = required


class InvalidTickerError(Exception):
    """Raised when ticker is not found in MOEX ISS."""

    def __init__(self, ticker: str):
        super().__init__(f"Invalid ticker: {ticker}")
        self.ticker = ticker


class ConvergenceWarning(UserWarning):
    """Warning emitted when EM algorithm did not fully converge."""


class DegenerateRegimeError(Exception):
    """Raised when an estimated regime has near-zero variance."""

    def __init__(self, k: int):
        super().__init__(f"Regime {k} is degenerate (sigma < 1e-6).")
        self.k = k


class OptimizationFailedError(Exception):
    """Raised when CVXPY optimization does not return an optimal solution."""

    def __init__(self, status: str, k: int = -1):
        super().__init__(f"Optimization failed (status={status}, regime={k}).")
        self.status = status
        self.k = k
