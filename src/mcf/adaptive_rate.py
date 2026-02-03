"""
Adaptive rate limiter for API requests.

Dynamically adjusts request rate based on API responses:
- Backs off aggressively on rate limits (429)
- Recovers slowly after sustained success
- Provides observable rate state for logging
"""

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class RateState:
    """Current state of the rate limiter."""
    current_rps: float
    success_streak: int
    total_rate_limits: int
    total_recoveries: int


class AdaptiveRateLimiter:
    """
    Manages dynamic request rate based on API responses.

    Behavior:
    - Starts at initial_rps (default 2.0)
    - On 429 (rate limit): Immediately cut rate by backoff_factor (50%)
    - After recovery_threshold consecutive successes: Increase by recovery_factor (10%)
    - Rate is clamped between min_rps and max_rps

    Example:
        limiter = AdaptiveRateLimiter(initial_rps=2.0)

        # On successful request
        new_rps = limiter.on_success()

        # On rate limit (429)
        new_rps = limiter.on_rate_limited()

        # Check current rate
        print(f"Current rate: {limiter.current_rps}")
    """

    def __init__(
        self,
        initial_rps: float = 2.0,
        min_rps: float = 0.5,
        max_rps: float = 5.0,
        backoff_factor: float = 0.5,
        recovery_factor: float = 1.1,
        recovery_threshold: int = 50,
    ):
        """
        Initialize the adaptive rate limiter.

        Args:
            initial_rps: Starting requests per second
            min_rps: Minimum RPS (never go slower)
            max_rps: Maximum RPS (never go faster)
            backoff_factor: Multiply RPS by this on rate limit (0.5 = halve)
            recovery_factor: Multiply RPS by this on recovery (1.1 = 10% increase)
            recovery_threshold: Consecutive successes before recovery
        """
        self.min_rps = min_rps
        self.max_rps = max_rps
        self.backoff_factor = backoff_factor
        self.recovery_factor = recovery_factor
        self.recovery_threshold = recovery_threshold

        self._current_rps = initial_rps
        self._success_streak = 0
        self._total_rate_limits = 0
        self._total_recoveries = 0

    @property
    def current_rps(self) -> float:
        """Current requests per second rate."""
        return self._current_rps

    def on_rate_limited(self) -> float:
        """
        Called when rate limited (429 response).

        Immediately backs off by backoff_factor and resets success streak.

        Returns:
            New RPS after backoff
        """
        old_rps = self._current_rps
        self._success_streak = 0
        self._total_rate_limits += 1
        self._current_rps = max(self.min_rps, self._current_rps * self.backoff_factor)

        logger.warning(
            f"Rate limited! Backing off: {old_rps:.2f} -> {self._current_rps:.2f} req/sec "
            f"(total rate limits: {self._total_rate_limits})"
        )

        return self._current_rps

    def on_success(self) -> float:
        """
        Called on successful request.

        Increments success streak and may increase rate after threshold.

        Returns:
            Current RPS (may be increased)
        """
        self._success_streak += 1

        if self._success_streak >= self.recovery_threshold:
            old_rps = self._current_rps
            self._current_rps = min(self.max_rps, self._current_rps * self.recovery_factor)
            self._success_streak = 0  # Reset after recovery
            self._total_recoveries += 1

            if self._current_rps > old_rps:
                logger.info(
                    f"Rate recovery: {old_rps:.2f} -> {self._current_rps:.2f} req/sec "
                    f"(recovery #{self._total_recoveries})"
                )

        return self._current_rps

    def on_error(self) -> float:
        """
        Called on non-rate-limit error.

        Partially resets success streak but doesn't change rate.

        Returns:
            Current RPS (unchanged)
        """
        # Reduce streak but don't reset completely
        self._success_streak = max(0, self._success_streak - 10)
        return self._current_rps

    def get_state(self) -> RateState:
        """Get current rate limiter state for monitoring."""
        return RateState(
            current_rps=self._current_rps,
            success_streak=self._success_streak,
            total_rate_limits=self._total_rate_limits,
            total_recoveries=self._total_recoveries,
        )

    def reset(self, initial_rps: float | None = None) -> None:
        """
        Reset the rate limiter to initial state.

        Args:
            initial_rps: New initial RPS (uses min_rps if not specified)
        """
        self._current_rps = initial_rps or (self.min_rps + self.max_rps) / 2
        self._success_streak = 0
        logger.info(f"Rate limiter reset to {self._current_rps:.2f} req/sec")
