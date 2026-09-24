"""Non-Decreasing Continuous PieceWise Linear Functions (NDCPWLF).

This module is the canonical, pure-Python reference for the function algebra
used by the time-dependent (TD) solution checker: arc arrival-time functions,
time-window ready-time functions, and their compositions. It is deliberately
kept simple and fully deterministic (plain IEEE-754 double arithmetic, no
epsilon comparisons) so that it can be reimplemented independently and produce
bit-identical results.

Representation: a function is the list of its breakpoints, split into two
parallel arrays ``xs`` (non-decreasing) and ``ys`` (non-decreasing). Between
two consecutive breakpoints the function is linear. Duplicate ``xs`` entries
encode a vertical step; evaluation at a step returns the smallest value.
Steps arise both as floating-point rounding artifacts of composition and as
genuine content of canonical ATFs consolidated from stepwise travel-time data
(e.g. Rifki2020, FIFO-restored by the arrival-time lower envelope: surviving
upward jumps are real steps, and the smallest-value convention is exactly the
envelope's value at the boundary). Duplicate ``ys`` entries encode a plateau
(e.g. waiting for a time window to open).

Composition follows the two-pointer event merge of Visser & Spliet (2020),
with two deliberate deviations that keep the canonical spec simple:

- no normalization (redundant breakpoints are kept; values are unchanged);
- interpolation is done directly between the two enclosing breakpoints
  instead of maintaining slope/intercept pairs, and emitted breakpoints are
  clamped monotone, so the non-decreasing invariant holds structurally under
  any rounding.

The checker's route fold (contract ``td-fold/2``, see ``td.checker``) adds two
exactness rules on top of this generic algebra, so that integer (or dyadic)
data is folded without any rounding at all:

- **slope-one rule** (``slope_one_exact=True`` on ``evaluate``/``compose``): on
  a piece whose rise equals its run (``y_hi - y_lo == x_hi - x_lo``), a point
  is interpolated as ``y_lo + (x - x_lo)`` and inverted as
  ``x_lo + (y - y_lo)`` instead of through the ratio ``t``;
- **vertex transforms** (``restrict_domain`` and ``apply_ready_time``) act on
  the accumulator's breakpoints directly (``max(y, earliest) + service_time``)
  instead of composing a ready-time function θ.

The generic defaults are unchanged and remain the reference used by the ATF
materializers (their sha256 pins depend on them).
"""

from __future__ import annotations

from bisect import bisect_left


class PWLFError(ValueError):
    """Raised when NDCPWLF invariants are violated."""


def _interpolate(x_lo: float, y_lo: float, x_hi: float, y_hi: float, x: float, slope_one_exact: bool) -> float:
    """Value at ``x`` of the piece ``(x_lo, y_lo)–(x_hi, y_hi)``, ``x_lo < x < x_hi``."""
    if slope_one_exact and y_hi - y_lo == x_hi - x_lo:
        return y_lo + (x - x_lo)
    t = (x - x_lo) / (x_hi - x_lo)
    return y_lo + t * (y_hi - y_lo)


def _crossing_x(x_lo: float, y_lo: float, x_hi: float, y_hi: float, value: float, slope_one_exact: bool) -> float:
    """Abscissa where the piece ``(x_lo, y_lo)–(x_hi, y_hi)`` reaches ``value``, ``y_lo < value < y_hi``.

    On a vertical piece (``x_lo == x_hi``) this is ``x_lo``.
    """
    if slope_one_exact and x_hi - x_lo == y_hi - y_lo:
        return x_lo + (value - y_lo)
    t = (value - y_lo) / (y_hi - y_lo)
    return x_lo + t * (x_hi - x_lo)


def _emit(hxs: list[float], hys: list[float], x: float, y: float) -> None:
    """Append a breakpoint, clamped monotone, dropping exact duplicates."""
    if hxs:
        if x < hxs[-1]:
            x = hxs[-1]
        if y < hys[-1]:
            y = hys[-1]
        if x == hxs[-1] and y == hys[-1]:
            return
    hxs.append(x)
    hys.append(y)


class NDCPWLF:
    """A non-decreasing continuous piecewise linear function.

    Immutable by convention: ``xs`` and ``ys`` must not be mutated after
    construction.
    """

    __slots__ = ("xs", "ys")

    def __init__(self, xs: list[float], ys: list[float], *, validate: bool = True) -> None:
        if validate:
            if len(xs) != len(ys):
                raise PWLFError(f"xs and ys must have the same length ({len(xs)} != {len(ys)})")
            for k in range(1, len(xs)):
                if xs[k] < xs[k - 1]:
                    raise PWLFError(f"xs must be non-decreasing (violated at index {k})")
                if ys[k] < ys[k - 1]:
                    raise PWLFError(f"ys must be non-decreasing (violated at index {k})")
        self.xs = xs
        self.ys = ys

    @classmethod
    def identity(cls, low: float, high: float) -> "NDCPWLF":
        """The identity function over ``[low, high]``."""
        if low > high:
            raise PWLFError(f"identity domain is empty: [{low}, {high}]")
        if low == high:
            return cls([low], [low], validate=False)
        return cls([low, high], [low, high], validate=False)

    @classmethod
    def empty(cls) -> "NDCPWLF":
        return cls([], [], validate=False)

    def is_empty(self) -> bool:
        return not self.xs

    def num_breakpoints(self) -> int:
        return len(self.xs)

    @property
    def min_domain(self) -> float:
        return self.xs[0]

    @property
    def max_domain(self) -> float:
        return self.xs[-1]

    @property
    def min_image(self) -> float:
        return self.ys[0]

    @property
    def max_image(self) -> float:
        return self.ys[-1]

    def evaluate(self, x: float, *, slope_one_exact: bool = False) -> float:
        """Evaluate the function at ``x`` (must lie within the domain).

        At a vertical step the smallest value is returned. ``slope_one_exact``
        selects the slope-one rule of the checker fold (module docstring).
        """
        if self.is_empty() or x < self.xs[0] or x > self.xs[-1]:
            raise PWLFError(f"x={x!r} is outside the domain of the function")
        i = bisect_left(self.xs, x)
        if self.xs[i] == x:
            return self.ys[i]
        return _interpolate(self.xs[i - 1], self.ys[i - 1], self.xs[i], self.ys[i], x, slope_one_exact)

    def __call__(self, x: float) -> float:
        return self.evaluate(x)

    def compose(self, g: "NDCPWLF", *, slope_one_exact: bool = False) -> "NDCPWLF":
        """Return ``h = self ∘ g`` restricted to ``{x in dom(g) : g(x) in dom(self)}``.

        Two-pointer event merge over the common value axis
        ``dom(self) ∩ img(g)``; O(len(self.xs) + len(g.xs)). ``slope_one_exact``
        selects the slope-one rule of the checker fold for both the forward
        interpolation of ``self`` and the inverse interpolation of ``g``.
        """
        f = self
        if f.is_empty() or g.is_empty():
            return NDCPWLF.empty()
        lo = max(f.xs[0], g.ys[0])
        hi = min(f.xs[-1], g.ys[-1])
        if lo > hi:
            return NDCPWLF.empty()

        fx, fy = f.xs, f.ys
        gx, gy = g.xs, g.ys
        nf, ng = len(fx), len(gx)

        i = 0
        while fx[i] < lo:
            i += 1
        j = 0
        while gy[j] < lo:
            j += 1

        hxs: list[float] = []
        hys: list[float] = []

        while True:
            has_f = i < nf and fx[i] <= hi
            has_g = j < ng and gy[j] <= hi
            if not has_f and not has_g:
                break
            if not has_g:
                u = fx[i]
            elif not has_f:
                u = gy[j]
            else:
                u = fx[i] if fx[i] <= gy[j] else gy[j]

            # Collect all f-breakpoints at value u (a step in f yields several ys).
            f_ys: list[float] = []
            while i < nf and fx[i] == u:
                f_ys.append(fy[i])
                i += 1
            # Collect all g-breakpoints whose image is u (a plateau in g yields several xs).
            g_xs: list[float] = []
            while j < ng and gy[j] == u:
                g_xs.append(gx[j])
                j += 1

            if not f_ys:
                # u lies strictly inside an f piece: fx[i-1] < u < fx[i].
                f_ys = [_interpolate(fx[i - 1], fy[i - 1], fx[i], fy[i], u, slope_one_exact)]
            if not g_xs:
                # u lies strictly inside a g piece image: gy[j-1] < u < gy[j].
                g_xs = [_crossing_x(gx[j - 1], gy[j - 1], gx[j], gy[j], u, slope_one_exact)]

            # Emission is clamped monotone so rounding can never break the ND
            # invariant; exact duplicates are dropped.
            for x_val in g_xs:
                _emit(hxs, hys, x_val, f_ys[0])
            for y_val in f_ys[1:]:
                _emit(hxs, hys, g_xs[-1], y_val)

        return NDCPWLF(hxs, hys, validate=False)

    def min_shifted_image(self) -> tuple[float, float]:
        """Return ``(min_k (ys[k] - xs[k]), xs[k*])`` with ``k*`` the earliest argmin.

        For a route ready-time function this is the optimal route duration and
        the associated optimal depot departure time. The minimum of ``y - x``
        over the whole domain is attained at a breakpoint because the shifted
        function is linear between breakpoints.
        """
        if self.is_empty():
            raise PWLFError("cannot minimize an empty function")
        best = self.ys[0] - self.xs[0]
        best_x = self.xs[0]
        for k in range(1, len(self.xs)):
            value = self.ys[k] - self.xs[k]
            if value < best:
                best = value
                best_x = self.xs[k]
        return best, best_x

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, NDCPWLF):
            return NotImplemented
        return self.xs == other.xs and self.ys == other.ys

    def __repr__(self) -> str:
        return f"NDCPWLF(xs={self.xs!r}, ys={self.ys!r})"


def _dedup_points(xs: list[float], ys: list[float]) -> tuple[list[float], list[float]]:
    """Drop every point exactly equal to the one before it.

    The chain contract forbids two equal consecutive breakpoints, and the
    degenerate ready-time functions below (a zero-width time window, a zero
    upper bound) naturally produce them.
    """
    dedup_xs: list[float] = []
    dedup_ys: list[float] = []
    for x, y in zip(xs, ys):
        if dedup_xs and x == dedup_xs[-1] and y == dedup_ys[-1]:
            continue
        dedup_xs.append(x)
        dedup_ys.append(y)
    return dedup_xs, dedup_ys


def make_theta(earliest: float, latest: float, service_time: float) -> NDCPWLF:
    """Vertex TW ready-time function θ over arrival times in ``[0, latest]``.

    ``θ(t) = max(t, earliest) + service_time``: arriving before ``earliest``
    waits (plateau), arriving after ``latest`` is infeasible (out of domain).

    The checker no longer composes θ (contract ``td-fold/2``): it applies the
    same map exactly with :func:`apply_ready_time`. θ stays available as the
    reference definition of the vertex transform.
    """
    if earliest > latest:
        raise PWLFError(f"invalid time window [{earliest}, {latest}]")
    xs = [0.0, earliest, latest]
    ys = [earliest + service_time, earliest + service_time, latest + service_time]
    return NDCPWLF(*_dedup_points(xs, ys))


def make_service_theta(upper: float, service_time: float) -> NDCPWLF:
    """Vertex ready-time function θ without a time window, over ``[0, upper]``.

    ``θ(t) = t + service_time``, no waiting. ``upper`` is the largest arrival
    time the vertex can see. It shares the dedup of ``make_theta``, so
    ``upper == 0`` yields the single point ``([0.0], [service_time])`` instead
    of two equal consecutive points.
    """
    xs = [0.0, upper]
    ys = [service_time, upper + service_time]
    return NDCPWLF(*_dedup_points(xs, ys))


def restrict_domain(f: NDCPWLF, low: float, high: float, *, slope_one_exact: bool = True) -> NDCPWLF:
    """Restrict ``f`` to ``[low, high] ∩ dom(f)`` (checker fold, ``td-fold/2``).

    Replaces ``f.compose(NDCPWLF.identity(low, high))``: every breakpoint of
    ``f`` inside the window is kept verbatim (both points of a step at a window
    end included), and a window end that falls strictly inside a piece is
    evaluated there. Returns the empty function when the window misses the
    domain.
    """
    if f.is_empty():
        return NDCPWLF.empty()
    lo = max(f.xs[0], low)
    hi = min(f.xs[-1], high)
    if lo > hi:
        return NDCPWLF.empty()
    xs, ys = f.xs, f.ys
    hxs: list[float] = []
    hys: list[float] = []
    k = bisect_left(xs, lo)
    if xs[k] != lo:
        _emit(hxs, hys, lo, _interpolate(xs[k - 1], ys[k - 1], xs[k], ys[k], lo, slope_one_exact))
    n = len(xs)
    while k < n and xs[k] <= hi:
        _emit(hxs, hys, xs[k], ys[k])
        k += 1
    # hxs is non-empty here (lo was emitted either as a breakpoint or evaluated).
    if hxs[-1] < hi:
        # hi lies strictly inside the piece (xs[k-1], xs[k]).
        _emit(hxs, hys, hi, _interpolate(xs[k - 1], ys[k - 1], xs[k], ys[k], hi, slope_one_exact))
    return NDCPWLF(hxs, hys, validate=False)


def apply_ready_time(
    acc: NDCPWLF,
    *,
    earliest: float | None,
    latest: float | None,
    service_time: float,
    slope_one_exact: bool = True,
) -> NDCPWLF:
    """Apply the vertex map ``t -> max(t, earliest) + service_time`` to the image of ``acc``.

    This is the exact form of ``θ ∘ acc`` used by the checker fold
    (``td-fold/2``). ``acc`` maps depot departure times to arrival times at the
    vertex; the result maps them to ready times. Specification, over the
    breakpoints ``(x_k, y_k)`` of ``acc`` in order:

    - each kept breakpoint becomes ``(x_k, max(y_k, earliest) + service_time)``
      (plain float operations, no interpolation);
    - where a piece strictly crosses ``earliest`` (``y_{k-1} < earliest < y_k``)
      the crossing ``(x*, earliest + service_time)`` is inserted, ``x*`` being
      the piece's inverse interpolation at ``earliest``;
    - at the first ``y_k > latest`` the fold stops, after inserting the crossing
      ``(x*, latest + service_time)`` when ``y_{k-1} < latest``; a piece that
      crosses both bounds emits the ``earliest`` crossing first. If already
      ``y_0 > latest`` the result is empty (time-window violation);
    - emission is clamped monotone and exact duplicates are dropped.

    ``earliest=None`` means no waiting (TDVRP vertices); ``latest=None`` means
    no due-date cut. The depot's return cut is
    ``apply_ready_time(acc, earliest=None, latest=due, service_time=0.0)``.
    """
    if earliest is not None and latest is not None and earliest > latest:
        raise PWLFError(f"invalid time window [{earliest}, {latest}]")
    if acc.is_empty():
        return NDCPWLF.empty()
    xs, ys = acc.xs, acc.ys
    if latest is not None and ys[0] > latest:
        return NDCPWLF.empty()
    hxs: list[float] = []
    hys: list[float] = []
    for k in range(len(xs)):
        y_k = ys[k]
        if k > 0:
            y_prev = ys[k - 1]
            if earliest is not None and y_prev < earliest < y_k:
                x_cross = _crossing_x(xs[k - 1], y_prev, xs[k], y_k, earliest, slope_one_exact)
                _emit(hxs, hys, x_cross, earliest + service_time)
            if latest is not None and y_k > latest:
                if y_prev < latest:
                    x_cross = _crossing_x(xs[k - 1], y_prev, xs[k], y_k, latest, slope_one_exact)
                    _emit(hxs, hys, x_cross, latest + service_time)
                break
        ready = earliest if earliest is not None and y_k < earliest else y_k
        _emit(hxs, hys, xs[k], ready + service_time)
    return NDCPWLF(hxs, hys, validate=False)
