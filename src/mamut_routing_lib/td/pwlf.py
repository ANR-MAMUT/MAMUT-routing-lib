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
encode a vertical step (a floating-point rounding artifact of composition);
evaluation at a step returns the smallest value. Duplicate ``ys`` entries
encode a plateau (e.g. waiting for a time window to open).

Composition follows the two-pointer event merge of Visser & Spliet (2020),
with two deliberate deviations that keep the canonical spec simple:

- no normalization (redundant breakpoints are kept; values are unchanged);
- interpolation is done directly between the two enclosing breakpoints
  instead of maintaining slope/intercept pairs, and emitted breakpoints are
  clamped monotone, so the non-decreasing invariant holds structurally under
  any rounding.
"""

from __future__ import annotations

from bisect import bisect_left


class PWLFError(ValueError):
    """Raised when NDCPWLF invariants are violated."""


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

    def evaluate(self, x: float) -> float:
        """Evaluate the function at ``x`` (must lie within the domain).

        At a vertical step the smallest value is returned.
        """
        if self.is_empty() or x < self.xs[0] or x > self.xs[-1]:
            raise PWLFError(f"x={x!r} is outside the domain of the function")
        i = bisect_left(self.xs, x)
        if self.xs[i] == x:
            return self.ys[i]
        x_lo, x_hi = self.xs[i - 1], self.xs[i]
        y_lo, y_hi = self.ys[i - 1], self.ys[i]
        t = (x - x_lo) / (x_hi - x_lo)
        return y_lo + t * (y_hi - y_lo)

    def __call__(self, x: float) -> float:
        return self.evaluate(x)

    def compose(self, g: "NDCPWLF") -> "NDCPWLF":
        """Return ``h = self ∘ g`` restricted to ``{x in dom(g) : g(x) in dom(self)}``.

        Two-pointer event merge over the common value axis
        ``dom(self) ∩ img(g)``; O(len(self.xs) + len(g.xs)).
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

        def emit(x: float, y: float) -> None:
            # Clamp monotone so rounding can never break the ND invariant,
            # then drop exact duplicates.
            if hxs:
                if x < hxs[-1]:
                    x = hxs[-1]
                if y < hys[-1]:
                    y = hys[-1]
                if x == hxs[-1] and y == hys[-1]:
                    return
            hxs.append(x)
            hys.append(y)

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
                x_lo, x_hi = fx[i - 1], fx[i]
                t = (u - x_lo) / (x_hi - x_lo)
                f_ys = [fy[i - 1] + t * (fy[i] - fy[i - 1])]
            if not g_xs:
                # u lies strictly inside a g piece image: gy[j-1] < u < gy[j].
                y_lo, y_hi = gy[j - 1], gy[j]
                t = (u - y_lo) / (y_hi - y_lo)
                g_xs = [gx[j - 1] + t * (gx[j] - gx[j - 1])]

            for x_val in g_xs:
                emit(x_val, f_ys[0])
            for y_val in f_ys[1:]:
                emit(g_xs[-1], y_val)

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


def make_theta(earliest: float, latest: float, service_time: float) -> NDCPWLF:
    """Vertex TW ready-time function θ over arrival times in ``[0, latest]``.

    ``θ(t) = max(t, earliest) + service_time``: arriving before ``earliest``
    waits (plateau), arriving after ``latest`` is infeasible (out of domain).
    """
    if earliest > latest:
        raise PWLFError(f"invalid time window [{earliest}, {latest}]")
    xs = [0.0, earliest, latest]
    ys = [earliest + service_time, earliest + service_time, latest + service_time]
    dedup_xs: list[float] = []
    dedup_ys: list[float] = []
    for x, y in zip(xs, ys):
        if dedup_xs and x == dedup_xs[-1] and y == dedup_ys[-1]:
            continue
        dedup_xs.append(x)
        dedup_ys.append(y)
    return NDCPWLF(dedup_xs, dedup_ys)
