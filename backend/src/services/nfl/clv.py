"""NFL closing-line value: LINE clv and PRICE clv, measured separately.

MLB needed only price CLV — a moneyline has no line to move. NFL spreads and
totals move on two axes simultaneously, and collapsing them into one number
destroys the distinction that matters most: taking -3.5 before it closed -4.5
is a different achievement from taking -110 before it closed -120, and only the
first can cross a key number.

Promotion is gated on CLV *before* win rate or P&L is looked at, so these are
load-bearing.

Pure functions; no I/O.
"""

from __future__ import annotations

# Margins cluster hard on 3 and 7 in the NFL. Moving a spread across one of
# these is worth far more than the same half-point elsewhere, so crossings are
# flagged rather than averaged away into raw line movement.
KEY_NUMBERS = (3, 7, 10, 14, 6, 4)


def line_clv(bet_line: float | None, closing_line: float | None, side: str) -> float | None:
    """Points of line value captured, positive when we got the better number.

    LINES ARE IN EACH SIDE'S OWN PERSPECTIVE, matching how the shadow store
    holds them (the away spread is mirrored: home -3.5 is stored as away +3.5).
    Getting this wrong inverts CLV, which is the metric promotion is gated on.

    Derived from what each side actually wants:

      spread home   -3.5 lays 3.5. Laying FEWER points is better, so a higher
                    (less negative) line is better ->  bet - close
      spread away   +3.5 receives 3.5. Receiving MORE is better, so a higher
                    line is better, and we are worse off when it rose
                                                      ->  bet - close
      total over    needs the total ABOVE the line. A LOWER line is easier, so
                    we gain when the close is higher  ->  close - bet
      total under   needs the total BELOW the line. A HIGHER line is easier, so
                    we gain when the close is lower   ->  bet - close

    Only `over` inverts. That asymmetry is real: for a total, the two sides
    want the number to move in opposite directions, whereas both spread sides
    want a higher stored line.

    Returns None with no closing line — an unmeasured pick must never read as
    neutral, or a broken capture pipeline looks like an absence of edge.
    """
    if bet_line is None or closing_line is None:
        return None

    bet, close = float(bet_line), float(closing_line)
    if side == "over":
        return close - bet
    if side in ("home", "away", "under"):
        return bet - close
    return None


def novig_two_way(odds_a: float | None, odds_b: float | None) -> tuple[float | None, float | None]:
    """Vig-free probabilities for a two-way market, multiplicative method.

    Same method the scorers use at bet time, so the two are directly
    comparable rather than merely similar.
    """
    if not odds_a or not odds_b:
        return (None, None)
    ia, ib = 1.0 / float(odds_a), 1.0 / float(odds_b)
    total = ia + ib
    if total <= 0:
        return (None, None)
    return (ia / total, ib / total)


def price_clv(bet_odds: float | None, closing_novig_prob: float | None) -> float | None:
    """Closing-line value in probability points, from price alone.

        price_clv = closing_novig_prob - (1 / bet_odds)

    The market's vig-free assessment of our side at close, minus the
    probability we needed to break even at the price we took. Positive means
    the closing market says our price was profitable, regardless of outcome.
    """
    if bet_odds is None or closing_novig_prob is None:
        return None
    odds = float(bet_odds)
    if odds <= 1.0:
        return None
    return closing_novig_prob - (1.0 / odds)


def crossed_key_number(bet_line: float | None, closing_line: float | None) -> bool | None:
    """True when the line moved across (or onto) a key number.

    Reported separately from raw line movement because a half-point through 3
    is worth several times a half-point through 5, and averaging the two
    understates the first and overstates the second.
    """
    if bet_line is None or closing_line is None:
        return None

    lo, hi = sorted((abs(float(bet_line)), abs(float(closing_line))))
    return any(lo <= float(k) <= hi for k in KEY_NUMBERS)
