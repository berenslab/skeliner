"""Off the reference platform, how far the Python and the Rust path differ:
the largest float difference each parity test saw (``_parity.py``), to set
the tolerances from."""

from ._parity import DIFFERENCES


def pytest_terminal_summary(terminalreporter) -> None:
    if not DIFFERENCES:
        return
    tr = terminalreporter
    tr.section("Python path against Rust path: largest float differences")
    tr.write_line(f"{'relative':>10} {'absolute':>10}  test")
    for test, (absolute, relative) in sorted(
        DIFFERENCES.items(), key=lambda kv: -kv[1][1]
    ):
        tr.write_line(f"{relative:10.2e} {absolute:10.2e}  {test}")
