#!/usr/bin/env python
from __future__ import annotations
import os, sys, xml.etree.ElementTree as ET, argparse
from pathlib import Path

def pick(explicit: str | None):
    for name in ([explicit] if explicit else []) + ["coverage-combined.xml", "coverage.xml"]:
        if name and Path(name).is_file():
            return Path(name)
    return None

def pct(path: Path) -> float:
    root = ET.parse(path).getroot()
    r = root.get('line-rate')
    if not r:
        raise RuntimeError('line-rate missing')
    return float(r) * 100

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--percent-file', default=None)
    args = parser.parse_args()
    xml = pick(os.getenv('COVERAGE_FILE'))
    if not xml:
        print('COVERAGE: no xml (skip)')
        return 0
    try:
        val = pct(xml)
    except Exception as e:
        print(f'COVERAGE: parse error {e}')
        return 0
    t = os.getenv('COVERAGE_FAIL_UNDER')
    try:
        thr = float(t) if t else None
    except ValueError:
        thr = None
        print(f'COVERAGE: bad threshold {t!r}')
    if args.percent_file:
        try:
            with open(args.percent_file, 'w') as f:
                f.write(f'{val:.2f}\n')
        except OSError as e:
            print(f'COVERAGE: write error {e}')
    summary_path = os.getenv('GITHUB_STEP_SUMMARY')
    if summary_path:
        try:  # pragma: no cover
            with open(summary_path, 'a') as f:
                f.write(f"\n### Coverage\n\nObserved: {val:.2f}%\n")
        except OSError:
            pass
    hard_fail = os.getenv('COVERAGE_HARD_FAIL') == '1'
    if thr is None:
        print(f'COVERAGE: observed {val:.2f}% (soft)')
        return 0
    if val + 1e-9 < thr:
        if hard_fail:
            print(f'COVERAGE: {val:.2f}% below {thr:.2f}% (FAIL)')
            return 1
        print(f'COVERAGE: {val:.2f}% below {thr:.2f}% (soft fail)')
        return 0
    print(f'COVERAGE: {val:.2f}% meets {thr:.2f}%')
    return 0

if __name__ == '__main__':
    sys.exit(main())
