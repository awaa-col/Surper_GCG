from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poc.probes import ShieldGemmaAuditor, collect_result_items, write_audit_results


ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
OUT_FILE = ROOT.parent / "shieldgemma_audit_results.json"


def main() -> None:
    auditor = ShieldGemmaAuditor()
    items = collect_result_items(RESULTS_DIR, limit_per_file=2)
    results = auditor.audit_items(items, truncate_response=500, progress=True)
    write_audit_results(results, OUT_FILE)
    print(f"Saved {OUT_FILE}")


if __name__ == "__main__":
    main()
