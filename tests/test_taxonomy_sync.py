"""CI guard: FailureCategory enum must exactly match the chaos taxonomy doc.

Extracts IDs from docs/chaos-taxonomy.md headings (### N. `id`) and asserts
they are identical to the set of enum values in FailureCategory. Any drift
between the two sources of truth fails loudly here, not silently in eval.
"""

import pathlib
import re

from hypothesis_graph import FailureCategory

_TAXONOMY = pathlib.Path(__file__).parent.parent / "docs" / "chaos-taxonomy.md"


def _extract_ids(path: pathlib.Path) -> set:
    text = path.read_text()
    return set(re.findall(r"^### \d+\. `([a-z_]+)`", text, re.MULTILINE))


def test_failure_category_matches_taxonomy():
    taxonomy_ids = _extract_ids(_TAXONOMY)
    enum_values  = {e.value for e in FailureCategory}
    assert enum_values == taxonomy_ids, (
        f"FailureCategory enum out of sync with chaos taxonomy.\n"
        f"  In taxonomy but not enum : {taxonomy_ids - enum_values}\n"
        f"  In enum but not taxonomy : {enum_values - taxonomy_ids}"
    )
