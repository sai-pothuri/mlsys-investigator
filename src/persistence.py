"""Snapshot the hypothesis graph to disk so editors can display live updates.

Call snapshot_graph() after every update_graph() call. VS Code and most
editors auto-reload JSON files on disk change, giving a live view of
likelihood shifts, hypothesis status changes, and evidence accumulation.
"""

import json
import pathlib

from hypothesis_graph import HypothesisGraph

OUTPUT_PATH = pathlib.Path(__file__).parent.parent / "hypothesis_graph.json"


def snapshot_graph(
    graph: HypothesisGraph,
    path: pathlib.Path = OUTPUT_PATH,
) -> None:
    path.write_text(
        json.dumps(graph.model_dump(mode="json"), indent=2),
        encoding="utf-8",
    )
