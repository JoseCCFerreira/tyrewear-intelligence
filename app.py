from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st


ROOT_DIR = Path(__file__).resolve().parent
SNAPSHOT_DIR = ROOT_DIR / "snapshots"
DL_RESULTS_PATH = SNAPSHOT_DIR / "deep_learning_test_results.json"
PROJECT_SNAPSHOT_PATH = SNAPSHOT_DIR / "project_snapshot.json"


st.set_page_config(
    page_title="TyreWear Intelligence Results",
    page_icon="TW",
    layout="wide",
)


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def status_badge(status: str) -> str:
    return "OK" if status == "passed" else "Review"


deep_learning = load_json(DL_RESULTS_PATH)
snapshot = load_json(PROJECT_SNAPSHOT_PATH)
tests = deep_learning.get("tests", [])
dataset = deep_learning.get("dataset", {})
files = snapshot.get("files", [])
repository = snapshot.get("repository", {})

st.title("TyreWear Intelligence")
st.caption("Streamlit results view for snapshots, deep learning smoke tests and project state.")

metric_cols = st.columns(4)
metric_cols[0].metric("Synthetic rows", f"{dataset.get('rows', 0):,}")
metric_cols[1].metric("Features", dataset.get("features", 0))
metric_cols[2].metric("DL frameworks tested", len(tests))
metric_cols[3].metric("Snapshot files tracked", len(files))

st.divider()

left, right = st.columns([1.1, 0.9])

with left:
    st.subheader("PyTorch and TensorFlow smoke tests")
    if tests:
        results_df = pd.DataFrame(tests)
        results_df["result"] = results_df["status"].map(status_badge)
        st.dataframe(results_df, use_container_width=True, hide_index=True)

        chart_df = results_df.set_index("framework")[["final_mse"]]
        st.bar_chart(chart_df)
    else:
        st.warning("No deep learning results found. Run `python3 scripts/test_deep_learning.py`.")

with right:
    st.subheader("Interpretation")
    st.write(
        """
        Both frameworks train a small neural network on synthetic tyre data.
        The target is standardized useful tyre life in kilometres. Lower MSE is better.
        These tests are smoke tests: they prove the local PyTorch and TensorFlow setup works.
        """
    )
    for item in tests:
        st.success(
            f"{item['framework'].title()} {item['version']} finished with MSE {item['final_mse']}."
        )

st.divider()

st.subheader("Repository snapshot")
repo_cols = st.columns(3)
repo_cols[0].metric("Latest commit", repository.get("latest_commit", "not available")[:12])
repo_cols[1].metric("Git status", "clean" if not repository.get("git_status") else "dirty")
repo_cols[2].metric("Remote", "configured" if repository.get("remote") else "not in snapshot")

if files:
    files_df = pd.DataFrame(files)
    files_df["size_kb"] = (files_df["size_bytes"] / 1024).round(2)
    st.dataframe(files_df[["path", "size_kb"]], use_container_width=True, hide_index=True)
    st.bar_chart(files_df.set_index("path")[["size_kb"]])

st.divider()

st.subheader("Project pages")
page_cols = st.columns(2)
page_cols[0].link_button("Open landing page", "http://localhost:8570/index.html")
page_cols[1].link_button("Open technical support", "http://localhost:8570/technical_support.html")

st.info(
    "To refresh these results, run `python3 scripts/test_deep_learning.py` and "
    "`python3 scripts/create_snapshot.py`, then reload this app."
)
