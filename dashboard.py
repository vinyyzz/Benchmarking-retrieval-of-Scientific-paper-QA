"""Read-only experiment explorer. Run with streamlit run dashboard.py."""
import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Scientific-paper retrieval benchmark", layout="wide")
st.title("Scientific-paper retrieval benchmark")
root = Path(os.environ.get("RETRIEVAL_RESULTS", "results"))
splits = [split for split in ("heldout", "dev") if (root / split / "summary.json").is_file()]
if not splits:
    st.info("Run the retrieval study and export a completed split to explore results.")
    st.stop()


@st.cache_data
def load_json(path, modified):
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data
def load_lines(path, modified):
    return pd.read_json(path, lines=True, dtype={"paper_id": str, "question_id": str})


def get_json(path):
    return load_json(path, path.stat().st_mtime_ns)


def get_lines(path):
    return load_lines(path, path.stat().st_mtime_ns)


split = st.sidebar.selectbox("Partition", splits, format_func={"heldout": "Held-out", "dev": "Development"}.get)
directory = root / split
summary = get_json(directory / "summary.json")
a, b, c = st.columns(3)
a.metric("Papers", summary["papers"])
b.metric("Eligible questions", summary["questions"])
c.metric("Excluded questions", sum(summary["exclusions"].values()))
leaderboard, costs, evidence, qa_tab = st.tabs(["Retrieval", "Context and cost", "Evidence trace", "Answer evaluation"])

with leaderboard:
    st.subheader("Configurations at k = 5")
    rows = pd.DataFrame(summary["leaderboard"])
    st.dataframe(rows[["config_id", "chunking", "retrieval", "embedding", "recall@5", "mrr@10"]],
                 hide_index=True, use_container_width=True)
    st.subheader("Paired changes from B0")
    metric = st.selectbox("Metric", ["recall@5", "mrr@10", "recall@1", "recall@3", "recall@10"])
    differences = pd.DataFrame(summary["paired_differences"])
    st.dataframe(differences.loc[differences.metric == metric, ["config_id", "delta", "ci_low", "ci_high"]],
                 hide_index=True, use_container_width=True)
    with st.expander("Question type and evidence length"):
        st.dataframe(summary["breakdowns"], hide_index=True, use_container_width=True)
    with st.expander("Evidence coverage sensitivity"):
        st.dataframe(summary["diagnostics"], hide_index=True, use_container_width=True)

with costs:
    st.subheader("B0 ranking prefixes")
    sweep = pd.DataFrame(summary["k_sweep"])
    st.line_chart(sweep, x="k", y="recall")
    st.dataframe(sweep, hide_index=True, use_container_width=True)
    st.subheader("Retrieval cost by configuration at k = 5")
    st.dataframe(rows[["config_id", "retrieval_ms_mean", "scoring_ms_mean", "retrieved_tokens_mean", "index_bytes"]],
                 hide_index=True, use_container_width=True)

with evidence:
    st.subheader("Inspect a ranked result")
    traces = get_lines(directory / "traces.jsonl")
    config = st.selectbox("Configuration", list(dict.fromkeys(traces.config_id)))
    trace = traces[traces.config_id == config]
    lookup = {row.question_id: f"{row.paper_id} · {row.question}" for row in trace.itertuples()}
    question_id = st.selectbox("Question", list(lookup), format_func=lookup.get)
    row = trace[trace.question_id == question_id].iloc[0]
    st.write(row.paper_title)
    st.dataframe([row.metrics], hide_index=True, use_container_width=True)
    for chunk in row.chunks:
        label = f"Rank {chunk['rank']} · {chunk['token_count']} tokens"
        if chunk["gold_hit"]:
            label += " · gold evidence hit"
        with st.expander(label):
            st.text(chunk["text"])
            st.dataframe(chunk["spans"], hide_index=True, use_container_width=True)

with qa_tab:
    qa_path = directory / "qa_summary.json"
    if qa_path.is_file():
        qa = get_json(qa_path)
        st.write(f"Human validation status — {qa['human_validation']}")
        for config, result in qa["by_config"].items():
            st.subheader(config)
            st.dataframe([{key: value for key, value in result.items() if key != "retrieval_answer_gap"}],
                         hide_index=True, use_container_width=True)
            st.dataframe(result["retrieval_answer_gap"], hide_index=True, use_container_width=True)
        agreement = directory / "judge_agreement.json"
        if agreement.is_file():
            st.subheader("Reviewer and judge agreement")
            st.json(get_json(agreement))
    else:
        st.info("Answer evaluation results are pending for this partition.")
