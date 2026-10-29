"""Streamlit Web Application for JioStar Contract Intelligence Assistant.

Evidence-Based Legal Clause Retrieval with verifiable character-offset evidence.
"""

import sys
from pathlib import Path
import streamlit as st

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.retrieval.pipeline import ContractRetrievalPipeline

# Page configuration
st.set_page_config(
    page_title="Contract Intelligence Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for clean, professional styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .disclaimer-box {
        background-color: #F3F4F6;
        border-left: 4px solid #3B82F6;
        padding: 0.8rem 1rem;
        border-radius: 4px;
        font-size: 0.88rem;
        color: #374151;
        margin-bottom: 1.5rem;
    }
    .result-card {
        background-color: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 8px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .badge-rank {
        background-color: #DBEAFE;
        color: #1E40AF;
        padding: 0.2rem 0.6rem;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-score {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 0.2rem 0.6rem;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-offsets {
        background-color: #E0E7FF;
        color: #3730A3;
        padding: 0.2rem 0.6rem;
        border-radius: 9999px;
        font-weight: 500;
        font-size: 0.85rem;
    }
    .passage-text {
        font-family: 'Consolas', 'Courier New', monospace;
        font-size: 0.92rem;
        background-color: #F9FAFB;
        border: 1px solid #E5E7EB;
        padding: 0.8rem;
        border-radius: 6px;
        white-space: pre-wrap;
        word-break: break-word;
        color: #111827;
        margin-top: 0.6rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading contract corpus and building retrieval index...")
def get_cached_pipeline(config_path: str = "configs/final.yaml") -> ContractRetrievalPipeline:
    """Load and cache the ContractRetrievalPipeline per configuration."""
    cfg_p = Path(config_path)
    if not cfg_p.is_absolute():
        cfg_p = project_root / cfg_p
    if not cfg_p.exists():
        cfg_p = project_root / "configs" / "final.yaml"
    pipeline = ContractRetrievalPipeline(config_path=cfg_p)
    pipeline.build_index()
    return pipeline


def main():
    # Header Section
    st.markdown('<div class="main-title">Contract Intelligence Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Evidence-Based Legal Clause Retrieval</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="disclaimer-box">
            <strong>System Notice:</strong> This software is an evidence retrieval assistant designed to locate relevant 
            clauses with exact character-level source offsets. It is not an automated legal decision-making system 
            and does not offer legal counsel.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Configuration Selection
    final_cfg_rel = "configs/final.yaml"
    baseline_cfg_rel = "configs/baseline.yaml"
    final_cfg_path = project_root / final_cfg_rel
    baseline_cfg_path = project_root / baseline_cfg_rel

    final_label = "Final Selected Pipeline (configs/final.yaml)"
    baseline_label = "Baseline BM25 Pipeline (configs/baseline.yaml)"

    available_configs = {}
    if final_cfg_path.exists():
        available_configs[final_label] = final_cfg_rel
    available_configs[baseline_label] = baseline_cfg_rel

    config_options = list(available_configs.keys())

    # Ensure default session state points to Final Selected Pipeline
    default_config_label = final_label if final_label in available_configs else config_options[0]

    if "active_config_label" not in st.session_state or st.session_state["active_config_label"] not in available_configs:
        st.session_state["active_config_label"] = default_config_label

    with st.sidebar:
        st.header("Pipeline Configuration")
        selected_config_label = st.selectbox(
            "Active Configuration",
            options=config_options,
            key="active_config_label",
            help="Select retrieval pipeline configuration. Defaults to the frozen final evaluated pipeline.",
        )
        config_path = available_configs[selected_config_label]

    # Initialize Pipeline
    try:
        pipeline = get_cached_pipeline(config_path)
    except Exception as e:
        st.error(f"Failed to initialize retrieval index: {e}")
        st.info("Ensure that 'python scripts/prepare_dataset.py' has been executed.")
        return

    # Sidebar Controls
    with st.sidebar:
        st.header("Search Settings")
        top_k = st.slider("Top-K Passages", min_value=1, max_value=10, value=pipeline.default_top_k)

        st.subheader("Abstention Controls")
        if pipeline.abstention_engine.strategy in ("normalized_score_coverage", "combined_norm_coverage"):
            st.markdown("• **Status**: `Active (Frozen Setting)`")
            st.markdown("• **Strategy**: Normalized Score + Evidence Coverage")
            st.markdown(f"• **Normalized Score Threshold**: `{pipeline.abstention_engine.min_norm_score:.2f}`")
            st.markdown(
                f"• **Coverage Guardrail**: Enabled (≥{pipeline.abstention_engine.min_words_for_guard} query words "
                f"and ≤{pipeline.abstention_engine.max_matched_for_guard} matched in passage)"
            )
            st.caption(
                "Normalized relevance is calculated as the top BM25 score divided by the number of query content words "
                f"(Score / |W_q| ≥ {pipeline.abstention_engine.min_norm_score:.2f}). Evidence coverage checks whether "
                "sufficient query terms appear in the retrieved passage."
            )
        else:
            st.markdown("• **Status**: `Active (Frozen Setting)`")
            st.markdown("• **Strategy**: Raw BM25 Score Threshold")
            st.markdown(f"• **Minimum BM25 Score**: `{pipeline.abstention_engine.threshold:.2f}`")
            st.caption(
                "Baseline abstention compares the top retrieved raw BM25 score against the configured minimum score "
                f"threshold (Score > {pipeline.abstention_engine.threshold:.2f})."
            )
        st.markdown(
            "<small style='color: #6B7280;'>🔒 Parameters are locked to the frozen evaluated specification.</small>",
            unsafe_allow_html=True,
        )

        st.markdown("---")
        st.subheader("Indexed Corpus Info")
        st.write(f"• **Contracts**: {pipeline.contracts_count} SEC EDGAR agreements")
        st.write(f"• **Indexed Chunks**: {len(pipeline.chunks):,} passages")
        chunking_label = (
            "Clause-aware boundary detection"
            if pipeline.chunk_strategy == "clause_aware"
            else "Fixed-character window"
        )
        st.write(f"• **Chunking Method**: {chunking_label}")
        st.write(f"• **Chunk Size**: {pipeline.chunk_size} chars (overlap: {pipeline.overlap})")
        st.write(f"• **BM25 Parameters**: k1={pipeline.k1}, b={pipeline.b}")
        norm_label = (
            "Legal morphological stemming"
            if pipeline.tokenizer.enhancement_mode == "legal_stem"
            else "Standard lowercase"
        )
        st.write(f"• **Token Normalization**: {norm_label}")
        qp_label = (
            "Two-stage preamble filter"
            if pipeline.query_mode == "two_stage"
            else "Direct query"
        )
        st.write(f"• **Query Processing**: {qp_label}")

        st.markdown("---")
        st.subheader("Example Queries")
        sample_queries = [
            "Highlight the parts related to Exclusivity.",
            "What is the governing law of this agreement?",
            "Highlight the parts related to Termination for Convenience.",
            "What is the airspeed velocity of an unladen European swallow? (Unanswerable)",
            "What are Microsoft's service level agreements for Azure? (Outside Corpus)",
        ]
        selected_sample = st.selectbox("Select a sample query to test:", ["(Custom Query)"] + sample_queries)

    # Search Bar
    default_text = "" if selected_sample == "(Custom Query)" else selected_sample
    query = st.text_input(
        "Enter legal query or contractual topic:",
        value=default_text,
        placeholder="e.g., Highlight clauses concerning non-solicitation or audit rights.",
    )

    col1, col2 = st.columns([1, 5])
    with col1:
        search_button = st.button("Search Clauses", type="primary", use_container_width=True)

    if search_button or query:
        if not query.strip():
            st.warning("Please enter a non-empty query string.")
            return

        with st.spinner("Searching indexed contractual clauses..."):
            result = pipeline.search(query=query.strip(), top_k=top_k)

        # Status and Results display
        if result["status"] == "invalid_query":
            st.warning(f"Invalid query: {result['abstention_rationale']}")
            return

        if result["status"] == "system_error":
            st.error(f"Search error: {result['abstention_rationale']}")
            return

        if result["abstained"] or not result["passages"]:
            st.info(
                f"**No Relevant Clause Found (Abstained)**\n\n"
                f"{result['abstention_rationale']}\n\n"
                f"*The system abstained from returning unverified evidence because no candidate clause met the calibrated relevance and evidence criteria.*"
            )
            return

        # Results Found
        st.success(
            f"Found **{result['returned_count']}** relevant passage(s) for query: *\"{query}\"*"
        )

        for p in result["passages"]:
            rank = p["rank"]
            doc = p["document"]
            score = p["score"]
            start_char = p["start_char"]
            end_char = p["end_char"]
            char_len = end_char - start_char
            text = p["text"]

            with st.container():
                st.markdown(
                    f"""
                    <div class="result-card">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                            <div>
                                <span class="badge-rank">Rank #{rank}</span> &nbsp;
                                <span class="badge-score">BM25 Score: {score:.4f}</span> &nbsp;
                                <span class="badge-offsets">Offsets: [{start_char:,} : {end_char:,}] ({char_len} chars)</span>
                            </div>
                        </div>
                        <div style="font-size: 0.95rem; font-weight: 600; color: #1F2937; margin-top: 0.4rem;">
                            Document: <code>{doc}</code>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                with st.expander(f"View Full Contract Passage (Rank #{rank})", expanded=True):
                    st.markdown(f'<div class="passage-text">{text}</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
