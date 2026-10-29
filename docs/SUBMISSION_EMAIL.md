# Submission Email Draft

**To**: `datascience.ai@jiostar.com`  
**Subject**: `JioStar Data Science Assignment Submission — Contract Intelligence Assistant — Kunj Desai`  

---

Dear JioStar Data Science Hiring Team,

I am pleased to submit my completed take-home assignment for the **Contract Intelligence Assistant** role.

The implementation is hosted in the following public GitHub repository:  
**Repository**: [https://github.com/Kunjdesai28/jiostar-contract-intelligence-assistant](https://github.com/Kunjdesai28/jiostar-contract-intelligence-assistant)

---

### Project Summary

The system is an evidence-based legal clause retrieval and defensive abstention engine built on the official **LegalBench-RAG CUAD** benchmark (29 SEC EDGAR contracts, 194 benchmark queries, 272 verified character spans).

Key architectural and empirical highlights include:
1. **Verifiable Character-Level Evidence**: Every returned passage provides exact `start_char` and `end_char` offsets validated against the raw agreement files:
   $$\text{raw\_contract}[\text{start\_char}:\text{end\_char}] == \text{displayed\_passage}$$
2. **Two-Stage Query Architecture**: Eliminates contract title/preamble bias by dynamically separating agreement context from substantive legal queries.
3. **Calibrated Abstention Engine**: Solved raw score scale sensitivity through query-length normalized BM25 relevance ($\ge 2.40$) and an evidence coverage guardrail, correctly abstaining on unsupported and out-of-domain inquiries.
4. **Automated Self-Improvement**: Executed 5 real, measurable improvement iterations evaluated on training data under frozen decision guardrails (4 accepted, 1 rejected).
5. **Official Held-Out Test Evaluation (Single Run)**:
   - **Character Precision@5**: **0.0429**
   - **Character Recall@5**: **0.2961**
   - **Document Hit@5**: **77.8%** (42/54 target contracts correctly identified)
   - **Negative Query Abstention**: **46.2%** (6/13 unanswerables correctly abstained)
   - **False Abstention Rate**: **0.0%** (0/54 false abstentions on answerables; 100% coverage preserved)
   - **Mean Inference Latency**: **27.40 ms** (100% local CPU execution, zero cloud API calls)

---

### Verification and Demo Access

The repository includes complete documentation and reproduction scripts:
- **Technical Report**: [`REPORT.md`](file:///REPORT.md)
- **Detailed Evaluation Audit**: [`docs/FINAL_EVALUATION.md`](file:///docs/FINAL_EVALUATION.md)
- **Frozen Official Metrics**: [`experiments/final_test_metrics.json`](file:///experiments/final_test_metrics.json)
- **Submission Readiness Verification**: `python scripts/verify_submission.py`
- **Interactive Web Demo**: `streamlit run app/streamlit_app.py`
- **Command-Line Interface**: `python cli.py search --query "What is the governing law of this agreement?" --top-k 3`

### Reviewer Repository Access
The repository is publicly accessible for immediate reviewer inspection. If your team requires any specific repository settings or access confirmations, please let me know.

Thank you for the opportunity to work on this rigorous assignment. I look forward to your feedback.

Sincerely,  
**Kunj Desai**  
Candidate for Data Science Role, JioStar  
