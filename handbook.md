# Internal Policy Handbook

## 1. Overview
This is the operational handbook for the RigorousRAG Research Orchestrator.

## 2. Operating Procedures
- **Integrity Checks**: All scientific claims must be passed through the `Scientific Debate` and `Conflict Detector` tools.
- **Visual Evidence**: Use `Visual Entailment` to cross-link claims with paper figures (requires GPT-4o).
- **External Data**: Web search results must follow domain-allowlist policies if configured.

## 3. Data Privacy (Goal 16)
- The system automatically redacts PII (Emails, Phones, Addresses) during ingestion.
- Do not bypass redaction layers when handling sensitive pre-publication data.
