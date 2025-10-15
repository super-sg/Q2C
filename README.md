# Query to Clause

_"From natural language to precise policy logic."_

## Synopsis

Query to Clause is an AI-powered system that bridges the gap between natural language queries and complex policy documents. Leveraging Large Language Models (LLMs), it automates the interpretation of insurance policies, contracts, HR policies, and compliance documents.

Manual analysis of unstructured documents is slow and error-prone. Query to Clause enables users to input queries like “46-year-old male, knee surgery, 3-month-old insurance policy,” and automatically parses the query, searches large documents, extracts relevant clauses, and applies domain-specific logic to generate an interpretable decision.

The system outputs a structured JSON containing the decision, payout amount (if applicable), and an explanation grounded in the retrieved clauses. It supports various document formats (PDFs, Word, emails), handles vague or incomplete queries, and ensures explainability and traceability—vital for claims processing and audits.

## Features

- **Natural Language Query Parsing:** Extracts key metadata (age, condition, policy details, etc.) from user queries.
- **Semantic Clause Retrieval:** Finds relevant clauses using semantic search, not just keywords.
- **Policy Logic Evaluation:** Applies domain-specific logic to evaluate conditions and eligibility.
- **Structured Output:** Returns decisions, payout amounts, and explanations in JSON format.
- **Multimodal Support:** Handles both text and image inputs (e.g., scanned documents, claim forms).
- **Explainability & Traceability:** Every decision references specific clauses for audit compliance.

## Use Cases

- Insurance claims automation
- Contract analysis
- HR policy interpretation
- Compliance audits

## How It Works

1. **User submits a query** (text and/or image).
2. **System parses the query** to extract entities and context.
3. **Semantic search** retrieves relevant clauses from uploaded documents.
4. **Policy logic is applied** to evaluate eligibility and conditions.
5. **Structured JSON response** is generated, including decision, payout, and clause references.

## Getting Started

1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/q2c.git
   cd q2c
   ```
2. Create and activate a Python virtual environment:
   ```sh
   python -m venv venv
   venv\Scripts\activate
   ```
3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
4. Run the Streamlit app:
   ```sh
   streamlit run streamlit_app_multilingual.py
   ```

## File Structure

- `streamlit_app_multilingual.py` — Main application code
- `requirements.txt` — Python dependencies
- `.gitignore` — Ignores virtual environments and other unnecessary files
- `.gitattributes` — Ensures consistent line endings

## License

MIT License

---

**For more details, see the documentation or contact the maintainers.**