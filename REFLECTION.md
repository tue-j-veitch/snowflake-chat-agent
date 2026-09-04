**Development Process & Architectural Decisions**
I began by validating the Snowflake database connection to ensure accurate data access and retrieval. Next, I designed the LangGraph architecture and experimented with different models. I initially tested Gemini but ultimately selected Groq LLM utilizing the `openai/gpt-oss-20b` model, as it provided significantly faster and cost-free inference. Finally, I built a lightweight user interface with Streamlit and deployed the application via Streamlit Community Cloud.

**Future Improvements**
Given more time, I would dive deeper into the SafeGraph dataset to better understand its nuances and maximize its AI integration. I would also add a suggestion node to generate relevant follow-up prompts for the user, and implement data visualization features to display demographic results on USA maps or analytical graphs.

**Edge Cases & Failure Modes**
A primary concern is that the current `guardrail_node` may be overly strict, potentially misclassifying and blocking valid demographic prompts. Balancing conversational safety with successful query execution remains a failure mode that requires further tuning.

**Testing Strategy**
My initial testing approach utilized a `test_run.py` script to execute three distinct prompts and manually verify the pipeline's responses. To improve this, I plan to test the system against a much larger, diverse set of prompt types to properly calibrate the guardrails. Additionally, I would integrate Langfuse to monitor system performance and analyze execution times across the graph.
