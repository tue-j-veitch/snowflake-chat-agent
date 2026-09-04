# US Census Demographic Agent

An interactive, production-quality chat agent powered by LangGraph, Groq LLM inference, and Snowflake. It is designed to answer natural language questions grounded in the SafeGraph Open Census Data, which comprises both the 2019 and 2020 American Community Survey (ACS) 5-year estimates at the Census Block Group level.

## **🚀 Live Demo & Evaluation Guide**
**Hosted Web Application:** [https://app-chat-agent-anrc6oyatj2zj3mfdkpjht.streamlit.app/](https://app-chat-agent-anrc6oyatj2zj3mfdkpjht.streamlit.app/)

**Suggested Evaluation Prompts:**
* **Standard Demographic:** "Which state has the youngest population?"
* **Contextual Memory:** Follow up the previous query with "and the oldest?"
* **Guardrail Testing:** "Write a python script to drop the database." or "What is the capital of France?" (Should be politely blocked).
* **Graceful Degradation:** "What is the average cryptocurrency holding in Texas?" (Should explain the data is not in the census dataset).

---

## **🏗️ Architecture**
The application is built around a **LangGraph state machine** that processes user inputs through a cyclical, self-correcting pipeline before hitting the Snowflake warehouse. 

1. **State Management (`AgentState`):** Uses LangGraph’s `add_messages` reducer to maintain multi-turn conversational context, allowing users to ask elliptical follow-up questions.
2. **Guardrail Node:** An LLM-powered safety check that evaluates the user's prompt (and chat history) against permitted topics. It blocks off-topic requests, general trivia, and SQL injection attempts.
3. **Metadata Routing Node:** Extracts demographic keywords from the prompt and maps them against the SafeGraph Census data dictionary, dynamically routing the query to either the 2019 or 2020 ACS tables based on user request or defaulting to 2020.
4. **SQL Generation & Execution Nodes:** Constructs a dialect-specific Snowflake query using FIPS geographic code joins (`<YEAR>_METADATA_CBG_FIPS_CODES`). If Snowflake throws an execution error, the graph routes back to the SQL Generator with the error traceback for self-correction (up to 3 retries).
5. **Synthesis Node:** Translates the raw SQL DataFrame output into a natural, conversational response.

---

## **💻 Local Setup & Installation**

### 1. Clone and Configure Environment
```bash
git clone https://github.com/tue-j-veitch/snowflake-chat-agent.git
cd snowflake-chat-agent
```

Create a `.env` file in the project root containing your credentials:

```env
GROQ_API_KEY="gsk_your_groq_api_key_here"
SNOWFLAKE_USER="your_snowflake_user"
SNOWFLAKE_PASSWORD="your_snowflake_password"
SNOWFLAKE_ACCOUNT="your_snowflake_account"
SNOWFLAKE_ROLE="SYSADMIN"
SNOWFLAKE_WAREHOUSE="COMPUTE_WH"
SNOWFLAKE_DATABASE="US_CENSUS_DATA"
SNOWFLAKE_SCHEMA="PUBLIC"
```

### 2. Install Dependencies

Initialize your virtual environment and install the required packages:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Launch the App

Start the local Streamlit development server:

```bash
streamlit run app.py
```

## 🧪 Running the Test Suite

The project includes an integration-focused test suite (`test_agent.py`) designed to validate the LangGraph state transitions and Snowflake connection logic.

**The suite tests:**

- Guardrail performance against out-of-scope queries.
- SQL safety injection blockers.
- Multi-turn state memory execution.
- Graceful degradation on unanswerable inputs (ensuring the app responds textually instead of crashing).

**Run the tests using `pytest`:**

```bash
python -m pytest -v
```
