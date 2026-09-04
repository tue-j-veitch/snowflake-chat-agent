import os
from typing import Annotated, List, Optional
from langchain_groq import ChatGroq
from typing_extensions import TypedDict
from dotenv import load_dotenv

import snowflake.connector
import pandas as pd
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)

def get_text(message: AIMessage) -> str:
    if isinstance(message.content, list):
        return "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in message.content
        )
    return str(message.content)

def get_snowflake_conn():
    return snowflake.connector.connect(
        user=os.getenv("SNOWFLAKE_USER"),
        password=os.getenv("SNOWFLAKE_PASSWORD"),
        account=os.getenv("SNOWFLAKE_ACCOUNT"),
        role=os.getenv("SNOWFLAKE_ROLE", "SYSADMIN"),
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH"),
        database=os.getenv("SNOWFLAKE_DATABASE", "US_CENSUS_DATA"),
        schema=os.getenv("SNOWFLAKE_SCHEMA", "PUBLIC")
    )

def query_metadata(keyword: str) -> str:
    conn = get_snowflake_conn()
    cur = conn.cursor()
    # Target table B19 for income-related queries specifically
    table_prefix = "B19" if "income" in keyword.lower() else "B01"
    sql = f"""
        SELECT TABLE_ID, TABLE_TITLE, FIELD_LEVEL_1 
        FROM "2020_METADATA_CBG_FIELD_DESCRIPTIONS"
        WHERE TABLE_ID ILIKE '{table_prefix}%' AND TABLE_TITLE ILIKE '%Median%'
        LIMIT 5;
    """
    try:
        cur.execute(sql)
        rows = cur.fetchall()
        cols = [desc[0] for desc in cur.description]
        df = pd.DataFrame(rows, columns=cols)
        return df.to_string(index=False) if not df.empty else "No matching fields found."
    finally:
        cur.close()
        conn.close()

def run_snowflake_sql(query: str) -> str:
    forbidden = ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "TRUNCATE"]
    if any(cmd in query.upper() for cmd in forbidden):
        raise ValueError("Mutation queries are strictly prohibited.")
    
    conn = get_snowflake_conn()
    cur = conn.cursor()
    try:
        cur.execute(query)
        rows = cur.fetchmany(15)
        cols = [desc[0] for desc in cur.description]
        df = pd.DataFrame(rows, columns=cols)
        return df.to_string(index=False)
    finally:
        cur.close()
        conn.close()

# --- State & Nodes ---

class AgentState(TypedDict):
    messages: List[BaseMessage]
    is_on_topic: bool
    metadata_context: str
    sql_query: Optional[str]
    sql_result: Optional[str]
    error_message: Optional[str]
    retry_count: int

def guardrail_node(state: AgentState) -> AgentState:
    conversation_history = "\n".join([f"{type(m).__name__}: {m.content}" for m in state["messages"]])
    prompt = (
        "Determine if the ongoing conversation and latest user query are asking about US Census statistics, "
        "demographics, population, income, education, housing, or US geographic regions.\n"
        "Return ONLY the single word VALID or INVALID. Do not add punctuation or explanation.\n\n"
        f"Conversation:\n{conversation_history}"
    )
    raw_msg = llm.invoke([HumanMessage(content=prompt)])
    decision = get_text(raw_msg).strip().upper()
    state["is_on_topic"] = ("VALID" in decision) and ("INVALID" not in decision)
    print(f"\n[DEBUG Guardrail] Context-aware check -> On-Topic: {state['is_on_topic']} (Raw: {decision})")
    return state

def metadata_node(state: AgentState) -> AgentState:
    conversation_history = "\n".join([f"{type(m).__name__}: {m.content}" for m in state["messages"]])
    prompt = (
        "Extract 1 single keyword or metric category (e.g., 'age', 'income', 'education', 'poverty', 'housing') "
        "from the conversation history to search the census dictionary. Output ONLY the single keyword:\n\n"
        f"{conversation_history}"
    )
    raw_msg = llm.invoke([HumanMessage(content=prompt)])
    keyword = get_text(raw_msg).strip().split()[0]
    print(f"[DEBUG Metadata] Searching keyword from context: '{keyword}'")
    
    state["metadata_context"] = query_metadata(keyword)
    state["retry_count"] = 0
    return state

def sql_gen_node(state: AgentState) -> AgentState:
    conversation_history = "\n".join([f"{type(m).__name__}: {m.content}" for m in state["messages"]])
    error_hint = f"\nPrevious SQL error: {state['error_message']}\nFix the query syntax." if state.get("error_message") else ""
    
    prompt = f"""
You are an expert Snowflake SQL generator for the US Census dataset (SafeGraph format).
Database: US_CENSUS_DATA, Schema: PUBLIC.

Available Demographic Columns from Metadata:
{state['metadata_context']}

Reference Tables:
1. FIPS Lookup Table: "2020_METADATA_CBG_FIPS_CODES" (Columns: STATE, COUNTY, STATE_FIPS, COUNTY_FIPS)
   - Note: f.STATE contains two-letter postal abbreviations like 'CA', 'TX', 'NY'.
2. Demographic Table: Use "2020_CBG_" + first 3 letters of TABLE_ID (e.g., "2020_CBG_B01"). Columns: CENSUS_BLOCK_GROUP, [TABLE_ID]

SQL Construction Rules:
1. JOIN the demographic table `d` with "2020_METADATA_CBG_FIPS_CODES" `f` using:
   SUBSTR(d."CENSUS_BLOCK_GROUP", 1, 2) = f."STATE_FIPS" AND SUBSTR(d."CENSUS_BLOCK_GROUP", 3, 3) = f."COUNTY_FIPS"
2. Handle conversational context: If the user asks for "the largest" or "highest" following a previous question, invert the ordering (e.g., ORDER BY metric DESC LIMIT 1) to find the maximum value across states or counties.
3. Use MEDIAN(d."<TABLE_ID>") or AVG(d."<TABLE_ID>") grouped by geographic dimensions.
4. Always enclose table and column names in double quotes.
5. Return ONLY the raw executable SQL statement without backticks, markdown formatting, or preamble.
{error_hint}

Full Conversation Context:
{conversation_history}
"""
    raw_msg = llm.invoke([HumanMessage(content=prompt)])
    clean_sql = get_text(raw_msg).replace("```sql", "").replace("```", "").strip()
    print(f"[DEBUG SQL Gen] Generated SQL:\n{clean_sql}")
    state["sql_query"] = clean_sql
    return state

def sql_exec_node(state: AgentState) -> AgentState:
    try:
        result = run_snowflake_sql(state["sql_query"])
        state["sql_result"] = result
        state["error_message"] = None
        print(f"[DEBUG SQL Exec] Success! Rows retrieved:\n{result[:200]}...")
    except Exception as e:
        state["sql_result"] = None
        state["error_message"] = str(e)
        state["retry_count"] = state.get("retry_count", 0) + 1
        print(f"[DEBUG SQL Exec Error (Attempt {state['retry_count']})]: {str(e)}")
    return state

def synthesize_node(state: AgentState) -> AgentState:
    last_user_msg = state["messages"][-1].content
    
    if not state.get("is_on_topic"):
        msg = "I can only answer questions related to US Census demographics, population data, housing, and socioeconomic statistics."
        state["messages"].append(AIMessage(content=msg))
        return state

    if state.get("error_message") and state["retry_count"] >= 2:
        msg = (
            f"I was unable to retrieve data from Snowflake. Internal diagnostic error: {state['error_message']}. "
            "Could you rephrase or ask about broader state/county metrics?"
        )
        state["messages"].append(AIMessage(content=msg))
        return state

    prompt = f"""
Answer the user's question accurately using the Snowflake query results below.

Question: {last_user_msg}
Data from Census tables:
{state['sql_result']}
"""
    raw_msg = llm.invoke([HumanMessage(content=prompt)])
    state["messages"].append(AIMessage(content=get_text(raw_msg)))
    return state

# --- Routing Logic ---
def route_after_guardrail(state: AgentState) -> str:
    return "metadata_node" if state.get("is_on_topic") else "synthesize_node"

def route_after_sql_exec(state: AgentState) -> str:
    if state.get("error_message") and state.get("retry_count", 0) < 2:
        return "sql_gen_node"  # Self-correcting loop
    return "synthesize_node"

# --- Build the Graph ---
workflow = StateGraph(AgentState)

workflow.add_node("guardrail_node", guardrail_node)
workflow.add_node("metadata_node", metadata_node)
workflow.add_node("sql_gen_node", sql_gen_node)
workflow.add_node("sql_exec_node", sql_exec_node)
workflow.add_node("synthesize_node", synthesize_node)

workflow.set_entry_point("guardrail_node")

workflow.add_conditional_edges(
        "guardrail_node",
        route_after_guardrail,
        {"metadata_node": "metadata_node", "synthesize_node": "synthesize_node"}
)
workflow.add_edge("metadata_node", "sql_gen_node")
workflow.add_edge("sql_gen_node", "sql_exec_node")
workflow.add_conditional_edges(
        "sql_exec_node",
        route_after_sql_exec,
        {"sql_gen_node": "sql_gen_node", "synthesize_node": "synthesize_node"}
)
workflow.add_edge("synthesize_node", END)

# In-memory checkpointer preserves multi-turn session state
app = workflow.compile(checkpointer=MemorySaver())
