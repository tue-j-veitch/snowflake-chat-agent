import pytest
from langchain_core.messages import HumanMessage
from agent import app, run_snowflake_sql, guardrail_node

def test_guardrail_off_topic():
    """Verify that unrelated queries are flagged as off-topic."""
    state = {"messages": [HumanMessage(content="What is the capital of France?")]}
    res = guardrail_node(state)
    assert res["is_on_topic"] is False

def test_guardrail_on_topic():
    """Verify that US Census demographic queries are flagged as on-topic."""
    state = {"messages": [HumanMessage(content="What is the median household income in Texas?")]}
    res = guardrail_node(state)
    assert res["is_on_topic"] is True

def test_sql_safety_blocker():
    """Verify that destructive SQL operations raise a ValueError."""
    dangerous_queries = [
        "DROP TABLE 2020_CBG_B19;",
        "DELETE FROM 2020_METADATA_CBG_FIPS_CODES WHERE STATE = 'CA';",
        "UPDATE 2020_METADATA_CBG_FIPS_CODES SET STATE = 'XX';"
    ]
    for query in dangerous_queries:
        with pytest.raises(ValueError, match="Mutation queries are strictly prohibited"):
            run_snowflake_sql(query)

def test_agent_graph_execution():
    """Verify that the LangGraph state machine completes a multi-turn thread invocation."""
    config = {"configurable": {"thread_id": "test-session-pytest"}}
    query = "What is the median income in California counties?"
    
    result = app.invoke(
        {"messages": [HumanMessage(content=query)], "retry_count": 0},
        config=config
    )
    
    assert len(result["messages"]) >= 2
    final_content = result["messages"][-1].content
    assert len(final_content) > 0
