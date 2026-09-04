import os
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from agent import app

st.set_page_config(
    page_title="US Census Demographic Agent", 
    page_icon="📊", 
    layout="centered"
)

st.title("US Census Demographic Agent")
st.caption("Ask natural language questions about US population, income, housing, and socioeconomic statistics.")

# Initialize multi-turn session state and thread tracking
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = "user-eval-session-1"

# Render historical conversation turns
for msg in st.session_state.messages:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

# Capture user input from chat box
if prompt := st.chat_input("e.g., What are the highest median income counties in California?"):
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    
    with st.chat_message("assistant"):
        with st.spinner("Analyzing Census metadata and querying Snowflake..."):
            result = app.invoke(
                {"messages": [HumanMessage(content=prompt)], "retry_count": 0},
                config=config
            )
            response_text = result["messages"][-1].content
            st.markdown(response_text)
            
            # Expose executed SQL in an expandable pane for transparency and debugging
            if result.get("sql_query"):
                with st.expander("Inspected Snowflake SQL Query"):
                    st.code(result["sql_query"], language="sql")

    st.session_state.messages.append(AIMessage(content=response_text))
