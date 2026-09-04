from agent import app
from langchain_core.messages import HumanMessage

config = {"configurable": {"thread_id": "demo-session-1"}}

queries = [
    "What is the capital of France?", # Out of scope
    "Can you tell me about median income in California counties?", # Valid query
    "How does that compare to Texas?" # Context-dependent follow-up
]

for query in queries:
    print(f"\nUser: {query}")
    events = app.invoke(
        {"messages": [HumanMessage(content=query)], "retry_count": 0},
        config=config
    )
    print(f"Agent: {events['messages'][-1].content}")
