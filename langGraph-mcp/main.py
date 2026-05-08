from agent.graph import run_agent

QUERIES = [
    "What is (123 * 456) + 789?",
    "What's the weather like in Lahore and Islamabad?",
    "Summarize this: Artificial intelligence is transforming every industry. "
    "From healthcare to finance, automation is reshaping how humans work, "
    "live, and interact. The pace of change is accelerating rapidly.",
]

if __name__ == "__main__":
    for q in QUERIES:
        print(f"\n🟡 Query: {q}")
        print(f"🟢 Answer: {run_agent(q)}")
        print("-" * 60)