# LangChain & LangGraph — Study Guide

> Concepts, Q&A, and runnable code examples for evaluation prep and hands-on practice.

---

## Table of Contents

- [Core Concepts](#core-concepts)
  - [LangChain Concepts](#langchain-concepts)
  - [LangGraph Concepts](#langgraph-concepts)
- [LangChain — Q&A](#langchain--qa)
- [LangGraph — Q&A](#langgraph--qa)
- [Comparison — Q&A](#comparison--qa)
- [Code Examples](#code-examples)
  - [LangChain Examples](#langchain-code-examples)
  - [LangGraph Examples](#langgraph-code-examples)

---

## Core Concepts

### LangChain Concepts

#### 1. LCEL — LangChain Expression Language
The modern way to compose chains using the `|` pipe operator. Connects components sequentially: output of one becomes input of the next. Supports streaming, batching, and async natively.

```
prompt | model | output_parser
```

#### 2. Chains vs. Agents
- **Chain** — fixed, pre-defined sequence of steps (deterministic)
- **Agent** — uses an LLM at runtime to decide which tools to call and in what order (dynamic)

#### 3. Retrieval Augmented Generation (RAG)
Pipeline: `Load → Split → Embed → Store → Retrieve → Generate`

Key components: `DocumentLoader`, `TextSplitter`, `Embeddings`, `VectorStore`, `Retriever`

#### 4. Memory Types

| Type | Description | Best For |
|------|-------------|----------|
| `ConversationBufferMemory` | Stores full chat history | Short conversations |
| `ConversationSummaryMemory` | LLM compresses older turns | Long conversations |
| `ConversationBufferWindowMemory` | Keeps last K turns | Fixed context budget |
| `VectorStoreMemory` | Semantic search over history | Long-term personalization |

#### 5. Tools & Toolkits
- Tools have a `name`, `description`, and callable function
- Agents select tools purely based on the description — write clear descriptions
- Use `@tool` decorator or `Tool` class to create custom tools
- Toolkits group related tools (e.g. SQL toolkit, file toolkit)

#### 6. Runnable Interface
Every LCEL component implements `Runnable` with `.invoke()`, `.stream()`, `.batch()` methods. `RunnablePassthrough` passes input unchanged — used to inject original input into later steps.

---

### LangGraph Concepts

#### 1. StateGraph
The main class. You define a `TypedDict` as state, instantiate `StateGraph(YourState)`, add nodes and edges, then compile. Every node receives the full state and returns updates to it.

#### 2. Nodes & Edges

| Type | Description |
|------|-------------|
| Node | A Python function: `(state) -> dict` |
| Normal edge | Always transitions from A → B |
| Conditional edge | Router function inspects state, returns a key that maps to the next node |

#### 3. START and END
Special constants marking the entry and exit of the graph.
```python
graph.add_edge(START, "first_node")
graph.add_edge("last_node", END)
```

#### 4. Checkpointing & Persistence
Attach a `checkpointer` (e.g. `MemorySaver`, `SqliteSaver`, `PostgresSaver`) to save state after every node. Identified by `thread_id` in config. Enables multi-turn memory, human-in-the-loop, and fault tolerance.

#### 5. Human-in-the-Loop
Set `interrupt_before=["node_name"]` at compile time. Graph pauses before that node. Resume with:
```python
graph.invoke(None, config)   # same thread_id
```
Optionally inject edits with `graph.update_state(config, {"key": value})` before resuming.

#### 6. Reducers & Annotated State
Controls how state fields are updated when a node returns.

| Pattern | Behavior |
|---------|----------|
| Default | Overwrites the field |
| `Annotated[list, operator.add]` | Appends to the list |
| `Annotated[list, add_messages]` | Appends messages, deduplicates by ID |

#### 7. MessagesState
Pre-built `TypedDict` with `messages: Annotated[list, add_messages]`. Subclass it instead of building from scratch for chat-based agents.

#### 8. ReAct Pattern in LangGraph
The loop is made explicit as graph nodes:
- `agent` node → calls LLM
- `tools` node → executes tool calls
- Conditional edge → routes back to `agent` if more tool calls needed, or to `END`

---

## LangChain — Q&A

### Easy

**Q: What is LangChain and what problem does it solve?**

A: LangChain is a framework for building LLM-powered applications. It solves the problem of orchestrating multiple components — prompts, models, memory, tools, retrievers — into cohesive pipelines, so developers don't have to wire everything from scratch.

---

**Q: What is LCEL and what is the pipe operator `|` used for?**

A: LCEL (LangChain Expression Language) is a declarative way to compose chains. The `|` operator connects components: the output of one becomes the input of the next. Example:
```python
chain = prompt | model | output_parser
```
It supports streaming, batching, and async natively on any chain built this way.

---

**Q: What is the difference between a `ChatModel` and an `LLM` in LangChain?**

A: An `LLM` takes a plain string and returns a string. A `ChatModel` takes a list of messages (`SystemMessage`, `HumanMessage`, `AIMessage`) and returns a message. Modern models like GPT-4 and Claude are `ChatModel`s and handle multi-turn context better.

---

**Q: What is a `PromptTemplate` and why is it useful?**

A: A `PromptTemplate` is a reusable prompt with named placeholders like `{input}`. It separates prompt logic from application logic and makes prompts composable in LCEL chains. Fill values at runtime with a dict or `.format()`.

---

### Medium

**Q: Explain the RAG pipeline in LangChain. What are the key components?**

A: RAG = Retrieval Augmented Generation.

1. **Load** documents with a `DocumentLoader`
2. **Split** with a `TextSplitter` (e.g. `RecursiveCharacterTextSplitter`)
3. **Embed** chunks using an `Embeddings` model
4. **Store** in a `VectorStore` (FAISS, Chroma, Pinecone)
5. **Retrieve** relevant chunks via a `Retriever` at query time
6. **Generate** — pass retrieved context + question to the LLM

`create_retrieval_chain` or a custom LCEL chain wires steps 5–6 together.

---

**Q: What are the types of Memory in LangChain and when would you use each?**

A:
- `ConversationBufferMemory` — stores full chat history; simple but grows unbounded
- `ConversationSummaryMemory` — LLM summarizes older turns; great for long conversations
- `ConversationBufferWindowMemory` — keeps only the last K turns; predictable token budget
- `VectorStoreMemory` — stores and retrieves memories by semantic similarity; useful for long-term personalized memory

---

**Q: How does a LangChain agent decide which tool to call?**

A: The agent passes the user query + list of tool names and descriptions to the LLM. The LLM reasons in ReAct format (Thought / Action / Observation) about which tool fits based on the description. The `AgentExecutor` calls the chosen tool, feeds the result back to the LLM, and repeats until a final answer is reached. Tool descriptions are critical — the LLM has no other signal.

---

**Q: What is the difference between `invoke()`, `stream()`, and `batch()`?**

A:
- `invoke()` — runs the chain once, returns complete response
- `stream()` — returns a generator yielding tokens as they arrive (real-time UIs)
- `batch()` — runs on a list of inputs concurrently (uses a thread pool)

All three work uniformly on any LCEL chain.

---

### Hard

**Q: How would you create a custom tool in LangChain?**

A: Two approaches:

```python
# Option 1 — @tool decorator (docstring = description the LLM sees)
@tool
def search_codebase(query: str) -> str:
    """Search the codebase for relevant functions and files."""
    return grep_codebase(query)

# Option 2 — StructuredTool with Pydantic schema for multiple inputs
class SearchInput(BaseModel):
    query: str
    file_extension: str = ".py"

tool = StructuredTool.from_function(
    func=search_codebase,
    name="search_codebase",
    description="Search the codebase for relevant functions and files.",
    args_schema=SearchInput,
)
```

---

**Q: What is a `Runnable` and how does `RunnablePassthrough` work?**

A: `Runnable` is the base interface for any LCEL component — provides `invoke`, `stream`, `batch`. `RunnablePassthrough` passes input through unchanged. Often used to keep the original question alive while adding retrieved context:

```python
chain = (
    RunnablePassthrough.assign(context=retriever)  # adds context, keeps question
    | prompt
    | model
)
```

---

**Q: How do you implement conversational RAG (keeping chat history in a RAG chain)?**

A: You need two LLM calls:
1. `create_history_aware_retriever` — rewrites the user's question using chat history into a standalone question so the retriever gets a self-contained query
2. `create_retrieval_chain` — the actual RAG chain with the reformulated question

Chat history is passed as a list of messages to each call. The reformulation step is what makes it work — without it, follow-up questions like "tell me more" would confuse the retriever.

---

## LangGraph — Q&A

### Easy

**Q: What is LangGraph and how does it differ from LangChain?**

A: LangGraph is a library built on LangChain for building stateful, cyclic, multi-actor applications. While LangChain excels at linear pipelines, LangGraph models workflows as a graph with nodes (functions) and edges (transitions), supporting loops, branching, and persistent state across runs.

---

**Q: What is a `StateGraph` and how do you define state in LangGraph?**

A: `StateGraph` is the main class. State is defined as a `TypedDict` specifying what flows through the graph:

```python
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    next: str

graph = StateGraph(AgentState)
```

Every node receives this state and returns a dict with only the fields it wants to update.

---

**Q: What are `START` and `END` nodes?**

A: Special constants imported from `langgraph.graph`. `START` marks the entry point; `END` terminates execution.

```python
from langgraph.graph import StateGraph, START, END

graph.add_edge(START, "planner")
graph.add_edge("coder", END)
```

---

### Medium

**Q: What is the difference between a normal edge and a conditional edge?**

A: A normal edge (`add_edge`) always transitions A → B. A conditional edge (`add_conditional_edges`) takes a router function that reads state and returns a string key, which maps to a target node. This is how branching and loops are expressed.

```python
def route(state) -> str:
    if state["needs_tools"]:
        return "tools"
    return END

graph.add_conditional_edges("agent", route, {"tools": "tools", END: END})
```

---

**Q: How does checkpointing work in LangGraph and why is it important?**

A: A checkpointer is attached when compiling the graph. After each node, state is saved with a `thread_id`. This enables:
- Multi-turn conversations (same `thread_id` restores prior state)
- Human-in-the-loop interrupts (pause and resume)
- Fault tolerance (replay from last checkpoint)

```python
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)
config = {"configurable": {"thread_id": "session-42"}}
app.invoke({"messages": [HumanMessage("Hello")]}, config)
```

---

**Q: Explain how human-in-the-loop works in LangGraph.**

A: Compile with `interrupt_before=["node_name"]`. The graph pauses before that node and returns control. Inspect state, optionally call `graph.update_state()`, then resume with `graph.invoke(None, config)` using the same `thread_id`.

```python
app = graph.compile(checkpointer=checkpointer, interrupt_before=["approval"])

# First call — pauses at "approval"
app.invoke(initial_state, config)

# Human inspects, optionally edits state
app.update_state(config, {"approved": True})

# Resume
app.invoke(None, config)
```

---

**Q: What is a reducer in LangGraph? Give an example.**

A: A reducer controls how a state field is updated when a node returns. Default is overwrite. Using `Annotated[list, operator.add]` appends instead:

```python
import operator
from typing import Annotated

class State(TypedDict):
    messages: Annotated[list, operator.add]  # appends, never overwrites
    result: str                               # overwrites each time
```

`add_messages` is a smarter reducer for chat messages — appends new ones but updates by ID if the message already exists.

---

### Hard

**Q: How do you build a multi-agent system in LangGraph?**

A: Two common patterns:

**Supervisor pattern** — one supervisor node routes to specialist agents using `Command`:
```python
def supervisor(state):
    decision = llm.invoke(routing_prompt)
    return Command(goto=decision.next_agent, update={"task": decision.task})
```

**Subgraph pattern** — each agent is a compiled graph used as a node in the parent:
```python
coder_graph = coder_builder.compile()
parent_graph.add_node("coder", coder_graph)
```

---

**Q: How do you implement a retry/reflection loop in LangGraph?**

A: Use a conditional edge after the agent node. The router checks quality in state. If below threshold, route back to the agent (creating a cycle). Set `recursion_limit` to prevent infinite loops.

```python
def should_retry(state) -> str:
    if state["quality_score"] < 0.8 and state["attempts"] < 3:
        return "agent"   # loop back
    return END

graph.add_conditional_edges("reviewer", should_retry)
app = graph.compile(checkpointer=checkpointer)
app.invoke(state, config={"recursion_limit": 10})
```

---

**Q: What is `add_messages` and how does `MessagesState` simplify things?**

A: `add_messages` is a built-in reducer that appends new messages but overwrites if a message with the same ID already exists (useful for updating tool results). `MessagesState` is a pre-built `TypedDict` with this wired in — subclass it instead of defining from scratch:

```python
from langgraph.graph import MessagesState

class MyState(MessagesState):
    extra_field: str  # add your own fields on top
```

---

## Comparison — Q&A

**Q: When would you choose LangGraph over a standard LangChain agent?**

A: Choose LangGraph when you need:
- Cycles / loops (retry logic, reflection, multi-step reasoning)
- Explicit branching that's hard to express in a chain
- Human-in-the-loop interrupts
- Multiple agents that coordinate
- Fine-grained control over state and transitions

LangChain agents are sufficient for simple single-agent ReAct loops without cycles.

---

**Q: How is state management different in LangChain vs LangGraph?**

A: In LangChain, memory is a side object attached to a chain and managed manually. In LangGraph, state is first-class — a typed schema every node reads and writes. LangGraph's checkpointer automatically persists state between calls, making it more explicit, traceable, and production-ready.

---

**Q: Can LangChain and LangGraph be used together? How?**

A: Yes — LangGraph is built on LangChain. A LangGraph node can call any LCEL chain, a LangChain agent executor, a retrieval chain, or any `Runnable`. Typical pattern: build LLM logic (prompts, tools, RAG) with LangChain/LCEL, orchestrate the high-level flow (loops, branching, human approval) with LangGraph.

---

**Q: What are the tradeoffs between a LangChain `AgentExecutor` and a LangGraph agent?**

A:

| | `AgentExecutor` | LangGraph agent |
|--|----------------|-----------------|
| Setup | Easy, less code | More boilerplate |
| Loop logic | Black-box | Fully explicit |
| Branching | Limited | First-class |
| Streaming | Per-token | Per-node + per-token |
| Checkpointing | Manual | Built-in |
| Human-in-loop | Hard | Native `interrupt()` |

For production multi-step agents, LangGraph's explicitness is worth the extra setup.

---

## Code Examples

### LangChain Code Examples

#### `01_basic_lcel_chain.py`
```python
"""
Basic LCEL chain: prompt | model | output_parser
Run: python 01_basic_lcel_chain.py
"""
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. Define components
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful coding assistant."),
    ("human", "{question}")
])

model = ChatAnthropic(model="claude-3-5-sonnet-20241022")
parser = StrOutputParser()

# 2. Compose chain with LCEL pipe operator
chain = prompt | model | parser

# 3. Invoke
response = chain.invoke({"question": "What is a decorator in Python?"})
print(response)

# 4. Stream tokens
print("\n--- Streaming ---")
for chunk in chain.stream({"question": "Explain list comprehensions briefly."}):
    print(chunk, end="", flush=True)

# 5. Batch
responses = chain.batch([
    {"question": "What is a generator?"},
    {"question": "What is a context manager?"},
])
for r in responses:
    print("\n---\n", r)
```

---

#### `02_custom_tools_agent.py`
```python
"""
Custom tools + ReAct agent
Run: python 02_custom_tools_agent.py
"""
from langchain_anthropic import ChatAnthropic
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.tools import tool
from langchain import hub

# 1. Define custom tools
@tool
def search_python_docs(query: str) -> str:
    """Search Python documentation for a given topic or function name."""
    # In a real project, call the actual docs API
    return f"Documentation result for '{query}': [mock result — integrate real API here]"

@tool
def run_python_snippet(code: str) -> str:
    """Execute a small Python snippet and return its output. Use for quick calculations."""
    try:
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            exec(code, {})
        return buf.getvalue() or "No output"
    except Exception as e:
        return f"Error: {e}"

# 2. LLM + tools
llm = ChatAnthropic(model="claude-3-5-sonnet-20241022")
tools = [search_python_docs, run_python_snippet]

# 3. Pull ReAct prompt from hub (or define your own)
prompt = hub.pull("hwchase17/react")

# 4. Build agent
agent = create_react_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 5. Run
result = executor.invoke({"input": "What does itertools.chain do? Also calculate 2**10."})
print(result["output"])
```

---

#### `03_rag_pipeline.py`
```python
"""
Full RAG pipeline: load → split → embed → store → retrieve → generate
Run: python 03_rag_pipeline.py
"""
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_anthropic import ChatAnthropic
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# 1. Load a document (create a sample.txt in the same folder to test)
loader = TextLoader("sample.txt")
docs = loader.load()

# 2. Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# 3. Embed and store in FAISS vector store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 4. Build the RAG chain
llm = ChatAnthropic(model="claude-3-5-sonnet-20241022")

system_prompt = (
    "You are a helpful assistant. Use the following retrieved context to answer "
    "the question. If the context doesn't contain the answer, say so.\n\n"
    "Context:\n{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# 5. Ask a question
response = rag_chain.invoke({"input": "What is this document about?"})
print("Answer:", response["answer"])
print("\nSource chunks:")
for doc in response["context"]:
    print(" -", doc.page_content[:100], "...")
```

---

#### `04_conversational_memory.py`
```python
"""
Conversation with memory (buffer + summary types)
Run: python 04_conversational_memory.py
"""
from langchain_anthropic import ChatAnthropic
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain.chains import ConversationChain

llm = ChatAnthropic(model="claude-3-5-sonnet-20241022")

# --- Buffer Memory (full history) ---
print("=== Buffer Memory ===")
buffer_memory = ConversationBufferMemory()
buffer_chain = ConversationChain(llm=llm, memory=buffer_memory, verbose=False)

buffer_chain.predict(input="My name is Ali and I'm learning LangChain.")
buffer_chain.predict(input="I also love Python.")
response = buffer_chain.predict(input="What do you know about me so far?")
print(response)

# Inspect the stored memory
print("\nStored memory:")
print(buffer_memory.load_memory_variables({}))

# --- Summary Memory (compressed) ---
print("\n=== Summary Memory ===")
summary_memory = ConversationSummaryMemory(llm=llm)
summary_chain = ConversationChain(llm=llm, memory=summary_memory, verbose=False)

summary_chain.predict(input="I'm building a RAG system for legal documents.")
summary_chain.predict(input="The main challenge is chunking long contracts correctly.")
response = summary_chain.predict(input="Summarize what we've discussed.")
print(response)
```

---

### LangGraph Code Examples

#### `05_simple_stategraph.py`
```python
"""
Minimal StateGraph: two nodes, one edge
Run: python 05_simple_stategraph.py
"""
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

# 1. Define state schema
class State(TypedDict):
    input: str
    result: str

# 2. Define node functions
def process_input(state: State) -> dict:
    print(f"[process_input] received: {state['input']}")
    return {"result": f"Processed: {state['input'].upper()}"}

def format_output(state: State) -> dict:
    print(f"[format_output] result: {state['result']}")
    return {"result": f"✓ {state['result']}"}

# 3. Build graph
builder = StateGraph(State)
builder.add_node("process", process_input)
builder.add_node("format", format_output)

builder.add_edge(START, "process")
builder.add_edge("process", "format")
builder.add_edge("format", END)

# 4. Compile and run
graph = builder.compile()
output = graph.invoke({"input": "hello langraph"})
print("\nFinal state:", output)
```

---

#### `06_conditional_edges.py`
```python
"""
Conditional edges: route based on state value
Run: python 06_conditional_edges.py
"""
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    number: int
    category: str
    message: str

def classify(state: State) -> dict:
    n = state["number"]
    if n < 0:
        return {"category": "negative"}
    elif n == 0:
        return {"category": "zero"}
    else:
        return {"category": "positive"}

def handle_negative(state: State) -> dict:
    return {"message": f"{state['number']} is negative"}

def handle_zero(state: State) -> dict:
    return {"message": "The number is zero"}

def handle_positive(state: State) -> dict:
    return {"message": f"{state['number']} is positive"}

# Router function — reads state, returns a string key
def route(state: State) -> Literal["negative", "zero", "positive"]:
    return state["category"]

# Build graph
builder = StateGraph(State)
builder.add_node("classify", classify)
builder.add_node("negative", handle_negative)
builder.add_node("zero", handle_zero)
builder.add_node("positive", handle_positive)

builder.add_edge(START, "classify")
builder.add_conditional_edges("classify", route)   # routes to node with matching name
builder.add_edge("negative", END)
builder.add_edge("zero", END)
builder.add_edge("positive", END)

graph = builder.compile()

for number in [-5, 0, 42]:
    result = graph.invoke({"number": number})
    print(result["message"])
```

---

#### `07_checkpointing_memory.py`
```python
"""
Multi-turn conversation using checkpointing (MemorySaver)
Run: python 07_checkpointing_memory.py
"""
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage

llm = ChatAnthropic(model="claude-3-5-sonnet-20241022")

# MessagesState has: messages: Annotated[list, add_messages]  built in
def chat_node(state: MessagesState) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

# Build
builder = StateGraph(MessagesState)
builder.add_node("chat", chat_node)
builder.add_edge(START, "chat")
builder.add_edge("chat", END)

# Attach checkpointer — persists state between calls
checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)

# thread_id ties all turns together
config = {"configurable": {"thread_id": "user-session-1"}}

# Turn 1
result = graph.invoke(
    {"messages": [HumanMessage("My name is Ali. I'm learning LangGraph.")]},
    config
)
print("Turn 1:", result["messages"][-1].content)

# Turn 2 — graph automatically recalls Turn 1 via checkpointer
result = graph.invoke(
    {"messages": [HumanMessage("What did I tell you about myself?")]},
    config
)
print("Turn 2:", result["messages"][-1].content)
```

---

#### `08_human_in_the_loop.py`
```python
"""
Human-in-the-loop: interrupt before approval node
Run: python 08_human_in_the_loop.py
"""
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

class State(TypedDict):
    plan: str
    approved: bool
    result: str

def create_plan(state: State) -> dict:
    plan = "Step 1: Analyse codebase\nStep 2: Write tests\nStep 3: Refactor"
    print(f"[Planner] Created plan:\n{plan}")
    return {"plan": plan}

def execute_plan(state: State) -> dict:
    print(f"[Executor] Running approved plan...")
    return {"result": "Execution complete!"}

# Build
builder = StateGraph(State)
builder.add_node("planner", create_plan)
builder.add_node("executor", execute_plan)
builder.add_edge(START, "planner")
builder.add_edge("planner", "executor")
builder.add_edge("executor", END)

checkpointer = MemorySaver()
# interrupt_before="executor" means: pause BEFORE executor runs
graph = builder.compile(checkpointer=checkpointer, interrupt_before=["executor"])

config = {"configurable": {"thread_id": "hitl-demo"}}

# First invoke — runs planner, then PAUSES before executor
print("=== First invoke (will pause) ===")
graph.invoke({"plan": "", "approved": False, "result": ""}, config)

# Human reviews the plan
current_state = graph.get_state(config)
print("\n[Human] Reviewing plan:", current_state.values["plan"])
human_input = input("[Human] Approve? (y/n): ")

if human_input.lower() == "y":
    # Inject approval into state, then resume
    graph.update_state(config, {"approved": True})
    print("\n=== Resuming graph ===")
    final = graph.invoke(None, config)
    print("Final result:", final["result"])
else:
    print("Plan rejected. Graph not resumed.")
```

---

#### `09_react_agent_langgraph.py`
```python
"""
ReAct agent built explicitly in LangGraph (loop made visible)
Run: python 09_react_agent_langgraph.py
"""
from typing import Annotated
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

# 1. Define tools
@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression. Example: '2 ** 10' or '100 / 4'."""
    try:
        return str(eval(expression, {"__builtins__": {}}))
    except Exception as e:
        return f"Error: {e}"

@tool
def get_current_time() -> str:
    """Get the current date and time."""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

tools = [calculator, get_current_time]

# 2. LLM bound to tools
llm = ChatAnthropic(model="claude-3-5-sonnet-20241022").bind_tools(tools)

# 3. Node functions
def agent_node(state: MessagesState) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

# ToolNode handles tool execution automatically from AIMessage tool_calls
tool_node = ToolNode(tools)

# 4. Router — if last message has tool_calls, go to tools; else END
def should_continue(state: MessagesState) -> str:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END

# 5. Build graph
builder = StateGraph(MessagesState)
builder.add_node("agent", agent_node)
builder.add_node("tools", tool_node)

builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", should_continue)
builder.add_edge("tools", "agent")   # loop back after tool use

graph = builder.compile()

# 6. Run
result = graph.invoke({
    "messages": [HumanMessage("What is 2^16? Also what time is it right now?")]
})
print(result["messages"][-1].content)
```

---

#### `10_multi_agent_supervisor.py`
```python
"""
Multi-agent supervisor pattern using Command
Run: python 10_multi_agent_supervisor.py
"""
from typing import Annotated, Literal
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.types import Command
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

llm = ChatAnthropic(model="claude-3-5-sonnet-20241022")

AGENTS = ["coder", "reviewer", "FINISH"]

# Supervisor decides which agent to route to
def supervisor_node(state: MessagesState) -> Command:
    system = SystemMessage(content=f"""You are a supervisor managing these agents: {AGENTS}.
Given the conversation, decide who should act next.
Reply with ONLY one word: coder, reviewer, or FINISH.""")

    response = llm.invoke([system] + state["messages"])
    next_agent = response.content.strip().lower()

    if next_agent not in [a.lower() for a in AGENTS]:
        next_agent = "FINISH"

    if next_agent == "finish":
        return Command(goto=END)

    return Command(goto=next_agent)

def coder_node(state: MessagesState) -> Command:
    response = llm.invoke([
        SystemMessage("You are an expert Python developer. Write clean, well-commented code."),
        *state["messages"]
    ])
    return Command(
        goto="supervisor",   # always return to supervisor
        update={"messages": [response]}
    )

def reviewer_node(state: MessagesState) -> Command:
    response = llm.invoke([
        SystemMessage("You are a senior code reviewer. Point out bugs, style issues, and improvements."),
        *state["messages"]
    ])
    return Command(
        goto="supervisor",
        update={"messages": [response]}
    )

# Build graph
builder = StateGraph(MessagesState)
builder.add_node("supervisor", supervisor_node)
builder.add_node("coder", coder_node)
builder.add_node("reviewer", reviewer_node)

builder.add_edge(START, "supervisor")
# No explicit edges needed — Command(goto=...) handles all routing

graph = builder.compile()

# Run
result = graph.invoke({
    "messages": [HumanMessage("Write a Python function to check if a string is a palindrome, then review it.")]
})

print("=== Final conversation ===")
for msg in result["messages"]:
    role = msg.__class__.__name__.replace("Message", "")
    print(f"\n[{role}]\n{msg.content}")
```

---

## Quick Reference

### Installation

```bash
pip install langchain langchain-anthropic langgraph
pip install langchain-community faiss-cpu sentence-transformers  # for RAG
pip install langgraph-checkpoint-sqlite  # for SQLite checkpointing
```

### Environment setup

```bash
export ANTHROPIC_API_KEY="your-key-here"
```

### Key imports cheat sheet

```python
# LangChain
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough

# LangGraph
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from langgraph.types import Command

# Types
from typing import TypedDict, Annotated, Literal
from langgraph.graph.message import add_messages
```