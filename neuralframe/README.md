# NeuralFrame

**A production-grade, AI-native web framework written in Rust.**

NeuralFrame is designed from the ground up to treat LLMs, agents, memory, and streaming as first-class citizens — not bolted-on afterthoughts. Built for industrial-scale AI applications.

---

## ⚡ Features

| Feature | Description |
|---------|-------------|
| **Zero-allocation Router** | Radix tree router with path params, wildcards, and method-based dispatch |
| **Typed Extractors** | Json, Query, PathParams, Headers — auto-deserialized |
| **Async Middleware** | Logging, CORS, rate limiting, compression — composable pipeline |
| **5 LLM Providers** | OpenAI, Anthropic, Google Gemini, Ollama, Groq — unified interface |
| **SSE Streaming** | First-class token streaming with automatic SSE formatting |
| **Retry & Failover** | Exponential backoff with jitter, automatic provider failover |
| **Tool/Function Calling** | Define tools, execute in parallel, feed results back |
| **HNSW Vector Store** | Built-in approximate nearest neighbor search with metadata filtering |
| **Multi-tier Memory** | Conversation, summary, vector, and entity memory types |
| **Semantic Cache** | Cache LLM responses by embedding similarity — reduce API costs |
| **Agent Orchestration** | ReAct agents, DAG planning, sequential & parallel execution |
| **Auth** | API key, JWT, per-key rate limiting |
| **Python Bindings** | PyO3/maturin — use from Python |

---

## 🏗️ Architecture

```
neuralframe/
├── Cargo.toml                     # Workspace root
├── crates/
│   ├── neuralframe-core/          # HTTP server, routing, middleware
│   ├── neuralframe-llm/           # LLM client, streaming, retry
│   ├── neuralframe-prompt/        # Prompt templating, versioning
│   ├── neuralframe-vector/        # HNSW vector store, metrics
│   ├── neuralframe-memory/        # Multi-tier memory engine
│   ├── neuralframe-cache/         # Semantic LLM response cache
│   ├── neuralframe-agents/        # Agent orchestration (ReAct, DAG)
│   ├── neuralframe-auth/          # Authentication & authorization
│   └── neuralframe-py/            # Python bindings (PyO3)
├── examples/
│   ├── chatbot_server.rs          # Full chatbot with streaming
│   ├── rag_pipeline.rs            # RAG with vector search
│   └── multi_agent.rs             # Multi-agent collaboration
└── README.md
```

---

## 🚀 Quick Start

### Hello World

```rust
use neuralframe_core::prelude::*;

#[tokio::main]
async fn main() {
    NeuralFrame::new()
        .get("/hello", |_req| async {
            Response::ok().json(serde_json::json!({"message": "Hello, NeuralFrame!"}))
        })
        .bind("0.0.0.0:8080")
        .run()
        .await
        .expect("server failed");
}
```

### LLM Chat Completion

```rust
use neuralframe_llm::prelude::*;

#[tokio::main]
async fn main() {
    let provider = OpenAIProvider::new("sk-your-api-key");

    let req = CompletionRequest::new("gpt-4o")
        .system("You are a helpful assistant.")
        .user("Explain Rust's ownership model in 3 sentences.")
        .max_tokens(200)
        .temperature(0.7);

    let response = provider.complete(req).await.unwrap();
    println!("{}", response.content);
}
```

### Streaming Tokens

```rust
use neuralframe_llm::prelude::*;
use futures::StreamExt;

#[tokio::main]
async fn main() {
    let provider = OpenAIProvider::new("sk-your-api-key");

    let req = CompletionRequest::new("gpt-4o")
        .user("Write a haiku about Rust.")
        .streaming();

    let mut stream = provider.stream(req).await.unwrap();
    while let Some(token) = stream.next().await {
        match token {
            Ok(t) => print!("{}", t.content),
            Err(e) => eprintln!("Error: {}", e),
        }
    }
}
```

### Resilient Client with Failover

```rust
use neuralframe_llm::prelude::*;
use neuralframe_llm::retry::{ResilientClient, RetryConfig};

let client = ResilientClient::new(OpenAIProvider::new("key"))
    .with_fallback(AnthropicProvider::new("key"))
    .with_fallback(GroqProvider::new("key"))
    .with_retry(RetryConfig::new(3));

let response = client.complete(
    CompletionRequest::new("gpt-4o").user("Hello!")
).await.unwrap();
```

### Vector Search

```rust
use neuralframe_vector::{VectorStore, DistanceMetric, Filter};

let store = VectorStore::new(384, DistanceMetric::Cosine);

store.insert("doc1", embedding_vec, serde_json::json!({"topic": "rust"})).unwrap();

let results = store.search(
    &query_embedding,
    10,
    Some(&Filter::Eq("topic".into(), serde_json::json!("rust"))),
).unwrap();
```

### Prompt Templates

```rust
use neuralframe_prompt::{PromptTemplate, PromptRegistry};
use std::collections::HashMap;

let template = PromptTemplate::new(
    "rag",
    "Context: {{ context }}\n\nQuestion: {{ question }}\n\nAnswer:"
);

let mut vars = HashMap::new();
vars.insert("context".to_string(), "NeuralFrame is an AI framework.".to_string());
vars.insert("question".to_string(), "What is NeuralFrame?".to_string());

let prompt = template.render(&vars).unwrap();
```

### Multi-Agent Pipeline

```rust
use neuralframe_agents::prelude::*;

let mut orchestrator = Orchestrator::new();
orchestrator.add_agent(ReActAgent::new(AgentConfig::new("researcher", "You research topics")));
orchestrator.add_agent(ReActAgent::new(AgentConfig::new("writer", "You write responses")));

let result = orchestrator
    .run_sequential(&["researcher", "writer"], "Explain quantum computing")
    .await
    .unwrap();

println!("{}", result.answer);
```

---

## 📦 Crate Overview

### `neuralframe-core`
HTTP server built on **Hyper** + **Tokio** with a radix tree router, async middleware pipeline (logging, CORS, rate limiting), typed extractors (Json, Query, PathParams, Headers), and SSE streaming responses.

### `neuralframe-llm`
Universal LLM client with a unified `LLMProvider` trait. Supports **OpenAI**, **Anthropic**, **Google Gemini**, **Ollama**, and **Groq**. Includes SSE streaming, retry with exponential backoff, provider failover, and function/tool calling.

### `neuralframe-prompt`
Prompt templating with Jinja2-like syntax (`{{ vars }}`, `{% if %}`, `{% for %}`). Token counting, truncation, versioned templates, A/B testing via `PromptRegistry`.

### `neuralframe-vector`
Built-in vector store with HNSW (Hierarchical Navigable Small World) index for approximate nearest neighbor search. SIMD-friendly similarity metrics (cosine, euclidean, dot product). Metadata filtering and persistent storage with WAL.

### `neuralframe-memory`
Multi-tier memory: Conversation (sliding window), Summary (auto-summarization), Vector (semantic retrieval), Entity (named entities). Backends: InMemory, Redis, PostgreSQL, SQLite.

### `neuralframe-cache`
Semantic LLM response caching. Exact and similarity-based matching. TTL expiration, LRU eviction, hit statistics.

### `neuralframe-agents`
Agent orchestration with the ReAct (Reasoning + Acting) pattern. DAG-based planning with dependency tracking. Multi-agent collaboration via sequential pipeline and parallel execution.

### `neuralframe-auth`
API key and JWT authentication. Per-key rate limiting with token bucket. Identity model with role-based access control.

### `neuralframe-py`
PyO3 Python bindings exposing CompletionRequest, CompletionResponse, and Message types.

---

## 🏃 Running Examples

```bash
# Chatbot server
cargo run --example chatbot_server

# RAG pipeline
cargo run --example rag_pipeline

# Multi-agent system
cargo run --example multi_agent
```

## 🧪 Testing

```bash
# Run all tests
cargo test --workspace --exclude neuralframe-py

# Run tests for a specific crate
cargo test -p neuralframe-core
cargo test -p neuralframe-llm
cargo test -p neuralframe-vector
```

## 📊 Benchmarks

```bash
cargo bench -p neuralframe-vector
```

---

## 🔧 Requirements

- **Rust** 1.75+
- **Tokio** runtime (included)

---

## 📄 License

MIT OR Apache-2.0
