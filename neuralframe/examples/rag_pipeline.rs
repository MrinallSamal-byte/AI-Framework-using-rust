//! # RAG Pipeline Example
//!
//! Retrieval-Augmented Generation pipeline using NeuralFrame with:
//! - Document ingestion and embedding
//! - Vector store for semantic search
//! - Context-aware LLM completion
//! - Prompt templating
//!
//! ## Running
//!
//! ```bash
//! OPENAI_API_KEY=your-key cargo run --example rag_pipeline
//! ```

use neuralframe_core::prelude::*;
use neuralframe_prompt::{PromptBuilder, PromptTemplate};
use neuralframe_vector::{DistanceMetric, VectorStore};

/// A document to be ingested into the RAG pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Document {
    id: String,
    title: String,
    content: String,
    source: String,
}

/// RAG pipeline state shared across handlers.
struct RagState {
    vector_store: VectorStore,
    prompt_template: PromptTemplate,
}

impl RagState {
    fn new() -> Self {
        let vector_store = VectorStore::new(384, DistanceMetric::Cosine);

        let prompt_template = PromptTemplate::new(
            "rag_prompt",
            "You are a knowledgeable assistant. Answer the user's question \
             based on the following context:\n\n\
             Context:\n{{ context }}\n\n\
             Question: {{ question }}\n\n\
             Provide a clear and accurate answer based on the context above. \
             If the context doesn't contain enough information, say so.",
        );

        Self {
            vector_store,
            prompt_template,
        }
    }
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt().with_env_filter("info").init();

    tracing::info!("Starting NeuralFrame RAG Pipeline");

    // Pre-load some sample documents
    let state = RagState::new();

    // Simulate document ingestion with mock embeddings
    let documents = vec![
        Document {
            id: "doc1".into(),
            title: "Rust Programming".into(),
            content: "Rust is a systems programming language focused on safety, \
                      speed, and concurrency. It achieves memory safety without \
                      garbage collection through its ownership system."
                .into(),
            source: "rust-lang.org".into(),
        },
        Document {
            id: "doc2".into(),
            title: "NeuralFrame Architecture".into(),
            content: "NeuralFrame is an AI-native web framework built in Rust. \
                      It provides first-class support for LLMs, agents, memory, \
                      and streaming. The core uses a radix tree router and \
                      async middleware pipeline."
                .into(),
            source: "neuralframe.dev".into(),
        },
        Document {
            id: "doc3".into(),
            title: "Vector Databases".into(),
            content: "Vector databases store high-dimensional vectors and enable \
                      fast similarity search. HNSW (Hierarchical Navigable Small \
                      World) is a popular algorithm for approximate nearest \
                      neighbor search with logarithmic complexity."
                .into(),
            source: "research-paper".into(),
        },
    ];

    // Ingest documents with mock embeddings
    for (i, doc) in documents.iter().enumerate() {
        // In production, embeddings come from the LLM provider's embed() method
        let mut embedding = vec![0.0f32; 384];
        // Create distinct mock embeddings
        for j in 0..384 {
            embedding[j] = ((i * 100 + j) as f32).sin() * 0.5 + 0.5;
        }

        state
            .vector_store
            .insert(
                &doc.id,
                embedding,
                serde_json::json!({
                    "title": doc.title,
                    "content": doc.content,
                    "source": doc.source,
                }),
            )
            .expect("failed to insert document");
    }

    tracing::info!("Ingested {} documents into vector store", documents.len());

    let app = NeuralFrame::new()
        .name("NeuralFrame RAG Pipeline")
        // Ingest a document
        .post("/v1/documents", |req| async move {
            #[derive(Deserialize)]
            struct IngestRequest {
                title: String,
                content: String,
                source: Option<String>,
            }

            match req.json::<IngestRequest>() {
                Ok(ingest) => {
                    let doc_id = uuid::Uuid::new_v4().to_string();

                    Response::created().json(serde_json::json!({
                        "id": doc_id,
                        "title": ingest.title,
                        "status": "ingested",
                        "note": "In production, embedding is generated via LLM provider"
                    }))
                }
                Err(_) => Response::bad_request("Invalid JSON"),
            }
        })
        // Query the RAG pipeline
        .post("/v1/query", |req| async move {
            #[derive(Deserialize)]
            struct QueryRequest {
                question: String,
                #[serde(default = "default_k")]
                top_k: usize,
            }

            fn default_k() -> usize {
                3
            }

            match req.json::<QueryRequest>() {
                Ok(query) => {
                    // In production:
                    // 1. Embed the question using LLM provider
                    // 2. Search the vector store
                    // 3. Build context from top-k results
                    // 4. Render the RAG prompt template
                    // 5. Send to LLM for completion

                    let prompt = PromptBuilder::new()
                        .system("You are a RAG-powered assistant.")
                        .user(&query.question)
                        .context("Retrieved documents would appear here")
                        .build();

                    Response::ok().json(serde_json::json!({
                        "question": query.question,
                        "answer": format!(
                            "RAG answer for: '{}'. In production, this retrieves \
                             top-{} relevant documents and generates an answer.",
                            query.question, query.top_k
                        ),
                        "sources": [],
                        "prompt_preview": prompt.full_text(),
                    }))
                }
                Err(_) => Response::bad_request("Invalid JSON"),
            }
        })
        // Search documents
        .post("/v1/search", |req| async move {
            #[derive(Deserialize)]
            struct SearchRequest {
                query: String,
                #[serde(default = "default_limit")]
                limit: usize,
            }

            fn default_limit() -> usize {
                5
            }

            match req.json::<SearchRequest>() {
                Ok(search) => Response::ok().json(serde_json::json!({
                    "query": search.query,
                    "results": [],
                    "note": "In production, this performs vector similarity search"
                })),
                Err(_) => Response::bad_request("Invalid JSON"),
            }
        })
        // Get document stats
        .get("/v1/stats", |_req| async {
            Response::ok().json(serde_json::json!({
                "total_documents": 3,
                "vector_dimensions": 384,
                "distance_metric": "cosine",
                "index_type": "HNSW"
            }))
        })
        .middleware(LoggingMiddleware::new())
        .bind("0.0.0.0:8081");

    tracing::info!("RAG pipeline server starting on http://0.0.0.0:8081");
    tracing::info!("Endpoints:");
    tracing::info!("  POST /v1/documents  - Ingest a document");
    tracing::info!("  POST /v1/query      - Ask a question (RAG)");
    tracing::info!("  POST /v1/search     - Semantic search");
    tracing::info!("  GET  /v1/stats      - Index statistics");

    if let Err(e) = app.run().await {
        tracing::error!(error = %e, "server failed");
    }
}
