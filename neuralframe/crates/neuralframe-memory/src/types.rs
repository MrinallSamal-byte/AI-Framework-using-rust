//! Memory types: Conversation, Summary, Vector, and Entity memory.

use crate::{MemoryEntry, MemoryError, MemoryStore};
use std::sync::Arc;

/// Conversation memory with a sliding window of the last N messages.
pub struct ConversationMemory {
    store: Arc<dyn MemoryStore>,
    window_size: usize,
}

impl ConversationMemory {
    /// Create a new conversation memory with the given window size.
    pub fn new(store: Arc<dyn MemoryStore>, window_size: usize) -> Self {
        Self { store, window_size }
    }

    /// Get the last N messages for a session.
    pub async fn get_context(&self, session: &str) -> Result<Vec<MemoryEntry>, MemoryError> {
        let all = self.store.get_all(session).await?;
        let start = if all.len() > self.window_size {
            all.len() - self.window_size
        } else {
            0
        };
        Ok(all[start..].to_vec())
    }

    /// Add a message to the conversation.
    pub async fn add(&self, session: &str, role: &str, content: &str) -> Result<(), MemoryError> {
        let entry = MemoryEntry::new(session, role, content);
        self.store.store(session, entry).await
    }
}

/// Summary memory that auto-summarizes old context.
pub struct SummaryMemory {
    store: Arc<dyn MemoryStore>,
    summary_threshold: usize,
}

impl SummaryMemory {
    /// Create a new summary memory.
    pub fn new(store: Arc<dyn MemoryStore>, summary_threshold: usize) -> Self {
        Self {
            store,
            summary_threshold,
        }
    }

    /// Get the current summary.
    pub async fn get_summary(&self, session: &str) -> Result<Option<String>, MemoryError> {
        let entries = self.store.retrieve(session, "summary", 1).await?;
        Ok(entries.first().map(|e| e.content.clone()))
    }

    /// Check if summarization is needed.
    pub async fn needs_summarization(&self, session: &str) -> Result<bool, MemoryError> {
        let count = self.store.count(session).await?;
        Ok(count > self.summary_threshold)
    }

    /// Summarize the conversation for a session using an LLM provider.
    pub async fn summarize(
        &self,
        session: &str,
        provider: &dyn neuralframe_llm::providers::LLMProvider,
        model: &str,
    ) -> Result<String, MemoryError> {
        let entries = self.store.get_all(session).await?;

        let transcript: String = entries
            .iter()
            .map(|e| format!("{}: {}", e.role, e.content))
            .collect::<Vec<_>>()
            .join("\n");

        let req = neuralframe_llm::types::CompletionRequest::new(model)
            .system("Summarize the following conversation in 3-5 sentences.")
            .user(&transcript);

        let response = provider
            .complete(req)
            .await
            .map_err(|e| MemoryError::StorageError(e.to_string()))?;

        let summary = response.content.clone();

        let summary_entry = MemoryEntry::new(session, "summary", &summary);
        self.store.store(session, summary_entry).await?;

        Ok(summary)
    }
}

/// Vector memory using semantic retrieval of relevant past context.
pub struct VectorMemory {
    store: Arc<dyn MemoryStore>,
    similarity_threshold: f32,
}

impl VectorMemory {
    /// Create a new vector memory.
    pub fn new(store: Arc<dyn MemoryStore>) -> Self {
        Self {
            store,
            similarity_threshold: 0.7,
        }
    }

    /// Set the similarity threshold for retrieval.
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.similarity_threshold = threshold;
        self
    }

    /// Retrieve semantically relevant memories.
    pub async fn retrieve_relevant(
        &self,
        session: &str,
        query: &str,
        limit: usize,
    ) -> Result<Vec<MemoryEntry>, MemoryError> {
        self.store.retrieve(session, query, limit).await
    }
}

/// Entity memory for extracting and remembering named entities.
pub struct EntityMemory {
    store: Arc<dyn MemoryStore>,
}

impl EntityMemory {
    /// Create a new entity memory.
    pub fn new(store: Arc<dyn MemoryStore>) -> Self {
        Self { store }
    }

    /// Store an entity.
    pub async fn store_entity(
        &self,
        session: &str,
        entity_name: &str,
        entity_type: &str,
        description: &str,
    ) -> Result<(), MemoryError> {
        let entry = MemoryEntry::new(session, "entity", description).with_metadata(
            serde_json::json!({
                "entity_name": entity_name,
                "entity_type": entity_type,
            }),
        );
        self.store.store(session, entry).await
    }

    /// Retrieve entities by name.
    pub async fn get_entities(
        &self,
        session: &str,
        query: &str,
    ) -> Result<Vec<MemoryEntry>, MemoryError> {
        self.store.retrieve(session, query, 10).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::inmemory::InMemoryStore;

    #[tokio::test]
    async fn test_conversation_memory() {
        let store = Arc::new(InMemoryStore::new());
        let mem = ConversationMemory::new(store, 3);

        mem.add("s1", "user", "First").await.unwrap();
        mem.add("s1", "assistant", "Second").await.unwrap();
        mem.add("s1", "user", "Third").await.unwrap();
        mem.add("s1", "assistant", "Fourth").await.unwrap();

        let ctx = mem.get_context("s1").await.unwrap();
        assert_eq!(ctx.len(), 3);
        assert_eq!(ctx[0].content, "Second");
    }

    #[tokio::test]
    async fn test_summary_memory() {
        let store = Arc::new(InMemoryStore::new());
        let mem = SummaryMemory::new(store, 5);

        let needs = mem.needs_summarization("s1").await.unwrap();
        assert!(!needs);
    }

    #[tokio::test]
    async fn test_vector_memory() {
        let store = Arc::new(InMemoryStore::new());
        let mem = VectorMemory::new(store).with_threshold(0.8);
        assert_eq!(mem.similarity_threshold, 0.8);
    }

    #[tokio::test]
    async fn test_entity_memory() {
        let store = Arc::new(InMemoryStore::new());
        let mem = EntityMemory::new(store);

        mem.store_entity("s1", "Alice", "person", "A software engineer")
            .await
            .unwrap();

        let entities = mem.get_entities("s1", "Alice").await.unwrap();
        assert!(!entities.is_empty());
    }

    // Mock LLM provider for testing SummaryMemory::summarize
    struct MockProvider {
        response: String,
    }

    impl MockProvider {
        fn new(response: &str) -> Self {
            Self {
                response: response.to_string(),
            }
        }
    }

    #[async_trait::async_trait]
    impl neuralframe_llm::providers::LLMProvider for MockProvider {
        async fn complete(
            &self,
            _req: neuralframe_llm::types::CompletionRequest,
        ) -> Result<neuralframe_llm::types::CompletionResponse, neuralframe_llm::error::LLMError>
        {
            Ok(neuralframe_llm::types::CompletionResponse {
                content: self.response.clone(),
                model: "mock".to_string(),
                usage: neuralframe_llm::types::Usage::default(),
                finish_reason: Some(neuralframe_llm::types::FinishReason::Stop),
                tool_calls: vec![],
            })
        }

        async fn stream(
            &self,
            _req: neuralframe_llm::types::CompletionRequest,
        ) -> Result<
            std::pin::Pin<
                Box<
                    dyn tokio_stream::Stream<
                            Item = Result<
                                neuralframe_llm::types::Token,
                                neuralframe_llm::error::LLMError,
                            >,
                        > + Send,
                >,
            >,
            neuralframe_llm::error::LLMError,
        > {
            Ok(Box::pin(tokio_stream::empty()))
        }

        async fn embed(
            &self,
            _text: &str,
            _model: &str,
        ) -> Result<Vec<f32>, neuralframe_llm::error::LLMError> {
            Ok(vec![0.0; 128])
        }

        fn name(&self) -> &str {
            "mock"
        }

        fn models(&self) -> Vec<String> {
            vec!["mock-model".to_string()]
        }
    }

    #[tokio::test]
    async fn test_summarize() {
        let store = Arc::new(InMemoryStore::new());
        let mem = SummaryMemory::new(store.clone(), 5);

        // Add some conversation entries
        store
            .store("s1", MemoryEntry::new("s1", "user", "Hello, how are you?"))
            .await
            .unwrap();
        store
            .store(
                "s1",
                MemoryEntry::new("s1", "assistant", "I'm doing well, thank you!"),
            )
            .await
            .unwrap();
        store
            .store(
                "s1",
                MemoryEntry::new("s1", "user", "Tell me about Rust programming."),
            )
            .await
            .unwrap();

        let mock_provider = MockProvider::new(
            "The user greeted the assistant and asked about Rust programming.",
        );

        let summary = mem.summarize("s1", &mock_provider, "mock-model").await.unwrap();
        assert_eq!(
            summary,
            "The user greeted the assistant and asked about Rust programming."
        );

        // The summary should have been stored
        let all = store.get_all("s1").await.unwrap();
        let summary_entries: Vec<_> = all.iter().filter(|e| e.role == "summary").collect();
        assert_eq!(summary_entries.len(), 1);
        assert_eq!(summary_entries[0].content, summary);
    }
}
