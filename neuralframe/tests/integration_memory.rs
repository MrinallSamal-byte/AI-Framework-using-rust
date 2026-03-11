use neuralframe_memory::{backends::inmemory::InMemoryStore, prelude::*, MemoryEntry};
use std::sync::Arc;

#[tokio::test]
async fn test_inmemory_sliding_window() {
    let store = InMemoryStore::new().with_max_entries(3);
    for i in 0..5usize {
        let _ = store
            .store("s1", MemoryEntry::new("s1", "user", &format!("msg {}", i)))
            .await;
    }
    let all = store.get_all("s1").await.unwrap_or_default();
    assert_eq!(all.len(), 3);
    assert!(all[0].content.contains('2'));
}

#[tokio::test]
async fn test_conversation_memory_window_size() {
    let store: Arc<dyn MemoryStore> = Arc::new(InMemoryStore::new());
    let mem = ConversationMemory::new(Arc::clone(&store), 2);
    let _ = mem.add("s1", "user", "first").await;
    let _ = mem.add("s1", "user", "second").await;
    let _ = mem.add("s1", "user", "third").await;
    let ctx = mem.get_context("s1").await.unwrap_or_default();
    assert_eq!(ctx.len(), 2);
    assert_eq!(ctx[0].content, "second");
    assert_eq!(ctx[1].content, "third");
}

#[tokio::test]
async fn test_entity_memory_store_and_retrieve() {
    let store: Arc<dyn MemoryStore> = Arc::new(InMemoryStore::new());
    let em = EntityMemory::new(Arc::clone(&store));
    let _ = em
        .store_entity("s1", "Alice", "person", "A software engineer")
        .await;
    let _ = em
        .store_entity("s1", "Rust", "language", "Systems programming language")
        .await;
    let results = em.get_entities("s1", "Alice").await.unwrap_or_default();
    assert!(!results.is_empty());
    assert!(results[0].content.contains("software engineer"));
}

#[tokio::test]
async fn test_context_builder_respects_token_limit() {
    let entries = vec![
        MemoryEntry {
            id: "1".into(),
            session_id: "s1".into(),
            role: "user".into(),
            content: "A".repeat(400),
            timestamp: chrono::Utc::now(),
            metadata: serde_json::json!({}),
            embedding: None,
            relevance: 1.0,
        },
        MemoryEntry {
            id: "2".into(),
            session_id: "s1".into(),
            role: "user".into(),
            content: "B".repeat(400),
            timestamp: chrono::Utc::now(),
            metadata: serde_json::json!({}),
            embedding: None,
            relevance: 0.5,
        },
    ];
    let ctx = ContextBuilder::new(50)
        .add_entries(entries)
        .build()
        .unwrap_or_default();
    assert!(!ctx.contains(&"B".repeat(400)));
}

#[tokio::test]
async fn test_separate_sessions_isolated() {
    let store = InMemoryStore::new();
    let _ = store
        .store("s1", MemoryEntry::new("s1", "user", "For session 1"))
        .await;
    let _ = store
        .store("s2", MemoryEntry::new("s2", "user", "For session 2"))
        .await;
    let s1 = store.get_all("s1").await.unwrap_or_default();
    let s2 = store.get_all("s2").await.unwrap_or_default();
    assert_eq!(s1.len(), 1);
    assert_eq!(s2.len(), 1);
    assert!(s1[0].content.contains("session 1"));
    assert!(s2[0].content.contains("session 2"));
}

#[tokio::test]
async fn test_sqlite_backend_full_cycle() {
    use neuralframe_memory::backends::sqlite::SqliteStore;
    let store = SqliteStore::in_memory();
    let _ = store
        .store("s1", MemoryEntry::new("s1", "user", "Hello Rust"))
        .await;
    let _ = store
        .store("s1", MemoryEntry::new("s1", "assistant", "Hello!"))
        .await;
    assert_eq!(store.count("s1").await.unwrap_or_default(), 2);
    let results = store.retrieve("s1", "Rust", 1).await.unwrap_or_default();
    assert!(!results.is_empty());
    assert!(results[0].content.contains("Rust"));
    let _ = store.clear("s1").await;
    assert_eq!(store.count("s1").await.unwrap_or_default(), 0);
}
