//! # NeuralFrame Python Bindings
//!
//! PyO3 bindings for the NeuralFrame framework.

use pyo3::prelude::*;
use std::sync::Arc;

fn to_completion_request(req: &PyCompletionRequest) -> neuralframe_llm::types::CompletionRequest {
    let mut cr = neuralframe_llm::types::CompletionRequest::new(&req.model);
    for msg in &req.messages {
        match msg.role.as_str() {
            "system" => {
                cr = cr.system(&msg.content);
            }
            "assistant" => {
                cr = cr.assistant(&msg.content);
            }
            _ => {
                cr = cr.user(&msg.content);
            }
        }
    }
    if let Some(mt) = req.max_tokens {
        cr.max_tokens = Some(mt);
    }
    if let Some(t) = req.temperature {
        cr.temperature = Some(t);
    }
    cr
}

fn to_py_response(resp: neuralframe_llm::types::CompletionResponse) -> PyCompletionResponse {
    PyCompletionResponse {
        content: resp.content,
        model: resp.model,
        prompt_tokens: resp.usage.prompt_tokens,
        completion_tokens: resp.usage.completion_tokens,
        total_tokens: resp.usage.total_tokens,
    }
}

/// NeuralFrame Python module.
#[pymodule]
fn neuralframe(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let mut builder = tokio::runtime::Builder::new_multi_thread();
    builder.enable_all();
    pyo3_asyncio_0_21::tokio::init(builder);
    m.add_class::<PyCompletionRequest>()?;
    m.add_class::<PyCompletionResponse>()?;
    m.add_class::<PyMessage>()?;
    m.add_class::<PyVectorStore>()?;
    m.add_class::<PySemanticCache>()?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(complete_openai, m)?)?;
    m.add_function(wrap_pyfunction!(complete_anthropic, m)?)?;
    m.add_function(wrap_pyfunction!(complete_groq, m)?)?;
    Ok(())
}

/// Return the NeuralFrame version.
#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// A completion request for the LLM.
#[pyclass(name = "CompletionRequest")]
#[derive(Debug, Clone)]
struct PyCompletionRequest {
    #[pyo3(get, set)]
    model: String,
    #[pyo3(get, set)]
    messages: Vec<PyMessage>,
    #[pyo3(get, set)]
    max_tokens: Option<u32>,
    #[pyo3(get, set)]
    temperature: Option<f32>,
    #[pyo3(get, set)]
    stream: bool,
}

#[pymethods]
impl PyCompletionRequest {
    #[new]
    fn new(model: String) -> Self {
        Self {
            model,
            messages: Vec::new(),
            max_tokens: None,
            temperature: None,
            stream: false,
        }
    }

    /// Add a system message.
    fn system(&mut self, content: String) {
        self.messages.push(PyMessage {
            role: "system".to_string(),
            content,
        });
    }

    /// Add a user message.
    fn user(&mut self, content: String) {
        self.messages.push(PyMessage {
            role: "user".to_string(),
            content,
        });
    }

    /// Add an assistant message.
    fn assistant(&mut self, content: String) {
        self.messages.push(PyMessage {
            role: "assistant".to_string(),
            content,
        });
    }

    fn __repr__(&self) -> String {
        format!(
            "CompletionRequest(model='{}', messages={}, stream={})",
            self.model,
            self.messages.len(),
            self.stream,
        )
    }
}

/// A chat message.
#[pyclass(name = "Message")]
#[derive(Debug, Clone)]
struct PyMessage {
    #[pyo3(get, set)]
    role: String,
    #[pyo3(get, set)]
    content: String,
}

#[pymethods]
impl PyMessage {
    #[new]
    fn new(role: String, content: String) -> Self {
        Self { role, content }
    }

    fn __repr__(&self) -> String {
        format!("Message(role='{}', content='{}')", self.role, self.content)
    }
}

/// A completion response from the LLM.
#[pyclass(name = "CompletionResponse")]
#[derive(Debug, Clone)]
struct PyCompletionResponse {
    #[pyo3(get)]
    content: String,
    #[pyo3(get)]
    model: String,
    #[pyo3(get)]
    prompt_tokens: u32,
    #[pyo3(get)]
    completion_tokens: u32,
    #[pyo3(get)]
    total_tokens: u32,
}

#[pymethods]
impl PyCompletionResponse {
    fn __repr__(&self) -> String {
        format!(
            "CompletionResponse(model='{}', tokens={})",
            self.model, self.total_tokens
        )
    }
}

/// Call the OpenAI chat completion API from Python.
#[pyfunction]
fn complete_openai<'py>(
    py: Python<'py>,
    req: PyCompletionRequest,
    api_key: String,
) -> PyResult<Bound<'py, PyAny>> {
    pyo3_asyncio_0_21::tokio::future_into_py(py, async move {
        use neuralframe_llm::providers::LLMProvider;
        let cr = to_completion_request(&req);
        let provider = neuralframe_llm::providers::openai::OpenAIProvider::new(&api_key);
        provider
            .complete(cr)
            .await
            .map(to_py_response)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    })
}

/// Call the Anthropic chat completion API from Python.
#[pyfunction]
fn complete_anthropic<'py>(
    py: Python<'py>,
    req: PyCompletionRequest,
    api_key: String,
) -> PyResult<Bound<'py, PyAny>> {
    pyo3_asyncio_0_21::tokio::future_into_py(py, async move {
        use neuralframe_llm::providers::LLMProvider;
        let cr = to_completion_request(&req);
        let provider = neuralframe_llm::providers::anthropic::AnthropicProvider::new(&api_key);
        provider
            .complete(cr)
            .await
            .map(to_py_response)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    })
}

/// Call the Groq chat completion API from Python.
#[pyfunction]
fn complete_groq<'py>(
    py: Python<'py>,
    req: PyCompletionRequest,
    api_key: String,
) -> PyResult<Bound<'py, PyAny>> {
    pyo3_asyncio_0_21::tokio::future_into_py(py, async move {
        use neuralframe_llm::providers::LLMProvider;
        let cr = to_completion_request(&req);
        let provider = neuralframe_llm::providers::groq::GroqProvider::new(&api_key);
        provider
            .complete(cr)
            .await
            .map(to_py_response)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    })
}

/// A vector store accessible from Python.
#[pyclass]
struct PyVectorStore {
    inner: Arc<neuralframe_vector::VectorStore>,
}

#[pymethods]
impl PyVectorStore {
    #[new]
    fn new(dimensions: usize) -> Self {
        Self {
            inner: Arc::new(neuralframe_vector::VectorStore::new(
                dimensions,
                neuralframe_vector::DistanceMetric::Cosine,
            )),
        }
    }

    /// Insert a vector with optional JSON metadata.
    fn insert(&self, id: String, vector: Vec<f32>, metadata: Option<String>) -> PyResult<()> {
        let meta: serde_json::Value = metadata
            .as_deref()
            .map(serde_json::from_str)
            .transpose()
            .map_err(|e: serde_json::Error| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
            .unwrap_or(serde_json::json!({}));
        self.inner
            .insert(&id, vector, meta)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Search for nearest vectors.
    fn search(&self, query: Vec<f32>, limit: usize) -> PyResult<Vec<PyObject>> {
        let results = self
            .inner
            .search(&query, limit, None)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Python::with_gil(|py| {
            results
                .into_iter()
                .map(|r| {
                    let d = pyo3::types::PyDict::new_bound(py);
                    d.set_item("id", &r.id)?;
                    d.set_item("score", r.score)?;
                    d.set_item("metadata", r.metadata.to_string())?;
                    Ok(d.into())
                })
                .collect()
        })
    }

    /// Delete a vector by ID.
    fn delete(&self, id: String) -> PyResult<()> {
        self.inner
            .delete(&id)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Return the number of stored vectors.
    fn __len__(&self) -> usize {
        self.inner.len()
    }
}

/// A semantic cache for LLM responses, accessible from Python.
#[pyclass]
struct PySemanticCache {
    inner: neuralframe_cache::SemanticCache,
}

#[pymethods]
impl PySemanticCache {
    #[new]
    #[pyo3(signature = (threshold=0.95, max_entries=10000, ttl_secs=3600))]
    fn new(threshold: f32, max_entries: usize, ttl_secs: u64) -> Self {
        Self {
            inner: neuralframe_cache::SemanticCache::new(neuralframe_cache::CacheConfig {
                similarity_threshold: threshold,
                max_entries,
                ttl: std::time::Duration::from_secs(ttl_secs),
                exact_match_first: true,
            }),
        }
    }

    /// Store a prompt-response pair with its embedding.
    fn store(&self, prompt: String, response: String, embedding: Vec<f32>, model: String) {
        self.inner.store(&prompt, &response, embedding, &model);
    }

    /// Look up a response by exact prompt match.
    fn get_exact(&self, prompt: String) -> Option<String> {
        self.inner.get_exact(&prompt).map(|r| r.response)
    }

    /// Look up a response by semantic similarity.
    fn get_semantic(&self, embedding: Vec<f32>) -> Option<String> {
        self.inner.get_semantic(&embedding).map(|r| r.response)
    }

    /// Return cache statistics as a dict.
    fn stats(&self) -> PyObject {
        let stats = self.inner.stats();
        Python::with_gil(|py| {
            let d = pyo3::types::PyDict::new_bound(py);
            let _ = d.set_item("total_entries", stats.total_entries);
            let _ = d.set_item("total_hits", stats.total_hits);
            let _ = d.set_item("max_entries", stats.max_entries);
            d.into()
        })
    }

    /// Clear all cached entries.
    fn clear(&self) {
        self.inner.clear();
    }
}
