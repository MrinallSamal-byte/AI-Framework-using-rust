//! # NeuralFrame Python Bindings
//!
//! PyO3 bindings for the NeuralFrame framework, allowing
//! Python applications to use NeuralFrame's Rust internals.
//!
//! ## Usage
//!
//! ```python
//! import neuralframe
//!
//! # Create an app
//! app = neuralframe.NeuralFrame()
//! app.get("/", lambda req: "Hello from NeuralFrame!")
//! app.run("0.0.0.0:8080")
//! ```

use pyo3::prelude::*;

/// NeuralFrame Python module.
#[pymodule]
fn neuralframe(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCompletionRequest>()?;
    m.add_class::<PyCompletionResponse>()?;
    m.add_class::<PyMessage>()?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    Ok(())
}

/// Return the NeuralFrame version.
#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// A completion request for the LLM.
#[pyclass]
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
#[pyclass]
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
#[pyclass]
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
