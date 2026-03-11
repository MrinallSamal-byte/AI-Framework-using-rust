//! # NeuralFrame Agent Orchestration
//!
//! ReAct-pattern agents with planning, tool use, memory integration,
//! multi-agent collaboration, and autonomous goal-pursuit.

pub mod builtin_tools;
pub mod orchestrator;
pub mod planning;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::sync::Arc;

/// Errors from the agent system.
#[derive(Debug)]
pub enum AgentError {
    /// LLM call failed.
    LLMError(String),
    /// Tool execution failed.
    ToolError(String),
    /// Planning failed.
    PlanningError(String),
    /// Agent reached max iterations.
    MaxIterationsReached(usize),
    /// Agent configuration error.
    ConfigError(String),
}

impl fmt::Display for AgentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LLMError(msg) => write!(f, "LLM error: {}", msg),
            Self::ToolError(msg) => write!(f, "tool error: {}", msg),
            Self::PlanningError(msg) => write!(f, "planning error: {}", msg),
            Self::MaxIterationsReached(n) => write!(f, "max iterations {} reached", n),
            Self::ConfigError(msg) => write!(f, "config error: {}", msg),
        }
    }
}

impl std::error::Error for AgentError {}

/// An agent's action: think, act, or observe.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AgentAction {
    /// Agent is thinking/reasoning.
    Think(String),
    /// Agent is calling a tool.
    Act {
        tool: String,
        input: serde_json::Value,
    },
    /// Agent observed tool output.
    Observe(String),
    /// Agent has a final answer.
    Answer(String),
}

/// A step in the agent's execution trace.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentStep {
    /// Step number.
    pub step: usize,
    /// The action taken.
    pub action: AgentAction,
    /// Timestamp.
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Duration in milliseconds.
    pub duration_ms: u64,
}

/// Configuration for an agent.
#[derive(Debug, Clone)]
pub struct AgentConfig {
    /// Agent name.
    pub name: String,
    /// System prompt / persona.
    pub system_prompt: String,
    /// Maximum iterations before stopping.
    pub max_iterations: usize,
    /// Temperature for LLM calls.
    pub temperature: f32,
    /// Model to use.
    pub model: String,
    /// Available tool names.
    pub tools: Vec<String>,
}

impl AgentConfig {
    /// Create a new agent config with a name and system prompt.
    pub fn new(name: &str, system_prompt: &str) -> Self {
        Self {
            name: name.to_string(),
            system_prompt: system_prompt.to_string(),
            max_iterations: 10,
            temperature: 0.7,
            model: "gpt-4o".to_string(),
            tools: Vec::new(),
        }
    }

    /// Set the model.
    pub fn with_model(mut self, model: &str) -> Self {
        self.model = model.to_string();
        self
    }

    /// Set the sampling temperature.
    pub fn with_temperature(mut self, temp: f32) -> Self {
        self.temperature = temp;
        self
    }

    /// Set max iterations.
    pub fn with_max_iterations(mut self, max: usize) -> Self {
        self.max_iterations = max;
        self
    }

    /// Add a tool.
    pub fn with_tool(mut self, tool_name: &str) -> Self {
        self.tools.push(tool_name.to_string());
        self
    }
}

/// Result of running an agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentResult {
    /// Final answer.
    pub answer: String,
    /// Number of steps taken.
    pub steps: usize,
    /// Execution trace.
    pub trace: Vec<AgentStep>,
    /// Total duration in milliseconds.
    pub total_duration_ms: u64,
    /// Token usage.
    pub tokens_used: u64,
}

/// The core Agent trait.
#[async_trait]
pub trait Agent: Send + Sync {
    /// Run the agent with a given task.
    async fn run(&self, task: &str) -> Result<AgentResult, AgentError>;

    /// Get the agent configuration.
    fn config(&self) -> &AgentConfig;

    /// Get the agent name.
    fn name(&self) -> &str {
        &self.config().name
    }
}

/// A ReAct agent implementation.
pub struct ReActAgent {
    config: AgentConfig,
    provider: Option<Arc<dyn neuralframe_llm::providers::LLMProvider>>,
    tool_registry: Option<Arc<neuralframe_llm::tools::ToolRegistry>>,
}

impl ReActAgent {
    /// Create a new ReAct agent.
    pub fn new(config: AgentConfig) -> Self {
        Self {
            config,
            provider: None,
            tool_registry: None,
        }
    }

    /// Configure an LLM provider for the agent.
    pub fn with_provider<P: neuralframe_llm::providers::LLMProvider + 'static>(
        mut self,
        provider: P,
    ) -> Self {
        self.provider = Some(Arc::new(provider));
        self
    }

    /// Configure a tool registry for the agent.
    pub fn with_tools(mut self, registry: neuralframe_llm::tools::ToolRegistry) -> Self {
        self.tool_registry = Some(Arc::new(registry));
        self
    }

    fn build_system_prompt(&self) -> String {
        if self.config.tools.is_empty() {
            return self.config.system_prompt.clone();
        }
        let tool_names = self.config.tools.join(", ");
        format!(
            "{}\n\nAvailable tools: {}\n\nRespond in this exact format:\nThought: <reasoning>\nAction: <tool_name>\nAction Input: <valid JSON object>\n\nWhen you have a final answer:\nThought: <reasoning>\nFinal Answer: <your complete answer>",
            self.config.system_prompt,
            tool_names
        )
    }

    fn parse_response(text: &str) -> AgentAction {
        let mut thought = String::new();
        let mut action_tool: Option<String> = None;
        let mut action_input_lines: Vec<String> = Vec::new();
        let mut collecting_input = false;
        let mut final_answer: Option<String> = None;

        for line in text.lines() {
            let trimmed = line.trim();
            if let Some(t) = trimmed.strip_prefix("Thought:") {
                collecting_input = false;
                thought = t.trim().to_string();
            } else if let Some(a) = trimmed.strip_prefix("Action:") {
                collecting_input = false;
                action_tool = Some(a.trim().to_string());
            } else if let Some(ai) = trimmed.strip_prefix("Action Input:") {
                collecting_input = true;
                action_input_lines.push(ai.trim().to_string());
            } else if let Some(fa) = trimmed.strip_prefix("Final Answer:") {
                collecting_input = false;
                final_answer = Some(fa.trim().to_string());
            } else if collecting_input {
                action_input_lines.push(trimmed.to_string());
            }
        }

        if let Some(answer) = final_answer {
            return AgentAction::Answer(answer);
        }

        if let Some(tool) = action_tool {
            let raw = action_input_lines.join("\n");
            let input =
                serde_json::from_str(&raw).unwrap_or_else(|_| serde_json::json!({"input": raw}));
            return AgentAction::Act { tool, input };
        }

        if !thought.is_empty() {
            return AgentAction::Think(thought);
        }

        AgentAction::Answer(text.trim().to_string())
    }
}

#[async_trait]
impl Agent for ReActAgent {
    async fn run(&self, task: &str) -> Result<AgentResult, AgentError> {
        use neuralframe_llm::types::{CompletionRequest, Message};

        let start = std::time::Instant::now();
        let mut trace: Vec<AgentStep> = Vec::new();
        let mut conversation: Vec<Message> = Vec::new();
        let system_prompt = self.build_system_prompt();
        conversation.push(Message::system(&system_prompt));
        conversation.push(Message::user(task));
        let mut iteration = 0usize;

        loop {
            iteration += 1;
            if iteration > self.config.max_iterations {
                return Err(AgentError::MaxIterationsReached(self.config.max_iterations));
            }

            let step_start = std::time::Instant::now();
            let llm_response = if let Some(ref provider) = self.provider {
                let mut req = CompletionRequest::new(&self.config.model);
                req.messages = conversation.clone();
                req.temperature = Some(self.config.temperature);
                provider
                    .complete(req)
                    .await
                    .map_err(|e| AgentError::LLMError(e.to_string()))?
                    .content
            } else {
                format!("Final Answer: Completed: {}", task)
            };

            let action = Self::parse_response(&llm_response);
            let duration_ms = step_start.elapsed().as_millis() as u64;
            trace.push(AgentStep {
                step: iteration,
                action: action.clone(),
                timestamp: chrono::Utc::now(),
                duration_ms,
            });

            match action {
                AgentAction::Answer(answer) => {
                    return Ok(AgentResult {
                        answer,
                        steps: trace.len(),
                        trace,
                        total_duration_ms: start.elapsed().as_millis() as u64,
                        tokens_used: 0,
                    });
                }
                AgentAction::Act {
                    ref tool,
                    ref input,
                } => {
                    conversation.push(Message::assistant(&llm_response));
                    let observation = if let Some(ref registry) = self.tool_registry {
                        registry
                            .execute(tool, input.clone())
                            .await
                            .unwrap_or_else(|e| format!("Tool error: {}", e))
                    } else {
                        format!("Tool registry not configured for tool: {}", tool)
                    };
                    trace.push(AgentStep {
                        step: iteration,
                        action: AgentAction::Observe(observation.clone()),
                        timestamp: chrono::Utc::now(),
                        duration_ms: 0,
                    });
                    conversation.push(Message::user(&format!("Observation: {}", observation)));
                }
                AgentAction::Think(_) => {
                    conversation.push(Message::assistant(&llm_response));
                }
                AgentAction::Observe(_) => {}
            }
        }
    }

    fn config(&self) -> &AgentConfig {
        &self.config
    }
}

/// Prelude re-exports.
pub mod prelude {
    pub use crate::builtin_tools::{CurrentTimeTool, EchoTool};
    pub use crate::orchestrator::*;
    pub use crate::planning::*;
    pub use crate::{
        Agent, AgentAction, AgentConfig, AgentError, AgentResult, AgentStep, ReActAgent,
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use neuralframe_llm::error::LLMError;
    use neuralframe_llm::providers::LLMProvider;
    use neuralframe_llm::tools::ToolRegistry;
    use neuralframe_llm::types::{
        CompletionRequest, CompletionResponse, FinishReason, Token, Usage,
    };
    use std::pin::Pin;
    use std::sync::Mutex;
    use tokio_stream::Stream;

    struct MockProvider {
        responses: Mutex<Vec<String>>,
    }

    impl MockProvider {
        fn new(responses: Vec<&str>) -> Self {
            Self {
                responses: Mutex::new(responses.into_iter().map(String::from).collect()),
            }
        }
    }

    #[async_trait::async_trait]
    impl LLMProvider for MockProvider {
        async fn complete(&self, _req: CompletionRequest) -> Result<CompletionResponse, LLMError> {
            let content = match self.responses.lock() {
                Ok(mut responses) => {
                    if responses.is_empty() {
                        "Final Answer: done".to_string()
                    } else {
                        responses.remove(0)
                    }
                }
                Err(_) => "Final Answer: done".to_string(),
            };
            Ok(CompletionResponse {
                content,
                model: "mock".to_string(),
                usage: Usage::default(),
                finish_reason: Some(FinishReason::Stop),
                tool_calls: vec![],
            })
        }

        async fn stream(
            &self,
            _req: CompletionRequest,
        ) -> Result<Pin<Box<dyn Stream<Item = Result<Token, LLMError>> + Send>>, LLMError> {
            Ok(Box::pin(tokio_stream::empty()))
        }

        async fn embed(&self, _text: &str, _model: &str) -> Result<Vec<f32>, LLMError> {
            Ok(vec![0.0; 128])
        }

        fn name(&self) -> &str {
            "mock"
        }

        fn models(&self) -> Vec<String> {
            vec!["mock".to_string()]
        }
    }

    #[test]
    fn test_agent_config() {
        let config = AgentConfig::new("test-agent", "You are helpful")
            .with_model("gpt-4o")
            .with_temperature(0.2)
            .with_max_iterations(5)
            .with_tool("search");
        assert_eq!(config.name, "test-agent");
        assert_eq!(config.max_iterations, 5);
        assert_eq!(config.temperature, 0.2);
        assert!(config.tools.contains(&"search".to_string()));
    }

    #[test]
    fn test_parse_multiline_action_input() {
        let text = "Thought: I need to search\nAction: search\nAction Input: {\n  \"query\": \"rust\",\n  \"limit\": 5\n}";
        let action = ReActAgent::parse_response(text);
        match action {
            AgentAction::Act { tool, input } => {
                assert_eq!(tool, "search");
                assert_eq!(input["query"], "rust");
                assert_eq!(input["limit"], 5);
            }
            _ => panic!("expected Act"),
        }
    }

    #[tokio::test]
    async fn test_react_agent_no_provider_returns_answer() {
        let agent = ReActAgent::new(AgentConfig::new("test", "Be helpful"));
        let result = agent.run("hello").await;
        assert!(result.is_ok());
        assert!(!result.expect("result").answer.is_empty());
    }

    #[tokio::test]
    async fn test_react_agent_with_mock_final_answer() {
        let provider = MockProvider::new(vec!["Final Answer: 42"]);
        let agent = ReActAgent::new(AgentConfig::new("test", "Be helpful")).with_provider(provider);
        let result = agent.run("What is the answer?").await;
        assert_eq!(result.expect("result").answer, "42");
    }

    #[tokio::test]
    async fn test_react_agent_with_echo_tool() {
        let provider = MockProvider::new(vec![
            "Thought: I should echo\nAction: echo\nAction Input: {\"input\": \"hello\"}",
            "Final Answer: hello",
        ]);
        let mut registry = ToolRegistry::new();
        registry.register(crate::builtin_tools::EchoTool);
        let agent = ReActAgent::new(AgentConfig::new("test", "Be helpful").with_tool("echo"))
            .with_provider(provider)
            .with_tools(registry);
        let result = agent.run("Echo hello").await.expect("result");
        assert_eq!(result.answer, "hello");
        assert!(result
            .trace
            .iter()
            .any(|s| matches!(s.action, AgentAction::Observe(_))));
    }

    #[tokio::test]
    async fn test_react_agent_max_iterations() {
        let provider = MockProvider::new(vec!["Thought: still thinking"; 20]);
        let agent = ReActAgent::new(AgentConfig::new("test", "Be helpful").with_max_iterations(3))
            .with_provider(provider);
        let result = agent.run("task").await;
        assert!(matches!(result, Err(AgentError::MaxIterationsReached(3))));
    }
}
