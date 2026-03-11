//! # NeuralFrame Agent Orchestration
//!
//! ReAct-pattern agents with planning, tool use, memory integration,
//! multi-agent collaboration, and autonomous goal-pursuit.

pub mod orchestrator;
pub mod planning;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

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
///
/// Agents implement the ReAct (Reasoning + Acting) pattern:
/// 1. Think about the task
/// 2. Decide on an action
/// 3. Observe the result
/// 4. Repeat until done
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

/// A simple ReAct agent implementation.
pub struct ReActAgent {
    config: AgentConfig,
}

impl ReActAgent {
    /// Create a new ReAct agent.
    pub fn new(config: AgentConfig) -> Self {
        Self { config }
    }

    /// Parse LLM response text into an AgentAction.
    ///
    /// Expected format:
    ///   Thought: <reasoning>
    ///   Action: <tool_name>
    ///   Action Input: <json_input>
    /// or:
    ///   Thought: <reasoning>
    ///   Final Answer: <answer>
    #[allow(dead_code)]
    fn parse_response(text: &str) -> AgentAction {
        let mut thought = String::new();
        let mut action_tool = None;
        let mut action_input = None;
        let mut final_answer = None;

        for line in text.lines() {
            let trimmed = line.trim();
            if let Some(t) = trimmed.strip_prefix("Thought:") {
                thought = t.trim().to_string();
            } else if let Some(a) = trimmed.strip_prefix("Action:") {
                action_tool = Some(a.trim().to_string());
            } else if let Some(ai) = trimmed.strip_prefix("Action Input:") {
                action_input = Some(ai.trim().to_string());
            } else if let Some(fa) = trimmed.strip_prefix("Final Answer:") {
                final_answer = Some(fa.trim().to_string());
            }
        }

        if let Some(answer) = final_answer {
            return AgentAction::Answer(answer);
        }

        if let Some(tool) = action_tool {
            let input = action_input
                .and_then(|s| serde_json::from_str(&s).ok())
                .unwrap_or(serde_json::json!({}));
            return AgentAction::Act { tool, input };
        }

        if !thought.is_empty() {
            return AgentAction::Think(thought);
        }

        // If we can't parse the response in the expected format,
        // treat the entire text as the final answer.
        AgentAction::Answer(text.trim().to_string())
    }
}

#[async_trait]
impl Agent for ReActAgent {
    async fn run(&self, task: &str) -> Result<AgentResult, AgentError> {
        let start = std::time::Instant::now();
        let mut trace = Vec::new();

        // Step 1: Think about the task
        let think_step = AgentStep {
            step: 1,
            action: AgentAction::Think(format!(
                "I need to complete this task: {}",
                task
            )),
            timestamp: chrono::Utc::now(),
            duration_ms: 0,
        };
        trace.push(think_step);

        // ReAct loop: iterate up to max_iterations
        // Without an LLM provider wired in, the agent produces a final answer
        // directly. When an LLM is connected, this loop would:
        //   1. Build a prompt with system prompt, tools, and history
        //   2. Call the LLM to get Thought/Action/Final Answer
        //   3. Execute tools and feed observations back
        //   4. Repeat until a Final Answer is produced or max iterations hit
        let mut iteration = 1;
        let answer = loop {
            iteration += 1;
            if iteration > self.config.max_iterations {
                return Err(AgentError::MaxIterationsReached(self.config.max_iterations));
            }

            // Without a live LLM, produce the final answer
            let step_start = std::time::Instant::now();
            let action = AgentAction::Answer(format!("Completed: {}", task));

            let step = AgentStep {
                step: iteration,
                action: action.clone(),
                timestamp: chrono::Utc::now(),
                duration_ms: step_start.elapsed().as_millis() as u64,
            };
            trace.push(step);

            if let AgentAction::Answer(ref ans) = action {
                break ans.clone();
            }
        };

        Ok(AgentResult {
            answer,
            steps: trace.len(),
            trace,
            total_duration_ms: start.elapsed().as_millis() as u64,
            tokens_used: 0,
        })
    }

    fn config(&self) -> &AgentConfig {
        &self.config
    }
}

/// Prelude re-exports.
pub mod prelude {
    pub use crate::orchestrator::*;
    pub use crate::planning::*;
    pub use crate::{Agent, AgentAction, AgentConfig, AgentError, AgentResult, AgentStep, ReActAgent};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_config() {
        let config = AgentConfig::new("test-agent", "You are helpful")
            .with_model("gpt-4o")
            .with_max_iterations(5)
            .with_tool("search");

        assert_eq!(config.name, "test-agent");
        assert_eq!(config.max_iterations, 5);
        assert!(config.tools.contains(&"search".to_string()));
    }

    #[tokio::test]
    async fn test_react_agent() {
        let config = AgentConfig::new("test", "Be helpful");
        let agent = ReActAgent::new(config);

        let result = agent.run("test task").await.unwrap();
        assert!(!result.answer.is_empty());
        assert!(result.steps > 0);
    }
}
