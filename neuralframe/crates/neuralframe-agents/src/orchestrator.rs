//! Multi-agent orchestration for collaborative task execution.

use crate::{Agent, AgentConfig, AgentError, AgentResult, AgentStep, AgentAction};
use std::collections::HashMap;
use std::sync::Arc;

/// Orchestrator that manages multiple agents working together.
///
/// Supports sequential and parallel agent execution.
pub struct Orchestrator {
    agents: HashMap<String, Arc<dyn Agent>>,
    max_rounds: usize,
}

impl Orchestrator {
    /// Create a new orchestrator.
    pub fn new() -> Self {
        Self {
            agents: HashMap::new(),
            max_rounds: 10,
        }
    }

    /// Set the maximum rounds of collaboration.
    pub fn with_max_rounds(mut self, rounds: usize) -> Self {
        self.max_rounds = rounds;
        self
    }

    /// Register an agent.
    pub fn add_agent<A: Agent + 'static>(&mut self, agent: A) {
        let name = agent.config().name.clone();
        self.agents.insert(name, Arc::new(agent));
    }

    /// Run a single agent by name.
    pub async fn run_agent(
        &self,
        name: &str,
        task: &str,
    ) -> Result<AgentResult, AgentError> {
        let agent = self
            .agents
            .get(name)
            .ok_or(AgentError::ConfigError(format!(
                "agent '{}' not found",
                name
            )))?;

        agent.run(task).await
    }

    /// Run agents sequentially, piping each output to the next.
    pub async fn run_sequential(
        &self,
        agent_names: &[&str],
        initial_task: &str,
    ) -> Result<AgentResult, AgentError> {
        let mut current_input = initial_task.to_string();
        let mut all_traces = Vec::new();
        let start = std::time::Instant::now();

        for name in agent_names {
            let result = self.run_agent(name, &current_input).await?;
            current_input = result.answer.clone();
            all_traces.extend(result.trace);
        }

        Ok(AgentResult {
            answer: current_input,
            steps: all_traces.len(),
            trace: all_traces,
            total_duration_ms: start.elapsed().as_millis() as u64,
            tokens_used: 0,
        })
    }

    /// Run agents in parallel and aggregate results.
    pub async fn run_parallel(
        &self,
        agent_names: &[&str],
        task: &str,
    ) -> Result<Vec<AgentResult>, AgentError> {
        let mut handles = Vec::new();

        for name in agent_names {
            let agent = self
                .agents
                .get(*name)
                .ok_or(AgentError::ConfigError(format!(
                    "agent '{}' not found",
                    name
                )))?
                .clone();
            let task = task.to_string();

            handles.push(tokio::spawn(async move { agent.run(&task).await }));
        }

        let mut results = Vec::new();
        for handle in handles {
            let result = handle
                .await
                .map_err(|e| AgentError::LLMError(e.to_string()))?;
            results.push(result?);
        }

        Ok(results)
    }

    /// List registered agents.
    pub fn list_agents(&self) -> Vec<String> {
        self.agents.keys().cloned().collect()
    }

    /// Get the number of registered agents.
    pub fn agent_count(&self) -> usize {
        self.agents.len()
    }
}

impl Default for Orchestrator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ReActAgent;

    #[tokio::test]
    async fn test_orchestrator() {
        let mut orch = Orchestrator::new();

        let agent1 = ReActAgent::new(
            AgentConfig::new("research", "You research topics"),
        );
        let agent2 = ReActAgent::new(
            AgentConfig::new("writer", "You write responses"),
        );

        orch.add_agent(agent1);
        orch.add_agent(agent2);

        assert_eq!(orch.agent_count(), 2);

        let result = orch.run_agent("research", "Test task").await.unwrap();
        assert!(!result.answer.is_empty());
    }

    #[tokio::test]
    async fn test_sequential() {
        let mut orch = Orchestrator::new();
        orch.add_agent(ReActAgent::new(
            AgentConfig::new("a1", "Agent 1"),
        ));
        orch.add_agent(ReActAgent::new(
            AgentConfig::new("a2", "Agent 2"),
        ));

        let result = orch
            .run_sequential(&["a1", "a2"], "Initial task")
            .await
            .unwrap();
        assert!(!result.answer.is_empty());
    }

    #[tokio::test]
    async fn test_parallel() {
        let mut orch = Orchestrator::new();
        orch.add_agent(ReActAgent::new(
            AgentConfig::new("a1", "Agent 1"),
        ));
        orch.add_agent(ReActAgent::new(
            AgentConfig::new("a2", "Agent 2"),
        ));

        let results = orch
            .run_parallel(&["a1", "a2"], "Shared task")
            .await
            .unwrap();
        assert_eq!(results.len(), 2);
    }
}
