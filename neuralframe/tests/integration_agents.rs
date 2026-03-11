use neuralframe_agents::prelude::*;

#[tokio::test]
async fn test_react_agent_no_provider_returns_answer() {
    let agent = ReActAgent::new(AgentConfig::new("a", "Be helpful"));
    let result = agent
        .run("test task")
        .await
        .unwrap_or_else(|_| panic!("agent failed"));
    assert!(!result.answer.is_empty());
    assert!(result.steps > 0);
}

#[tokio::test]
async fn test_orchestrator_sequential_pipeline() {
    let mut orch = Orchestrator::new();
    orch.add_agent(ReActAgent::new(AgentConfig::new("a1", "Agent 1")));
    orch.add_agent(ReActAgent::new(AgentConfig::new("a2", "Agent 2")));
    let result = orch
        .run_sequential(&["a1", "a2"], "Initial task")
        .await
        .unwrap_or_else(|_| panic!("sequential failed"));
    assert!(!result.answer.is_empty());
    assert!(result.steps >= 2);
}

#[tokio::test]
async fn test_orchestrator_parallel_count() {
    let mut orch = Orchestrator::new();
    orch.add_agent(ReActAgent::new(AgentConfig::new("a1", "Agent 1")));
    orch.add_agent(ReActAgent::new(AgentConfig::new("a2", "Agent 2")));
    let results = orch
        .run_parallel(&["a1", "a2"], "Shared task")
        .await
        .unwrap_or_else(|_| panic!("parallel failed"));
    assert_eq!(results.len(), 2);
}

#[tokio::test]
async fn test_orchestrator_parallel_tolerant_partial_failure() {
    let mut orch = Orchestrator::new();
    orch.add_agent(ReActAgent::new(AgentConfig::new("good", "Good")));
    let results = orch
        .run_parallel_tolerant(&["good", "missing"], "task")
        .await;
    assert_eq!(results.len(), 2);
    assert!(results[0].is_ok());
    assert!(results[1].is_err());
}

#[tokio::test]
async fn test_plan_dag_dependency_ordering() {
    let mut plan = Plan::new("Goal");
    let s0 = plan.add_step(PlanStep::new("Step 0"));
    let _ = plan.add_step(PlanStep::new("Step 1").depends_on(s0));
    let s2 = plan.add_step(PlanStep::new("Step 2"));
    let next = plan.next_steps();
    assert!(next.contains(&s0));
    assert!(next.contains(&s2));
    assert!(!next.contains(&1));
    plan.complete_step(s0, "done");
    let next2 = plan.next_steps();
    assert!(next2.contains(&1));
}

#[tokio::test]
async fn test_agent_max_iterations_error() {
    let config = AgentConfig::new("a", "prompt").with_max_iterations(2);
    let agent = ReActAgent::new(config);
    let result = agent.run("task").await;
    assert!(matches!(
        result,
        Ok(_) | Err(AgentError::MaxIterationsReached(_))
    ));
}

#[tokio::test]
async fn test_echo_tool_execute() {
    use neuralframe_agents::builtin_tools::EchoTool;
    use neuralframe_llm::tools::Tool;
    let tool = EchoTool;
    let result = tool
        .execute(serde_json::json!({"input": "hello world"}))
        .await
        .unwrap_or_default();
    assert_eq!(result, "hello world");
}

#[tokio::test]
async fn test_current_time_tool_returns_valid_iso8601() {
    use neuralframe_agents::builtin_tools::CurrentTimeTool;
    use neuralframe_llm::tools::Tool;
    let tool = CurrentTimeTool;
    let result = tool
        .execute(serde_json::json!({}))
        .await
        .unwrap_or_default();
    assert!(chrono::DateTime::parse_from_rfc3339(&result).is_ok());
}
