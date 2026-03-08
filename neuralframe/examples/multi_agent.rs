//! # Multi-Agent System Example
//!
//! Demonstrates a multi-agent system with:
//! - Research agent that gathers information
//! - Analysis agent that processes data
//! - Writer agent that composes responses
//! - Orchestrator that coordinates the pipeline
//!
//! ## Running
//!
//! ```bash
//! cargo run --example multi_agent
//! ```

use neuralframe_agents::prelude::*;
use neuralframe_core::prelude::*;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter("info")
        .init();

    tracing::info!("Starting NeuralFrame Multi-Agent System");

    // ── Define agents ──────────────────────────────────────────────

    let research_agent = ReActAgent::new(
        AgentConfig::new(
            "researcher",
            "You are a research agent. Your job is to gather relevant \
             information about a topic. Use the search tool to find data, \
             then summarize your findings.",
        )
        .with_model("gpt-4o")
        .with_max_iterations(5)
        .with_tool("web_search")
        .with_tool("document_lookup"),
    );

    let analysis_agent = ReActAgent::new(
        AgentConfig::new(
            "analyst",
            "You are a data analysis agent. Your job is to take research \
             findings and extract key insights, patterns, and actionable \
             recommendations.",
        )
        .with_model("gpt-4o")
        .with_max_iterations(3),
    );

    let writer_agent = ReActAgent::new(
        AgentConfig::new(
            "writer",
            "You are a technical writer agent. Your job is to take analyzed \
             insights and compose a clear, well-structured response for the \
             end user.",
        )
        .with_model("gpt-4o")
        .with_max_iterations(3),
    );

    // ── Set up orchestrator ────────────────────────────────────────

    let mut orchestrator = Orchestrator::new().with_max_rounds(10);
    orchestrator.add_agent(research_agent);
    orchestrator.add_agent(analysis_agent);
    orchestrator.add_agent(writer_agent);

    tracing::info!(
        agents = orchestrator.agent_count(),
        "orchestrator initialized"
    );

    // ── Create a plan ──────────────────────────────────────────────

    let mut plan = Plan::new("Research and write about NeuralFrame's architecture");
    let step0 = plan.add_step(
        PlanStep::new("Research NeuralFrame's core components")
            .with_tool("web_search", serde_json::json!({"query": "NeuralFrame Rust AI framework"})),
    );
    let step1 = plan.add_step(
        PlanStep::new("Analyze the architecture patterns used").depends_on(step0),
    );
    let _step2 = plan.add_step(
        PlanStep::new("Write a summary document").depends_on(step1),
    );

    tracing::info!(
        steps = plan.steps.len(),
        "execution plan created"
    );

    // ── Execute sequential pipeline ────────────────────────────────

    tracing::info!("Running sequential agent pipeline: researcher → analyst → writer");

    let result = orchestrator
        .run_sequential(
            &["researcher", "analyst", "writer"],
            "Explain the architecture of NeuralFrame, focusing on how it \
             handles LLM integration, vector search, and agent orchestration.",
        )
        .await;

    match result {
        Ok(agent_result) => {
            tracing::info!(
                steps = agent_result.steps,
                duration_ms = agent_result.total_duration_ms,
                "sequential pipeline completed"
            );
            println!("\n═══════════════════════════════════════════");
            println!("  Sequential Pipeline Result");
            println!("═══════════════════════════════════════════");
            println!("Answer: {}", agent_result.answer);
            println!("Steps: {}", agent_result.steps);
            println!("Duration: {}ms", agent_result.total_duration_ms);
        }
        Err(e) => {
            tracing::error!(error = %e, "sequential pipeline failed");
        }
    }

    // ── Execute parallel agents ────────────────────────────────────

    tracing::info!("Running parallel agents: researcher + analyst");

    let results = orchestrator
        .run_parallel(
            &["researcher", "analyst"],
            "What are the key design decisions in building an AI-native web framework?",
        )
        .await;

    match results {
        Ok(parallel_results) => {
            println!("\n═══════════════════════════════════════════");
            println!("  Parallel Execution Results");
            println!("═══════════════════════════════════════════");
            for (i, result) in parallel_results.iter().enumerate() {
                println!("Agent {}: {} ({}ms)", i + 1, result.answer, result.total_duration_ms);
            }
        }
        Err(e) => {
            tracing::error!(error = %e, "parallel execution failed");
        }
    }

    // ── Plan execution tracking ────────────────────────────────────

    println!("\n═══════════════════════════════════════════");
    println!("  Plan Progress");
    println!("═══════════════════════════════════════════");

    let next = plan.next_steps();
    println!("Next actionable steps: {:?}", next);

    // Simulate completing steps
    for step_idx in next {
        plan.complete_step(step_idx, "completed");
    }

    println!("Progress: {:.0}%", plan.progress());
    println!("Complete: {}", plan.is_complete());

    // ── Expose via HTTP ────────────────────────────────────────────

    let app = NeuralFrame::new()
        .name("NeuralFrame Multi-Agent")
        .post("/v1/agents/run", |req| async move {
            #[derive(Deserialize)]
            struct AgentRequest {
                task: String,
                agents: Option<Vec<String>>,
                #[serde(default = "default_mode")]
                mode: String,
            }

            fn default_mode() -> String {
                "sequential".to_string()
            }

            match req.json::<AgentRequest>() {
                Ok(agent_req) => {
                    let agent_req = agent_req.into_inner();
                    Response::ok().json(serde_json::json!({
                        "task": agent_req.task,
                        "mode": agent_req.mode,
                        "agents": agent_req.agents.unwrap_or_else(|| {
                            vec!["researcher".into(), "analyst".into(), "writer".into()]
                        }),
                        "status": "completed",
                        "result": format!("Processed: {}", agent_req.task)
                    }))
                }
                Err(_) => Response::bad_request("Invalid JSON"),
            }
        })
        .get("/v1/agents", |_req| async {
            Response::ok().json(serde_json::json!({
                "agents": [
                    {"name": "researcher", "model": "gpt-4o", "tools": ["web_search", "document_lookup"]},
                    {"name": "analyst", "model": "gpt-4o", "tools": []},
                    {"name": "writer", "model": "gpt-4o", "tools": []}
                ]
            }))
        })
        .middleware(LoggingMiddleware::new())
        .bind("0.0.0.0:8082");

    tracing::info!("Multi-agent server starting on http://0.0.0.0:8082");

    if let Err(e) = app.run().await {
        tracing::error!(error = %e, "server failed");
    }
}
