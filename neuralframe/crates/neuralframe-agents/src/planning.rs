//! Planning module for goal decomposition and task planning.

use serde::{Deserialize, Serialize};

/// A plan node representing a step in achieving a goal.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanStep {
    /// Step description.
    pub description: String,
    /// Dependencies (step indices that must complete first).
    pub depends_on: Vec<usize>,
    /// Whether this step is complete.
    pub completed: bool,
    /// The tool to use (if applicable).
    pub tool: Option<String>,
    /// Input for the tool.
    pub input: Option<serde_json::Value>,
    /// Output from execution.
    pub output: Option<String>,
}

impl PlanStep {
    /// Create a new plan step.
    pub fn new(description: &str) -> Self {
        Self {
            description: description.to_string(),
            depends_on: Vec::new(),
            completed: false,
            tool: None,
            input: None,
            output: None,
        }
    }

    /// Add a dependency.
    pub fn depends_on(mut self, step: usize) -> Self {
        self.depends_on.push(step);
        self
    }

    /// Set the tool for this step.
    pub fn with_tool(mut self, tool: &str, input: serde_json::Value) -> Self {
        self.tool = Some(tool.to_string());
        self.input = Some(input);
        self
    }
}

/// A complete execution plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Plan {
    /// The goal being pursued.
    pub goal: String,
    /// Ordered steps to achieve the goal.
    pub steps: Vec<PlanStep>,
}

impl Plan {
    /// Create a new plan for a goal.
    pub fn new(goal: &str) -> Self {
        Self {
            goal: goal.to_string(),
            steps: Vec::new(),
        }
    }

    /// Add a step to the plan.
    pub fn add_step(&mut self, step: PlanStep) -> usize {
        let idx = self.steps.len();
        self.steps.push(step);
        idx
    }

    /// Get the next actionable steps (whose dependencies are met).
    pub fn next_steps(&self) -> Vec<usize> {
        self.steps
            .iter()
            .enumerate()
            .filter(|(_, step)| {
                !step.completed
                    && step
                        .depends_on
                        .iter()
                        .all(|dep| self.steps.get(*dep).is_some_and(|s| s.completed))
            })
            .map(|(i, _)| i)
            .collect()
    }

    /// Mark a step as complete.
    pub fn complete_step(&mut self, index: usize, output: &str) {
        if let Some(step) = self.steps.get_mut(index) {
            step.completed = true;
            step.output = Some(output.to_string());
        }
    }

    /// Check if all steps are complete.
    pub fn is_complete(&self) -> bool {
        self.steps.iter().all(|s| s.completed)
    }

    /// Get completion percentage.
    pub fn progress(&self) -> f32 {
        if self.steps.is_empty() {
            return 100.0;
        }
        let done = self.steps.iter().filter(|s| s.completed).count();
        (done as f32 / self.steps.len() as f32) * 100.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plan_creation() {
        let mut plan = Plan::new("Build a website");
        plan.add_step(PlanStep::new("Design wireframes"));
        plan.add_step(PlanStep::new("Implement frontend").depends_on(0));
        plan.add_step(PlanStep::new("Deploy").depends_on(1));

        assert_eq!(plan.steps.len(), 3);
        assert!(!plan.is_complete());
    }

    #[test]
    fn test_next_steps() {
        let mut plan = Plan::new("Test");
        plan.add_step(PlanStep::new("Step 1"));
        plan.add_step(PlanStep::new("Step 2").depends_on(0));
        plan.add_step(PlanStep::new("Step 3"));

        let next = plan.next_steps();
        assert_eq!(next, vec![0, 2]); // Steps without unmet deps
    }

    #[test]
    fn test_completing_steps() {
        let mut plan = Plan::new("Test");
        plan.add_step(PlanStep::new("Step 1"));
        plan.add_step(PlanStep::new("Step 2").depends_on(0));

        assert_eq!(plan.next_steps(), vec![0]);

        plan.complete_step(0, "Done");
        assert_eq!(plan.next_steps(), vec![1]);

        plan.complete_step(1, "Done");
        assert!(plan.is_complete());
    }

    #[test]
    fn test_progress() {
        let mut plan = Plan::new("Test");
        plan.add_step(PlanStep::new("S1"));
        plan.add_step(PlanStep::new("S2"));
        plan.add_step(PlanStep::new("S3"));
        plan.add_step(PlanStep::new("S4"));

        assert_eq!(plan.progress(), 0.0);

        plan.complete_step(0, "done");
        assert!((plan.progress() - 25.0).abs() < 0.01);

        plan.complete_step(1, "done");
        assert!((plan.progress() - 50.0).abs() < 0.01);
    }
}
