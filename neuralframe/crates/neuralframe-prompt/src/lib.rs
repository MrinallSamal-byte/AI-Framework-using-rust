//! # NeuralFrame Prompt Engine
//!
//! Prompt templating with variable injection, conditionals, loops,
//! token counting, truncation, versioning, and A/B testing.
//!
//! ## Examples
//!
//! ```rust
//! use neuralframe_prompt::{PromptTemplate, PromptBuilder};
//!
//! let template = PromptTemplate::new(
//!     "greeting",
//!     "Hello, {{ name }}! You are a {{ role }}."
//! );
//! let mut vars = std::collections::HashMap::new();
//! vars.insert("name".to_string(), "Alice".to_string());
//! vars.insert("role".to_string(), "helpful assistant".to_string());
//! let rendered = template.render(&vars).unwrap();
//! assert_eq!(rendered, "Hello, Alice! You are a helpful assistant.");
//! ```

use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

/// Errors from the prompt engine.
#[derive(Debug)]
pub enum PromptError {
    /// A required variable was not provided.
    MissingVariable(String),
    /// Template syntax error.
    SyntaxError(String),
    /// Token limit exceeded.
    TokenLimitExceeded { count: usize, limit: usize },
    /// Version not found.
    VersionNotFound(String),
}

impl fmt::Display for PromptError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingVariable(name) => write!(f, "missing variable: {}", name),
            Self::SyntaxError(msg) => write!(f, "syntax error: {}", msg),
            Self::TokenLimitExceeded { count, limit } => {
                write!(f, "token count {} exceeds limit {}", count, limit)
            }
            Self::VersionNotFound(v) => write!(f, "version not found: {}", v),
        }
    }
}

impl std::error::Error for PromptError {}

/// A prompt template with Jinja2-like syntax.
///
/// Supports:
/// - `{{ variable }}` - Variable substitution
/// - `{% if condition %}...{% endif %}` - Conditionals
/// - `{% for item in list %}...{% endfor %}` - Loops
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptTemplate {
    /// Template name/identifier.
    pub name: String,
    /// Raw template string.
    pub template: String,
    /// Template version.
    pub version: String,
    /// Description of the template.
    pub description: Option<String>,
}

impl PromptTemplate {
    /// Create a new prompt template.
    pub fn new(name: &str, template: &str) -> Self {
        Self {
            name: name.to_string(),
            template: template.to_string(),
            version: "1.0.0".to_string(),
            description: None,
        }
    }

    /// Set the version.
    pub fn with_version(mut self, version: &str) -> Self {
        self.version = version.to_string();
        self
    }

    /// Set the description.
    pub fn with_description(mut self, desc: &str) -> Self {
        self.description = Some(desc.to_string());
        self
    }

    /// Render the template with the given variables.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use neuralframe_prompt::PromptTemplate;
    /// use std::collections::HashMap;
    ///
    /// let tpl = PromptTemplate::new("test", "Hello, {{ name }}!");
    /// let mut vars = HashMap::new();
    /// vars.insert("name".to_string(), "World".to_string());
    /// assert_eq!(tpl.render(&vars).unwrap(), "Hello, World!");
    /// ```
    pub fn render(&self, variables: &HashMap<String, String>) -> Result<String, PromptError> {
        let mut result = self.template.clone();

        // Process conditionals: {% if varname %}...{% endif %}
        result = self.process_conditionals(&result, variables)?;

        // Process loops: {% for item in listvar %}...{% endfor %}
        result = self.process_loops(&result, variables)?;

        // Process variable substitution: {{ varname }}
        result = self.process_variables(&result, variables)?;

        // Clean up whitespace
        result = result
            .lines()
            .map(|l| l.trim_end())
            .collect::<Vec<_>>()
            .join("\n")
            .trim()
            .to_string();

        Ok(result)
    }

    /// Process `{{ variable }}` substitutions.
    fn process_variables(
        &self,
        input: &str,
        variables: &HashMap<String, String>,
    ) -> Result<String, PromptError> {
        let mut result = input.to_string();
        let mut start = 0;

        while let Some(open) = result[start..].find("{{") {
            let open = start + open;
            if let Some(close) = result[open..].find("}}") {
                let close = open + close;
                let var_name = result[open + 2..close].trim();
                let value = variables.get(var_name).ok_or_else(|| {
                    PromptError::MissingVariable(var_name.to_string())
                })?;
                result = format!("{}{}{}", &result[..open], value, &result[close + 2..]);
                start = open + value.len();
            } else {
                return Err(PromptError::SyntaxError(
                    "unclosed {{ delimiter".to_string(),
                ));
            }
        }

        Ok(result)
    }

    /// Process `{% if varname %}...{% endif %}` conditionals.
    fn process_conditionals(
        &self,
        input: &str,
        variables: &HashMap<String, String>,
    ) -> Result<String, PromptError> {
        let mut result = input.to_string();

        while let Some(if_start) = result.find("{% if ") {
            let if_end = result[if_start..].find("%}").ok_or_else(|| {
                PromptError::SyntaxError("unclosed {% if %} tag".to_string())
            })?;
            let if_end = if_start + if_end + 2;

            let condition = result[if_start + 6..if_end - 2].trim();
            let is_negated = condition.starts_with("not ");
            let var_name = if is_negated {
                condition.strip_prefix("not ").unwrap_or(condition).trim()
            } else {
                condition
            };

            let endif_tag = "{% endif %}";
            let endif_pos = result[if_end..].find(endif_tag).ok_or_else(|| {
                PromptError::SyntaxError("missing {% endif %}".to_string())
            })?;
            let endif_pos = if_end + endif_pos;

            let body = &result[if_end..endif_pos];
            let var_exists = variables
                .get(var_name)
                .map(|v| !v.is_empty())
                .unwrap_or(false);
            let condition_met = if is_negated { !var_exists } else { var_exists };

            let replacement = if condition_met {
                body.to_string()
            } else {
                String::new()
            };

            result = format!(
                "{}{}{}",
                &result[..if_start],
                replacement,
                &result[endif_pos + endif_tag.len()..]
            );
        }

        Ok(result)
    }

    /// Process `{% for item in listvar %}...{% endfor %}` loops.
    fn process_loops(
        &self,
        input: &str,
        variables: &HashMap<String, String>,
    ) -> Result<String, PromptError> {
        let mut result = input.to_string();

        while let Some(for_start) = result.find("{% for ") {
            let for_end = result[for_start..].find("%}").ok_or_else(|| {
                PromptError::SyntaxError("unclosed {% for %} tag".to_string())
            })?;
            let for_end = for_start + for_end + 2;

            let for_expr = result[for_start + 7..for_end - 2].trim();
            let parts: Vec<&str> = for_expr.split(" in ").collect();
            if parts.len() != 2 {
                return Err(PromptError::SyntaxError(
                    "invalid for loop syntax".to_string(),
                ));
            }
            let item_var = parts[0].trim();
            let list_var = parts[1].trim();

            let endfor_tag = "{% endfor %}";
            let endfor_pos = result[for_end..].find(endfor_tag).ok_or_else(|| {
                PromptError::SyntaxError("missing {% endfor %}".to_string())
            })?;
            let endfor_pos = for_end + endfor_pos;

            let body = &result[for_end..endfor_pos];

            // List items are comma-separated in the variable value
            let list_value = variables.get(list_var).cloned().unwrap_or_default();
            let items: Vec<&str> = list_value.split(',').map(|s| s.trim()).collect();

            let mut expanded = String::new();
            for item in items {
                if item.is_empty() {
                    continue;
                }
                let item_rendered = body.replace(&format!("{{{{ {} }}}}", item_var), item);
                expanded.push_str(&item_rendered);
            }

            result = format!(
                "{}{}{}",
                &result[..for_start],
                expanded,
                &result[endfor_pos + endfor_tag.len()..]
            );
        }

        Ok(result)
    }

    /// Extract all variable names referenced in the template.
    pub fn variables(&self) -> Vec<String> {
        let mut vars = Vec::new();
        let mut pos = 0;
        let template = &self.template;

        while let Some(open) = template[pos..].find("{{") {
            let open = pos + open;
            if let Some(close) = template[open..].find("}}") {
                let close = open + close;
                let var_name = template[open + 2..close].trim().to_string();
                if !vars.contains(&var_name) {
                    vars.push(var_name);
                }
                pos = close + 2;
            } else {
                break;
            }
        }

        vars
    }
}

/// Builder for constructing multi-part prompts (system + user + context).
#[derive(Debug, Clone)]
pub struct PromptBuilder {
    system: Option<String>,
    user: Option<String>,
    context: Vec<String>,
    max_tokens: Option<usize>,
}

impl PromptBuilder {
    /// Create a new prompt builder.
    pub fn new() -> Self {
        Self {
            system: None,
            user: None,
            context: Vec::new(),
            max_tokens: None,
        }
    }

    /// Set the system message.
    pub fn system(mut self, msg: &str) -> Self {
        self.system = Some(msg.to_string());
        self
    }

    /// Set the user message.
    pub fn user(mut self, msg: &str) -> Self {
        self.user = Some(msg.to_string());
        self
    }

    /// Add context information.
    pub fn context(mut self, ctx: &str) -> Self {
        self.context.push(ctx.to_string());
        self
    }

    /// Set max token limit for the prompt.
    pub fn max_tokens(mut self, limit: usize) -> Self {
        self.max_tokens = Some(limit);
        self
    }

    /// Build the final prompt parts.
    pub fn build(self) -> BuiltPrompt {
        BuiltPrompt {
            system: self.system,
            user: self.user,
            context: self.context,
        }
    }
}

impl Default for PromptBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// A fully constructed prompt.
#[derive(Debug, Clone)]
pub struct BuiltPrompt {
    /// System message.
    pub system: Option<String>,
    /// User message.
    pub user: Option<String>,
    /// Context pieces.
    pub context: Vec<String>,
}

impl BuiltPrompt {
    /// Get the full prompt as a single string.
    pub fn full_text(&self) -> String {
        let mut parts = Vec::new();
        if let Some(sys) = &self.system {
            parts.push(format!("System: {}", sys));
        }
        for ctx in &self.context {
            parts.push(format!("Context: {}", ctx));
        }
        if let Some(usr) = &self.user {
            parts.push(format!("User: {}", usr));
        }
        parts.join("\n\n")
    }
}

/// Prompt version store with A/B testing support.
pub struct PromptRegistry {
    templates: Arc<DashMap<String, Vec<PromptTemplate>>>,
}

impl PromptRegistry {
    /// Create a new prompt registry.
    pub fn new() -> Self {
        Self {
            templates: Arc::new(DashMap::new()),
        }
    }

    /// Register a prompt template.
    pub fn register(&self, template: PromptTemplate) {
        self.templates
            .entry(template.name.clone())
            .or_default()
            .push(template);
    }

    /// Get the latest version of a template.
    pub fn get_latest(&self, name: &str) -> Option<PromptTemplate> {
        self.templates
            .get(name)
            .and_then(|versions| versions.last().cloned())
    }

    /// Get a specific version of a template.
    pub fn get_version(&self, name: &str, version: &str) -> Option<PromptTemplate> {
        self.templates.get(name).and_then(|versions| {
            versions.iter().find(|t| t.version == version).cloned()
        })
    }

    /// Select a random template for A/B testing.
    pub fn ab_select(&self, name: &str) -> Option<PromptTemplate> {
        self.templates.get(name).and_then(|versions| {
            if versions.is_empty() {
                return None;
            }
            let idx = rand::random::<usize>() % versions.len();
            Some(versions[idx].clone())
        })
    }

    /// List all registered template names.
    pub fn list_templates(&self) -> Vec<String> {
        self.templates
            .iter()
            .map(|entry| entry.key().clone())
            .collect()
    }
}

impl Default for PromptRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Count tokens for a string using the selected model tokenizer when available.
pub fn count_tokens(text: &str, model: &str) -> usize {
    tiktoken_rs::get_bpe_from_model(model)
        .or_else(|_| tiktoken_rs::get_bpe_from_model("gpt-4"))
        .map(|bpe| bpe.encode_with_special_tokens(text).len())
        .unwrap_or_else(|_| (text.len() as f64 / 4.0).ceil() as usize)
}

/// Estimate token count for a string.
pub fn estimate_tokens(text: &str) -> usize {
    count_tokens(text, "gpt-4")
}

/// Truncate text to fit within a token limit.
pub fn truncate_to_tokens(text: &str, max_tokens: usize) -> String {
    truncate_to_tokens_for_model(text, max_tokens, "gpt-4")
}

/// Truncate text to fit within a token limit for a specific model.
pub fn truncate_to_tokens_for_model(text: &str, max_tokens: usize, model: &str) -> String {
    match tiktoken_rs::get_bpe_from_model(model)
        .or_else(|_| tiktoken_rs::get_bpe_from_model("gpt-4"))
    {
        Ok(bpe) => {
            let tokens = bpe.encode_with_special_tokens(text);
            if tokens.len() <= max_tokens {
                return text.to_string();
            }
            let truncated = &tokens[..max_tokens.saturating_sub(1)];
            bpe.decode(truncated.to_vec())
                .unwrap_or_else(|_| text.chars().take(max_tokens * 4).collect())
        }
        Err(_) => {
            let max_chars = max_tokens * 4;
            if text.len() <= max_chars {
                text.to_string()
            } else {
                format!("{}...", &text[..max_chars.saturating_sub(3)])
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_variable_substitution() {
        let tpl = PromptTemplate::new("test", "Hello, {{ name }}!");
        let mut vars = HashMap::new();
        vars.insert("name".to_string(), "World".to_string());
        assert_eq!(tpl.render(&vars).unwrap(), "Hello, World!");
    }

    #[test]
    fn test_multiple_variables() {
        let tpl = PromptTemplate::new(
            "test",
            "{{ greeting }}, {{ name }}! You are a {{ role }}.",
        );
        let mut vars = HashMap::new();
        vars.insert("greeting".to_string(), "Hi".to_string());
        vars.insert("name".to_string(), "Alice".to_string());
        vars.insert("role".to_string(), "developer".to_string());
        assert_eq!(
            tpl.render(&vars).unwrap(),
            "Hi, Alice! You are a developer."
        );
    }

    #[test]
    fn test_missing_variable() {
        let tpl = PromptTemplate::new("test", "Hello, {{ name }}!");
        let vars = HashMap::new();
        assert!(matches!(
            tpl.render(&vars),
            Err(PromptError::MissingVariable(_))
        ));
    }

    #[test]
    fn test_conditional_true() {
        let tpl = PromptTemplate::new(
            "test",
            "Hello{% if role %}, {{ role }}{% endif %}!",
        );
        let mut vars = HashMap::new();
        vars.insert("role".to_string(), "admin".to_string());
        let result = tpl.render(&vars).unwrap();
        assert!(result.contains("admin"));
    }

    #[test]
    fn test_conditional_false() {
        let tpl = PromptTemplate::new(
            "test",
            "Hello{% if role %}, {{ role }}{% endif %}!",
        );
        let vars = HashMap::new();
        let result = tpl.render(&vars).unwrap();
        assert!(!result.contains("role"));
    }

    #[test]
    fn test_for_loop() {
        let tpl = PromptTemplate::new(
            "test",
            "Items: {% for item in items %}[{{ item }}]{% endfor %}",
        );
        let mut vars = HashMap::new();
        vars.insert("items".to_string(), "a,b,c".to_string());
        let result = tpl.render(&vars).unwrap();
        assert!(result.contains("[a]"));
        assert!(result.contains("[b]"));
        assert!(result.contains("[c]"));
    }

    #[test]
    fn test_extract_variables() {
        let tpl = PromptTemplate::new("test", "{{ a }} and {{ b }} and {{ a }}");
        let vars = tpl.variables();
        assert_eq!(vars.len(), 2);
        assert!(vars.contains(&"a".to_string()));
        assert!(vars.contains(&"b".to_string()));
    }

    #[test]
    fn test_prompt_builder() {
        let prompt = PromptBuilder::new()
            .system("You are helpful.")
            .user("Hello!")
            .context("Previous conversation...")
            .build();

        let text = prompt.full_text();
        assert!(text.contains("System: You are helpful."));
        assert!(text.contains("User: Hello!"));
        assert!(text.contains("Context: Previous conversation..."));
    }

    #[test]
    fn test_prompt_registry() {
        let registry = PromptRegistry::new();
        registry.register(
            PromptTemplate::new("greet", "Hello {{ name }}!")
                .with_version("1.0.0"),
        );
        registry.register(
            PromptTemplate::new("greet", "Hi {{ name }}!")
                .with_version("2.0.0"),
        );

        let latest = registry.get_latest("greet").unwrap();
        assert_eq!(latest.version, "2.0.0");

        let v1 = registry.get_version("greet", "1.0.0").unwrap();
        assert_eq!(v1.template, "Hello {{ name }}!");
    }

    #[test]
    fn test_count_tokens_nonempty() {
        assert!(count_tokens("Hello world", "gpt-4") > 0);
    }

    #[test]
    fn test_count_tokens_empty() {
        assert_eq!(count_tokens("", "gpt-4"), 0);
    }

    #[test]
    fn test_estimate_tokens() {
        assert_eq!(estimate_tokens(""), 0);
        assert!(estimate_tokens("Hello, world!") > 0);
    }

    #[test]
    fn test_truncate_preserves_short_text() {
        let text = "one two three four five";
        assert_eq!(truncate_to_tokens(text, 100), text);
    }

    #[test]
    fn test_truncate_reduces_long_text() {
        let text = "x".repeat(1000);
        let truncated = truncate_to_tokens(&text, 10);
        assert!(truncated.len() < text.len());
    }

    #[test]
    fn test_template_with_version() {
        let tpl = PromptTemplate::new("test", "Hello!")
            .with_version("2.0.0")
            .with_description("A greeting template");
        assert_eq!(tpl.version, "2.0.0");
        assert_eq!(tpl.description.unwrap(), "A greeting template");
    }
}
