//! # Retry and Failover Engine
//!
//! Exponential backoff with jitter, automatic provider failover,
//! and per-provider rate limiting.

use crate::error::LLMError;
use crate::providers::LLMProvider;
use crate::types::{CompletionRequest, CompletionResponse, Token};
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;
use tokio_stream::Stream;

/// Configuration for retry behavior.
#[derive(Debug, Clone)]
pub struct RetryConfig {
    /// Maximum number of retry attempts.
    pub max_retries: u32,
    /// Initial delay between retries.
    pub initial_delay: Duration,
    /// Maximum delay between retries.
    pub max_delay: Duration,
    /// Backoff multiplier.
    pub backoff_multiplier: f64,
    /// Whether to add jitter to delays.
    pub jitter: bool,
    /// Request timeout.
    pub timeout: Duration,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            initial_delay: Duration::from_millis(500),
            max_delay: Duration::from_secs(30),
            backoff_multiplier: 2.0,
            jitter: true,
            timeout: Duration::from_secs(120),
        }
    }
}

impl RetryConfig {
    /// Create a new retry config with the given max retries.
    pub fn new(max_retries: u32) -> Self {
        Self {
            max_retries,
            ..Default::default()
        }
    }

    /// Set no retries.
    pub fn no_retry() -> Self {
        Self {
            max_retries: 0,
            ..Default::default()
        }
    }

    /// Calculate the delay for a given attempt (0-based).
    pub fn delay_for_attempt(&self, attempt: u32) -> Duration {
        let base = self.initial_delay.as_millis() as f64
            * self.backoff_multiplier.powi(attempt as i32);
        let capped = base.min(self.max_delay.as_millis() as f64);

        if self.jitter {
            let jitter_range = capped * 0.1;
            let jitter = rand::random::<f64>() * jitter_range * 2.0 - jitter_range;
            Duration::from_millis((capped + jitter).max(0.0) as u64)
        } else {
            Duration::from_millis(capped as u64)
        }
    }
}

/// An LLM client with retry logic and optional provider failover.
///
/// Wraps one or more `LLMProvider` instances and automatically
/// retries or falls over on transient failures.
///
/// # Examples
///
/// ```rust,no_run
/// use neuralframe_llm::retry::{RetryConfig, ResilientClient};
/// use neuralframe_llm::providers::openai::OpenAIProvider;
///
/// let client = ResilientClient::new(OpenAIProvider::new("key"))
///     .with_retry(RetryConfig::new(3));
/// ```
pub struct ResilientClient {
    /// Primary provider.
    primary: Arc<dyn LLMProvider>,
    /// Fallback providers (tried in order if primary fails).
    fallbacks: Vec<Arc<dyn LLMProvider>>,
    /// Retry configuration.
    retry_config: RetryConfig,
}

impl ResilientClient {
    /// Create a new resilient client with a primary provider.
    pub fn new<P: LLMProvider + 'static>(provider: P) -> Self {
        Self {
            primary: Arc::new(provider),
            fallbacks: Vec::new(),
            retry_config: RetryConfig::default(),
        }
    }

    /// Set the retry configuration.
    pub fn with_retry(mut self, config: RetryConfig) -> Self {
        self.retry_config = config;
        self
    }

    /// Add a fallback provider.
    pub fn with_fallback<P: LLMProvider + 'static>(mut self, provider: P) -> Self {
        self.fallbacks.push(Arc::new(provider));
        self
    }

    /// Complete a request with retry and failover logic.
    pub async fn complete(&self, req: CompletionRequest) -> Result<CompletionResponse, LLMError> {
        // Try primary provider with retries
        match self
            .try_with_retries(&*self.primary, req.clone())
            .await
        {
            Ok(response) => return Ok(response),
            Err(primary_err) => {
                if self.fallbacks.is_empty() {
                    return Err(primary_err);
                }

                tracing::warn!(
                    provider = self.primary.name(),
                    error = %primary_err,
                    "primary provider failed, trying fallbacks"
                );

                let mut errors = vec![primary_err];

                // Try each fallback
                for fallback in &self.fallbacks {
                    match self
                        .try_with_retries(&**fallback, req.clone())
                        .await
                    {
                        Ok(response) => {
                            tracing::info!(
                                provider = fallback.name(),
                                "fallback provider succeeded"
                            );
                            return Ok(response);
                        }
                        Err(err) => {
                            tracing::warn!(
                                provider = fallback.name(),
                                error = %err,
                                "fallback provider failed"
                            );
                            errors.push(err);
                        }
                    }
                }

                Err(LLMError::AllProvidersFailed(errors))
            }
        }
    }

    /// Stream tokens with the primary provider (no failover for streams).
    pub async fn stream(
        &self,
        req: CompletionRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<Token, LLMError>> + Send>>, LLMError> {
        self.primary.stream(req).await
    }

    /// Try a request with retries against a single provider.
    async fn try_with_retries(
        &self,
        provider: &dyn LLMProvider,
        req: CompletionRequest,
    ) -> Result<CompletionResponse, LLMError> {
        let mut last_error = None;

        for attempt in 0..=self.retry_config.max_retries {
            if attempt > 0 {
                let delay = self.retry_config.delay_for_attempt(attempt - 1);
                tracing::debug!(
                    provider = provider.name(),
                    attempt,
                    delay_ms = delay.as_millis() as u64,
                    "retrying after delay"
                );
                tokio::time::sleep(delay).await;
            }

            match provider.complete(req.clone()).await {
                Ok(response) => return Ok(response),
                Err(err) => {
                    if !Self::is_retryable(&err) {
                        return Err(err);
                    }
                    tracing::warn!(
                        provider = provider.name(),
                        attempt,
                        error = %err,
                        "retryable error"
                    );
                    last_error = Some(err);
                }
            }
        }

        Err(last_error.unwrap_or(LLMError::ProviderUnavailable(
            provider.name().to_string(),
        )))
    }

    /// Check if an error is retryable.
    fn is_retryable(err: &LLMError) -> bool {
        matches!(
            err,
            LLMError::RequestFailed { status, .. }
                if *status >= 500 || *status == 429
        ) || matches!(err, LLMError::RateLimited { .. })
            || matches!(err, LLMError::StreamError(_))
            || matches!(err, LLMError::Timeout { .. })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_retry_config_default() {
        let config = RetryConfig::default();
        assert_eq!(config.max_retries, 3);
        assert!(config.jitter);
    }

    #[test]
    fn test_retry_config_no_retry() {
        let config = RetryConfig::no_retry();
        assert_eq!(config.max_retries, 0);
    }

    #[test]
    fn test_delay_calculation() {
        let config = RetryConfig {
            initial_delay: Duration::from_secs(1),
            backoff_multiplier: 2.0,
            max_delay: Duration::from_secs(60),
            jitter: false,
            ..Default::default()
        };

        assert_eq!(config.delay_for_attempt(0), Duration::from_secs(1));
        assert_eq!(config.delay_for_attempt(1), Duration::from_secs(2));
        assert_eq!(config.delay_for_attempt(2), Duration::from_secs(4));
    }

    #[test]
    fn test_delay_capped_at_max() {
        let config = RetryConfig {
            initial_delay: Duration::from_secs(1),
            backoff_multiplier: 10.0,
            max_delay: Duration::from_secs(5),
            jitter: false,
            ..Default::default()
        };

        assert_eq!(config.delay_for_attempt(5), Duration::from_secs(5));
    }

    #[test]
    fn test_is_retryable() {
        assert!(ResilientClient::is_retryable(&LLMError::RequestFailed {
            status: 500,
            message: "".into()
        }));
        assert!(ResilientClient::is_retryable(&LLMError::RequestFailed {
            status: 429,
            message: "".into()
        }));
        assert!(ResilientClient::is_retryable(&LLMError::RateLimited {
            retry_after: None
        }));
        assert!(!ResilientClient::is_retryable(&LLMError::AuthError(
            "bad key".into()
        )));
        assert!(!ResilientClient::is_retryable(&LLMError::RequestFailed {
            status: 400,
            message: "".into()
        }));
    }
}
