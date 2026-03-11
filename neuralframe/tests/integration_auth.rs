use neuralframe_auth::*;

#[tokio::test]
async fn test_api_key_valid_identity() {
    let auth = ApiKeyAuth::new();
    auth.add_key(ApiKeyConfig {
        key: "valid-key".into(),
        name: "Test App".into(),
        roles: vec!["user".into(), "admin".into()],
        rate_limit: None,
    });
    let id = auth
        .authenticate("valid-key")
        .await
        .unwrap_or_else(|_| panic!("auth failed"));
    assert_eq!(id.id, "valid-key");
    assert!(id.has_role("admin"));
    assert!(!id.has_role("superadmin"));
}

#[tokio::test]
async fn test_api_key_invalid() {
    let auth = ApiKeyAuth::new();
    assert!(matches!(
        auth.authenticate("nonexistent").await,
        Err(AuthError::InvalidApiKey)
    ));
}

#[tokio::test]
async fn test_api_key_revoke() {
    let auth = ApiKeyAuth::new();
    auth.add_key(ApiKeyConfig {
        key: "temp-key".into(),
        name: "Temp".into(),
        roles: vec![],
        rate_limit: None,
    });
    assert!(auth.authenticate("temp-key").await.is_ok());
    auth.revoke_key("temp-key");
    assert!(auth.authenticate("temp-key").await.is_err());
}

#[tokio::test]
async fn test_jwt_full_roundtrip() {
    let auth = JwtAuth::new("super-secret-key");
    let id = Identity {
        id: "user-42".into(),
        name: Some("Bob".into()),
        email: Some("bob@example.com".into()),
        roles: vec!["editor".into()],
        api_key: None,
        claims: serde_json::json!({}),
    };
    let token = auth
        .sign(&id, 3600)
        .unwrap_or_else(|_| panic!("sign failed"));
    assert!(!token.is_empty());
    let verified = auth
        .authenticate(&token)
        .await
        .unwrap_or_else(|_| panic!("verify failed"));
    assert_eq!(verified.id, "user-42");
    assert_eq!(verified.name, Some("Bob".into()));
    assert!(verified.has_role("editor"));
}

#[tokio::test]
async fn test_jwt_expired_token_rejected() {
    let auth = JwtAuth::new("secret");
    let id = Identity {
        id: "u1".into(),
        name: None,
        email: None,
        roles: vec![],
        api_key: None,
        claims: serde_json::json!({}),
    };
    let token = auth.sign(&id, 0).unwrap_or_else(|_| panic!("sign failed"));
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    let result = auth.authenticate(&token).await;
    assert!(matches!(result, Err(AuthError::InvalidToken(_))));
}

#[tokio::test]
async fn test_jwt_tampered_token_rejected() {
    let auth = JwtAuth::new("secret");
    let id = Identity {
        id: "u1".into(),
        name: None,
        email: None,
        roles: vec![],
        api_key: None,
        claims: serde_json::json!({}),
    };
    let token = auth
        .sign(&id, 3600)
        .unwrap_or_else(|_| panic!("sign failed"));
    let tampered = format!("{}X", &token[..token.len().saturating_sub(1)]);
    assert!(auth.authenticate(&tampered).await.is_err());
}

#[test]
fn test_rate_limiter_per_key_isolation() {
    let limiter = KeyRateLimiter::new(2);
    assert!(limiter.check("key-a").is_ok());
    assert!(limiter.check("key-a").is_ok());
    assert!(limiter.check("key-a").is_err());
    assert!(limiter.check("key-b").is_ok());
    assert!(limiter.check("key-b").is_ok());
    assert!(limiter.check("key-b").is_err());
}
