# Google OAuth Authentication

Authenticate a user via Google Sign-In by verifying the Google ID token.

## Request

- **token** (required): The Google ID token obtained from the frontend Google Sign-In flow.

## Behavior

1. Verifies the Google ID token against Google's public keys and the configured `GOOGLE_CLIENT_ID`.
2. Extracts user information (email, name, picture) from the verified token.
3. Looks up the user by email in the database.
4. If the user exists and was inactive (first login), activates the account.
5. Generates a JWT access token and refresh token, set as **HTTP-only secure cookies**.
6. If the user has exactly one project, it is auto-selected and embedded in the JWT.
7. If the user has multiple projects, `requires_project_selection: true` is returned with the list.

## Response Format

All responses follow the standard `APIResponse` format:
```json
{
  "success": true,
  "data": {
    "access_token": "...",
    "token_type": "bearer",
    "user": { ... },
    "google_profile": { ... },
    "requires_project_selection": false,
    "available_projects": [ ... ]
  }
}
```

## Error Responses

- **400**: Invalid or expired Google token, or email not verified by Google.
- **401**: No account found for the Google email address.
- **500**: `GOOGLE_CLIENT_ID` is not configured.
