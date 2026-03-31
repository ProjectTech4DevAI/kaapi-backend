# Google OAuth Authentication

Authenticate a user via Google Sign-In by verifying the Google ID token.

## Request

- **token** (required): The Google ID token obtained from the frontend Google Sign-In flow.

## Behavior

1. Verifies the Google ID token against Google's public keys and the configured `GOOGLE_CLIENT_ID`.
2. Extracts user information (email, name, picture) from the verified token.
3. Looks up the user by email in the database.
4. If the user exists and is active, generates a JWT access token.
5. Sets the access token as an **HTTP-only secure cookie** (`access_token`) in the response.
6. Returns the access token, user details, and Google profile information.

## Error Responses

- **400**: Invalid or expired Google token, or email not verified by Google.
- **401**: No account found for the Google email address.
- **403**: User account is inactive.
