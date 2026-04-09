# Verify Magic Link

Verify a magic link login token and log the user in.

## Query Parameters

- **token** (required): The login JWT token from the email link.

## Behavior

1. Validates the magic link token (checks signature, expiry, and type).
2. Looks up the user by the email embedded in the token.
3. Verifies the user is active.
4. If the user has exactly one project, it is auto-selected and embedded in the JWT.
5. Returns a JWT access token and sets HTTP-only cookies.

## Error Responses

- **400**: Invalid or expired login link.
- **403**: User account is inactive.
- **404**: User account not found.
