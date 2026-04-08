# Verify Invitation

Verify an invitation token from a magic link email and log the user in.

## Query Parameters

- **token** (required): The invitation JWT token from the email link.

## Behavior

1. Validates the invitation token (checks signature, expiry, and type).
2. Looks up the user by the email embedded in the token.
3. If the user exists and is inactive (first login), activates the account.
4. Returns a JWT access token with the org/project from the invitation embedded.
5. Sets `access_token` and `refresh_token` as HTTP-only cookies.

## Frontend Usage

The frontend should have a route like `/invite?token=xxx` that:
1. Reads the `token` query parameter.
2. Calls `GET /api/v1/auth/invite/verify?token=xxx`.
3. On success, the user is logged in with cookies set — redirect to dashboard.
4. On error, show an appropriate message.

## Error Responses

- **400**: Invalid or expired invitation link.
- **404**: User account not found.
