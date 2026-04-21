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

## Error Responses

- **400**: Invalid or expired invitation link.
- **404**: User account not found.
