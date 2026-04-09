# Request Magic Link Login

Send a magic link login email to the user's email address.

## Request Body

- **email** (required): The user's email address.

## Behavior

1. Checks if the user exists and is active.
2. Generates a short-lived login token (15 minutes).
3. Sends an email with a "Sign In Now" button linking to the frontend.
4. Returns the same success message regardless of whether the user exists (prevents email enumeration).

## Response

Always returns:
```json
{
  "success": true,
  "data": {
    "message": "If an account exists, a login link has been sent."
  }
}
```

## Error Responses

- **500**: Email service is not configured or failed to send.
