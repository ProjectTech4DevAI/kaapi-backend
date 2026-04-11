# Request Magic Link Login

Send a magic link login email to the user's email address.

## Request Body

- **email** (required): The user's email address.

## Behavior

1. Checks if the user exists — returns 404 if not.
2. Generates a short-lived login token (15 minutes).
3. Sends an email with a "Sign In Now" button linking to the frontend.

## Error Responses

- **404**: No account found for this email.
- **500**: Email service is not configured or failed to send.
