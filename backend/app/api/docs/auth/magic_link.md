# Request Magic Link Login

Send a magic link login email to the user's email address.

## Request Body

- **email** (required): The user's email address.

## Behavior

1. Checks if the user exists — returns 404 if not.
2. Checks if the user is active — returns 403 if inactive.
3. Generates a short-lived login token (15 minutes).
4. Sends an email with a "Sign In Now" button linking to the frontend.

## Error Responses

- **403**: User account is inactive.
- **404**: No account found for this email.
- **500**: Email service is not configured or failed to send.
