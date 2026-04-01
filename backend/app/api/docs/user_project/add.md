Add one or more users to the current project by email.

Pass a single email or multiple emails in the `emails` array:
- **Single user**: `{"emails": ["user@gmail.com"]}`
- **Multiple users**: `{"emails": ["a@gmail.com", "b@gmail.com"]}`

For each email:
- If the user does not exist, a new account is created with `is_active: false`. The user will be activated on their first Google login.
- If the user already exists and is already in this project, they are skipped.
- If the user exists but is not in this project, they are added.
