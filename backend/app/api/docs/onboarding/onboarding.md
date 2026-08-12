# Onboarding API Behavior

## 🏢 Organization Handling
- If `organization_name` does **not exist**, a new organization will be created.
- If `organization_name` already exists, the request will proceed to create the project under that organization.

---

## 📂 Project Handling
- If `project_name` does **not exist** in the organization, it will be created.
- If the project already exists in the same organization, the API will return **409 Conflict**.

---

## 👤 User Handling
**Fields:**
- `email` (required): User's email address
- `password` (optional): Password for the primary user (must be at least 8 characters)
- `username` (optional): Full name of the primary user

**Behavior:**
- If `email` does not exist, a new user is created and linked to the project.
- If the user already exists, they are simply attached to the project.
---

## 🔑 Credentials (Optional)
- If provided, the given credentials will be **encrypted** and stored as project credentials.
- The `credentials` parameter accepts a list of one or more credentials (e.g., an OpenAI key, Langfuse credentials, etc.).
- If omitted, the project will be created **without credentials**.
- We’ve also included a list of the providers currently supported by kaapi.

   ### Supported Providers
   - **LLM:** openai, google (v1 only, deprecated), google-aistudio, google-gcp (v2 only), anthropic, sarvamai
   - **Observability:** langfuse
   - **Audio:** elevenlabs

   `google-gcp` (Gemini on Vertex AI) requires all of: `api_key`, `project_id`, `location`,
   `sa_key` (the service-account key JSON as an object), `gcs_bucket`.

   ### Example: For sending multiple credentials -
   ```
   "credentials": [
     {
       "openai": {
         "api_key": "sk-proj-..."
       }
     },
     {
       "google": {
         "api_key": "AIzaSy..."
       }
     },
     {
       "sarvamai": {
         "api_key": "sarvam-..."
       }
     },
     {
       "elevenlabs": {
         "api_key": "sk_..."
       }
     },
     {
       "google-gcp": {
         "api_key": "AQ.Ab8...",
         "project_id": "my-gcp-project",
         "location": "us-central1",
         "sa_key": {
           "type": "service_account",
           "project_id": "my-gcp-project",
           "private_key": "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n",
           "client_email": "svc@my-gcp-project.iam.gserviceaccount.com"
         },
         "gcs_bucket": "my-audio-staging-bucket"
       }
     },
     {
       "langfuse": {
         "public_key": "pk-lf-....",
         "secret_key": "sk-lf-...",
         "host": "https://cloud.langfuse.com"
       }
     }
   ]
   ```
---

## 🆕 v2 (`/api/v2/onboard`)
This v1 route is **deprecated** — prefer `/api/v2/onboard` (see its own docs). v2 rejects vanilla `google` (use `google-aistudio`/`google-gcp`) and caps the body at 32 KB; conversely `google-gcp` is rejected here on v1.

---

## 🔄 Transactional Guarantee
The onboarding process is **all-or-nothing**:
- If any step fails (e.g., invalid password), **no organization, project, or user will be persisted**.
