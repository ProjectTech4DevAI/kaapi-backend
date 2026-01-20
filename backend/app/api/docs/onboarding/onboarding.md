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
- If `email` does **not exist**, a new user is created and linked to the project.
- If the user already exists, they are simply attached to the project.

---

## 🔑 Credentials (Optional)
- If provided, the given credentials will be **encrypted** and stored as project credentials.
- The `credentials` parameter accepts a list of one or more credentials (e.g., an OpenAI key, Langfuse credentials, etc.).
- If omitted, the project will be created **without credentials**.
- We’ve also included a list of the providers currently supported by kaapi.

   ### Supported Providers
   - **LLM:** openai
   - **Observability:** langfuse

   ### Example: For sending multiple credentials -
   ```
   "credentials": [
     {
       "openai": {
         "api_key": "sk-proj-..."
       }
     },
     {
       "langfuse": {
         "public_key": "pk-lf-...",
         "secret_key": "sk-lf-...",
         "host": "https://cloud.langfuse.com"
       }
     }
   ]
   ```
---

## 🔄 Transactional Guarantee
The onboarding process is **all-or-nothing**:
- If any step fails (e.g., invalid password), **no organization, project, or user will be persisted**.
