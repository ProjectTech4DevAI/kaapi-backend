Refreshes the status of a fine-tuning job by retrieving the latest information from OpenAI.
If there are any changes in status, fine-tuned model, or error message, the local job record is updated accordingly.
Returns the latest state of the job.

When a job is completed and updated in the database, model evaluation for that fine-tuned model will start automatically.

OpenAI's job status is retrieved using their [Fine-tuning Job Retrieve API](https://platform.openai.com/docs/api-reference/fine_tuning/retrieve).
