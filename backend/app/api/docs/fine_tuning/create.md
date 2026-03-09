Initiate fine-tuning of an OpenAI model using a CSV dataset. The CSV must include:

- A column named `query`, `question`, or `message` containing user inputs or messages.
- A column named `label` indicating whether a given message is a genuine query or not (e.g., casual conversation or small talk).

The `split_ratio` form field is a comma-separated string that determines how your data is divided between training and testing. For example, `"0.5"` means 50% training and 50% testing. You can provide multiple ratios—for instance, `"0.7,0.9"`. This will trigger multiple fine-tuning jobs, one for each ratio. You would also need to specify a `base_model` that you want to fine-tune.

The `system_prompt` form field allows you to define an initial instruction or context-setting message that will be included in the training data. This message helps the model learn how it is expected to behave when responding to user inputs. It is prepended as the first message in each training example during fine-tuning.

The system handles the fine-tuning process by interacting with OpenAI's APIs under the hood. These include:

- [Openai File create to upload your training and testing files](https://platform.openai.com/docs/api-reference/files/create)

- [Openai Fine Tuning Job create to initiate each fine-tuning job](https://platform.openai.com/docs/api-reference/fine_tuning/create)

If successful, the response will include a message along with a list of fine-tuning jobs that were initiated. Each job object includes:

```json
{
    "id": "the internal ID of the fine-tuning job",
    "document_id": "the ID of the document used for fine-tuning",
    "split_ratio": "the data split used for that job",
    "status": "the initial status of the job (usually 'pending')"
}
```
