Upload a CSV file containing golden Q&A pairs for evaluation.

Datasets allow you to store reusable question-answer pairs for systematic LLM testing with automatic validation, duplication for statistical significance, and Langfuse integration. Response includes dataset ID, sanitized name, item counts, Langfuse dataset ID, and object store URL (the cloud storage location where your CSV file is stored).

**Key Features:**
* Validates CSV format and required columns (question, answer)
* Automatic dataset name sanitization for Langfuse compatibility
* Optional item duplication for statistical significance (1-5x, default: 1x)
* Uploads to object store and syncs with Langfuse
* Skips rows with missing values automatically


**CSV Format Requirements:**
* Required columns: `question`, `answer`
* Additional columns are allowed (will be ignored)
* Missing values in required columns are automatically skipped


**Dataset Name Sanitization:**

Your dataset name will be automatically sanitized for Langfuse compatibility:
* Spaces → underscores
* Special characters removed
* Converted to lowercase
* Example: `"My Dataset 01!"` → `"my_dataset_01"`


**Duplication Factor:**

Control how many times each Q&A pair is duplicated (1-5x, default: 1x):
* Higher duplication = better statistical significance
* Useful for batch evaluation reliability
* `1` = no duplication (original dataset only)
