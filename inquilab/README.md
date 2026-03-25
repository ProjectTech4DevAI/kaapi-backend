
## Setup

```bash
cd inquilab
pip install -r requirements.txt
```

Create a `.env` file in this folder:

```
OPENAI_API_KEY=your-openai-key
GEMINI_API_KEY=your-gemini-key
```

## Scripts

### Score + Feedback

Generates scores across 5 dimensions and detailed mentor feedback for each submission.

```bash
python run_evaluation_score_feedback.py --provider openai --model gpt-4o-mini --both
```

Output folder: `output_score_feedback/`

### Score Only

Generates scores across 5 dimensions without feedback.

```bash
python run_evaluation_score.py --provider openai --model gpt-4o-mini --both
```

Output folder: `output_score/`

## CLI Usage

```bash
# default: runs both without and with attachments using OpenAI gpt-4o-mini
python run_evaluation_score_feedback.py

# use Google Gemini instead
python run_evaluation_score.py --provider google --model gemini-2.5-flash

# run with attachments only
python run_evaluation_score_feedback.py --provider openai --model gpt-4o-mini --attachment

# run without attachments only
python run_evaluation_score.py --wo-attachment

# run both modes (without attachments first, then with attachments)
python run_evaluation_score_feedback.py --both

# custom input file
python run_evaluation_score.py --input my_data.xlsx

# custom output file name
python run_evaluation_score_feedback.py --output my_results.xlsx

# custom temperature
python run_evaluation_score.py --temperature 0.2

# full example with all options
python run_evaluation_score_feedback.py --provider google --model gemini-2.5-flash --input data.xlsx --output results.xlsx --temperature 0.3 --attachment
```
