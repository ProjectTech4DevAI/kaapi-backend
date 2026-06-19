Save post-processing config for a single assessment run.

Stores the config inside the run's `input` JSON blob (key
`post_processing_config`). It is applied at export/preview time and never
re-runs the LLM, so it can be edited after the run completes.

The config has three optional sections:

- `computed_columns`: derived columns from formulas, e.g.
  `{"name": "Total_Score", "formula": "@Novelty_score + @Usefulness_score"}`.
  Formulas reference columns with `@` and support `+ - * /` and parentheses.
- `filter`: row filters combined with AND logic.
- `sort`: sort rules applied in priority order.

Pass `null` (or an empty body) to clear post-processing for the run.
