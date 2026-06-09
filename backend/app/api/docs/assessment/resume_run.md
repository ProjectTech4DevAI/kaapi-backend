Resume a failed assessment run from its failed stage.

Re-runs the same child run in place, starting at the stage that failed.
Stages that already completed are reused (their batch results are not
recomputed). Only valid when the run is in a failed state.
