<!--
  Copyright 2026 Can Deniz Kaya

  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
-->

# Conventions

This SUM uses the following notation and syntax conventions:

- **Monospace** (`code font`) denotes literal user input, command names, file paths, configuration
  keys and environment variables — e.g. `eo-data-embedding extract`, `artifacts/embeddings.parquet`,
  `GEO_LOG_LEVEL`.
- In command syntax, `<placeholder>` marks a value the user supplies, and `[args]` marks optional
  arguments. Flags use the `--flag value` form; a vertical bar (`a | b`) lists mutually exclusive
  choices.
- Phase numbers (Phase 0–5) refer to the pipeline stages implemented by the `scripts/phaseN_*.py`
  scripts and exposed as CLI subcommands.
- Cross-references to other manual pages use linked page titles (e.g.
  [Reference manual](reference-manual)).
