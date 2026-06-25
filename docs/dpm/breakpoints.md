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

# Breakpoints

Not applicable. `eo-data-embedding` runs a batch-then-serve embedding pipeline
with no operator-driven intermediate checkpoints: the encode pass writes its
output once to the Parquet store, and the downstream consumers (search, probe,
change) read that store directly. There are therefore no internal processing
breakpoints — no intermediate products are exposed for operator inspection or
resumption between processing steps. The end-to-end flow is documented in the
[DPM Introduction](./introduction). See SDP §3.
