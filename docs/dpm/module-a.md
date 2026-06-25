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

# Module A

Not applicable. `eo-data-embedding` is an ML embedding library, not a
per-instrument data processor; the classical DPM notion of a named, per-instrument
processing module (with theoretical description, algorithm inputs/outputs and
instrument-specific equations) does not apply. The actual processing chain
(tile → embed → store → fan-out) is documented in the
[DPM Introduction](./introduction) and [Context overview](./context), with the
underlying algorithms specified in the
[Software Design Document](../sdd/index). See SDP §3.
