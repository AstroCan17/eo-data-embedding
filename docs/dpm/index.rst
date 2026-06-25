.. Copyright 2026 Can Deniz Kaya

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

Detailed processing model
=========================

This section describes the Detailed processing model (DPM) for the
``eo-data-embedding`` pipeline: how raw Sentinel imagery is turned into
reusable vector embeddings and how those embeddings feed the downstream
search, probe and change-detection capabilities. The processing chain is
documented end-to-end in the :doc:`introduction` and :doc:`context` pages;
the key processing parameters and data items are tabulated in
:doc:`parameters-data-list`.

This software is an ML embedding library rather than a per-instrument
data processor, so the classical DPM notions of named algorithm modules
and processing breakpoints do not apply. The corresponding template slots
(``module-a``, ``module-b``, ``breakpoints``) record that rationale instead;
see SDP §3 for the tailoring decision.


.. toctree::
   :maxdepth: 1
   :caption: Contents:

   introduction
   conventions
   context
   module-a
   module-b
   breakpoints
   parameters-data-list
