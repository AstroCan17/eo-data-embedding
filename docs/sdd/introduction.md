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

# Introduction

## Purpose

This Software Design Document (SDD) describes the architectural and detailed design of
`eo-data-embedding` — a research-grade pipeline that produces dense embeddings of Sentinel-1/2
imagery from a **frozen** Clay v1.5 foundation model and exercises them through four downstream
tasks: FAISS similarity retrieval, a few-shot linear/CNN probe, and bitemporal change detection.
It is the design counterpart to the Software Requirements Specification (SRS): every `REQ-*`
allocated to the design resolves here to a component and an interface.

## Objective

The objective is to record the design rationale at a level that lets a reader reconstruct *why*
the software is shaped the way it is — most centrally the one load-bearing decision that the heavy
encoder pass runs **once** and is persisted, so every downstream capability operates cheaply over
the stored embeddings. The SDD captures the static architecture, the per-module detailed design,
and the design decisions (frozen encoder, exact FAISS index, held-out threshold methodology) that
underpin the project's reproducibility, honest-evaluation, and CPU/GPU-portability goals.

## Scope

`eo-data-embedding` is non-flight, non-real-time ground software: a single-developer research /
portfolio project. Reproducibility, honest evaluation, and portability are first-class goals;
operational deployment is out of scope (see SDP §1). The ECSS real-time, scheduling, and formal
review-board provisions are tailored out accordingly (SDP §2–3); where a DRD section does not
apply, this is stated explicitly rather than left blank.

## Content

The remainder of this SDD is organised as follows: the [design overview](overview) presents the
static architecture, behaviour, interfaces context, long-lifetime and budget considerations, and
the adopted design standards; the [software design](design) gives the per-module detailed design
and the key design decisions; and the [traceability](traceability) page maps requirements to
design, code and tests. External interfaces are specified normatively in the ICD (`icd`).
