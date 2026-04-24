# InferEdgeForge Roadmap

This roadmap defines the early implementation path for InferEdgeForge as a reproducible build and optimization system for edge inference targets. The emphasis is on reliable artifact production, traceable metadata, and clean handoff into external validation workflows.

## Phase 1: Project Foundation

- [x] Establish the baseline CLI entry point and command layout
- [x] Define the core builder interface for backend-specific implementations
- [x] Create the initial output directory conventions for deployment artifacts
- [x] Add documentation for architecture, CLI behavior, and repository structure
- [x] Set up initial test scaffolding for CLI and builder integration points

## Phase 2: Preset and Metadata System

- [x] Define preset file conventions for backend and target selection
- [x] Implement preset discovery and validation
- [x] Add `list-presets` and `show-preset` CLI commands
- [x] Define the structured JSON metadata schema for build outputs
- [x] Record source model, preset, target, backend, and produced files in metadata

## Phase 3: RKNN Build Pipeline

- [x] Implement the first end-to-end `build` flow for RKNN targets
- [x] Map preset fields into RKNN builder configuration
- [x] Standardize artifact naming for RKNN outputs
- [x] Emit metadata for every RKNN build result
- [x] Add tests for successful builds and expected failure paths

## Phase 4: TensorRT Build Pipeline

- [x] Implement an initial TensorRT builder behind the shared build interface
- [x] Define initial TensorRT preset fields and validation rules
- [x] Standardize artifact naming for TensorRT outputs
- [x] Emit metadata consistent with the RKNN build contract
- [x] Add backend-specific tests without breaking the common CLI surface
- [x] Replace the placeholder TensorRT artifact path with `trtexec`-based engine generation
- [x] Validate TensorRT FP16 and FP32 builds on Jetson hardware
- [x] Record Jetson TensorRT validation results with artifact and benchmark traceability

## Phase 5: Validation Handoff to InferEdgeLab

- [x] Define the artifact handoff contract expected by InferEdgeLab
- [x] Ensure metadata includes the fields required for downstream comparison
- [x] Document the boundary between artifact production and validation analysis
- [x] Add CLI handoff commands for Lab profile input and command generation
- [x] Verify that generated outputs can be consumed without manual restructuring
- [x] Add `run-benchmark` to execute the downstream Lab profile command from metadata
- [x] Persist Forge-side `run_summary.json` output after downstream execution
- [x] Include persisted run summaries in `inspect-build`
- [x] Add experiment-level build listing and compare candidate discovery
- [x] Add compare command preview from persisted structured result paths
- [x] Document accuracy evidence handoff through InferEdgeLab enrich commands
- [x] Validate manifest-based rebuilds on Jetson as functionally reproducible and benchmarkable
- [ ] Expand handoff coverage for additional downstream validation workflows
- [ ] Add task-matched TensorRT accuracy evidence collection guidance and examples
- [ ] Close the loop on accuracy-aware TensorRT compare workflows

## Phase 6: Developer Experience Improvements

- [ ] Improve CLI help text and error messages
- [x] Add dry-run and inspect modes for build planning and review
- [x] Expand test coverage across presets, metadata, builders, handoff, and execution paths
- [x] Add example presets and sample workflows for contributors
- [ ] Improve packaging, release hygiene, and local development setup
- [ ] Expand broader device coverage beyond the current validated paths

## Notes

- `build` remains the primary contract of the project.
- Metadata quality is a core engineering concern, not an optional add-on.
- InferEdgeForge builds and traces artifacts, including fingerprints, preset snapshots, manifests, and persisted run summaries.
- InferEdgeLab profiles, compares, and can attach accuracy evidence through downstream enrich commands.
- The current Jetson validation proves latency-oriented FP16/FP32 handoff and compare flow, not a full accuracy-aware deployment judgement.
- Jetson rebuild validation currently supports functional reproducibility of the TensorRT recipe, not guaranteed bitwise identity of serialized engine hashes.
