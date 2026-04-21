# InferEdgeForge Roadmap

This roadmap defines the early implementation path for InferEdgeForge as a reproducible build and optimization system for edge inference targets. The emphasis is on reliable artifact production, traceable metadata, and clean handoff into external validation workflows.

## Phase 1: Project Foundation

- [ ] Establish the baseline CLI entry point and command layout
- [ ] Define the core builder interface for backend-specific implementations
- [ ] Create the initial output directory conventions for deployment artifacts
- [ ] Add documentation for architecture, CLI behavior, and repository structure
- [ ] Set up initial test scaffolding for CLI and builder integration points

## Phase 2: Preset and Metadata System

- [ ] Define preset file conventions for backend and target selection
- [ ] Implement preset discovery and validation
- [ ] Add `list-presets` and `show-preset` CLI commands
- [ ] Define the structured JSON metadata schema for build outputs
- [ ] Record source model, preset, target, backend, and produced files in metadata

## Phase 3: RKNN Build Pipeline

- [ ] Implement the first end-to-end `build` flow for RKNN targets
- [ ] Map preset fields into RKNN builder configuration
- [ ] Standardize artifact naming for RKNN outputs
- [ ] Emit metadata for every RKNN build result
- [ ] Add tests for successful builds and expected failure paths

## Phase 4: TensorRT Build Pipeline

- [ ] Implement the TensorRT builder behind the shared build interface
- [ ] Define TensorRT-specific preset fields and validation rules
- [ ] Standardize artifact naming for TensorRT outputs
- [ ] Emit metadata consistent with the RKNN build contract
- [ ] Add backend-specific tests without breaking the common CLI surface

## Phase 5: Validation Handoff to InferEdgeLab

- [ ] Define the artifact handoff contract expected by InferEdgeLab
- [ ] Ensure metadata includes the fields required for downstream comparison
- [ ] Document the boundary between artifact production and validation analysis
- [ ] Add export or packaging conventions for handing build outputs to InferEdgeLab
- [ ] Verify that generated outputs can be consumed without manual restructuring

## Phase 6: Developer Experience Improvements

- [ ] Improve CLI help text and error messages
- [ ] Add dry-run or inspect modes for build planning
- [ ] Expand test coverage across presets, metadata, and builders
- [ ] Add example presets and sample workflows for contributors
- [ ] Improve packaging, release hygiene, and local development setup

## Notes

- `build` is the first implementation priority because it defines the primary contract of the project.
- Metadata quality is a core engineering concern, not an optional add-on.
- Validation and deployment decision analysis remain the responsibility of InferEdgeLab, not InferEdgeForge.
