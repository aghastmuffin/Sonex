# Proposed Change: Make Sonex Run Flawlessly on Windows

Proposed by GPT 3.5 High Thinking
Proofread by Levi Taeson Kim Brown, 4/10/26 
Implemented on/at: 4/10/26
## Goal
Ensure Sonex installs, launches, and runs end-to-end on Windows 10/11 with no manual patching, while preserving current behavior on Darwin-like("macOS") or unix-like systems.

## Definition of "Flawless" for Windows
- Fresh install succeeds on a standard Windows machine.
- All core pipelines complete without OS-specific crashes.
- GUI launches correctly and can process real project inputs.
- Required third-party tools are auto-detected or clearly installed by guided setup.
- Logs and error messages are actionable for non-developer users.

## What Must Change

### 1. Platform-Aware Path and Process Handling
Current codebase likely assumes POSIX behavior in several places. To run cleanly on Windows, the program needs:
- Full migration to pathlib-based path handling in all runtime-critical modules.
- No hardcoded "/" separators, no shell-only path assumptions.
- Subprocess usage rewritten to avoid shell-specific quoting edge cases.
- Safe handling of paths with spaces (common on Windows user profiles).

### 2. Dependency and Toolchain Compatibility
Windows success depends on controlling external tools and Python dependencies:
- Audit every dependency for Windows wheel/support status.
- Replace or pin packages that are unstable on Windows.
- Identify external binaries used by workflows (for example, ffmpeg, MFA, demucs, alignment tools).
- Create a Windows-first dependency matrix with supported versions.

### 3. Installer and Environment Bootstrap
The installer must become deterministic for Windows users:
- Add a dedicated Windows setup path in installer docs/scripts.
- Detect Python version and venv status before install.
- Validate required system tools and provide install hints if missing.
- Add post-install verification command that checks all mandatory components.

### 4. Runtime OS Abstraction Layer
A small runtime compatibility layer should centralize OS behavior:
- OS detection and capability checks in one module.
- Central helper functions for temp paths, executable lookup, and command construction.
- Prevent scattered platform-specific logic across modules.

### 5. Encoding, Locale, and File I/O Safety
Windows systems often expose encoding and newline edge cases:
- Force UTF-8-safe file reads/writes for lyrics, JSON, and logs.
- Normalize newline handling for cross-platform consistency.
- Guard against Unicode path and filename issues.

### 6. GUI and Worker Process Stability on Windows
UI/worker behavior can differ on Windows multiprocessing rules:
- Ensure worker start method and process spawning are Windows-safe.
- Validate that background tasks and subprocesses do not deadlock.
- Confirm asset path resolution works from packaged and source runs.

### 7. Packaging and Distribution for Windows
Users need a supported delivery path:
- Decide supported distribution strategy: source-only, frozen app, or both.
- If frozen app is required, add Windows packaging config and smoke tests.
- Bundle or document required runtime redistributables.

### 8. CI and QA Gates for Windows
Windows support is not complete without automated verification:
- Add Windows CI jobs for install, import checks, and smoke pipeline tests.
- Add integration test fixtures from deptest for representative workflows.
- Define pass/fail release gate that includes Windows-specific checks.

## Delivery Plan

### Phase 1: Discovery and Gap List
- Inventory OS-sensitive code paths.
- Inventory all Python and binary dependencies.
- Produce a blocker list with severity and owner.

### Phase 2: Core Compatibility Refactor
- Implement path/process abstraction and replace fragile call sites.
- Resolve dependency incompatibilities and update lock files.
- Add robust error handling around external command execution.

### Phase 3: Installer + Docs
- Ship Windows setup script/documentation.
- Add verification script to confirm environment readiness.
- Publish troubleshooting section for common Windows failures.

### Phase 4: Test, Package, and Release Gate
- Enable Windows CI matrix.
- Run full workflow regression tests.
- Release only when all Windows acceptance criteria are green.

## Acceptance Criteria
- Clean install on Windows from documented steps, no manual fixes.
- At least one full end-to-end analysis workflow passes on test assets.
- GUI workflow completes without crash on Windows 10 and 11.
- CI includes required Windows checks and blocks regressions.
- Documentation includes exact setup, verification, and troubleshooting.

## Risks and Mitigations
- Risk: third-party tool incompatibility on Windows.
	Mitigation: pin known-good versions and add explicit preflight checks.
- Risk: hidden POSIX assumptions in less-used paths.
	Mitigation: add targeted integration tests across major workflows.
- Risk: packaging complexity for binary-heavy stack.
	Mitigation: start with source-based support, then add frozen distribution as a controlled phase.

## Recommended Outcome
Proceed with a formal Windows compatibility project rather than ad hoc fixes. This will prevent recurring breakages and make Windows support a first-class, test-enforced capability.
