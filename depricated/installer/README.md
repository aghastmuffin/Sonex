# Sonex Installer Pipeline

This folder contains cross-platform packaging scaffolding for Sonex.

## Goals

- Build native installers for macOS, Windows, and Linux
- Keep one shared dependency definition (`installer/environment.yml`)
- Avoid user-global Python dependency drift

## Files

- `installer/environment.yml`: shared Conda environment definition
- `installer/constructor/construct.yaml`: Constructor config used to build native installers
- `.github/workflows/build-installers.yml`: CI workflow that builds per-platform artifacts

## Local Build

1. Install packaging tools:
   - `conda install -c conda-forge constructor conda-lock`
2. (Optional) generate lock files for reproducibility:
   - `conda-lock lock -f installer/environment.yml -p osx-64 -p osx-arm64 -p linux-64 -p win-64`
3. Build an installer for your current platform:
   - `constructor installer/constructor --output-dir dist`
4. Build for a specific target platform:
   - `constructor installer/constructor --platform osx-arm64 --output-dir dist`

## Runtime Notes

- Sonex worker outputs are now written to a user-writable app data folder by default:
  - macOS: `~/Library/Application Support/Sonex/outputs`
  - Windows: `%APPDATA%/Sonex/outputs`
  - Linux: `$XDG_DATA_HOME/Sonex/outputs` (fallback `~/.local/share/Sonex/outputs`)

- The worker can be forced to use a custom output root by passing an explicit path as its final CLI argument.
