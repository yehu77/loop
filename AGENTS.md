# Repository Guidelines

## Project Structure & Module Organization

This workspace currently contains no application source, tests, assets, or project documentation. The existing hidden directories (`.agents/`, `.aws/`, `.codex/`, and `.git/`) are environment metadata; do not use them for application code or commit their contents.

When introducing the initial implementation, group source code, tests, and assets into clearly named directories appropriate to the chosen stack. Add a root `README.md` explaining the layout, project purpose, and local setup.

## Build, Test, and Development Commands

No build system, dependency manifest, or development commands are currently configured. Do not assume commands such as `npm test` or `make build` are available.

When adding tooling, document the exact dependency installation, local development, build, and test commands in `README.md`. Prefer reproducible commands backed by checked-in configuration and applicable lockfiles.

## Coding Style & Naming Conventions

No language, indentation standard, formatter, or linter has been established. Follow the conventions of the selected language and keep formatting consistent within each file. Use descriptive names for modules, functions, and variables.

Introduce formatter and linter configuration alongside the first implementation, and document how to run both.

## Testing Guidelines

No test framework or coverage threshold is configured. Add tests for new behavior and regression tests for bug fixes once executable code exists. Use descriptive test names that identify the behavior and expected outcome. Document test locations and execution commands when selecting a framework.

## Commit & Pull Request Guidelines

Git history is unavailable in this workspace, so no existing commit convention can be verified. Use concise, imperative commit subjects, such as `Add initial project setup`, and keep changes focused.

Pull requests should explain the change, link relevant issues, and report validation performed. Include screenshots for visible interface changes and note any setup or configuration changes.
