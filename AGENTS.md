# Working in this experiment

- README.md describes the public project. Local implementation notes may exist in docs/development_spec.md and are excluded from Git. Keep three independent entry points.
- All project files, environments, weights, caches, logs and experiment products stay under this project.
- You may read/edit experiment.env and models.env. They must never contain an actual API key.
- NEVER open, cat, copy, search inside, archive or print private/**.
- NEVER request, display or inspect private API settings, including credentials, remote endpoint and remote model ID, via environment dumps, debugger inspection, exception output or config snapshots.
- Only the runtime for a stage set to api loads that stage's profile from private/remote_api.env. Evaluation and local model stages use public settings alone.
- remote_api.env.example is a public template. The user maintains the actual remote settings in the protected file.
- Do not include private/, .env secrets, weights, caches, raw datasets or runtime databases in source deliverables.
- Local model paths, names, loopback endpoints and generation settings in models.env are readable/editable. The stage selectors are GRAPH_MODEL_MODE and AGENT_MODEL_MODE, each local or api.
- Keep tests and fixture products under tests/ or reports/. Do not modify other users' jobs or services.
- After a completed code change and appropriate verification, inspect the diff, commit only source changes with a clear message, and push to the configured GitHub remote. Respect user instructions to defer a push. Avoid empty commits.
- Before staging, check that private settings, datasets, models, experiment outputs, unpublished source materials and original coarsening bytecode are excluded. Stage explicit paths and review the staged file list.
- Never open, copy or print Git credential stores, including .git/github-credentials. Use Git's configured credential helper for authenticated operations.
