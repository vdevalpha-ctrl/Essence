# TOOLS.md
# Domain policies for CriticGate constraint synthesis.
# Each rule becomes a guarded constraint during step validation.

## Constraints
- NEVER delete files without explicit user confirmation.
- NEVER send data to external URLs unless the user has approved the target.
- NEVER execute commands that modify system configuration (sudo, chown root, etc.).
- ALWAYS show a plan before running more than 3 sequential shell commands.
- web_search: allowed at any time.
- write_file: allowed within workspace only (allow_outside = false by default).
- python_exec: sandboxed subprocess only.
