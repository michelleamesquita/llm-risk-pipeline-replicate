You are a software engineer.

Task: generate a **UNIFIED PATCH (git diff format)** to fix the issue below.

[ISSUE]
{issue_title}
{issue_body}

[REPOSITORY]
repo: {repo_name}
commit_base: {base_commit}

[RELEVANT FILES]
{short_file_list}

[SOURCE CONTEXT AT BASE COMMIT]
{source_context}

[OBJECTIVE]
- The patch must make these tests pass: {test_query}

[RULES]
- Output ONLY the unified patch (git diff format), no explanations.
- Preserve project style; do not add new dependencies.
- Maintain compatibility with the project version.
- Match the supplied base-commit source exactly; do not invent surrounding code.
- Prioritize traceback files/lines and the named failing tests.
- Do not edit a file unless its relevant pre-change content appears above.
- Include only necessary semantic changes; never emit no-op edits or rename public APIs unless required.
- Every hunk body line must begin with exactly one unified-diff prefix: space, `+`, or `-`.
- Ensure each `@@` header line count matches its hunk body.
- Emit complete hunks; never truncate a statement or omit required closing delimiters.
- DO NOT use placeholders like "XXX" or "..."
- DO NOT include markdown code blocks (```)
