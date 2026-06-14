# External documentation

Documents that originate **outside this repository** — third-party reports, vendor
write-ups, collaborator notes, and reference PDFs — kept here as *input* to our work,
not as artifacts we authored.

Keeping them under a single `external/` boundary preserves provenance: a reader (or an
auditor) can tell at a glance what we wrote from what we received. This is the same
"who authored this" clarity the rest of the tech file depends on.

## Conventions

- Prefix files with the date they were received or published: `YYYY-MM-DD-<short-title>.<ext>`.
- Prefer `.md` or `.pdf`. Convert `.docx`/`.pptx` to one of those so the content is
  diff-friendly and directly readable.
- Do not edit the body of an external document. If you need to annotate it, add a
  sibling `*-notes.md` rather than altering the original.
- If volume grows, subfolder by topic inside here (e.g. `external/segmentation/`).
