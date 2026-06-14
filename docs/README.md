# DAPIDL documentation

Docs are organized by **provenance and kind**, so it is always obvious where a new
document belongs and whether it was authored here or received from outside.

```
docs/
├── external/      Third-party material we received (reports, vendor docs, PDFs).
│                  Never authored here — see external/README.md.
├── reference/     Durable "how it works" docs: algorithms, pipeline overviews,
│                  platform references. Long-lived; updated in place as the system changes.
├── reports/       Point-in-time analyses: code reviews, benchmarks, audits.
│                  Dated snapshots; not kept evergreen.
├── thesis/        Melanoma-DAPI thesis / dissertation material and the project review
│                  that frames it.
├── manuscript_draft.md   The manuscript (stays at docs/ root — see note below).
├── figures/       Figures embedded by the manuscript and metrics dossier.
├── plans/         Pre-superpowers implementation plans (legacy).
└── superpowers/   Brainstorming-workflow artifacts, already structured:
    ├── specs/         design docs
    ├── plans/         implementation plans
    └── reviews/       code / multi-agent reviews
```

## Why `manuscript_draft.md` and `figures/` stay at the root

They are a coupled bundle wired to fixed paths:

- `scripts/manuscript_figures.py` writes PNGs to `docs/figures/`.
- `scripts/sync_manuscript_to_obsidian.sh` reads `docs/manuscript_draft.md` and
  `docs/figures/` to mirror them into the Obsidian vault.
- `manuscript_draft.md` embeds figures with paths relative to `docs/` (`figures/…`).

Moving either would break those scripts and the figure links, so they are kept where
the tooling expects them rather than foldered for tidiness alone.

## Adding a document

- Wrote it yourself, describes how something works → `reference/`.
- Wrote it yourself, a dated analysis/review/benchmark → `reports/`.
- Received it from outside → `external/` (date-prefixed).
- A design or implementation plan from the brainstorming workflow → `superpowers/`.
