# Upgrade the uploaded Windows installation safely

Extract the new source archive into a **separate folder**. Do not extract it over
an existing installation. The update manifest is relative to the source archive
uploaded on 2026-10-16, not the older public GitHub branch.

Stop your project API and dashboard in their terminals with Ctrl+C. Leave your
models, `.env`, `.venv`, data and backups in the existing project directory.

Open a terminal in the newly extracted update folder:

```powershell
$target = "C:\path\to\your\existing-project"
py -3.11 scripts/apply_update.py --target $target --check
py -3.11 scripts/apply_update.py --target $target
```

Set `$target` to the actual installation directory. Python 3.12 also works.
The check refuses any changed file whose contents differ from both the uploaded
baseline, an explicitly recorded previous release, and the new version. Review/merge any refused files instead of forcing
an overwrite. No files are written until every source precondition passes.

Before applying, the helper creates a ZIP of replaced source files under
`reports/source-updates/`, including a restore manifest. Runtime directories and
`.env` are protected. A write failure restores attempted source changes. The
helper neither installs packages nor launches services. It is idempotent.

Then return to the existing project:

```powershell
Set-Location $target
.\.venv\Scripts\python.exe -m pytest -q
.\scripts\start.ps1
```

If local PowerShell policy prevents running scripts, use the two manual server
commands in the README; do not change system policy just for this launcher.
The launcher opens separate terminals and reuses recognizable existing services.
It never kills an arbitrary port owner or downloads a model. Restart existing
project terminals to load changed code or configuration.

## Index compatibility and optional migration

Embedding keys and existing SQLite records retain compatibility. This update
**does not reindex automatically**. The v4 parser preserves native single-column reading order, recognizes indented
abstract headings and avoids cutting ordinary words at overlap boundaries. It also
retains the earlier Roman-numbered author-list support for new uploads. Table geometry is read from the
retained original PDF and needs no vector reindex to receive caption/selection fixes.

Uploading an identical PDF again can return its existing duplicate entry without
migrating chunks. If an older document needs v4 section/chunk boundaries, stop the
API and use:

```powershell
.\.venv\Scripts\python.exe scripts/reindex_document.py DOCUMENT_SHA256
```

A SQLite backup is made first. Retained original bytes must match the document
hash; otherwise migration refuses. For legacy uploads without retained bytes,
supply `--pdf "C:\path\to\original.pdf"`. Existing document identity/name and
other documents are preserved. Changed chunk IDs invalidate old passage-level
annotations; regenerate them for the migrated document. Restart the API after
CLI migration, because a second API process cannot refresh its in-memory index.

## Portable source archives

```powershell
.\.venv\Scripts\python.exe scripts/package_source.py --output project-source.zip
```

This selects source/tests/docs and synthetic evaluation assets, excludes model
weights, environments, corpora, `.env`, bytecode and backups, checks ZIP integrity,
and includes a SHA-256 inventory. Review documentation/screenshots before sharing
if you added private material. The inventory detects accidental changes; it is
not a publisher signature.

## Preparing another source release (developers)

In the Git checkout, preserve the previous shipped `UPDATE-MANIFEST.json` before
editing. Then regenerate against the original uploaded baseline, accepting only
recorded previous release hashes:

```bash
python -m scripts.prepare_update --baseline 6581c08 --previous-manifest /path/to/previous/UPDATE-MANIFEST.json
python scripts/package_source.py --output /path/to/project-source-updated.zip
```

The baseline ref applies to this checkout. This release helper requires Git; end
users applying an update do not. Unknown edits still cause preflight refusal.
Deleted or reverted shipped files require a reviewed migration rather than silent
deletions. Existing backups and runtime directories remain protected.
