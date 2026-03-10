# Desktop App Improvements for Nontechnical Researchers

**Status:** Current
**Last updated:** 2026-03-17

Prioritized backlog of UX improvements for the Batchalign desktop app (Tauri).
The target audience is researchers (linguists, speech pathologists) who are not
comfortable with terminals.

## High Impact

### 1. Output preview / result verification
After processing completes, show a quick summary: "Added %mor and %gra tiers
to 45 files." For a sample file, show a before/after diff or the first few
lines of output. Researchers need to trust the output is correct before they
use it in their research. Right now they have to open every file manually.

### 2. Job cancellation — DONE (2026-03-17)
Cancel button added to ProcessingProgress. Wired to `POST /jobs/{id}/cancel`.

### 3. Plain-language command descriptions — DONE (2026-03-17)
Rewrote all 6 command descriptions in CommandPicker to explain *when* and *why*
a researcher would use each command. Expanded HelpPanel FAQ from 6 to 12
entries covering CHAT format basics, grammar/alignment use cases, timing
expectations, language support, cancellation, and settings.

### 4. Settings page — DONE (2026-03-17)
SettingsModal component added, accessible from gear icon in header (desktop
only). Reads/writes ~/.batchalign.ini via Tauri config commands. Shows
current ASR engine with toggle, Rev.AI key field.

## Medium Impact

### 5. Time estimates
Track per-file processing time, extrapolate remaining. Show "~12 minutes
remaining" instead of just "8 of 45 files." Long jobs (hours of audio) are
anxiety-inducing without an ETA.

### 6. Drag-and-drop
The FolderPicker has a dashed border (the universal "drop here" affordance)
but no actual drop handler. Users will try it, it won't work, and they'll
think the app is broken.

### 7. Richer FAQ / contextual help — DONE (2026-03-17)
Expanded HelpPanel FAQ from 6 to 12 entries. Added: CHAT format explanation,
when to use grammar/alignment, processing time expectations, multi-language
support, cancellation, settings access. Covered as part of item 3.

### 8. Multi-step pipelines
Researchers often need to chain commands: transcribe → align → morphotag.
Right now they have to do each step manually, re-picking folders each time.
A "pipeline" mode that chains 2-3 commands on the same folder would save
significant friction.

## Lower Impact (good polish)

### 9. CHAT file preview
A built-in viewer that shows CHAT files with syntax highlighting (speaker
tiers in one color, dependent tiers in another). Not an editor — just enough
to verify output looks right without leaving the app.

### 10. Batch history with re-run
The "Recent Tasks" list shows status but you can't re-run a previous job with
the same settings. A "Run again" button on completed jobs would save time for
repeated workflows.

### 11. Language auto-detection
For commands that need a language, default to "English" but offer "Auto-detect
from file" for `.cha` files that already have `@Languages` headers. Reduces
one decision point.

### 12. Progress notifications
When a long job finishes and the app is in the background, send a native OS
notification ("Transcription complete — 45 files processed"). Tauri has
`tauri-plugin-notification` for this.

### 13. Accessibility
Keyboard navigation through the command cards, focus management in the setup
wizard, screen reader labels on the progress dots. The target audience includes
researchers with disabilities.
