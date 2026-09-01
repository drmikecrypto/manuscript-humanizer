# Format support matrix

Manuscript Humanizer reads and writes the **same format** as the input file. Cross-format conversion (e.g. DOCX to PDF) is not supported for layout-preserving output.

## Input and output

| Input | Output (default) | In-place (`-i`) | Layout preserved |
|-------|------------------|-----------------|------------------|
| `.md`, `.txt`, `.tex` | `<name>_humanized.<ext>` | Overwrite + `.bak` | N/A (plain text) |
| `.docx` | `<name>_humanized.docx` | Overwrite + `.bak` | Paragraph text only (no runs/styles) |
| `.pdf` | `<name>_humanized.pdf` | Overwrite + `.bak` | Line geometry, fonts, justified body lines |

## PDF specifics

- Body lines are redrawn in-place; letterhead and sign-offs stay frozen.
- Justified paragraphs keep left and right margins (Word Ctrl+J behavior).
- Redaction uses white fill; colored backgrounds may need manual touch-up.
- Scanned/image-only PDFs are not supported.

## DOCX specifics

- Paragraph text is replaced; bold, italic, tables, and headers are not preserved.
- Paragraph count is preserved via word-weight distribution when line counts drift.

## Quality gates

- **Short-form** (letters, abstracts): strict overlap (≥0.72), length 0.85–1.15x. CLI skips save on failure unless `--force-save`.
- **Long-form** manuscripts: overlap ≥0.55, length ≥0.78x, protected domain terms preserved.

## Verification

The built-in score is an **offline proxy**. Confirm with [ZeroGPT](https://www.zerogpt.com/) manually after humanizing.

```powershell
mh score examples\demo_manuscript.md
mh examples\demo_manuscript.md -o tests\output\humanized.md
python scripts\calibrate_proxy.py examples\demo_manuscript.md
```
