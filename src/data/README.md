# Law data pipeline

`src/data/raw/egyptian_civil_code.pdf` is the canonical source document. It contains English in the
left column and Arabic in the right column. The extraction uses `pdfplumber`
word coordinates to separate columns and reconstruct reading order; a plain
text extractor cannot reliably do that for this PDF.

## Conda and UV

Create the environment once:

```powershell
conda env create -f environment.yml
conda activate legal-rag
uv sync --extra dev
```

For an existing environment:

```powershell
conda activate legal-rag
uv pip install -e ".[dev]"
```

## Prepare articles and chunks

The project handbook is stored under `src/data/docs/`. Run the preparation command
from the repository root:

```powershell
uv run python src/data/prepare_law.py src/data/raw/egyptian_civil_code.pdf
```

The command writes:

- `src/data/processed/law_articles.jsonl`: one bilingual record per article.
- `src/data/processed/law_chunks.jsonl`: citation-preserving article/paragraph chunks.

Each article contains `article_number`, `book`, `chapter`, `section`, `topic`,
`text_ar`, `text_en`, `is_repealed`, `source_page`, and `citation`.

The parser reports source gaps instead of fabricating text. In this document,
Article 54 has no normal Arabic article header and Article 1022 has no Arabic
body in the source; these must remain explicit data-quality findings.

The supplied PDF contains `السجل` in the relevant passage, not `بالسجلات`.
The pipeline does not invent `بالسجلات` when it is absent from the source.