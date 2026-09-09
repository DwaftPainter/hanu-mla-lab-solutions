---
name: mla-lab-report
description: Prepare a weekly HANU Machine Learning Applications lab submission from its instruction PDF, notebooks, and source files; produce one compliant group PDF with reproducible results and a code appendix. Use for recurring MLA lab-report work, not generic PDF authoring.
metadata:
  short-description: Prepare weekly MLA lab PDF submissions
---

# MLA lab report workflow

Use this skill when the user asks to complete, assemble, check, or submit a weekly Machine Learning Applications lab report. The deliverable is a single PDF for the group, normally named like `Group01_CLC01_Lab3.pdf`.

## Non-negotiable constraints

- Treat the official `Lab <Week N> - Instructions` PDF as the source of truth for required tasks, section order, questions, figures, metrics, and submission rules. Do not invent a report structure when that PDF is available.
- Work from the user's notebooks and source files. Execute and report the actual code and outputs; never invent metrics, plots, observations, citations, or completed answers.
- Do not silently complete missing graded notebook code or present agent-generated solutions as the group's original work. If required implementation is incomplete, identify the gap and ask the user to provide or confirm their own implementation. Automation may organize, execute, test, explain, and format that work.
- Respect the academic-integrity notice: do not copy code between groups, reuse unrelated previous solutions as if they were current work, or create several group's submissions. Discussion of concepts is fine; the submitted implementation must be the user's work.
- Produce one group submission only. Do not use a report generator's default `both`/individual mode. The final PDF must contain one submitting member, with other members listed only when the user supplies their details.
- Do not overwrite the original notebooks, starter code, datasets, or instruction PDFs. Execute notebooks from copies or write only generated artifacts to a dedicated report/output directory.

## Discover the current lab

1. Find the target week/lab from the user's request. If no week is stated, inspect the repository for the newest `week-*` directory that contains notebooks or a lab instruction PDF, but state the assumption before proceeding.
2. Search that week and its parent for:
   - instruction PDFs, especially names containing `Instruction`, `Instructions`, `Tutorial`, or `Week`;
   - `*.ipynb` notebooks, excluding `.ipynb_checkpoints`;
   - `*.py` source files, existing `generate_report.py` helpers, README files, data directories, figures, and prior report PDFs.
3. If the official instruction PDF is not present locally or as a user-provided attachment, stop before generating the final report and ask the user to provide it. A README or a previous report cannot replace the official requirements.
4. Extract the instruction PDF text and inspect its page count. Use a PDF reader/rendering tool when visual layout, tables, or embedded examples matter. Record the required task-to-section mapping before writing report prose.
5. Inspect the Week 1 and Week 2 reports if they are available. In this repository, generated PDFs may be ignored; when they are absent, use `week-1/Lab1_KNN_Starter/generate_report.py` and `week-2/Lab2_Starter/generate_report.py` as the available style references. Reuse their visual language—A4 layout, Arial/Unicode support, navy section headings, tables, plots, running header/footer, page numbers, and code appendix—without copying their claims or hard-coded results.

## Resolve submission metadata once

Read metadata from an existing `mla-report-config.json` in the repository if present; otherwise use values explicitly supplied by the user. The reusable schema is in [references/report-config.md](references/report-config.md).

Required before a final PDF is created:

- exact group prefix used for the filename, such as `Group01_CLC01`;
- lab number, confirmed from the official instructions;
- submitting member's name and student ID;
- any additional title-page fields required by the instruction PDF.

Do not guess the group/CLC number, member identity, or lab number. If metadata is missing, the agent may prepare and validate draft artifacts, but must not present a submission-ready PDF with placeholders or borrowed defaults from a starter script.

The final basename is `<group-prefix>_Lab<N>.pdf`, preserving the user's prefix and removing only accidental spaces if the repository convention uses compact names. For example: `Group01_CLC01_Lab3.pdf`. Do not substitute `report.pdf` as the only deliverable.

## Execute and collect evidence

Create a generated-artifacts directory inside the target lab directory, preferably `report_artifacts/` or the repository's existing ignored equivalent. Keep originals untouched.

- Run the supplied Python programs from the directory they expect so relative `data/` paths resolve correctly.
- Execute each relevant notebook in a temporary copy, in the order specified by the instructions or filename order. Preserve outputs, seeds, warnings, and errors. If execution requires unavailable packages, report the exact dependency rather than fabricating output.
- Prefer deterministic runs and record the environment assumptions that affect results: data split, random seed, hyperparameters, preprocessing, and evaluation metric.
- Reuse existing report-generation helpers when they correctly reflect the current lab. Patch them only to use actual current inputs, the required group metadata, and the required output filename.
- If the report is assembled from notebooks rather than a helper, convert the executed notebook or a generated HTML/Markdown document to PDF only after checking that equations, code, plots, Unicode text, and page breaks render correctly.

## Build the report

Follow the instruction PDF's structure and task numbering. Unless it specifies another structure, organize the single PDF as:

1. title/metadata block with course, lab, group, and submitting member;
2. dataset and setup;
3. one section per instructed task, including method, relevant code/output, actual tables or figures, and a concise interpretation;
4. requested comparison, analysis, reflection, limitations, or conclusion;
5. appendix containing the complete submitted code, or a clearly named accompanying `.py` file only when the code cannot reasonably fit in the PDF.

Use the prior reports only as a formatting reference. Do not reuse their hard-coded student identity, task answers, metrics, plots, or conclusions. Every numeric claim in the new report must be traceable to the current execution output or the instruction PDF.

Keep explanations technically honest:

- distinguish training, validation, and test results;
- explain model-selection decisions using validation data only;
- label baselines and comparisons clearly;
- state limitations when the data or representation makes a conclusion uncertain;
- never claim that a plot or experiment was produced if it was not executed.

## Validate before handoff

Do not call the report submission-ready until all of these checks pass:

- the PDF opens successfully and has nonzero pages;
- the basename exactly matches `<group-prefix>_Lab<N>.pdf`;
- the title, lab number, group prefix, and submitting member are present and correct;
- every required task from the instruction PDF appears in the required order;
- all required figures/tables are present, readable, captioned where appropriate, and based on current outputs;
- the appendix contains the complete current code, or the separately attached `.py` file is explicitly named and included in the handoff;
- there are no unresolved `TODO`, `TBD`, placeholder names, default student IDs, or unexplained hard-coded metrics;
- page headers, footers, page numbers, Unicode text, code wrapping, and final/appendix page breaks render cleanly;
- the source code/notebooks used for the report run successfully, or any execution blocker is reported rather than hidden;
- only one final group PDF is offered to the user.

Use PyMuPDF, `pdfinfo`, or equivalent to check PDF metadata/text, and render representative first, middle, and appendix pages for visual inspection. Run repository-appropriate checks such as `git diff --check` when files were changed.

## Handoff

Link the final PDF with its absolute local path. Briefly state the lab, exact filename, execution/validation checks performed, and any remaining user action such as attaching the separate `.py` file. If a required input or metadata value is missing, report that blocker clearly and do not label a draft as final.
