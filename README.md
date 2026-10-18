# Replicate: Bounded Self-Replication of a Locally Inferring Software Agent

Lars Tray · Georgetown University · October 18, 2026

Replicate is a systems testbed for bounded self-replication across
owner-prepared receivers. Each admitted parent supplies a complete signed
software package and local model. Its child repeats the operation after the
seed is removed.

**[One-page overview](overview/replicate-overview.pdf)** ·
**[Short sharing blurb](overview/replicate-share-blurb.txt)** ·
**[Read the paper](paper/replicate.pdf)** ·
**[LaTeX source](paper/replicate.tex)** ·
**[arXiv source archive](submission/arxiv-source.zip)**

## Original motivation

The starting idea was a factory floor with hundreds of connected machines.
Authorized peers would supply newly installed devices with the current local
inference engine and supporting software. A self-healing mesh would distribute
approved updates and restore service without relying on an always-on central
orchestrator. The [short overview](overview/replicate-overview.pdf) explains
that vision and the bounded replication milestone demonstrated here. Large
populations, automatic version updates, and cooperative fault recovery remain
future work.

## What the experiments establish

- Three fresh seed–child–grandchild lineages completed six births on prepared
  8 GiB x86-64 Linux guests. Seed removal preceded grandchild eligibility.
- In a separately frozen cohosted comparison, trained and deterministic
  policies each completed two births and 15/15 diagnostics. Fixed policy
  used zero model calls.
- On forty post-development synthetic diagnostic cases, trained and fixed
  policies passed 40/40; the stock model passed 32/40 under the same bounded,
  non-thinking model configuration.
- Separate-code auditors accepted 24/24 valid controls and rejected 108/108
  prescribed inconsistencies in retained records.

These results establish replication feasibility and deterministic sufficiency
for the prepared task. They do not establish general autonomous replication,
model necessity, or population reliability. Auditing uses separate project
code and is not an unaffiliated replication.

## Source and complete research archive

This GitHub repository provides the paper, application and research source,
documentation, derived results, and an evidence index. The full model files,
carried runtimes, signed payloads, experiment records, historical attempts,
third-party notices, corresponding sources, and integrity manifests belong
to the complete reproduction archive.

**[Download the complete research archive](https://doi.org/10.5281/zenodo.23454911)** · DOI: `10.5281/zenodo.23454911`

No arXiv submission has been made through this repository.

The [reproduction guide](reproduction/README.md) describes the complete archive.
Its asset and evidence paths resolve in that archive; they are intentionally
absent from this source repository. Use the complete archive for reconstruction.
The [evidence index](reproduction/EVIDENCE-INDEX.md) maps the paper's E1–E22
citations to its package-relative files.

## Repository layout

- `overview/`: shareable one-page introduction, editable text, and short blurb.
- `paper/`: current manuscript and standalone LaTeX source.
- `submission/`: source upload ZIP, abstract, and submission metadata.
- `reproduction/src/organism/`: application, auditors, and research programs.
- `reproduction/docs/`: retained plans and operational documentation.
- `reproduction/results/`: derived results and historical receipts.
- `licenses/` and `LICENSES.md`: license scope and complete license texts.

The code and experimental records are preserved from the reviewed release.
Publication preparation does not rerun inference, training, native deployments,
or evidence replay. Historical plans describe intended work and are not claims
that all listed qualifications were completed.

## Citation and licenses

Use [CITATION.cff](CITATION.cff) for the author and manuscript title. The complete archive DOI is [10.5281/zenodo.23454911](https://doi.org/10.5281/zenodo.23454911). An arXiv identifier has not yet been assigned.

Original code is Apache-2.0. The manuscript and original authored data are
CC-BY-4.0. Third-party materials retain their own terms; see
[LICENSES.md](LICENSES.md).
