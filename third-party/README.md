# Third-party notices and runtime source

Frozen experiment payloads remain byte-identical. Their original notice index
and text files are surfaced in `frozen-notices/`; the same files remain inside
`reproduction/assets/shared/common/notices.tar.gz`. A historical
`redistribution_review_complete: false` flag is preserved rather than rewritten.
The additional material below accompanies this publication bundle.

| Bundled component | Notice/source support supplied here |
|---|---|
| IBM Granite stock and merged compact model | Existing foundation/model notices in `frozen-notices/`; original model provenance remains with the release records |
| Linux and Mac llama.cpp b9850 | `notices/llama.cpp/` and the pinned llama.cpp source archive in `sources/`; the same upstream license covers the retained Mac runtime |
| Carried Linux CPython 3.12.14 | Full-distribution `PYTHON.json` metadata and authentic dependency licenses in `notices/`; CPython and exact build definitions in `sources/` |
| Berkeley DB 6.0.19 linked by Python's `_dbm` | Authentic Sleepycat notice, complete DB source archive and CPython/build definitions |
| Ubuntu GCC 14 runtime libraries | Existing GPL/runtime-exception notices; exact gcc-14 original source, Ubuntu patch archive and source descriptor |
| Ubuntu libxcrypt | Existing LGPL notices; exact source, Ubuntu patch archive and source descriptor |
| Python SDK dependencies | Their own license files inside both dependency archives, plus the original frozen notice index |

`sources/SOURCE-PROVENANCE.json` records names, hashes, official source locations
and verification authorities. Eight source archives match preexisting digest
pins; the Python build-definition archive is identified by the exact release
commit and its newly recorded digest. Ubuntu descriptor signatures were not
independently verified. `notices/extraction-receipt.json` identifies license
members extracted from the two hash-verified full Python distributions.

The Python build metadata also references zlib-ng because its generator matches
both zlib implementations by library name. The pinned Unix build selects zlib;
the reference alone does not establish a zlib-ng Linux dependency. Its authentic
notice/source is nevertheless supplied, with the supporting build definitions.

The original and additional licenses remain authoritative for their respective
components. This inventory documents what is supplied; it is not a blanket
claim that every possible redistribution use has received legal clearance.
