# Documentation naming and recorded exceptions

## Contents
- [Existing guidance](#existing-guidance)
- [Exceptions](#exceptions)
- [Change inventory](#change-inventory)

## Existing guidance
| scope | existing practice or rule | source |
| --- | --- | --- |
| Ownership | Living model documents belong under their model directory; engine documents belong under `docs/core/`; cross-model records stay under `docs/` | `docs/glm/MOVE-TABLE.md:3` |
| Filenames | Uppercase hyphen-separated topics are existing practice, not a newly imposed filename rule; no global filename rule was found | Existing `docs/glm/MOVE-TABLE.md` and `docs/core/STYLE-STE100.md` |
| Technical content | Preserve technical meaning, uncertainty, historical context and measurement conditions | `docs/core/STYLE-STE100.md:7` |
| Protected text | Preserve paths, identifiers and commands during style edits | `docs/core/STYLE-STE100.md:62` |

## Exceptions
| path | exception | reason | review status |
| --- | --- | --- | --- |
| `docs/glm/PARALLEL-ANALYSIS.md` | Preserved reference retains prose, its original tables and 246 lines instead of the compact checklist format | Byte-identical preservation of the requested calculations, assumptions, source citations and evolution detail from `5ede9ea:docs/glm/PARALLEL-DESIGN.md`; prevents a formatting edit from deleting analytical evidence | Recorded for cc-o55 review; not an owner-approved waiver |
| `docs/glm/PARALLEL-DESIGN.md` | Command cells link to fenced blocks under Commands rather than embedding multiline fences inside table cells | Standard Markdown pipe tables cannot contain fenced blocks; each command retains one unique ID and a complete pasteable block | Format checked locally; review acceptance remains unverified |

## Change inventory
| item | result |
| --- | --- |
| Review entry | `docs/glm/PARALLEL-DESIGN.md`; 82 lines; 11 numbered step rows; six fields in the required order |
| Preserved reference | `docs/glm/PARALLEL-ANALYSIS.md`; SHA256 `8af91eea4e41e17313583b4886de6d1e39ea746c12e07d1f75811c69779b6d41` |
| Files deleted | 0 relative to `5ede9ea`; the review path remains present and the analysis is preserved separately |
| Existing referrers rewritten | 0; no external tracked referrers targeted the former analysis path |
| New reference targets | The review entry reads `PARALLEL-ANALYSIS.md`; format tests target `PARALLEL-DESIGN.md` and check that the reference exists |
| Remaining broken referrers | 0 found in the scoped tracked-reference audit |
| Remaining checklist violations | 0 detected by the local format checks; semantic review remains unverified |
| Remaining exception | The preserved reference does not satisfy the compact format; explicit review disposition is required |
| Local validation | 232 tests passed; 17 skipped; six require accelerate and eleven require the local gfx942 toolchain |
