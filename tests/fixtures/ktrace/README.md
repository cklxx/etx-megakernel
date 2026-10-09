# Historical ktrace fixtures

Lossless copies of the original kernel traces and imported plan used by the
kernel-comparison regression tests. No rows or columns were removed.

`provenance.json` records repository-relative source paths, uncompressed byte
counts and SHA-256 hashes. Files were compressed with Python
`gzip.compress(data, mtime=0)` and are unpacked only into pytest temporary storage.
The tests verify each checksum before parsing; missing fixtures fail the suite.

These are historical profiler measurements, not new performance evidence.
