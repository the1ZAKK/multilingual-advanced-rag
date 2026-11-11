# Preserved quality-cycle wire recordings

Each `.jsonl.gz` is a lossless copy of the corresponding hosted model artifact's
`translation-trace/generation-wire.jsonl`, compressed with a zero gzip timestamp.
The [manifest](manifest.json) binds the uncompressed and compressed SHA-256s to
the exact source commit, model, generation settings and artifact identity.

These public-paper development controls include raw generation proposals,
validation refusals and any recorded translation stages. Rejected outputs are
untrusted model proposals. They are not accepted answers, human reviews or
evidence of unseen accuracy. An exact quotation alone does not establish
semantic support. The recording's compact report contains the actual returned
answers and citation/provenance flags.

Use any gzip reader to inspect a recording. For example, from this directory:

```sh
python -c "import gzip; print(gzip.open('41188100282.jsonl.gz', 'rt', encoding='utf-8').read())"
```

The original Actions artifacts retain the complete original ZIP and runtime
files during GitHub's artifact-retention period. These checked-in traces keep
the failed proposals available beyond that period. Original source PDFs and
previous submitted reviews are preserved separately.
