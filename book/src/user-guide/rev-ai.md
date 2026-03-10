# Rev.AI Integration

Rev.AI is the default ASR engine for `batchalign3 transcribe`, and the default
UTR engine for `batchalign3 align`.

In server mode, those Rev.AI paths are now Rust-owned end to end: the server
submits or polls Rev.AI jobs directly and keeps Python reserved for engines
that genuinely require Python-hosted model libraries.

## Configure a Rev.AI key

Interactive setup:

```bash
batchalign3 setup
```

Non-interactive setup:

```bash
batchalign3 setup --non-interactive --engine rev --rev-key <YOUR_REV_AI_KEY>
```

This writes the key and default engine selection to `~/.batchalign.ini`.

## Use Rev.AI explicitly

```bash
batchalign3 transcribe recordings/ -o transcripts/ --asr-engine rev --lang eng
batchalign3 align corpus/ -o aligned/ --utr-engine rev
```

## Use a local model instead

If you do not want cloud ASR, use a local Whisper model:

```bash
batchalign3 transcribe recordings/ -o transcripts/ --asr-engine whisper --lang eng
```

For the OpenAI Whisper API instead of the local model:

```bash
batchalign3 transcribe recordings/ -o transcripts/ --asr-engine whisper-oai --lang eng
```

## Privacy note

Using Rev.AI sends audio to an external service. If your workflow has data-use
or IRB constraints, review your Rev.AI account settings and your local policy
before sending production data.
