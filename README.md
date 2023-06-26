# memos

1. Fork me on github.
2. Make the fork private.
3. Fill openai keys in `gpt3_keys.yaml`
4. `make dev` to make environment
5. `make transcribe` to transcribe all voice-memos synced to your mac,
   `make summarize` to run gpt-based summarization of existing transcripts and write markdown output files upon success,
   `make markdown` to just write markdown output files,
   or `make commit` to commit all output text files back to git.
   Run all four in sequence, e.g. by hand or on crontab.
   Invoking `memos` with the make file assumes default output location, `./out`.
