# voice-memos

1. fork me on github
2. make the fork private
3. fill openai keys in `gpt3_keys.yaml`
4. `make dev` to make environment
5. `make full` to transcribe all voice-memos synced to your mac,
   `make transcribe` to just run the transcription,
   `make summarize` to run gpt-based summarization of existing transcripts,
   or `make commit` to commit all output text files back to git.
   Invoking `memos` with the make file assumes default output location, `./out`.
