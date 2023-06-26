import argparse
import datetime
import logging
import itertools
import os.path
import re
import glob
import yaml

import git


from memos.transcribe import transcribe_file
import memos.gpt as gpt

from memos.logutils import logstack, logindent

RAW_FILENAME = "raw.txt"
META_FILENAME = "meta.yaml"
CLEAN_FILENAME = "clean.txt"
AUDIO_EXTS = ["m4a", "mp3"]
MEMO_ORIG_FILENAME = "memo-original.txt"
MEMO_CLEANUP_FILENAME = "memo-cleanup.txt"
MEMO_GRAMMAR_FILENAME = "memo-grammar.txt"
MEMO_SUMMARY_SHORT_FILENAME = "memo-summary-short.txt"
MEMO_SUMMARY_MEDIUM_FILENAME = "memo-summary-medium.txt"
MEMO_SUMMARY_LONG_FILENAME = "memo-summary-long.txt"

KEYS_FILENAME = "openai-keys.yaml"


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--memos-dir-path", default="./out", help="path to the memos output directory"
  )
  subparsers = parser.add_subparsers(required=True, dest="cmd")
  parser_add = subparsers.add_parser(
    "transcribe-from-dir",
    help=(
      "discover files in a given directory that were not previously trascribed ",
      "and saved in the memos directory, transcribe them, ",
      "and save them in the memos directory",
    ),
  )
  parser_add.add_argument(
    "--input-dir-path",
    required=True,
    type=str,
    help="path to the input directory with audio files",
  )
  parser_add.add_argument(
    "--no-ignore-errors",
    action="store_true",
    default=False,
    help="break on summarization errors, such as they happen sometimes",
  )
  parser_add.add_argument("--keys-file-path", default=KEYS_FILENAME)
  parser_summarize = subparsers.add_parser(
    "summarize", help="summarize all transcribed files"
  )
  parser_summarize.add_argument("--keys-file-path", default=KEYS_FILENAME)
  parser_summarize.add_argument("--extra", action="store_true", default=False)
  parser_summarize.add_argument("--overwrite", action="store_true", default=False)
  parser_markdown = subparsers.add_parser("markdown")
  parser_markdown.add_argument("--memos-dir-path", default="./out")
  parser_commit = subparsers.add_parser(
    "commit", help="commit all files in the output memos directory back to git"
  )
  parser_commit.add_argument(
    "--repo-dir-path", default=".", help="path to the git repo root"
  )
  return parser.parse_args()


def read_meta(memo_dir_path):
  file_path = os.path.join(memo_dir_path, META_FILENAME)
  with open(file_path) as f:
    x = yaml.safe_load(f)
  return x


def write_meta(
  memo_dir_path,
  filename,
  datetime,
  **kwargs,
):
  file_path = os.path.join(memo_dir_path, META_FILENAME)
  meta = {
    "filename": filename,
    "datetime": datetime,
  }
  meta.update(kwargs)
  with open(file_path, "w") as f:
    yaml.dump(meta, f)


def read_all_meta(memos_dir_path):
  dir_paths = [
    os.path.dirname(x)
    for x in glob.glob(os.path.join(memos_dir_path, "*", META_FILENAME))
  ]
  return {dir_path: read_meta(dir_path) for dir_path in sorted(dir_paths)}


def next_integer_dir(dir_path):
  old_dir_paths = os.listdir(dir_path)
  old_dir_paths = [
    old_dir_path
    for old_dir_path in old_dir_paths
    if re.match("^[0-9]+$", old_dir_path)
  ]
  old_dir_paths = [int(old_dir_path) for old_dir_path in old_dir_paths]
  if not old_dir_paths:
    next_i = 1
  else:
    next_i = max(old_dir_paths) + 1
  return os.path.join(dir_path, f"{next_i:05}")


def read_creation_date(file_path):
  return datetime.datetime.fromtimestamp(os.stat(file_path).st_birthtime)


@logstack
def transcribe_from_dir(
  memos_dir_path, input_dir_path, no_ignore_errors, keys_file_path
):
  gpt.load_keys(keys_file_path)
  memos_dir_path = os.path.join(memos_dir_path, "memos")
  if not os.path.isdir(memos_dir_path):
    os.makedirs(memos_dir_path)
  assert os.path.isdir(input_dir_path)
  old_metas = read_all_meta(memos_dir_path)
  all_old_filenames = set([meta["filename"] for meta in old_metas.values()])

  new_filenames = os.listdir(input_dir_path)
  new_filenames = [
    path for path in new_filenames if any(path.endswith(ext) for ext in AUDIO_EXTS)
  ]
  logging.info(f"found {len(new_filenames)} audio files in {input_dir_path}")
  new_filenames = [
    file_name
    for file_name in new_filenames
    if os.path.basename(file_name) not in all_old_filenames
  ]
  logging.info(f"found {len(new_filenames)} new audio files in {input_dir_path}")

  new_filenames = sorted(new_filenames)

  for new_filename in new_filenames:
    with logindent(f"working on {new_filename}", "done", spacing=True):
      target_dir_path = next_integer_dir(memos_dir_path)
      logging.info(f"outout dir {target_dir_path}")

      creation_datetime = read_creation_date(
        os.path.join(input_dir_path, new_filename)
      ).isoformat()

      try:
        text = transcribe_file(os.path.join(input_dir_path, new_filename))
      except KeyboardInterrupt:
        raise
      except:
        print("error")
        if no_ignore_errors:
          raise
        logging.info(f"file {new_filename} failed")
        os.makedirs(target_dir_path)
        write_meta(
          target_dir_path,
          new_filename,
          creation_datetime,
          error="transcription",
        )
        continue

      os.makedirs(target_dir_path)
      with open(os.path.join(target_dir_path, MEMO_ORIG_FILENAME), "w") as f:
        f.writelines(text)

      write_meta(target_dir_path, new_filename, creation_datetime)

  print("done.")


@logstack
def process_single(dir_path, meta, overwrite=False, extra=False):
  logging.info(f"working on {dir_path}...")
  with open(os.path.join(dir_path, MEMO_ORIG_FILENAME)) as f:
    text_raw = "".join(f.readlines())

  summary_medium_path = os.path.join(dir_path, MEMO_SUMMARY_MEDIUM_FILENAME)
  summary_long_path = os.path.join(dir_path, MEMO_SUMMARY_LONG_FILENAME)
  if not os.path.isfile(summary_medium_path) or overwrite:
    summary_medium, summary_long = gpt.summarize(text_raw)
    with open(summary_medium_path, "w") as f:
      f.writelines(summary_medium)
    with open(summary_long_path, "w") as f:
      f.writelines(summary_long)
  else:
    with open(summary_medium_path) as f:
      summary_medium = "".join(f.readlines())
    with open(summary_long_path) as f:
      summary_long = "".join(f.readlines())

  summary_short_path = os.path.join(dir_path, MEMO_SUMMARY_SHORT_FILENAME)
  if not os.path.isfile(summary_short_path) or overwrite:
    if len(summary_medium) > 500:
      summary_short = gpt.short_summarize(summary_medium)
    else:
      summary_short = summary_medium
    with open(summary_short_path, "w") as f:
      f.writelines(summary_short)
  else:
    with open(summary_short_path) as f:
      summary_short = "".join(f.readlines())

  if "title" not in meta:
    meta["title"] = gpt.make_title(summary_long)
    logging.info(f'title: "{meta["title"]}"')
    write_meta(dir_path, **meta)

  if "keywords" not in meta or overwrite:
    meta["keywords"] = gpt.extract_keywords(summary_long)
    write_meta(dir_path, **meta)

  if not extra:
    return

  cleanup_path = os.path.join(dir_path, MEMO_CLEANUP_FILENAME)
  if not os.path.isfile(cleanup_path):
    text_cleanup = gpt.cleanup_monologue(text_raw)
    with open(cleanup_path, "w") as f:
      f.writelines(text_cleanup)
  else:
    with open(cleanup_path) as f:
      text_cleanup = "".join(f.readlines())

  return

  # print("----------")
  # print(text_raw)
  # print("----------")
  # print(text_cleanup)
  # print("----------")

  # text = gpt.fix_transcription(text_raw)
  # if text is None:
  #   return
  # with open(os.path.join(dir_path, MEMO_FIXUP_FILENAME), 'w') as f:
  #   f.writelines(text)
  write_meta(dir_path, **meta)
  text_grammar = gpt.fix_grammar(text_cleanup)
  with open(os.path.join(dir_path, MEMO_GRAMMAR_FILENAME), "w") as f:
    f.writelines(text_grammar)
  meta["grammar"] = True
  write_meta(dir_path, **meta)


@logstack
def summarize(memos_dir_path, keys_file_path, extra, overwrite):
  memos_dir_path = os.path.join(memos_dir_path, "memos")
  gpt.load_keys(keys_file_path)
  metas = read_all_meta(memos_dir_path)
  for i, (dir_path, meta) in enumerate(metas.items()):
    logging.info("")
    process_single(dir_path, meta, extra=extra, overwrite=overwrite)
  markdown(memos_dir_path)


@logstack
def markdown(memos_dir_path):
  metas = read_all_meta(os.path.join(memos_dir_path, "memos"))

  for dir_path, meta in metas.items():
    meta["dir"] = dir_path
    meta["datetime"] = datetime.datetime.fromisoformat(meta["datetime"])
  metas = metas.values()
  metas = [meta for meta in metas if not meta.get("error")]

  for meta in metas:
    dir_path = meta["dir"]
    memo_date = meta["datetime"].date()

    summary_short_path = os.path.join(dir_path, MEMO_SUMMARY_SHORT_FILENAME)
    summary_medium_path = os.path.join(dir_path, MEMO_SUMMARY_MEDIUM_FILENAME)
    summary_long_path = os.path.join(dir_path, MEMO_SUMMARY_LONG_FILENAME)

    page_md = f"# {meta['title']}\n\n\n"
    page_md += f"{memo_date}\n\n"
    page_md += "\n\n"
    if "keywords" in meta:
      meta["keywords"] = " ".join(f"`{keyword}`" for keyword in meta["keywords"])
      page_md += meta["keywords"]
      page_md += "\n\n"
    blurb = None
    if os.path.isfile(os.path.join(dir_path, MEMO_SUMMARY_SHORT_FILENAME)):
      page_md += "## summary (short) \n\n"
      with open(os.path.join(dir_path, MEMO_SUMMARY_SHORT_FILENAME)) as f:
        blurb = "".join(f.readlines())
        meta["short_summary"] = blurb
        page_md += blurb
      page_md += "\n\n"
    if os.path.isfile(summary_medium_path):
      page_md += "## summary (medium)\n\n"
      with open(summary_medium_path) as f:
        page_md += "".join(f.readlines())
      page_md += "\n\n"
    if os.path.isfile(summary_long_path):
      page_md += "## summary (long)\n\n"
      with open(summary_long_path) as f:
        page_md += "".join(f.readlines())
      page_md += "\n\n"
    if os.path.isfile(os.path.join(dir_path, MEMO_GRAMMAR_FILENAME)):
      page_md += "## grammar rewrite\n\n"
      with open(os.path.join(dir_path, MEMO_GRAMMAR_FILENAME)) as f:
        page_md += "".join(f.readlines())
      page_md += "\n\n"
    if os.path.isfile(os.path.join(dir_path, MEMO_CLEANUP_FILENAME)):
      page_md += "## cleaned-up original\n\n"
      with open(os.path.join(dir_path, MEMO_CLEANUP_FILENAME)) as f:
        page_md += "".join(f.readlines())
      page_md += "\n\n"
    page_md += "## original\n\n"
    with open(os.path.join(dir_path, MEMO_ORIG_FILENAME)) as f:
      page_md += "".join(f.readlines())
    with open(os.path.join(meta["dir"], "README.md"), "w") as f:
      f.write(page_md)

  main_md = "# memos\n\n\n"
  for date, metas in itertools.groupby(
    reversed(sorted(metas, key=lambda meta: meta["datetime"])),
    key=lambda meta: meta["datetime"].date(),
  ):
    main_md += f"## {date}\n\n"
    for meta in metas:
      memo_date = meta["datetime"].date()
      main_md += f"### {meta['title']}\n\n"
      if "keywords" in meta:
        main_md += meta["keywords"]
        main_md += "\n\n"
      if "short_summary" in meta:
        main_md += f"{meta['short_summary']}\n\n"
      main_md += f"[full transcript](./memos/{os.path.basename(meta['dir'])}/README.md)\n"
      main_md += "\n"
    main_md += "\n\n"

  with open(os.path.join(memos_dir_path, "README.md"), "w") as f:
    f.write(main_md)


@logstack
def commit(memos_dir_path, repo_dir_path):
  memos_dir_path = os.path.abspath(memos_dir_path)
  repo_dir_path = os.path.abspath(repo_dir_path)
  assert memos_dir_path.startswith(repo_dir_path)
  logging.info("memos dir: %s", memos_dir_path)
  logging.info("repo dir: %s", repo_dir_path)
  repo = git.Repo(repo_dir_path)
  repo.git.add(repo_dir_path)


def main():
  logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
  args = parse_args()
  cmd = args.cmd
  del args.cmd
  if not os.path.isdir(args.memos_dir_path):
    os.makedirs(args.memos_dir_path)
  return globals()[cmd.replace("-", "_")](**vars(args))


if __name__ == "__main__":
  main()
