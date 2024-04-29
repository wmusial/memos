import argparse
import datetime
import logging
import itertools
import json
import os.path
import re
import glob
import yaml

import git

from memos.deepgram import transcribe_file
from memos.audio import get_duration
import memos.gpt as gpt


from memos.logutils import logstack, logindent

RAW_FILENAME = "raw.txt"
META_FILENAME = "meta.yaml"
CLEAN_FILENAME = "clean.txt"
AUDIO_EXTS = ["m4a", "mp3"]
MEMOS_DIR = "_memos"
MEMO_ORIG_FILENAME = "memo-original.txt"
MEMO_CLEANUP_FILENAME = "memo-cleanup.txt"
MEMO_GRAMMAR_FILENAME = "memo-grammar.txt"
MEMO_SUMMARY_FILENAME = "memo-summary.txt"
MEMO_SUMMARY_SHORT_FILENAME = "memo-summary-short.txt"
MEMO_SUMMARY_MEDIUM_FILENAME = "memo-summary-medium.txt"
MEMO_SUMMARY_LONG_FILENAME = "memo-summary-long.txt"

KEYS_DEEPGRAM_FILENAME = "deepgram-keys.yaml"
KEYS_OPENAI_FILENAME = "openai-keys.yaml"


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--out-dir-path", default="./out", help="path to the output directory"
  )
  subparsers = parser.add_subparsers(required=True, dest="cmd")
  parser_add = subparsers.add_parser(
    "transcribe",
    help=(
      "discover files in a given directory that were not previously trascribed ",
      "and saved in the memos directory, transcribe them, ",
      "and save them in the memos directory",
    ),
  )
  parser_add.add_argument(
    "--in-dir-path", default="./in", help="path to the input directory"
  )
  parser_add.add_argument(
    "--done-dir-path", default="./done", help="path to the done directory"
  )
  parser_add.add_argument(
    "--error-dir-path", default="./error", help="path to the error directory"
  )
  parser_add.add_argument(
    "--no-ignore-errors",
    action="store_true",
    default=False,
    help="break on summarization errors, such as they happen sometimes",
  )
  parser_add.add_argument("--deepgram-keys-file-path", default=KEYS_DEEPGRAM_FILENAME)
  parser_markdown = subparsers.add_parser("markdown")
  parser_summarize = subparsers.add_parser(
    "summarize", help="summarize all transcribed files"
  )
  parser_summarize.add_argument("--openai-keys-file-path", default=KEYS_OPENAI_FILENAME)
  parser_summarize.add_argument("--overwrite", action="store_true", default=False)
  if False:
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
  duration,
  **kwargs,
):
  file_path = os.path.join(memo_dir_path, META_FILENAME)
  meta = {
    "filename": filename,
    "datetime": datetime,
    "duration": duration,
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


def next_integer_dir(dir_path, prefix):
  old_dir_paths = os.listdir(dir_path)
  old_dir_paths = [
    old_dir_path
    for old_dir_path in old_dir_paths
    if old_dir_path.startswith(prefix + "_")
  ]
  old_dir_paths = [int(old_dir_path[(len(prefix) + 1):]) for old_dir_path in old_dir_paths]
  if not old_dir_paths:
    next_i = 1
  else:
    next_i = max(old_dir_paths) + 1
  return os.path.join(dir_path, f"{prefix}_{next_i:03}")


def read_creation_date(file_path):
  if re.findall(r"[0-9]{8,8} [0-9]{6,6}-.{8,8}.m4a", file_path):
    return datetime.datetime.strptime(os.path.basename(file_path)[:15], "%Y%m%d %H%M%S")
  return datetime.datetime.fromtimestamp(os.stat(file_path).st_birthtime)


@logstack
def transcribe(
  out_dir_path, in_dir_path, done_dir_path, error_dir_path, no_ignore_errors, deepgram_keys_file_path
):
  if not os.path.isdir(done_dir_path):
    os.makedirs(done_dir_path)
  if not os.path.isdir(error_dir_path):
    os.makedirs(error_dir_path)

  out_dir_path = os.path.join(out_dir_path, MEMOS_DIR)
  if not os.path.isdir(out_dir_path):
    os.makedirs(out_dir_path)
  assert os.path.isdir(in_dir_path)
  old_metas = read_all_meta(out_dir_path)

  in_filenames = os.listdir(in_dir_path)
  in_filenames = [
    path for path in in_filenames if any(path.endswith(ext) for ext in AUDIO_EXTS)
  ]
  logging.info(f"found {len(in_filenames)} audio files in {in_dir_path}")
  in_filenames = sorted(in_filenames)

  for in_filename in in_filenames:
    with logindent(f"working on {in_filename}", "done", spacing=True):

      in_path = os.path.join(in_dir_path, in_filename)

      creation_datetime = read_creation_date(in_path)
      logging.info(f"creation time: {creation_datetime.isoformat()}")

      duration = get_duration(in_path)
      logging.info(f"duration: {duration}")
      if duration.seconds > 1.5 * 60 * 60:
        logging.info("skipping for now")
        continue

      target_dir_path = next_integer_dir(out_dir_path, creation_datetime.strftime("%Y%m%dT%H%M%SZ"))
      logging.info(f"output dir {target_dir_path}")

      try:
        result, transcript, summary = transcribe_file(in_path, deepgram_keys_file_path)
      except KeyboardInterrupt:
        raise
      except Exception as e:
        print("error")
        if no_ignore_errors:
          print(e)
          raise e
        logging.info(f"file {in_filename} failed")
        os.rename(in_path, os.path.join(error_dir_path, in_filename))
        print(e)
        os.makedirs(target_dir_path)
        write_meta(
          target_dir_path,
          in_filename,
          creation_datetime.isoformat(),
          error="transcription",
        )
        continue

      os.rename(in_path, os.path.join(done_dir_path, in_filename))
      os.makedirs(target_dir_path)
      with open(os.path.join(target_dir_path, MEMO_ORIG_FILENAME), "w") as f:
        f.writelines(transcript)
      with open(os.path.join(target_dir_path, MEMO_SUMMARY_FILENAME), "w") as f:
        f.writelines(summary)
      write_meta(target_dir_path, in_filename, creation_datetime.isoformat(), duration.seconds)



@logstack
def summarize_single(keys_file_path, dir_path, meta, overwrite=False, extra=False):
  logging.info(f"working on {dir_path}...")
  with open(os.path.join(dir_path, MEMO_ORIG_FILENAME)) as f:
    text_raw = "".join(f.readlines())

  summary_medium_path = os.path.join(dir_path, MEMO_SUMMARY_MEDIUM_FILENAME)
  if not os.path.isfile(summary_medium_path) or overwrite:
    summary_medium = gpt.summarize(keys_file_path, text_raw)
    with open(summary_medium_path, "w") as f:
      f.writelines(summary_medium)
  else:
    with open(summary_medium_path) as f:
      summary_medium = "".join(f.readlines())

  if "title" not in meta:
    meta["title"] = gpt.make_title(keys_file_path, summary_medium)
    logging.info(f'title: "{meta["title"]}"')
    write_meta(dir_path, **meta)


@logstack
def summarize(out_dir_path, openai_keys_file_path, overwrite):
  sub_dir_paths = [x for x in os.listdir(out_dir_path) if os.path.isdir(os.path.join(out_dir_path, x)) and x != MEMOS_DIR]
  for sub_dir_path in sub_dir_paths:
    summarize(os.path.join(out_dir_path, sub_dir_path), openai_keys_file_path, overwrite)
  logging.info(f"summarize, dir: {out_dir_path}")
  metas = read_all_meta(os.path.join(out_dir_path, MEMOS_DIR))
  for i, (dir_path, meta) in enumerate(metas.items()):
    logging.info("")
    summarize_single(openai_keys_file_path, dir_path, meta, overwrite=overwrite)


@logstack
def markdown(out_dir_path):
  sub_dir_paths = [x for x in os.listdir(out_dir_path) if os.path.isdir(os.path.join(out_dir_path, x)) and x != MEMOS_DIR]
  for sub_dir_path in sub_dir_paths:
    markdown(os.path.join(out_dir_path, sub_dir_path))
  logging.info(f"out dir path: {out_dir_path}")
  dir_paths = [
    os.path.dirname(x)
    for x in glob.glob(os.path.join(out_dir_path, MEMOS_DIR, "*", META_FILENAME))
  ]
  metas = read_all_meta(os.path.join(out_dir_path, MEMOS_DIR))

  for dir_path, meta in metas.items():
    meta["dir"] = dir_path
    meta["datetime"] = datetime.datetime.fromisoformat(meta["datetime"])
    meta["duration"] = datetime.timedelta(seconds=meta["duration"])
  metas = metas.values()
  metas = [meta for meta in metas if not meta.get("error")]
  logging.info(f"{len(metas)} metas")

  for meta in metas:
    dir_path = meta["dir"]
    memo_date = meta["datetime"].date()
    # this is the summary done by deepgram
    summary_path = os.path.join(dir_path, MEMO_SUMMARY_FILENAME)
    # and this is the gpt one
    summary_medium_path = os.path.join(dir_path, MEMO_SUMMARY_MEDIUM_FILENAME)

    page_md = f"# {meta['datetime']}{'' if 'title' not in meta else ' - ' + meta['title']}\n\n\n"
    page_md += f"duration: {meta['duration']}\n\n"
    if "keywords" in meta:
      meta["keywords"] = " ".join(f"`{keyword}`" for keyword in meta["keywords"])
      page_md += meta["keywords"]
      page_md += "\n\n"
    blurb = None
    if os.path.isfile(summary_medium_path):
      page_md += "## summary (gpt)\n\n"
      with open(summary_medium_path) as f:
        summary = "\n\n".join(f.readlines())
        meta["summary"] = summary
        page_md += summary
      page_md += "\n\n"
    elif os.path.isfile(summary_path):
      page_md += "## summary (deepgram)\n\n"
      with open(summary_path) as f:
        summary = "\n\n".join(f.readlines())
        meta["summary"] = summary
        page_md += summary
      page_md += "\n\n"
    page_md += "## original\n\n"
    with open(os.path.join(dir_path, MEMO_ORIG_FILENAME)) as f:
      page_md += "".join(f.readlines())
    with open(os.path.join(meta["dir"], "README.md"), "w") as f:
      f.write(page_md)

  main_md = "# memos\n\n\n"
  for meta in reversed(sorted(metas, key=lambda meta: meta["datetime"])):
    #main_md += f"## {meta['datetime']}\n\n"
    main_md += f"## {meta['datetime']}{'' if 'title' not in meta else ' - ' + meta['title']}\n\n\n"
    main_md += f"duration: {meta['duration']}\n\n"
    #main_md += f"### {meta['title']}\n\n"
    if "keywords" in meta:
      main_md += meta["keywords"]
      main_md += "\n\n"
    if "summary" in meta:
      main_md += f"{meta['summary']}\n\n"
    main_md += f"[full transcript](./{MEMOS_DIR}/{os.path.basename(meta['dir'])}/README.md)\n"
    main_md += "\n\n"

  with open(os.path.join(out_dir_path, "README.md"), "w") as f:
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
  if not os.path.isdir(args.out_dir_path):
    os.makedirs(args.out_dir_path)
  return globals()[cmd.replace("-", "_")](**vars(args))


if __name__ == "__main__":
  main()
