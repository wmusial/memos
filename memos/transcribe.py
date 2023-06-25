import tempfile
import os.path
import logging
import datetime

import pydub
import openai

from memos.logutils import logstack


@logstack
def transcribe_file(file_path, max_file_size=26432142):
  target_name = "input.mp3"
  seg = pydub.AudioSegment.from_file(file_path)
  logging.info("opened file %s", file_path)
  logging.info("duration: %s sec", datetime.timedelta(seconds=seg.duration_seconds))
  with tempfile.TemporaryDirectory() as tmpdir:
    target_file_path = os.path.join(tmpdir, target_name)
    seg.export(target_file_path, format="mp3")
    target_file_size = os.path.getsize(target_file_path)
    if max_file_size is not None and target_file_size > max_file_size:
      logging.info(
        "exported size is %s, max is %s, as a percent of max: %s",
        target_file_size,
        max_file_size,
        target_file_size / max_file_size,
      )
      logging.info("exporting with 16k bitrate")
      seg = seg.set_frame_rate(24000)
      seg.export(target_file_path, format="mp3", bitrate="16k")
    logging.info("exported audio file as %s", target_file_path)
    target_file_size = os.path.getsize(target_file_path)
    logging.info(
      "exported size is %s, max is %s, as a percent of max: %s",
      target_file_size,
      max_file_size,
      target_file_size / max_file_size,
    )
    with open(target_file_path, "rb") as f:
      text = openai.Audio.transcribe(
        model="whisper-1",
        file=f,
        format="json",
        prompt="Hello, this recording is a transcript.",
      )["text"]
  return text
