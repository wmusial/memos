import deepgram
from memos.logutils import logstack
import yaml


def load_key(key_path):
  with open(key_path) as f:
    x = yaml.safe_load(f)
  return x["secret"]


@logstack
def transcribe_file(audio_path, key_path):
  secret = load_key(key_path)
  client = deepgram.DeepgramClient(secret)
  options = deepgram.PrerecordedOptions(
    model="nova-2",
    detect_language=True,
    #language="en",
    smart_format=True, 
    punctuate=True,
    paragraphs=True,
    diarize=True,
    summarize="v2",
  )
  with open(audio_path, "rb") as f:
    buffer_data = f.read()
  payload = {"buffer": buffer_data}

  response = client.listen.prerecorded.v("1").transcribe_file(payload, options, timeout=60*10)
  response = response.to_dict()
  assert response["results"]["summary"]["result"] == "success"
  [x] = response["results"]["channels"]
  [x] = x["alternatives"]
  return response, x["transcript"], response["results"]["summary"]["short"]

