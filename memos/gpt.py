import collections
import openai
import tiktoken
import json
import logging
import slugify
import requests
import fuzzysearch
import yaml
from memos.logutils import logstack
from cache_decorator import Cache
from langchain.text_splitter import TokenTextSplitter

CHAT_MODELS = [
  "gpt-3.5-turbo",
  "gpt-3.5-turbo-16k",
  "gpt-4",
  "gpt-4-32k",
  "gpt-3.5-turbo-0613",
  "gpt-4-0613",
]
FUNCTION_MODELS = ["gpt-4-0613", "gpt-3.5-turbo-0613"]

MODEL_MAX_TOKENS = {
  "gpt-3.5-turbo": 4096,
  "gpt-3.5-turbo-16k": 16384,
  "gpt-4": 8192,
  "gpt-4-32k": 32768,
}
"""
* fix punctuation, insert missing punctuation, remove excessive punctuation.
* join lines, join sentences across lines, insert empty space.
* insert paragraph breaks. separate text into many small paragraphs.
* delete non-verbal sound descriptions, like "[indistinct chatter]".
* organize text into paragraphs.
"""

PROMPT_CLEANUP_MONOLOGUE = """
You are a professional text editor.
Your job is to "tokenize" the following text.
You must split the existing text into sentences and paragraphs.
Put each sentence on a new line, clean up punctuation.
Put an additional empty line between paragraphs (groups of sentences).
Remove non-verbal sound descriptions.


---- BEGIN ----


{text}


---- END ----
"""

PROMPT_TITLE = """
Come up with a one-noun-phrase title or a headline for the text summarized below.
Do not make it sound pompous or better than it actually is.
Make it realistic and reflective of the contents. Use simple words.
Compose the title of two parts separated by a colon.
Each part is a noun phrase, possibly using a gerund.
The first part is more general and concerns the large theme in the text.
The second part is more specific.
Example: "Philosophy of science: questions about what makes arguments valid"


---- BEGIN ----


{text}


---- END ----
"""
PROMPT_SHORTSUMMARY = """
"Shorten the summary to one concise but easy to read paragraph.


---- BEGIN ----


{text}


---- END ----
"""

PROMPT_SUMMARY = """
The following text is a transcript of a monologue or a dialogue between two people.
It may contain occasional transcription errors and missing punctuation.
Please summarize the text in one or a few paragraphs.
Use efficient language, but make the summary very easy to read.
Be pedagogical.
Optimize for recall of details contained within.


---- BEGIN ----


{text}


---- END ----
"""

PROMPT_KEYWORDS = """
Extract keywords from the following text.
Keywords should be nouns.
Keywords should aid in recall of information contained in the text.
Include both general and specific keywords.
For example, extract a keyword that captures the topic on the text,
and also a keyword that captures a particular important particular subject discussed in the text.
Give 10 keywords.
Order the keywords in order of decreasing importance, most important first.


---- BEGIN ----


{text}


---- END ----
"""

FUNCTIONS_KEYWORDS = [
  {
    "name": "extracted_keywords",
    "description": "return a list with the given keywords",
    "parameters": {
      "type": "object",
      "properties": {
        "keywords": {
          "type": "array",
          "items": {
            "type": "string",
          },
          "description": "extracted keywords",
        },
      },
    },
  },
]


def load_keys(keys_file_path):
  with open(keys_file_path) as f:
    keys = yaml.safe_load(f)
  openai.api_key = keys["secret"]
  openai.organization = keys["organization"]


@Cache()
def get_single_completion(prompt, model, temperature, sequence_id=0):
  if model in CHAT_MODELS:
    if isinstance(prompt, list):
      completion = openai.ChatCompletion.create(
        model=model,
        messages=prompt,
        temperature=temperature,
      )
    else:
      completion = openai.ChatCompletion.create(
        model=model,
        messages=[
          {
            "role": "user",
            "content": prompt,
          }
        ],
        temperature=temperature,
      )
    ret = completion["choices"][0]["message"]["content"]
  else:
    completion = openai.Completion.create(
      model=model,
      prompt=prompt,
    )
    ret = completion["choices"][0]["text"]
  return ret


@Cache()
def stream_single_completion(
  prompt, model, temperature, sequence_id=0, do_print=False, functions=None
):
  assert model in CHAT_MODELS
  if functions is not None:
    assert model in FUNCTION_MODELS
  do_stream = functions is None
  params = {
    "model": model,
    "messages": [
      {
        "role": "user",
        "content": prompt,
      }
    ],
    "temperature": temperature,
    "stream": do_stream,
  }
  if functions is not None:
    params["functions"] = functions
    # params['call'] = "auto"

    completion = openai.ChatCompletion.create(**params)
  if do_stream:
    ret = ""
    try:
      for chunk in completion:
        delta = chunk["choices"][0]["delta"]
        if "content" in delta:
          if do_print:
            print(delta["content"], end="", flush=True)
          ret += delta["content"]
    except requests.exceptions.ChunkedEncodingError:
      pass
  else:
    if functions is None:
      ret = completion["choices"][0]["message"]["context"]
    else:
      ret = completion["choices"][0]["message"]["function_call"]
      if isinstance(ret["arguments"], str):
        ret["arguments"] = json.loads(ret["arguments"])
  return ret


def merge_overlapping_chunks(chunks, max_overlap_len=150):
  # for i, chunk in enumerate(chunks):
  #   print(f"---------- CHUNK {i} ----------")
  #   print(chunk)
  #   print(f"---------- END OF CHUNK {i} ----------")
  result = chunks[0]
  for i in range(len(chunks) - 1):
    chunk_suffix = chunks[i][-max_overlap_len:]
    x = []
    frac_denom = 6
    while not x and frac_denom > 3:
      logging.info(f"matching chunk {i} with fraction denominator {frac_denom}")
      x = fuzzysearch.find_near_matches(
        chunk_suffix,
        chunks[i + 1],
        max_l_dist=int(max_overlap_len / frac_denom),
      )
      frac_denom -= 1
    if not x:
      print(f"CHUNK {i}+1")
      print(chunks[i + 1])
      print("")
      print(f"CHUNK {i}")
      print(chunks[i])
      print("")
      print(f"CHUNK SUFFIX (end of {i})")
      print(chunk_suffix)
    assert len(x) == 1, x
    result += chunks[i + 1][x[0].end :]
  # print("----------- FINAL RESULT ---------")
  # print(result)
  # print("----------- END OF FINAL RESULT ---------")
  return result


def split_text_lines_by_max_tokens(text, model, chunk_size, chunk_overlap):
  splitter = TokenTextSplitter(
    model_name=model, chunk_size=int(chunk_size), chunk_overlap=int(chunk_overlap)
  )
  return splitter.split_text(text)


@logstack
def cleanup_monologue(text, model="gpt-3.5-turbo", temperature=0.7):
  emb = tiktoken.encoding_for_model(model)
  max_tokens = MODEL_MAX_TOKENS[model]
  avail_tokens = max_tokens - len(emb.encode(PROMPT_CLEANUP_MONOLOGUE)) - 100
  # the remaining tokens need to be able to encode chunk twice
  avail_tokens /= 2
  avail_tokens /= 2
  # text = re.sub("-\+", "-", text)
  chunks = split_text_lines_by_max_tokens(text, model, avail_tokens, avail_tokens / 4)
  logging.info(f"{len(chunks)} chunks")
  cleanups = []
  iters = 10
  for i, chunk in enumerate(chunks):
    cleanup = None
    for it in range(iters):
      logging.info(f"chunk {i+1}/{len(chunks)}, try {it+1}/{iters}")
      x = stream_single_completion(
        PROMPT_CLEANUP_MONOLOGUE.format(text=chunk), model, temperature, it
      )
      n = 1000
      print("-----")
      print(chunk[-n:])
      print("-----")
      print(x[-n:])
      print("-----")
      standardize = lambda x: x.replace(" ", "").replace("\n", "")
      len_ratio = 1.0 * len(standardize(x)) / len(standardize(chunk))
      logging.info(
        f"chunk {i+1}/{len(chunks)}, try {it+1}/{iters}, length ratio {len_ratio}"
      )
      if abs(len_ratio - 1) < 0.1:
        cleanup = x
        break
    assert cleanup is not None
    cleanups += [cleanup]
    if len(cleanups) > 1:
      cleanups = [merge_overlapping_chunks(cleanups)]
  if len(chunks) > 1:
    text_cleanup = merge_overlapping_chunks(cleanups)
  else:
    text_cleanup = cleanups[0]
  return text_cleanup


def merge_summaries(summaries, model="gpt-3.5-turbo-16k", temperature=0.7):
  if len(summaries) > 1:
    prompt = """
The following are summaries written about different parts of a single text.

Each summary is written without knowledge of other parts of the text,
as if the summarized text was complete. Consequently, summaries frequently contain
descriptions such as "The text begin with", etc.

Any two consecutive summaries overlap, e.g. summary 2 overlaps partially with summary 3, etc.

Align the summaries to figure out which portions ovelap.
Rewrite the parts into one contiguous whole, consolidating the overlapping portions,
and reframing one contiguous summary of a single text.

Preserve all details of the original summaries, do not skip details.


----


"""
    for sid, summary in enumerate(summaries):
      prompt += f"""


SUMMARY {sid+1}


"""
      prompt += summary
    summary = get_single_completion(prompt, model, temperature)
    for i, s in enumerate(summaries):
      print("")
      print(i)
      print(s)
    print("")
    print("final:")
    print(summary)
  else:
    summary = summaries[0]
  return summary


@logstack
def summarize(text, model="gpt-3.5-turbo-16k", temperature=0.7):
  if not text.strip():
    return '', ''
  emb = tiktoken.encoding_for_model(model)
  max_tokens = MODEL_MAX_TOKENS[model]
  avail_tokens = max_tokens - len(emb.encode(PROMPT_SUMMARY)) - 100
  avail_tokens /= 2 + 0.5

  chunks = split_text_lines_by_max_tokens(
    text, model, chunk_size=avail_tokens, chunk_overlap=avail_tokens / 2
  )
  logging.info(f"{len(chunks)} chunks")
  short_summaries = [
    get_single_completion(PROMPT_SUMMARY.format(text=chunk), model, temperature)
    for chunk in chunks
  ]
  long_summaries = [
    get_single_completion(
      [
        {
          "role": "user",
          "content": PROMPT_SUMMARY.format(text=chunk, summary=summary),
        },
        {"role": "assistant", "content": summary},
        {
          "role": "user",
          "content": "Great. Do it again, give more details from the original text.",
        },
      ],
      model,
      temperature,
    )
    for chunk, summary in zip(chunks, short_summaries)
  ]
  short_summary = merge_summaries(short_summaries)
  long_summary = merge_summaries(long_summaries)
  return short_summary, long_summary


@logstack
def short_summarize(text, model="gpt-3.5-turbo-0613", temperature=0.7):
  prompt = PROMPT_SHORTSUMMARY.format(text=text)
  return get_single_completion(prompt, model, temperature)


@logstack
def make_title(text, model="gpt-4", temperature=0.7):
  prompt = PROMPT_TITLE.format(text=text)
  return get_single_completion(prompt + text, model, temperature)


@logstack
def extract_keywords(text, model="gpt-3.5-turbo-0613", temperature=0.7, n=10, min_n=3):
  # TODO: support for longer text
  all_keywords = []
  keyword_orders = collections.defaultdict(list)
  for i in range(n):
    prompt = PROMPT_KEYWORDS.format(text=text)
    keywords = stream_single_completion(
      prompt + text,
      model,
      temperature,
      do_print=True,
      sequence_id=i,
      functions=FUNCTIONS_KEYWORDS,
    )["arguments"].get("keywords", [])
    keywords = [slugify.slugify(keyword) for keyword in keywords]
    all_keywords += keywords
    for i, keyword in enumerate(keywords):
      keyword_orders[keyword] += [i]
  keyword_order = {
    keyword: sum(orders) / len(orders) for keyword, orders in keyword_orders.items()
  }
  keyword_count = collections.Counter(all_keywords)
  all_keywords = sorted(
    set(all_keywords),
    key=lambda keyword: (-keyword_count[keyword], keyword_order[keyword]),
  )
  all_keywords = [
    keyword for keyword in all_keywords if keyword_count[keyword] >= min_n
  ]
  return all_keywords
