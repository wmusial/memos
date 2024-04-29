import collections
from openai import OpenAI

import tiktoken
import json
import logging
import slugify
import requests
import yaml
from memos.logutils import logstack
from cache_decorator import Cache

CHAT_MODELS = [
  "gpt-3.5-turbo",
  "gpt-3.5-turbo-16k",
  "gpt-4",
  "gpt-4-32k",
  "gpt-3.5-turbo-0613",
  "gpt-4-0613",
  "gpt-4-turbo-2024-04-09",
]
FUNCTION_MODELS = ["gpt-4-0613", "gpt-3.5-turbo-0613"]

MODEL_MAX_TOKENS = {
  "gpt-3.5-turbo": 4096,
  "gpt-3.5-turbo-16k": 16384,
  "gpt-4": 8192,
  "gpt-4-32k": 32768,
  "gpt-4-turbo-2024-04-09": 128000,
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
  return keys["secret"], keys["organization"]


@Cache()
def get_single_completion(secret, organization, prompt, model, temperature, sequence_id=0):
  client = OpenAI(api_key=secret, organization=organization)
  if model in CHAT_MODELS:
    if isinstance(prompt, list):
      completion = client.chat.completions.create(model=model,
      messages=prompt,
      temperature=temperature)
    else:
      completion = client.chat.completions.create(model=model,
      messages=[
        {
          "role": "user",
          "content": prompt,
        }
      ],
      temperature=temperature)
    ret = completion.choices[0].message.content
  else:
    completion = client.completions.create(model=model,
    prompt=prompt)
    ret = completion.choices[0].text
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

    completion = client.chat.completions.create(**params)
  if do_stream:
    ret = ""
    try:
      for chunk in completion:
        delta = chunk.choices[0].delta
        if "content" in delta:
          if do_print:
            print(delta["content"], end="", flush=True)
          ret += delta["content"]
    except requests.exceptions.ChunkedEncodingError:
      pass
  else:
    if functions is None:
      ret = completion.choices[0].message.context
    else:
      ret = completion.choices[0].message.function_call
      if isinstance(ret["arguments"], str):
        ret["arguments"] = json.loads(ret["arguments"])
  return ret


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
    summary = get_single_completion(secret, organization, prompt, model, temperature)
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
def summarize(keys_file_path, text, model="gpt-4-turbo-2024-04-09", temperature=0.7):
  secret, organization = load_keys(keys_file_path)
  if not text.strip():
    return '', ''
  emb = tiktoken.encoding_for_model(model)
  max_tokens = MODEL_MAX_TOKENS[model]
  prompt = PROMPT_SUMMARY.format(text=text)
  prompt_len = len(emb.encode(prompt))
  logging.info(f"prompt tokens: {prompt_len}")
  assert prompt_len < max_tokens
  short_summary = get_single_completion(secret, organization, prompt, model, temperature)
  # long_summaries = [
  #   get_single_completion(secret, organization,
  #     [
  #       {
  #         "role": "user",
  #         "content": PROMPT_SUMMARY.format(text=chunk, summary=summary),
  #       },
  #       {"role": "assistant", "content": summary},
  #       {
  #         "role": "user",
  #         "content": "Great. Do it again, give more details from the original text.",
  #       },
  #     ],
  #     model,
  #     temperature,
  #   )
  #   for chunk, summary in zip(chunks, short_summaries)
  # ]
  # short_summary = merge_summaries(short_summaries)
  # long_summary = merge_summaries(long_summaries)
  return short_summary #, long_summary


@logstack
def short_summarize(text, model="gpt-3.5-turbo-0613", temperature=0.7):
  prompt = PROMPT_SHORTSUMMARY.format(text=text)
  return get_single_completion(secret, organization, prompt, model, temperature)


@logstack
def make_title(keys_file_path, text, model="gpt-4", temperature=0.7):
  secret, organization = load_keys(keys_file_path)
  prompt = PROMPT_TITLE.format(text=text)
  return get_single_completion(secret, organization, prompt + text, model, temperature)


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
