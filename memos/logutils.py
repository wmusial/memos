import functools
import logging
import contextlib
import time


class IndentedLogContext(contextlib.AbstractContextManager):
  def __init__(self, preamble, epilogue, spacing):
    self._preamble = preamble
    self._epilogue = epilogue
    self._spacing = spacing

  def __enter__(self):
    self._old_factory = logging.getLogRecordFactory()
    self._tbeg = time.time()
    if self._spacing:
      logging.info("")
    logging.info(f"┌ {self._preamble}")

    def record_factory(*args, **kwargs):
      record = self._old_factory(*args, **kwargs)
      record.msg = "│ " + record.msg
      return record

    logging.setLogRecordFactory(record_factory)
    if self._spacing:
      logging.info("")

  def __exit__(self, *exc_info):
    if exc_info[0] is None and self._spacing:
      logging.info("")
    logging.setLogRecordFactory(self._old_factory)
    if exc_info[0] is None:
      sec = time.time() - self._tbeg
      logging.info(f"└ {self._epilogue} in {sec} sec.")
      if self._spacing:
        logging.info("")


def logindent(preamble, epilogue=None, spacing=False):
  assert preamble is not None
  if epilogue is None:
    epilogue = preamble
  return IndentedLogContext(preamble, epilogue, spacing)


def logstack(func):
  @functools.wraps(func)
  def wrapper(*args, **kwargs):
    with logindent(f"{func.__name__} enter.", f"{func.__name__} done.", False):
      result = func(*args, **kwargs)
    return result

  return wrapper
