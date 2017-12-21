import six
from xnmt.vocab import Vocab

class Output(object):
  '''
  A template class to represent all output.
  '''
  def __init__(self, actions=None):
    ''' Initialize an output with actions. '''
    self.actions = actions or []

  def to_tokens(self):
    raise NotImplementedError('All outputs must implement to_tokens.')

class TextOutput(Output):
  def __init__(self, actions=None, vocab=None, score=None):
    self.actions = actions or []
    self.vocab = vocab
    self.score = score
    self.filtered_tokens = set([Vocab.SS, Vocab.ES])

  def to_tokens(self):
    map_func = lambda wi: self.vocab[wi] if self.vocab != None else str
    return six.moves.map(map_func, filter(lambda wi: wi not in self.filtered_tokens, self.actions))

class OutputProcessor(object):
  def process_outputs(self, outputs):
    raise NotImplementedError()

class PlainTextOutputProcessor(OutputProcessor):
  '''
  Handles the typical case of writing plain text,
  with one sent per line.
  '''
  def process_outputs(self, outputs):
    for output in outputs:
      output.tokentext = self.make_tok_string(output.to_tokens())
      output.plaintext = self.make_postproc_string(output.to_tokens())

  def postproc_file(self, filename_in, filename_out):
    """
    Postprocess an entire file
    """
    with io.open(filename_in, encoding=encoding) as stream_in:
      with io.open(filename_out, 'wt', encoding=encoding) as stream_out:
        for line in stream_in:
          stream_out.write(self.make_postproc_string(line.strip().split()) + u"\n")

  def does_postproc(self):
    """
    Defines whether the output of make_tok_string and make_postrpoc_string are different
    """
    return False

  def make_tok_string(self, word_list):
    """
    Make a string of tokens in plain text without any postprocessing
    """
    return u" ".join(word_list)

  def make_postproc_string(self, word_list):
    """
    Convert the output to plain text after postprocessing
    """
    return u" ".join(word_list)

class JoinedCharTextOutputProcessor(PlainTextOutputProcessor):
  '''
  Assumes a single-character vocabulary and joins them to form words;
  per default, unicode underscores '▁' (used by sentencepiece) are treated
  as word separating tokens
  '''
  def __init__(self, space_token=u"▁"):
    self.space_token = space_token

  def make_postproc_string(self, word_list):
    return u"".join(map(lambda s: u" " if s==self.space_token else s, word_list))

  def does_postproc(self):
    return True

class JoinedBPETextOutputProcessor(PlainTextOutputProcessor):
  '''
  Assumes a bpe-based vocabulary and outputs the merged words;
  per default, the '@' postfix indicates subwords that should be merged
  '''
  def __init__(self, merge_indicator=u"@@"):
    self.merge_indicator_with_space = merge_indicator + u" "

  def make_postproc_string(self, word_list):
    return u" ".join(word_list).replace(self.merge_indicator_with_space, u"")

  def does_postproc(self):
    return True

class JoinedPieceTextOutputProcessor(PlainTextOutputProcessor):
  '''
  Assumes a sentence-piece vocabulary and joins them to form words;
  space_token could be the starting character of a piece
  per default, the u'\u2581' indicates spaces
  '''
  def __init__(self, space_token=u"\u2581"):
    self.space_token = space_token

  def make_postproc_string(self, word_list):
    return u"".join(word_list).replace(self.space_token, u" ").strip()

  def does_postproc(self):
    return True
