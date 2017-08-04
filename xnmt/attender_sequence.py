import dynet as dy
from decorators import TimeIt
from serializer import Serializable

class SequenceAttender(Serializable):
  def atten_sequence(self, src_sequence, trg_sequence):
    raise RuntimeError("Should call atten_sequence from the child class instead")

class StandardSequenceAttender(SequenceAttender):
  yaml_tag = u"!StandardSequenceAttender"

  def __init__(self, attender=None, lmbd=0.5):
    self.lmbd = lmbd
    self.attender = attender

  def atten_sequence(self, src_sequence, trg_sequence, full=True):
    if full:
      batch_size = trg_sequence.dim()[1]
      attention = []
      for i in range(batch_size):
        attention.append(self.calc_context(src_sequence,
                                           dy.pick_batch_elem(trg_sequence, i)))
      return dy.transpose(dy.concatenate(attention, d=0))
    else:
      return self.calc_context(src_sequence, trg_sequence)

  def calc_context(self, src, trg):
    attn = self.calc_attention(src, trg)
    dot = dy.transpose(src) * trg
    score = dy.cmult(attn, dot)
    return dy.sum_elems(score)

  def calc_attention(self, src, trg):
    self.attender.start_sent(src)
    trg = dy.transpose(trg)
    seq_len = trg.dim()[0][0]
    batch_size = src[0].dim()[1]
    att = []
    for j in range(seq_len):
      att.append(self.attender.calc_attention(dy.pick(trg, j), normalized=False))

    att = dy.concatenate_cols(att)
    # sum(alpha, d=0) == 1
    alpha = dy.softmax(att)
    # sum(beta, d=1) == 1
    beta = dy.transpose(dy.softmax(dy.transpose(att)))
    coef = self.lmbd * beta + (1-self.lmbd) * alpha

    return coef
