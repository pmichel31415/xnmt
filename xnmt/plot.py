import numpy as np
import matplotlib.pyplot as plt
import six
import pdb

def plot_attention(src_words, trg_words, attention_matrix, file_name=None):
  """This takes in source and target words and an attention matrix (in numpy format)
  and prints a visualization of this to a file.
  :param src_words: a list of words in the source
  :param trg_words: a list of target words
  :param attention_matrix: a two-dimensional numpy array of values between zero and one,
    where rows correspond to source words, and columns correspond to target words
  :param file_name: the name of the file to which we write the attention
  """
  fig, ax = plt.subplots()

  # put the major ticks at the middle of each cell
  ax.set_xticks(np.arange(attention_matrix.shape[1]) + 0.5, minor=False)
  ax.set_yticks(np.arange(attention_matrix.shape[0]) + 0.5, minor=False)
  ax.invert_yaxis()

  # label axes by words
  ax.set_xticklabels(trg_words, minor=False)
  ax.set_yticklabels(src_words, minor=False)
  ax.xaxis.tick_top()

  # draw the heatmap
  plt.pcolor(attention_matrix, cmap=plt.cm.Blues, vmin=0, vmax=1)
  plt.colorbar()

  if file_name != None:
    plt.savefig(file_name, dpi=100)
  else:
    plt.show()
  plt.close()


def plot_attention_continuous(src, trg_seq, attention_matrix, file_name=None):
  """This takes in source and target words and an attention matrix (in numpy format)
  and prints a visualization of this to a file.
  :param src: numpy matrix representing the input sequence of vectors
  :param trg_seq: target sequence
  :param attention_matrix: a two-dimensional numpy array of values between zero and one,
    where rows correspond to source embedding after encoder, and columns correspond to target words 
  :param file_name: the name of the file to which we write the attention
  """
  #remark: Alignement is not done directly from the source vectors, so the display
  #can be mistaken
  #fig, ax = plt.subplots()
  fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
  # put the major ticks at the middle of each cell
  ax2.set_xticks(np.arange(attention_matrix.shape[1]) + 0.5, minor=False)
  #ax.set_yticks(np.arange(attention_matrix.shape[0]) + 0.5, minor=False)
  ax2.invert_yaxis()

  # label axes by words for attention
  ax2.set_xticklabels(trg_seq, minor=False)
  ax2.xaxis.tick_top()
  ax2.yaxis.set_visible(False)    

  # plot matrix of input sequence
  ax1.xaxis.set_visible(False)
  ax1.yaxis.set_visible(False)
  #ax1.axis('off')  
  feat_im = ax1.imshow(src.T, interpolation='nearest', cmap=plt.cm.coolwarm, origin='upper')

  # draw the heatmap for attention
  att_im = plt.pcolor(attention_matrix, cmap=plt.cm.Blues, vmin=0, vmax=1)
  fig.colorbar(att_im, ax=ax2)
  #TODO Same shape for input seq and att matrix, remove border, stick plots
  #plt.subplots_adjust(wspace=0, hspace=0)
  if file_name != None:
    plt.savefig(file_name, dpi=100)
  else:
    plt.show()
  plt.close()
