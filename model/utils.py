
import torch
from torch.autograd import Variable

from collections import Counter

from torchtext import data
from torchtext import datasets
from torchtext.vocab import GloVe, Vectors, Vocab

def tokenizer(text): # create a tokenizer function
    return text.split(' ')
    
class GloVeChar(Vectors):
    url = {
        '840B': 'https://raw.githubusercontent.com/minimaxir/char-embeddings/master/glove.840B.300d-char.txt',
    }

    def __init__(self, name='840B', dim=300, **kwargs):
        url = self.url[name]
        name = 'glove.{}.{}d-char.txt'.format(name, str(dim))
        super(GloVeChar, self).__init__(name, url=url, **kwargs)

class Paraphrase():
    def __init__(self, args):
        self.args = args

    def build_char_vocab(self):
        char_counter = Counter()
        for word, count in self.TEXT.vocab.freqs.items():
            for ch in word:
                char_counter[ch] += count
        
        self.char_vocab = Vocab(char_counter, specials=['<unk>', '<pad>'], vectors = GloVeChar())

    def characterize_word(self, word_id, max_word_len):
        word_text = self.TEXT.vocab.itos[word_id]
        result = [self.char_vocab.stoi[ch] for ch in word_text[:max_word_len]]
        result.extend([1] * (max_word_len - len(result)))
        return result
        
    def characterize(self, batch, max_word_len):
        """
        :param batch: Pytorch Variable with shape (batch, seq_len)
        :return: Pytorch Variable with shape (batch, seq_len, max_word_len)
        """
        batch = batch.tolist()
        actual_max_word_len = max([max([len(self.TEXT.vocab.itos[w]) for w in words]) for words in batch])
        
        if max_word_len <= 0:
            max_word_len = actual_max_word_len
        else:
            max_word_len = min(actual_max_word_len, max_word_len)
                
        assert(max_word_len > 0)
                
        return [[self.characterize_word(w, max_word_len) for w in words] for words in batch]

    def get_features(self, batch, max_sent_len = -1, max_word_len = -1):
        if self.args.data_type == 'SNLI':
            s1, s2 = 'premise', 'hypothesis'
        else:
            s1, s2 = 'q1', 'q2'
            
        s1, s2 = getattr(batch, s1), getattr(batch, s2)
        
        if max_sent_len > 0:
            s1 = s1[:, :max_sent_len]
            s2 = s2[:, :max_sent_len]
        
        kwargs = {'p': s1, 'h': s2}

        char_p = Variable(torch.LongTensor(self.characterize(s1, max_word_len)))
        char_h = Variable(torch.LongTensor(self.characterize(s2, max_word_len)))

        if self.args.gpu > -1:
            char_p = char_p.cuda(self.args.gpu)
            char_h = char_h.cuda(self.args.gpu)

        kwargs['char_p'] = char_p
        kwargs['char_h'] = char_h
            
        return kwargs

class SNLI(Paraphrase):
    def __init__(self, args):
    
        super().__init__(args)
        
        self.TEXT = data.Field(batch_first=True, tokenize="spacy", lower=True)
        self.LABEL = data.Field(sequential=False, unk_token=None)

        self.train, self.dev, self.test = datasets.SNLI.splits(self.TEXT, self.LABEL)

        self.TEXT.build_vocab(self.train, self.dev, self.test, vectors=GloVe(name='840B', dim=300))
        self.build_char_vocab()
        self.LABEL.build_vocab(self.train)

        self.train_iter, self.dev_iter, self.test_iter = \
            data.BucketIterator.splits((self.train, self.dev, self.test),
                                       batch_sizes=[args.batch_size] * 3,
                                       device=torch.device('cuda', args.gpu) if args.gpu >= 0 else torch.device('cpu'),
                                       repeat=False)


class Quora(Paraphrase):
    def __init__(self, args):
        super().__init__(args)

        self.RAW = data.RawField()
        self.TEXT = data.Field(batch_first=True, tokenize=tokenizer)
        self.LABEL = data.Field(sequential=False, unk_token=None)

        self.train, self.dev, self.test = data.TabularDataset.splits(
            path='.data/quora',
            train='train.tsv',
            validation='dev.tsv',
            test='test.tsv',
            format='tsv',
            fields=[('label', self.LABEL),
                    ('q1', self.TEXT),
                    ('q2', self.TEXT),
                    ('id', self.RAW)])

        self.TEXT.build_vocab(self.train, self.dev, self.test, vectors=GloVe(name='840B', dim=300))
        self.build_char_vocab()
        self.LABEL.build_vocab(self.train)

        sort_key = lambda x: data.interleave_keys(len(x.q1), len(x.q2))

        self.train_iter, self.dev_iter, self.test_iter = \
            data.BucketIterator.splits((self.train, self.dev, self.test),
                                       batch_sizes=[args.batch_size] * 3,
                                       device=torch.device('cuda', args.gpu) if args.gpu >= 0 else torch.device('cpu'),
                                       sort_key=sort_key,
                                       repeat=False)

