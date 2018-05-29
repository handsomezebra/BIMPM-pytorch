import torch
from torch.autograd import Variable

from torchtext import data
from torchtext import datasets
from torchtext.vocab import GloVe

def tokenizer(text): # create a tokenizer function
    return text.split(' ')

class Paraphrase():
    def __init__(self, args):
        self.args = args

    def build_char_vocab(self):
        # for <pad> and <unk>
        self.char_vocab = {'<pad>': 0, '<unk>': 1}
        
        # for normal words
        for word in self.TEXT.vocab.itos[2:]:
            for c in list(word):
                if c not in self.char_vocab:
                    self.char_vocab[c] = len(self.char_vocab)

    def characterize_word(self, word_id, max_word_len):
        word_text = self.TEXT.vocab.itos[word_id]
        result = [self.char_vocab.get(ch, 1) for ch in word_text]
        result.extend([0] * (max_word_len - len(word_text)))
        return result
        
    def characterize(self, batch):
        """
        :param batch: Pytorch Variable with shape (batch, seq_len)
        :return: Pytorch Variable with shape (batch, seq_len, max_word_len)
        """
        batch = batch.tolist()
        max_word_len = max([max([len(self.TEXT.vocab.itos[w]) for w in words]) for words in batch])
                
        assert(max_word_len > 0)
                
        return [[self.characterize_word(w, max_word_len) for w in words] for words in batch]

    def get_features(self, batch):
        if self.args.data_type == 'SNLI':
            s1, s2 = 'premise', 'hypothesis'
        else:
            s1, s2 = 'q1', 'q2'
            
        s1, s2 = getattr(batch, s1), getattr(batch, s2)

        kwargs = {'p': s1, 'h': s2}

        char_p = Variable(torch.LongTensor(self.characterize(s1)))
        char_h = Variable(torch.LongTensor(self.characterize(s2)))

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
        self.LABEL.build_vocab(self.train)

        self.train_iter, self.dev_iter, self.test_iter = \
            data.BucketIterator.splits((self.train, self.dev, self.test),
                                       batch_sizes=[args.batch_size] * 3,
                                       device=torch.device('cuda', args.gpu) if args.gpu >= 0 else torch.device('cpu'),
                                       repeat=False)

        self.build_char_vocab()

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
        self.LABEL.build_vocab(self.train)

        sort_key = lambda x: data.interleave_keys(len(x.q1), len(x.q2))

        self.train_iter, self.dev_iter, self.test_iter = \
            data.BucketIterator.splits((self.train, self.dev, self.test),
                                       batch_sizes=[args.batch_size] * 3,
                                       device=torch.device('cuda', args.gpu) if args.gpu >= 0 else torch.device('cpu'),
                                       sort_key=sort_key,
                                       repeat=False)

        self.build_char_vocab()
