import torch
from torch.utils.data import Dataset
from nltk.util import ngrams


class NgramModel(torch.nn.Module):
    def __init__(self, vocab_size=-1, emb_size=256, window_size=5):
        super(NgramModel, self).__init__()
        self.window_size = window_size
        self.vocab_size = vocab_size
        self.embedding_layer = torch.nn.Embedding(self.vocab_size, emb_size)
        self.flatten = torch.nn.Flatten()
        self.hidden_layer1 = torch.nn.Linear(window_size * emb_size, 256)
        self.batch_norm1 = torch.nn.BatchNorm1d(256)
        self.hidden_layer2 = torch.nn.Linear(256, 512)
        self.batch_norm2 = torch.nn.BatchNorm1d(512)
        self.hidden_layer3 = torch.nn.Linear(512, 256)
        self.batch_norm3 = torch.nn.BatchNorm1d(256)
        self.hidden_layer4 = torch.nn.Linear(256, vocab_size)
        self.batch_norm4 = torch.nn.BatchNorm1d(vocab_size)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.activation = torch.nn.ReLU()

    def forward(self, input_tokens, temperature=1.0):
        y = self.embedding_layer(input_tokens)
        y = self.flatten(y)

        # window_size * emb_size -> 256
        y = self.hidden_layer1(y)
        y = self.batch_norm1(y)
        y = self.activation(y)

        # 256 -> 512
        y = self.hidden_layer2(y)
        y = self.batch_norm2(y)
        y = self.activation(y)

        # 512 -> 256
        y = self.hidden_layer3(y)
        y = self.batch_norm3(y)
        y = self.activation(y)

        # 256 -> vocab_size
        y = self.hidden_layer4(y)
        y = self.batch_norm4(y)
        logits_ = self.activation(y)

        # temperature
        logits_ = logits_ / temperature
        probs_ = self.softmax(logits_)
        return probs_, logits_


class GutenbergDataset(Dataset):
    def __init__(self, poems, ngram_size=5):
        self.ngram_size = ngram_size
        self.vocabulary = Vocabulary()
        words = [word for poem in poems for word in poem]
        most_common_words = self.calc_most_common(words)
        self.vocabulary.add_words(most_common_words)
        poems = self.filter_poems(poems)
        self.ngrams_ = self.build_ngrams(poems)

    def build_ngrams(self, poems):
        ngrams_ = []
        for poem in poems:
            poem = [self.vocabulary.sos] * self.ngram_size + poem
            ngram_ = ngrams(poem, self.ngram_size + 1)
            ngram_ = list(ngram_)
            ngrams_.extend(ngram_)
        return ngrams_

    def __len__(self):
        return len(self.ngrams_)

    def __getitem__(self, idx):
        x = [self.vocabulary.word_to_idx(word) for word in self.ngrams_[idx][:-1]]
        y = self.vocabulary.word_to_idx(self.ngrams_[idx][-1])
        x = torch.tensor(x, dtype=torch.long)
        y = torch.tensor(y, dtype=torch.long)
        return x, y

    def calc_most_common(self, words):
        word_freqs = dict()
        # Calculate word frequencies

        for token in words:
            if token in word_freqs.keys():
                word_freqs[token] += 1
            else:
                word_freqs[token] = 1

        # Get n most common words
        n = 3000
        most_common = list(reversed(sorted(word_freqs, key=word_freqs.get)))[:n]
        most_common = list(sorted(most_common))
        return most_common

    def filter_poems(self, poems):
        filtered_poems = []
        for poem in poems:
            unks = 0.
            for word in poem:
                word_idx = self.vocabulary.word_to_idx(word)
                if word_idx == self.vocabulary.word_to_idx(self.vocabulary.unk):
                    unks += 1
            if unks / len(poem) <= 0.1:
                filtered_poems.append(poem)
        num_poems = len(poems)
        num_removed = num_poems - len(filtered_poems)
        print("Removed {} poems of {}".format(num_removed, num_poems))
        return filtered_poems


class Vocabulary:
    def __init__(self, unk="<unk>", sos="<sos>", eol="<eol>", pad="<pad>"):
        self.unk = unk
        self.sos = sos
        self.eol = eol
        self.pad = pad

        self.special_words = [self.unk, self.sos, self.eol, self.pad]

        self._word_to_idx = dict()
        self._idx_to_word = dict()
        self.add_words(self.special_words)

        self.unk_idx = self.word_to_idx(self.unk)
        self.sos_idx = self.word_to_idx(self.sos)
        self.eol_idx = self.word_to_idx(self.eol)
        self.pad_idx = self.word_to_idx(self.pad)

    def add_words(self, words):
        # Eliminate duplicates
        words = set(words)
        words = sorted(list(words))
        for word in words:
            if word not in self._word_to_idx.keys():
                self.add_word(word)

    def add_word(self, word):
        curr_len = len(self._word_to_idx)
        idx = curr_len
        self._word_to_idx[word] = idx
        self._idx_to_word[idx] = word

    def sequence_to_indices(self, sequence):
        return [self.word_to_idx(word) for word in sequence]

    def indices_to_sequence(self, indices):
        return [self.idx_to_word(idx) for idx in indices]

    def word_to_idx(self, word):
        try:
            return self._word_to_idx[word]
        except KeyError:
            return self._word_to_idx[self.unk]

    def idx_to_word(self, idx):
        return self._idx_to_word[idx]

    def __getitem__(self, word):
        return self.word_to_idx(word)

    def __setitem__(self, word, idx):
        self._word_to_idx[word] = idx

    def __len__(self):
        return len(self._word_to_idx.keys())

    def get_char_vocab(self):
        char_vocab = Vocabulary()
        char_vocab.add_words(self.special_words)
        for word in self._word_to_idx.keys():
            char_vocab.add_words([word])
        return char_vocab

    def __str__(self):
        return str(self._word_to_idx)