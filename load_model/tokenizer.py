
import torch


class Tokenizer():
    def __init__(self, vocab) -> None:
        self.vocab = vocab

    def tokenize_sentence(self, sentence, delimiter=' ', dtype=torch.Tensor):
        return self.tokenize_wordlist(sentence.split(delimiter), dtype)

    def tokenize_wordlist(self, sentence, dtype=torch.Tensor):
        unk_token = self.vocab.word2id['<unk>']
        tokens = []
        for word in sentence:
            token = self.vocab.word2id.get(word)
            if token is not None:
                tokens.append(token)
            else:
                tokens.append(unk_token)
        if dtype == torch.Tensor:
            return torch.tensor(tokens)
        return tokens

    # TODO tokenize word with batch data 
    def tokenize_batch(self, inputs):
        raise NotImplementedError