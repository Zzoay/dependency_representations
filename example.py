
import sys
sys.path.append(".")

from load_model.tokenizer import Tokenizer
from load_model.dep_reprs import DepReprs
from common_module.vocab import Vocab
from common_module.utils import load_config, set_seed


if __name__ == '__main__':
    config = load_config("config/train.ini")
    data_config = config['data']
    model_config = config['model']
    trainer_config = config['trainer']

    set_seed(trainer_config["seed"])

    vocab = Vocab(data_config)

    dep_reprs = DepReprs(parser_file = "pretrained_model/parser.pt",
                         parser_vocab=vocab,
                         parser_config=model_config)
    
    tokenizer = Tokenizer(vocab=vocab)

    sentence = "这 是 测试 句子"

    tokens = tokenizer.tokenize_sentence(sentence)

    tokens = tokens.view((1, 4))

    x_syns = dep_reprs(tokens, seq_lens=[len(tokens)])

    print(x_syns)