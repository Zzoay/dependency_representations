
import torch
from torch import nn

from load_model.scale_mix import ScalarMix
from train_model.parser import DependencyParser
from common_module.modules import NonLinear


class DepReprs(nn.Module):
    def __init__(self, parser_file, parser_vocab, parser_config) -> None:
        super(DepReprs, self).__init__()

        word_embed_dim:int = parser_config["word_embed_dim"]
        tag_embed_dim:int = parser_config["tag_embed_dim"]

        lstm_hiddens:int = parser_config["lstm_hiddens"]

        mlp_arc_size:int = parser_config["mlp_arc_size"]
        mlp_rel_size:int = parser_config["mlp_rel_size"]

        self.hidden_dims = 2 * lstm_hiddens

        self.parser = DependencyParser(vocab_size = parser_vocab.word_size, 
                                       tag_size = parser_vocab.tag_size, 
                                       rel_size = parser_vocab.rel_size, 
                                       config = parser_config)
        
        self.load_parser_parameters(parser_file)

        self.transform_emb = NonLinear(word_embed_dim + tag_embed_dim, self.hidden_dims, activation=nn.ReLU)

        parser_dim = 2 * lstm_hiddens
        # self.transformer_lstm = nn.ModuleList([NonLinear(parser_dim, self.hidden_dims, activation=nn.ReLU)
        #                                         for i in range(lstm_layers)])
        self.transform_lstm = NonLinear(parser_dim, self.hidden_dims, activation=nn.ReLU)

        parser_mlp_dim = mlp_arc_size + mlp_rel_size
        self.transform_dep = NonLinear(parser_mlp_dim, self.hidden_dims, activation=nn.ReLU)
        self.transform_head = NonLinear(parser_mlp_dim, self.hidden_dims, activation=nn.ReLU)

        self.synscale = ScalarMix(mixture_size=4)
    
    def load_parser_parameters(self, parser_file: str) -> None:
        parser_state_dict = torch.load(parser_file)
        self.parser.load_state_dict(parser_state_dict)

    def forward(self, input_tokens, seq_lens) -> torch.Tensor:
        # batch_size, seq_len, embed_dim
        embed_word = self.parser.word_emb(input_tokens)   # type: ignore
        extra_tag_zeros = torch.zeros_like(embed_word)  # set tag embedding as zeros
        embed_word = torch.cat([embed_word, extra_tag_zeros], dim=2)

        lstm_output, _ = self.parser.bilstm(embed_word, seq_lens)  # type: ignore

        all_dep = self.parser.mlp_arc_dep(lstm_output)  # type: ignore
        all_head = self.parser.mlp_arc_head(lstm_output)  # type: ignore

        x_syns = []
        x_syn_emb = self.transform_emb(embed_word)
        x_syns.append(x_syn_emb)

        # for layer in range(self.parser_lstm_layers):
        #     x_syn_lstm = self.transformer_lstm[layer].forward(synxs[syn_idx])  
        #     x_syns.append(x_syn_lstm)
        # TODO DepSAWR use the every layers of LSTM, here only use the last layer output

        x_syn_lstm = self.transform_lstm(lstm_output)  
        x_syns.append(x_syn_lstm)

        x_syn_dep = self.transform_dep(all_dep)
        x_syns.append(x_syn_dep)

        x_syn_head = self.transform_head(all_head)
        x_syns.append(x_syn_head)

        x_syn = self.synscale(x_syns)  # (batch_size, seq_len, 2 * lstm_hiddens)

        return x_syn
