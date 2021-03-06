
# data configuration
data_path =  "data/ctb8.0/dep/",
vocab_path = "data/ctb8.0/vocab",

word_vocab_file = "data/ctb8.0/vocab/word_vocab.txt"
tag_vocab_file = "data/ctb8.0/vocab/tag_vocab.txt"
head_vocab_file = "data/ctb8.0/vocab/head_vocab.txt"
rel_vocab_file = "data/ctb8.0/vocab/rel_vocab.txt"

encoding = "utf-8"

min_freq = 2

shuffle = True
batch_size = 64

val_ratio = 0.3

# model configuration
word_embed_dim = 100
tag_embed_dim = 100

lstm_hiddens = 400
lstm_num_layers = 3

mlp_arc_size = 500
mlp_rel_size = 100

dropout = 0.33

# trainer configuration
seed = 1314
cuda = True
epochs = 30

optimizer = 'Adam'
lr = 0.001

update_every = 4
print_every = 100
eval_every = 500
clip = 5
