
import sys
sys.path.append(".")

import torch
from torch.utils.data import DataLoader

from train_model.data_helper import CTBDataset,SortPadCollator, train_val_split
from train_model.trainer import MyTrainer
from train_model.parser import DependencyParser
from train_model.metrics import arc_rel_loss, uas_las
from common_module.vocab import Vocab
from common_module.utils import load_config, set_seed


if __name__ == "__main__":
    data_path = "data/ctb8.0/dep/"
    vocab_file = "data/ctb8.0/vocab.txt"

    config = load_config("config/train.ini")
    data_config = config['data']
    model_config = config['model']
    trainer_config = config['trainer']

    set_seed(trainer_config["seed"])

    vocab = Vocab(data_config)
    dataset = CTBDataset(vocab, data_config)

    train_dataset, val_dataset = train_val_split(dataset, data_config["val_ratio"])

    sp_collator = SortPadCollator(sort_key=lambda x:x[5], ignore_indics=5)   

    train_iter = DataLoader(dataset=train_dataset,  
                            batch_size=data_config["batch_size"], 
                            shuffle=data_config["shuffle"], 
                            collate_fn=sp_collator)

    val_iter = DataLoader(dataset=val_dataset,  
                          batch_size=data_config["batch_size"], 
                          shuffle=data_config["shuffle"], 
                          collate_fn=sp_collator)

    model  = DependencyParser(vocab_size=vocab.word_size, 
                              tag_size=vocab.tag_size, 
                              rel_size=vocab.rel_size, 
                              config=model_config)
    
    trainer = MyTrainer(loss_fn=arc_rel_loss, metrics_fn=uas_las, config=trainer_config)

    trainer.train(model=model, train_iter=train_iter, val_iter=val_iter)
    trainer.eval(model=model, eval_iter=val_iter)
    
    torch.save(model.state_dict(), "parser.pt")
    print("finished.")