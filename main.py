from model import GPT1
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer
from torch.nn import TransformerDecoderLayer, TransformerDecoder, LayerNorm
import torch.optim as optim
from torch.utils.data import TensorDataset, random_split
from datasets import load_dataset, Dataset
from torch.utils.data.sampler import Sampler
import logging
import os

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')

device = 'cuda'
batch_size = 24
checkpoint_dir = './checkpoint'


def train(model: nn.Module, train_generator, dev_generator, tokenizer):
    model.train()
    criterion = torch.nn.NLLLoss()
    optimizer = optim.Adam(model.parameters())

    print_batch = 100
    for epoch in range(1):
        running_loss = 0.0
        model.train()
        for i, data in enumerate(train_generator, 0):
            inputs, labels = data[:-1, :], data[1:, :]
            optimizer.zero_grad()
            outputs = model(inputs, labels)
            outputs = outputs.view(-1, tokenizer.get_vocab_size())
            labels = labels.flatten(0, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % print_batch == (print_batch - 1):
                logging.info(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 10:.3f}')
                running_loss = 0.0
        #print("epoch %d, test accurary: %f" % (epoch, evaluate(model, dev_generator)))

def evaluate(model: nn.Module, dev_generator):
    model.eval()
    sums = 0
    length = 0
    for x, y in dev_generator():
        yo = model(x)
        yo = torch.argmax(yo, 1)
        target = torch.argmax(y, 1)
        acc = torch.eq(yo, target)
        sums = sums + sum(acc) 
        length = length + len(acc)
    return sums / length

def batchify(generator, batch_size):
    i = 0
    out = None
    for data in generator():
        if i == 0:
            out = torch.unsqueeze(data, 1)
        else:
            out = torch.cat((out, torch.unsqueeze(data, 1)), -1)
        i += 1
        if i == batch_size:
            yield out
            i = 0
        
def preprocess(dataset, tokenizer, token_saved_file):
    if not os.path.exists(token_saved_file):
        tokens = dataset.map(lambda x: {'text': tokenizer.encode(x['text']).ids}, num_proc=16)
        logging.debug('dataset `%s`, encoded `%s`', dataset[0]['text'], tokens[0]['text'])
        tokens.save_to_disk(token_saved_file)
    else:
        tokens = Dataset.load_from_disk(token_saved_file)
    def batch_iterator():
        target_tokens = 512
        next_sample = []
        for token in tokens:
            to_concat = min(len(token['text']), target_tokens - len(next_sample))
            next_sample = next_sample + token['text'][:to_concat]
            if len(next_sample) == target_tokens:
                yield torch.tensor(next_sample, device=device)
                next_sample = token['text'][to_concat:]
    return batchify(batch_iterator, batch_size)


def load_dataset():
    tokenizer = Tokenizer.from_file('./byte-level-bpe.tokenizer.json')
    dev_tokens = checkpoint_dir + '/dev_tokens' 
    train_tokens = checkpoint_dir + '/train_tokens' 
    if not os.path.exists(dev_tokens):
        dataset = load_dataset("bookcorpus", split="train")
        #dataset = dataset.select(range(0,1000000))
        train_size = math.floor(0.7 * len(dataset)) 
        train_set = dataset.select(range(0, train_size))
        dev_set = dataset.select(range(train_size, len(dataset) - train_size))
    else:
        train_set = None
        dev_set = None
    train_generator = preprocess(train_set, tokenizer, train_tokens)
    dev_generator = preprocess(dev_set, tokenizer, dev_tokens)
    return train_generator, dev_generator, tokenizer
        

def main():
    train_generator, dev_generator, tokenizer = load_dataset()
    model = GPT1(device=device, vocab_size=tokenizer.get_vocab_size())
    logging.info('start training')
    train(model, train_generator, dev_generator, tokenizer)


if __name__ == '__main__':
    main()
