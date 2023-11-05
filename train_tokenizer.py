from tokenizers import ByteLevelBPETokenizer
from datasets import load_dataset, Dataset
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers


def train_tokenizer():
    # Build a tokenizer
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.normalizer = normalizers.Lowercase()

    # Initialize a dataset
    dataset = load_dataset("bookcorpus", split="train")

    # Build an iterator over this dataset
    def batch_iterator():
        batch_size = 1000
        for batch in dataset.iter(batch_size=batch_size):
            yield batch["text"]
    
    # And finally train
    trainer = trainers.BpeTrainer(vocab_size=40000)
    #dataset = {'train': Dataset.from_list([{'text': "hello world"}])}
    tokenizer.train_from_iterator(batch_iterator(), trainer, length=len(dataset))

    tokenizer.save("byte-level-bpe.tokenizer.json", pretty=True)
    print(tokenizer.encode('usually , he would be tearing around the living room , playing with his toys .').tokens)
    print(tokenizer.encode('but just one look at a minion sent him practically catatonic .').tokens)

if __name__ == '__main__':
    #train_tokenizer()
    tokenizer = Tokenizer.from_file('./byte-level-bpe.tokenizer.json')
    print(tokenizer.encode('usually , he would be tearing around the living room , playing with his toys .').tokens)

