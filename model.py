import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer
from torch.nn import TransformerDecoderLayer, TransformerDecoder, LayerNorm


class GPT1(nn.Module):
    def __init__(self, d_model=768, nhead=12, num_layers=12, dim_feedforward=3072, vocab_size=40000, device=None, dtype=None):
        layer_norm_eps = 1e-5
        dropout = 0.1
        activation = F.gelu 
        batch_first = False
        norm_first = False
        num_decoder_layers = num_layers
        bias = True
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model, **factory_kwargs)
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout,
                                                    activation, layer_norm_eps, batch_first, norm_first,
                                                    bias, **factory_kwargs)
        decoder_norm = LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)
        self.linear = nn.Linear(d_model, vocab_size, **factory_kwargs)
  
    def forward(self, src, tgt):
        src = self.embedding(src)
        tgt = self.embedding(tgt)
        out = self.decoder(tgt, src)
        out = self.linear(out)
        #print(self.generate_square_subsequent_mask(src))
        return F.log_softmax(out, dim=-1)

if __name__ == '__main__':
    sentence = ['usually , he would be tearing around the living room , playing with his toys .']
    tokenizer = Tokenizer.from_file('./byte-level-bpe.tokenizer.json')
    tokens = torch.as_tensor([tokenizer.encode(x).ids for x in sentence])

    print(tokens)
    model = GPT1(vocab_size=tokenizer.get_vocab_size())
    out = model(tokens)
