#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn

import numpy as np
import math
from transformers import ViTModel
from dataloader import get_loader
from torchvision import transforms
import pickle
from build_vocab import Vocabulary
from tqdm import tqdm
import argparse

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224')
        self.hid_dim=768
        self.proj = nn.Linear(768, 768)
        
    def forward(self, src):
        #return = [batch size, patch len, hid dim]
        return self.proj(self.vit(pixel_values=src).last_hidden_state)

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()
        
        assert hid_dim % n_heads == 0
        
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        
        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)
        
        self.fc_o = nn.Linear(hid_dim, hid_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)
        
    def forward(self, query, key, value, mask = None):
        
        batch_size = query.shape[0]
        
        #query = [batch size, query len, hid dim]
        #key = [batch size, key len, hid dim]
        #value = [batch size, value len, hid dim]
                
        """
        ToDo: Encode your Q, K, V
            #Q = [batch size, query len, hid dim]
            #K = [batch size, key len, hid dim]
            #V = [batch size, value len, hid dim]
            e.g. 
                Q = self.fc_q(query) (batch_size, query len, hid dim)
                ....
        """
                
        """
        ToDo: Separate dimensions for each attention-head
        Result:
            #Q = [batch size, n heads, query len, head dim]
            #K = [batch size, n heads, key len, head dim]
            #V = [batch size, n heads, value len, head dim]
            e.g. 
                Q = torch.reshape(Q, (batch_size, self.n_heads, Q.size(1), self.head_dim))
                ...
        """
                
        """
        ToDo: Compute Attention Weight for each head (We named it as energy but you can call it whatever you want)
        energy = [batch size, n heads, query len, key len]
        hint: remember to divide energy with self.scale
        e.g.
            energy = torch.matmul(???, K.permute(?, ?, ?, ?)) ???
        """
        
        """
        ToDo: Mask unused tokens with variable: mask and Normalize attention weight
        hint: you should the attention weight of mask unused tokens to -inf (or you can mask it with a very negative number like: -1e10)
        hint: softmax function 
        #attention = [batch size, n heads, query len, key len]
        if mask is not None:
            energy = energy.masked_fill(mask == ???, ???)
        attention = ????(???, dim = -1) # apply softmax on energy
        """
        
        """
        ToDo:  multiply attention weight to V
        # x = ..... (x is the weighted sum of V)
        #x = [batch size, n heads, query len, head dim]
        x = torch.matmul(???, ???)
        """
        
        """
        We wrote these lines for you, contatenating separated heads together and output.
        You only need to finish the above parts (+ DecoderLayer class)  
        """
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, -1, self.hid_dim)
        #x = [batch size, query len, hid dim]
        
        x = self.fc_o(x)
        #x = [batch size, query len, hid dim]
        
        return x, attention

class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()
        
        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        
        x = self.dropout(torch.relu(self.fc_1(x)))
        x = self.fc_2(x)
        return x

class Decoder(nn.Module):
    def __init__(self, output_dim, hid_dim, n_layers, n_heads, pf_dim, dropout, device, max_length = 100):
        super().__init__()
        
        self.device = device
        
        self.tok_embedding = nn.Embedding(output_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)
        # self.pos_encoding = PositionalEncoding(hid_dim, max_length)
        
        self.layers = nn.ModuleList([DecoderLayer(hid_dim, n_heads, pf_dim, dropout, device) for _ in range(n_layers)])
        
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
        
    def forward(self, trg, enc_src, trg_mask, src_mask):
        
        #trg = [batch size, trg len]
        #enc_src = [batch size, src len, hid dim]
        #trg_mask = [batch size, 1, trg len, trg len]
        #src_mask = [batch size, 1, 1, src len]
                
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        
        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        trg = self.dropout((self.tok_embedding(trg) * self.scale) + self.pos_embedding(pos))

        for layer in self.layers:
            trg, attention = layer(trg, enc_src, trg_mask, src_mask)
        output = self.fc_out(trg)
            
        return output, attention

class DecoderLayer(nn.Module):
    def __init__(self, 
                 hid_dim, 
                 n_heads, 
                 pf_dim, 
                 dropout, 
                 device):
        super().__init__()
        
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.enc_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.encoder_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, pf_dim, dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, trg, enc_src, trg_mask, src_mask):
        
        #trg = [batch size, trg len, hid dim]
        #enc_src = [batch size, src len, hid dim]
        #trg_mask = [batch size, 1, trg len, trg len]
        #src_mask = [batch size, 1, 1, src len]
        
        """
        ToDo: 
            1. Call your implemented multi-head attention for self-attention (self.self_attention) & Layer Norm
            2. Cross attention with encoder output & layer norm
            3. positionwise feed forward & layer norm
        Hint: you can refer transformer block image in README.md
        don't forget the residual connection~~
        """
        
        return trg, attention

class Img2Seq(nn.Module):
    def __init__(self, encoder, decoder, trg_pad_idx, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.trg_pad_idx = trg_pad_idx
        self.device = device
        
    def make_src_mask(self, src):
        
        src_mask = torch.ones(src.size(0), src.size(1)).unsqueeze(1).unsqueeze(2).to(src.device)
        return src_mask
    
    def make_trg_mask(self, trg):
        
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device = self.device)).bool()
        trg_mask = trg_pad_mask & trg_sub_mask

        return trg_mask

    def forward(self, src, trg):
                
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src)
        src_mask = self.make_src_mask(enc_src)    
        output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)
        
        return output, attention

def train(model, iterator, optimizer, criterion, clip, log_step=10):
    
    model.train()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    epoch_loss = 0
    
    p_bar = tqdm(enumerate(iterator), total=len(iterator))
    
    for i, batch in p_bar:
        
        src, tgt, l = batch
        src = src.to(device)
        trg = tgt.to(device)
        
        optimizer.zero_grad()
        
        output, _ = model(src, trg[:,:-1])

        output_dim = output.shape[-1]
            
        output = output.contiguous().view(-1, output_dim)
        trg = trg[:,1:].contiguous().view(-1)
        
        loss = criterion(output, trg)
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        
        if i+1 % log_step == 0:
            p_bar.set_description(f'STEP {i+1} | Loss: {(epoch_loss/(i+1)):.3f} | Train PPL: {math.exp(epoch_loss/(i+1)):7.3f}')
        
    return epoch_loss / len(iterator)

def inference(src, tgt, model, device, vocab, max_len = 100):
    
    '''
    src: single image from dataloader
    tgt: single sequence of word_id from dataloader
    model: Img2Seq model
    max_len: max length of decoded sentence (int)
    '''

    model.eval()
    
    gold = tgt.tolist()
    gold_sent = [vocab.idx2word[i] for i in gold]
    trg_indexes = [vocab.word2idx['<start>']]
    
    enc_src = model.encoder(src)
    src_mask = model.make_src_mask(enc_src)
    
    for i in range(max_len):

        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)

        trg_mask = model.make_trg_mask(trg_tensor)
        
        with torch.no_grad():
            output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)
        
        pred_token = output.argmax(2)[:,-1].item()
        
        trg_indexes.append(pred_token)

        if pred_token == vocab.word2idx['<end>']:
            break
    
    trg_tokens = [vocab.idx2word[i] for i in trg_indexes]
    
    print('gold sent', gold_sent)
    print('pred sent', trg_tokens)

    return trg_tokens[1:], attention

def main(args):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([ 
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                            (0.229, 0.224, 0.225))])

    with open('data/vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)

    train_loader = get_loader(args.image_dir, args.caption_path, vocab, transform, args.batch_size, shuffle=True, num_workers=args.num_workers)

    print('vocab size',len(vocab))
    enc = Encoder()
    dec = Decoder(len(vocab), args.hidden_size, args.dec_layers, args.num_heads, args.hidden_size, args.dropout, device)
    model = Img2Seq(enc, dec, vocab.word2idx['<pad>'], 'cuda')

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f'The model has {count_parameters(model):,} trainable parameters')

    optimizer = torch.optim.Adam(model.parameters(), lr = args.learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index = 0)
    model = model.to(device)

    for epoch in range(args.num_epochs):
        
        train_loss = train(model, train_loader, optimizer, criterion, args.clip)
        torch.save(model.state_dict(), f'{args.model_path}/ViT_captioning_epoch{epoch}.pt')
        print(f'EPOCH {epoch}\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')

    # inference_loader = get_loader(args.image_dir, args.caption_path, vocab, transform, 1, shuffle=True, num_workers=args.num_workers)
    # for img, tgt in inference_loader:
    #     inference(img, tgt, model, device, vocab, args.max_len)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='models/' , help='path for saving trained models')
    parser.add_argument('--crop_size', type=int, default=224 , help='size for randomly cropping images')
    parser.add_argument('--vocab_path', type=str, default='data/vocab.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--image_dir', type=str, default='data/resized2014', help='directory for resized images')
    parser.add_argument('--caption_path', type=str, default='data/annotations/captions_train2014.json', help='path for train annotation json file')
    parser.add_argument('--log_step', type=int , default=10, help='step size for prining log info')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--max_len', type=int, default=100)

    # Model parameters
    parser.add_argument('--hidden_size', type=int , default=768, help='dimension of lstm hidden states')
    parser.add_argument('--dec_layers', type=int , default=6, help='number of decoder layers in transformer')
    parser.add_argument('--num_heads', type=int, default=8, help='amount of attention heads')
    parser.add_argument('--clip', type=int, default=1, help='gradient clipping value')
    
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=3e-5)
    args = parser.parse_args()
    main(args)
