from PIL import Image
from captioning import Seq2Seq, Encoder, Decoder, inference
import matplotlib.pyplot as plt
from torchvision import transforms
import numpy as np
import argparse
import pickle
import torch
from build_vocab import Vocabulary

def plot_attention(image, result, attention_plot):
    
    result = result[1:]
    print(result)
    temp_image = image
    attention_plot = attention_plot.squeeze(0).transpose(0,1).cpu().numpy()
    fig = plt.figure(figsize=(15, 15))
    fig.subplots_adjust(top = 0.8, bottom=0.01, hspace=1.5, wspace=0.4)
    len_result = len(result)
    for i in range(len_result):
        temp_att = np.resize(attention_plot[i], (8, 8))
        grid_size = max(np.ceil(len_result/2), 2)
        ax = fig.add_subplot(grid_size, grid_size, i+1)
        ax.set_title(result[i], fontsize=12)
        img = ax.imshow(temp_image)
        ax.imshow(temp_att, cmap='gray', alpha=0.6, extent=img.get_extent())
    plt.savefig('test.png')
#     plt.tight_layout()
#     plt.show()

def sample_image(image_path):
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                            (0.229, 0.224, 0.225))])
    
    image = Image.open(image_path)
    raw_image = image.resize([224, 224], Image.LANCZOS)
    
    image = transform(image).unsqueeze(0)
    return image, raw_image

def main(args):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open('data/vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)

    enc = Encoder()
    dec = Decoder(len(vocab), args.hidden_size, args.dec_layers, args.num_heads, args.hidden_size, args.dropout, device)
    model = Seq2Seq(enc, dec, vocab.word2idx['<pad>'], 'cuda')
    model.load_state_dict(torch.load(args.model_path))
    src, raw_image = sample_image(args.image_path)

    src = src.to(device)
    model = model.to(device)
    enc_src = model.encoder(src)
    src_mask = model.make_src_mask(enc_src)
    trg_indexes = [vocab.word2idx['<start>']]

    for i in range(100):

        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)
        trg_mask = model.make_trg_mask(trg_tensor)
        with torch.no_grad():
            output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)
        pred_token = output.argmax(2)[:,-1].item()
        trg_indexes.append(pred_token)
        if pred_token == vocab.word2idx['<end>']:
            break

    decoded_sent = [vocab.idx2word[i] for i in trg_indexes]
    plot_attention(raw_image, decoded_sent, attention)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, default='./data/train2014/COCO_train2014_000000581921.jpg')
    parser.add_argument('--model_path', type=str, default='models/ViT_captioning_epoch4.pt' , help='path for saving trained models')
    parser.add_argument('--crop_size', type=int, default=224 , help='size for randomly cropping images')
    parser.add_argument('--vocab_path', type=str, default='data/vocab.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--image_dir', type=str, default='data/resized2014', help='directory for resized images')
    parser.add_argument('--caption_path', type=str, default='data/annotations/captions_train2014.json', help='path for train annotation json file')
    parser.add_argument('--log_step', type=int , default=10, help='step size for prining log info')
    
    # Model parameters
    parser.add_argument('--hidden_size', type=int , default=768, help='dimension of lstm hidden states')
    parser.add_argument('--dec_layers', type=int , default=6, help='number of decoder layers in transformer')
    parser.add_argument('--num_heads', type=int, default=8, help='amount of attention heads')
    parser.add_argument('--clip', type=int, default=1, help='gradient clipping value')
    parser.add_argument('--dropout', type=float,default=0.1)
    
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    args = parser.parse_args()
    main(args)
    # COCO_val2014_000000436138.jpg
    # COCO_val2014_000000436111.jpg
    # COCO_val2014_000000435937.jpg




