import torch
from torch import nn

import torchvision
import torchvision.transforms as transforms

from tqdm import tqdm

from transformers import ViTFeatureExtractor

# feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.dim_head = dim_head
        self.norm = nn.LayerNorm(dim)

        self.softmax = nn.Softmax(dim = -1)
        self.q = nn.Linear(dim, inner_dim, bias=False)
        self.k = nn.Linear(dim, inner_dim, bias=False)
        self.v = nn.Linear(dim, inner_dim, bias=False)

        self.proj = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):

        x = self.norm(x)
        b, c, h = x.shape
        q, k, v, = self.q(x).reshape(b, -1, c, self.dim_head), self.k(x).reshape(b, -1, c, self.dim_head), self.v(x).reshape(b, -1, c, self.dim_head)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale # dot product attention

        attn = self.softmax(dots) # attention weights

        out = torch.matmul(attn, v)
        out = out.reshape(b, c, -1)

        return self.proj(out)

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.1):
        super().__init__()
        
        self.attn = Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)
        self.ff = FeedForward(dim, mlp_dim, dropout=dropout)

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.ff(x)
        return x

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([TransformerBlock(dim, heads, dim_head, mlp_dim, dropout) for _ in range(depth)])
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class VisionTransformer(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = image_size
        patch_height, patch_width = patch_size
        self.patch_height, self.patch_width = patch_height, patch_width
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        print('num patches', num_patches)
        patch_dim = channels * patch_height * patch_width
        print('patch dim', patch_dim)
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.patch_embedding = nn.Linear(patch_dim, dim)

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def to_patch(self, x):
        batch_size, c, h, w = x.shape
        x = x.reshape(batch_size, -1, self.patch_height*self.patch_width*c) # flatten patches
        return x

    def forward(self, img):
        # x = self.to_patch_embedding(img)
        x = self.to_patch(img)
        x = self.patch_embedding(x)
        b, n, _ = x.shape

        cls_tokens = self.cls_token.repeat(b, 1, 1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        return self.mlp_head(x)

def train(model, optimizer, train_loader, device, epoch, log_freq=10):

    model.train()
    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_sample = 0
    total_step = 0
    p_bar = tqdm(train_loader)

    for img_batch, label in p_bar:

        optimizer.zero_grad()
        img_batch, label = img_batch.to(device), label.to(device)
        pred = model(img_batch)
        loss = loss_fn(pred, label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_sample += img_batch.size(0)
        total_correct += (pred.argmax(-1) == label).sum().item()
        total_step += 1

        if total_step % log_freq == 0:
            p_bar.set_description(f'EPOCH{epoch} | train loss: {(total_loss/total_step):.4f} | train acc: {(total_correct/total_sample):.4f}')
            raise Exception()
    return total_loss/total_step, total_correct/total_sample

def test(model, test_loader, device, epoch, log_freq=10):

    model.eval()
    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_sample = 0
    total_step = 0
    p_bar = tqdm(test_loader)

    for img_batch, label in p_bar:

        img_batch, label = img_batch.to(device), label.to(device)
        pred = model(img_batch)
        loss = loss_fn(pred, label)

        total_loss += loss.item()
        total_sample += img_batch.size(0)
        total_correct += (pred.argmax(-1) == label).sum().item()
        total_step += 1

        if total_step % log_freq == 0:
            p_bar.set_description(f'EPOCH{epoch} | test loss: {(total_loss/total_step):.4f} | test acc: {(total_correct/total_sample):.4f}')
    return total_loss/total_step, total_correct/total_sample

def get_dataloader(batch_size):
    transform = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.RandomCrop(256, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    transform_test = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.Places365(root='./data_256_standard ', small=True ,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=0)

    testset = torchvision.datasets.Places365(root='./data_256_standard', split='val', small=True,
                                        download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=0)

    return trainloader, testloader

def run(epochs=20, lr=0.0005, batch_size=64):

    # classes = ('plane', 'car', 'bird', 'cat', 'deer',
    #         'dog', 'frog', 'horse', 'ship', 'truck')

    train_loader, test_loader = get_dataloader(batch_size)

    model = VisionTransformer(
        image_size = (256, 256),
        patch_size = (32, 32),
        num_classes = 10177,
        dim = 768,
        depth = 12,
        heads = 8,
        mlp_dim = 2048,
        dropout = 0.1,
        emb_dropout = 0.1
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    best_test_acc = 0
    for e in range(epochs):
        train_loss, train_acc = train(model, optimizer, train_loader, device, epoch=e)
        test_loss, test_acc = test(model, test_loader, device, epoch=e)
        
        if test_acc >= best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), './ViT_places365.pt')
        
        scheduler.step()

    print('best testing accuracy', best_test_acc)

run()
