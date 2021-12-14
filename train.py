import argparse
import os

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.cuda.amp import autocast
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

import wandb
from loader import TextImageDataset
from rudalle import get_tokenizer, get_vae
from rudalle.dalle.fp16 import FP16Module
from rudalle.dalle.model import DalleModel
from rudalle.utils import seed_everything


def preprocess_vqgan(x):
    x = 2.*x - 1.
    return x


def reconstruct_with_vqgan(x, model):
    z, _, [_, _, _] = model.encode(x)
    print(f'VQGAN --- {model.__class__.__name__}: latent shape: {z.shape[2:]}')
    xrec = model.decode(z)
    return xrec


# constants
def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='Malevich')
    parser.add_argument('--pretrained', type=bool, default=True)
    parser.add_argument('--fp16', type=bool, default=True)
    parser.add_argument('--cache_dir', type=str, default='/tmp/rudalle')
    parser.add_argument('--image_text_folder', type=str, default='/home/samsepiol/DatasetWorkspace/CurrentDatasets/COCO')
    parser.add_argument('--resume_from', type=str, default='')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--learning_rate', type=int, default=3e-4)
    return parser.parse_args()

def init_rudalle_module(fp16=False, device='cpu', cache_dir='/tmp/rudalle'):
    if fp16 and device == 'cpu':
        print('Warning! Using both fp16 and cpu doesnt support. You can use cuda device or turn off fp16.')
    model = DalleModel(device=device, 
        num_layers=6,
        hidden_size=768,
        num_attention_heads=6,
        embedding_dropout_prob=0.1,
        output_dropout_prob=0.1,
        attention_dropout_prob=0.1,
        image_tokens_per_dim=32,
        text_seq_length=128,
        cogview_sandwich_layernorm=True,
        cogview_pb_relax=True,
        vocab_size=8192 + 128,
        image_vocab_size=8192)
    if fp16:
        model = FP16Module(model)
    model.eval()
    model = model.to(device)
    return model


def main():
    _args = _parse_args()
    _device = 'cuda' if torch.cuda.is_available() else 'cpu'

    assert os.path.exists(_args.image_text_folder), f'{_args.image_text_folder} does not exist'
    assert os.path.exists(_args.cache_dir), f'{_args.cache_dir} does not exist'

    os.makedirs("./outputs", exist_ok=True)
    os.makedirs("./training", exist_ok=True)

    seed_everything(42)

    yttm_tokenizer = get_tokenizer(path='/home/samsepiol/CommonCheckpoints/yttm/allcaps.bpe', cache_dir='/home/samsepiol/CommonCheckpoints/rudalle')

    pretrained_vae = get_vae(dwt=True).to(_device).float().requires_grad_(False)
    pretrained_vae.eval()

    dalle = init_rudalle_module(fp16=False, device=_device, cache_dir=_args.cache_dir)
    if len (_args.resume_from) > 0:
        print(f'Loading model from {_args.resume_from}')
        checkpoint = torch.load(os.path.join(_args.cache_dir, _args.resume_from), map_location='cpu')
        dalle.load_state_dict(checkpoint)
    dalle = dalle.to(_device).train()

    text_seq_length = dalle.get_param('text_seq_length')
    total_seq_length = dalle.get_param('total_seq_length')
    device = dalle.get_param('device')

    optimizer = torch.optim.AdamW(dalle.parameters(), lr=_args.learning_rate, weight_decay=1e-4, amsgrad=True, betas=(0.9, 0.96))
    dataset = TextImageDataset(_args.image_text_folder, text_len=text_seq_length, image_size=256, truncate_captions=True, resize_ratio=0.75, tokenizer=yttm_tokenizer)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=_args.batch_size, shuffle=False, num_workers=0)
    tqdm.write(f'{len(dataset)} images in dataset')
    wandb.init(project='rudalle')


    for idx, (texts, text_input_ids, images) in tqdm(enumerate(dataloader)):
        with autocast(enabled=True, dtype=torch.float16, cache_enabled=True):
            images = images.to(device)
            text_input_ids = text_input_ids.to(device)
            attention_mask = torch.tril(torch.ones((1, 1, total_seq_length, total_seq_length), device=device))
        with autocast(enabled=True, dtype=torch.float32): # vae is unstable, so we use float32 
            image_input_ids = pretrained_vae.get_codebook_indices(images).to(device)
        with autocast(enabled=True, dtype=torch.float16, cache_enabled=True):
            input_ids = torch.cat((text_input_ids, image_input_ids), dim=1)
            loss, loss_values = dalle(input_ids, attention_mask, return_loss=True)
        
        loss.backward()
        clip_grad_norm_(dalle.parameters(), 1.)
        optimizer.step()
        optimizer.zero_grad()

        tqdm.write(f'{idx}/{len(dataset)}\tloss: {loss.item():.4f}')
        wandb.log({'loss': loss.item()})

        if idx % 1000 == 0:
            torch.save(dalle.state_dict(), f'training/dalle_step{idx}.pt')
            # _pil_images, _scores = generate_images(texts[0], yttm_tokenizer, dalle, pretrained_vae, 2048, 0.995, 3)
            # save all pil images
            # for batch_idx, pil_image in enumerate(_pil_images):
            #     tqdm.write(f'{batch_idx}/{len(_pil_images)}')
            #     pil_image.save(f'outputs/{idx}_{batch_idx}.png')
            #     wandb.log({ "image": wandb.Image(pil_image, caption=f'{idx}_{batch_idx}_{_scores[batch_idx]}') })


if __name__ == '__main__':
    main()
