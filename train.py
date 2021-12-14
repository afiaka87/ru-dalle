from tqdm.auto import tqdm
import os

from rudalle import get_rudalle_model, get_tokenizer, get_vae
from rudalle.utils import seed_everything


import torchvision.transforms.functional as TF
import torchvision.transforms as T
import torch
import argparse
from loader import TextImageDataset

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
    parser.add_argument('--model_name', type=str, default='small')
    parser.add_argument('--pretrained', type=bool, default=True)
    parser.add_argument('--fp16', type=bool, default=True)
    parser.add_argument('--cache_dir', type=str, default='/tmp/rudalle')
    parser.add_argument('--image_text_folder', type=str, default='/home/samsepiol/DatasetWorkspace/CurrentDatasets/COCO')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--deepspeed', action='store_true')
    parser.add_argument('--deepspeed_config', type=str, default='deepspeed_config.yaml')
    return parser.parse_args()

def main():
    _args = _parse_args()
    _device = 'cuda' if torch.cuda.is_available() else 'cpu'


    assert os.path.exists(_args.image_text_folder), f'{_args.image_text_folder} does not exist'
    assert os.path.exists(_args.cache_dir), f'{_args.cache_dir} does not exist'

    yttm_tokenizer = get_tokenizer(path='/home/samsepiol/CommonCheckpoints/yttm/allcaps.bpe', cache_dir='/home/samsepiol/CommonCheckpoints/rudalle')
    pretrained_vae = get_vae(dwt=True).half().to(_device)
    dalle = get_rudalle_model(_args.model_name, pretrained=_args.pretrained, fp16=_args.fp16, device=_device).half().to(_device)

    seed_everything(42)
    text_seq_length = dalle.get_param('text_seq_length')
    total_seq_length = dalle.get_param('total_seq_length')
    device = dalle.get_param('device')


    dataset = TextImageDataset(_args.image_text_folder, text_len=text_seq_length, image_size=256, truncate_captions=True, resize_ratio=0.75, tokenizer=yttm_tokenizer)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    tqdm.write(f'{len(dataset)} images in dataset')

    for idx, (texts, images) in tqdm(enumerate(dataloader)):
        images = images.half().to(device)
        text_input_ids = texts.to(device)
        attention_mask = torch.tril(torch.ones((1, 1, total_seq_length, total_seq_length), device=device))
        image_input_ids = pretrained_vae.get_codebook_indices(images)
        input_ids = torch.cat((text_input_ids, image_input_ids), dim=1)
        loss, loss_values = dalle.forward(input_ids, attention_mask, return_loss=True)
        loss = loss.data.detach().item()
        tqdm.write(f'{idx}: {loss}')

if __name__ == '__main__':
    main()