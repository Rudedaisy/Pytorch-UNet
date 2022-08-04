import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from utils.data_loading import BasicDataset
from unet import UNet
from utils.utils import plot_img_and_mask

from NM_pruned_layers import NMSparseConv, NMSparseLinear
from extract import export

def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5,
                extract=False):
    net.eval()

    if len(full_img) == 1:
        img = full_img[0]
        img = torch.from_numpy(BasicDataset.preprocess(img, scale_factor, is_mask=False))
        img = img.unsqueeze(0)
    else:
        imgs = []
        for img in full_img:
            img = torch.from_numpy(BasicDataset.preprocess(img, scale_factor, is_mask=False))
            img = img.unsqueeze(0)
            imgs.append(img)
        img = torch.cat(imgs, 0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)
        if extract:
            print("Finished prediction")
            exit(0)

        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)[0]
        else:
            probs = torch.sigmoid(output)[0]

        tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((full_img.size[1], full_img.size[0])),
            transforms.ToTensor()
        ])

        full_mask = tf(probs.cpu()).squeeze()

    if net.n_classes == 1:
        return (full_mask > out_threshold).numpy()
    else:
        return F.one_hot(full_mask.argmax(dim=0), net.n_classes).permute(2, 0, 1).numpy()


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images', required=True)
    parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+', help='Filenames of output images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=0.5,
                        help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--extract', action='store_true', default=False, help='extract contents of the model')
    
    return parser.parse_args()


def get_output_filenames(args):
    def _generate_name(fn):
        return f'{os.path.splitext(fn)[0]}_OUT.png'

    return args.output or list(map(_generate_name, args.input))


def mask_to_image(mask: np.ndarray):
    if mask.ndim == 2:
        return Image.fromarray((mask * 255).astype(np.uint8))
    elif mask.ndim == 3:
        return Image.fromarray((np.argmax(mask, axis=0) * 255 / mask.shape[0]).astype(np.uint8))

def replace_with_pruned(m, name):    
    #print(m)
    print("{}, {}".format(name, str(type(m))))
    #if type(m) == NMSparseConv or type(m) == NMSparseLinear:
    #    return

    # HACK: directly replace conv layers of downsamples
    if name == "downsample":
        m[0] = NMSparseConv(m[0])
    if name == "double_conv":
        m[0] = NMSparseConv(m[0])
        m[3] = NMSparseConv(m[3])
        
    for attr_str in dir(m):
        target_attr = getattr(m, attr_str)
        if type(target_attr) == torch.nn.Conv2d:
            print("Replaced CONV")
            setattr(m, attr_str, NMSparseConv(target_attr))
        elif type(target_attr) == torch.nn.Linear:
            print("Replaced Linear")
            setattr(m, attr_str, NMSparseLinear(target_attr))

    for n, ch in m.named_children():
        replace_with_pruned(ch, n)
    

if __name__ == '__main__':
    args = get_args()
    in_files = args.input
    out_files = get_output_filenames(args)

    net = UNet(n_channels=3, n_classes=2, bilinear=args.bilinear)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    replace_with_pruned(net, "net")
    net.to(device=device)
    if args.model == "MODEL.pth":
        net = torch.hub.load('milesial/Pytorch-UNet', 'unet_carvana', pretrained=True, scale=0.5).to(device)
    else:
        net.load_state_dict(torch.load(args.model, map_location=device))

    logging.info('Model loaded!')

    img = []
    for i, filename in enumerate(in_files):
        logging.info(f'\nPredicting image {filename} ...')
        img.append(Image.open(filename))

    if args.extract:
        print("Extracting model.")
        inference_func = predict_img
        export(net, "UNet", "extract/", inference_func, img, device)
        
    mask = predict_img(net=net,
                       full_img=img,
                       scale_factor=args.scale,
                       out_threshold=args.mask_threshold,
                       device=device)

    if not args.no_save:
        out_filename = out_files[i]
        result = mask_to_image(mask)
        result.save(out_filename)
        logging.info(f'Mask saved to {out_filename}')

    if args.viz:
        logging.info(f'Visualizing results for image {filename}, close to continue...')
        plot_img_and_mask(img, mask)
