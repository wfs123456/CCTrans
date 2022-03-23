from typing import final
from Networks import ALTGVT
import numpy as np
from torch.autograd import Variable
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
import scipy.io as io
import torch
import os
import argparse
import torch.nn.functional as F
import cv2
import random
import string


############
# This piece of code is a demonstration of inference on multiple image at the same time
# ! TEST PURPOSE ! Tested on a custom dataset (no sha, crowd ecc). (we only take a random subset of images with length batch_size, no dataloader
# Work with square images, adapt your code if not.
############


def load_model(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    device = torch.device('cuda')

    model = ALTGVT.alt_gvt_large(pretrained=False)
    model.to(device)
    model.load_state_dict(torch.load(args.weight_path, device))
    model.eval()
    return model


def vis(args, images_list, model):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    device = torch.device('cuda')

    height, width = args.img_size, args.img_size
    inputs = torch.zeros(args.batch_size, 3, height, width, device=device)

    print("The model output will be:", inputs.shape)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    for i, img in enumerate(images_list):
        if i == args.batch_size:
            break
        image_path = os.path.join(args.images_path, img)
        image = Image.open(image_path).convert("RGB").resize((height, width))
        image = transform(image)

        inputs[i, ...] = image

    print(inputs.shape)
    with torch.no_grad():
      
        # for every image, we crop that image in patches of size args.cropsize
        # NOTE! we consider only square image, if not adapt your code
        batch, c, height, width = inputs.size()
        total_patches_for_each_image = int(height / args.crop_size) ** 2
        total_number_of_patches = total_patches_for_each_image * args.batch_size

        crop_h, crop_w = args.crop_size, args.crop_size
        # we want total_number_of_patches, c, h, w
        crop_imgs = torch.zeros(
            total_number_of_patches, c, crop_h, crop_w, device=device)

        # we traverse the image from left to right, and we take args.crop_size * args.crop_size patch
        patch = 0
        for img in range(batch):
            for i in range(0, height, crop_h):
                start_height, end_height = i, min(height, i + crop_h)
                for j in range(0, width, crop_w):
                    start_width, end_width = j, min(width, j + crop_w)
                    
                    crop_imgs[patch] = inputs[img, :, start_height:end_height, start_width:end_width]
                    patch += 1

        # inference
        crop_pred, _ = model(crop_imgs)

        crop_reduction = int(args.crop_size / 8) # downsample hyperparams
        # we return to the original patches size
        patch_predictions = F.interpolate(crop_pred, size=(
            crop_reduction * 8, crop_reduction * 8), mode='bilinear', align_corners=True) / 64
        
        idx = 0
        pred_map = torch.zeros(args.batch_size, 1, height, width, device=device)
        # recreate the original prediction image by combing the patches
        for b in range((args.batch_size)):
            for i in range(0, height, crop_h):
                gis, gie = max(min(height - crop_h, i),
                               0), min(height, i + crop_h)
                for j in range(0, width, crop_w):
                    gjs, gje = max(min(width - crop_w, j),
                                   0), min(width, j + crop_w)
                    pred_map[b, :, gis:gie, gjs:gje] += patch_predictions[idx]
                    idx += 1

        
        pred_map = pred_map.cpu().detach().numpy()
        
        for i in range(args.batch_size):
            pred_image_at_i = round(
                pred_map[i, :].sum())
            print("Image #", images_list[i], "pred", pred_image_at_i)
    return pred_map


def save_image(pred_map, gt_count, image_name, save_path, args):

    if not os.path.exists(save_path):
        print("Making dir...", save_path)
        os.makedirs(save_path)

    print(f"Predicted:{ round(pred_map.sum())}, GT:{gt_count}")
    pred_map = pred_map[0, :]
    vis_img = pred_map
    # normalize density map values from 0 to 1, then map it to 0-255.
    vis_img = (vis_img - vis_img.min()) / \
        (vis_img.max() - vis_img.min() + 1e-5)
    vis_img = (vis_img * 255).astype(np.uint8)
    vis_img = cv2.applyColorMap(vis_img, cv2.COLORMAP_JET)

    # Draw gt and inference result into the colormap
    w_image, h_image, _ = vis_img.shape
    x, y, w, h = 0, h_image - 1, 175, h_image - 50
    # Draw black background rectangle
    cv2.rectangle(vis_img, (x, h_image - 50),
                  (x + w, h_image - 1), (0, 0, 0), -1)

    cv2.putText(vis_img, f"Predicted: { round(pred_map.sum())}", (
        x + int(w/10), h_image - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(vis_img, f"GT       : {gt_count}", (x + int(w/10),
                h_image - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    final_path = os.path.join(save_path, image_name)
    print(f"Saved in {final_path}")
    alpha = 0.5
    beta = (1.0 - alpha)

    original_image_path = os.path.join(args.images_path, image_name)
    original_image = cv2.imread(original_image_path)
    original_image = cv2.resize(original_image, (vis_img.shape[0], vis_img.shape[1]))

    if args.blend:
        vis_img = cv2.addWeighted(vis_img, alpha, original_image, beta, 0.0)
    cv2.imwrite(final_path, vis_img)


def save_images(pred_maps, img_lists, save_path, args):
    for i, pred_map in enumerate(pred_maps):
        gt_people_count = 0
        save_image(pred_map, gt_people_count, img_lists[i], save_path, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0', help='assign device')
    parser.add_argument("--images-path", type=str, required=True,
                        help="images path")
    parser.add_argument("--weight-path", type=str, required=True,
                        help="the weight path to be loaded")
    parser.add_argument('--crop-size', type=int, default=256,
                        help='the crop size of the train image')
    parser.add_argument('--batch-size', type=int,
                        default=1, help='train batch size')
    parser.add_argument('--dataset', default='sha',
                        help='dataset name: qnrf, nwpu, sha, shb, custom')
    parser.add_argument('--img-size', default=512, type=int,
                        help='img size that the model expects. Is an intenger (es 512). The images are resized as squared')
    parser.add_argument('--blend', default=1, type=int,
                        help='blend togheter the density map and the original image')
    args = parser.parse_args()
    print(args)

    model = load_model(args)

    dest_folder = args.images_path.split("/")[-1]
    str_random = ''.join(random.choice(
        string.ascii_uppercase + string.digits) for _ in range(4))
    save_path = f"vis/{dest_folder}/{str_random}"

    # traverse images
    img_dir_list = [item for item in os.listdir(args.images_path)
                    if os.path.isfile(os.path.join(args.images_path, item))]

    import time
    start = time.time()
    pred_map = vis(args, img_dir_list, model)
    print("after refactoring", round(time.time() - start, 2))

    #save_images(pred_map, img_dir_list, save_path, args)

    gpu_tensor = round(torch.cuda.max_memory_allocated(0) / 1073741824, 2)
    gpu_total = round(torch.cuda.max_memory_reserved(0) / 1073741824, 2)
    print("GPU - tensor occupation", gpu_tensor, "GB")
    print("GPU - total (without model)", gpu_total, "GB")