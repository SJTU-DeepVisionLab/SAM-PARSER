# %% load environment
import numpy as np
import matplotlib.pyplot as plt
import os
join = os.path.join
import torch
from segment_anything import sam_model_registry
#from segment_anything_lora import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
from tqdm import tqdm
import argparse
import traceback
import svf_torch
from utils.SurfaceDice import compute_dice_coefficient
# visualization functions
# source: https://github.com/facebookresearch/segment-anything/blob/main/notebooks/predictor_example.ipynb
# change color to avoid red and green
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([251/255, 252/255, 30/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='blue', facecolor=(0,0,0,0), lw=2))    


def compute_dice(mask_gt, mask_pred):
    """Compute soerensen-dice coefficient.
    Returns:
    the dice coeffcient as float. If both masks are empty, the result is NaN
    """
    volume_sum = mask_gt.sum() + mask_pred.sum()
    if volume_sum == 0:
        return np.NaN
    volume_intersect = (mask_gt & mask_pred).sum()
    return 2*volume_intersect / volume_sum

def finetune_model_predict(img_np, box_np, sam_trans, sam_model_tune, device='cuda:0'):
    H, W = img_np.shape[:2]
    resize_img = sam_trans.apply_image(img_np)
    resize_img_tensor = torch.as_tensor(resize_img.transpose(2, 0, 1)).to(device)
    input_image = sam_model_tune.preprocess(resize_img_tensor[None,:,:,:]) # (1, 3, 1024, 1024)
    with torch.no_grad():
        image_embedding = sam_model_tune.image_encoder(input_image.to(device)) # (1, 256, 64, 64)
        # convert box to 1024x1024 grid
        box = sam_trans.apply_boxes(box_np, (H, W))
        box_torch = torch.as_tensor(box, dtype=torch.float, device=device)
        if len(box_torch.shape) == 2:
            box_torch = box_torch[:, None, :] # (B, 1, 4)
        
        sparse_embeddings, dense_embeddings = sam_model_tune.prompt_encoder(
            points=None,
            boxes=box_torch,
            masks=None,
        )
        medsam_seg_prob, _ = sam_model_tune.mask_decoder(
            image_embeddings=image_embedding.to(device), # (B, 256, 64, 64)
            image_pe=sam_model_tune.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
            multimask_output=False,
            )
        medsam_seg_prob = torch.sigmoid(medsam_seg_prob)
        # convert soft mask to hard mask
        medsam_seg_prob = medsam_seg_prob.cpu().numpy().squeeze()
        medsam_seg = (medsam_seg_prob > 0.5).astype(np.uint8)
    return medsam_seg

#%% run inference
# set up the parser
parser = argparse.ArgumentParser(description='run inference on testing set based on MedSAM')
parser.add_argument('-i', '--data_path', type=str, default='data/Npz_files/CT_Abd-Gallbladder/test', help='path to the data folder')
parser.add_argument('-o', '--seg_path_root', type=str, default='data/Test_MedSAMBaseSeg', help='path to the segmentation folder')
parser.add_argument('--seg_png_path', type=str, default='data/sanity_test/Test_MedSAMBase_png', help='path to the segmentation folder')
parser.add_argument('--model_type', type=str, default='vit_b', help='model type')
parser.add_argument('--device', type=str, default='cuda:2', help='device')
parser.add_argument('-chk', '--checkpoint', type=str, default='./work_dir_lora_rank_32/CT_Abd-Gallbladder/sam_model_best.pth', help='path to the trained model')
parser.add_argument('-chk_l', '--checkpoint_b_l', type=str, default='./work_dir_SVD_encoder_true_global_rank_1.0_v3_only_1x1_3x3_no_linear/CT_Abd-Gallbladder/sam_model_best.pth', help='path to the trained model')
parser.add_argument('-chk_baseline', '--checkpoint_b', type=str, default='./work_dir_lr1.25e-6_batchsize1_epoch25/CT_Abd-Gallbladder/sam_model_best.pth', help='path to the trained model')
parser.add_argument('-chki', '--checkpoint_initial', type=str, default='./work_dir/SAM/sam_vit_b_01ec64.pth', help='path to the trained model')
args = parser.parse_args()

#% load MedSAM model
device = args.device
sam_model_tune_baseline = sam_model_registry[args.model_type](checkpoint=args.checkpoint_b).to(device)
sam_model_tune_baseline.eval()


# sam_model_tune2 = sam_model_registry[args.model_type](checkpoint=args.checkpoint_initial).to(device)
# sam_model_tune2.image_encoder = svf_torch.resolver(sam_model_tune2.image_encoder,global_low_rank_ratio=1,  # no need to change
#                     skip_1x1_linear=True,  # we will decompose 1x1 conv layers
#                     skip_1x1=True,  # we will decompose 1x1 conv layers
#                     skip_3x3=False  # we will decompose 3x3 conv layers
#                                 )
# sam_model_tune2.load_state_dict(torch.load(args.checkpoint_b_l))
# sam_model_tune2 = sam_model_tune2.to(device)
# sam_model_tune2.eval()


sam_model_tune2 = sam_model_registry[args.model_type](checkpoint=args.checkpoint_initial).to(device)
sam_model_tune2.image_encoder = svf_torch.resolver(sam_model_tune2.image_encoder,global_low_rank_ratio=1,  # no need to change
                    skip_1x1_linear=True,  # we will decompose 1x1 conv layers
                    skip_1x1=True,  # we will decompose 1x1 conv layers
                    skip_3x3=False  # we will decompose 3x3 conv layers
                                )
sam_model_tune2.load_state_dict(torch.load(args.checkpoint_b_l))
sam_model_tune2 = sam_model_tune2.to(device)
sam_model_tune2.eval()

from segment_anything_lora import sam_model_registry
sam_model_tune = sam_model_registry[args.model_type](checkpoint=args.checkpoint_initial).to(device)
#same_model_tune = sam_model_registry[args.model_type](checkpoint=args.checkpoint_initial).to(device)

#sam_model_tune.image_encoder = svf_torch.resolver(sam_model_tune.image_encoder,global_low_rank_ratio=1,  # no need to change
#                    skip_1x1_linear=True, 
#                    skip_1x1=True,  # we will decompose 1x1 conv layers
#                    skip_3x3=False  # we will decompose 3x3 conv layers
#                                )
sam_model_tune.load_state_dict(torch.load(args.checkpoint))
sam_model_tune = sam_model_tune.to(device)
sam_model_tune.eval()



sam_trans = ResizeLongestSide(sam_model_tune.image_encoder.img_size)



test_npzs = sorted(os.listdir(args.data_path))
# random select a test case
npz_idx = np.random.randint(0, len(test_npzs))
npz = np.load(join(args.data_path, test_npzs[npz_idx]))
imgs = npz['imgs']
gts = npz['gts']

def get_bbox_from_mask(mask):
    '''Returns a bounding box from a mask'''
    y_indices, x_indices = np.where(mask > 0)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    # add perturbation to bounding box coordinates
    H, W = mask.shape
    x_min = max(0, x_min - np.random.randint(20, 40))
    x_max = min(W, x_max + np.random.randint(20, 40))
    y_min = max(0, y_min - np.random.randint(20, 40))
    y_max = min(H, y_max + np.random.randint(20, 40))

    return np.array([x_min, y_min, x_max, y_max])

ori_sam_segs = []
medsam_segs = []
medsam_segs_l = []
bboxes = []
for img, gt in zip(imgs, gts):
    bbox_initial = get_bbox_from_mask(gt)
    bboxes.append(bbox_initial)
    # predict the segmentation mask using the fine-tuned model
    H, W = img.shape[:2]
    resize_img = sam_trans.apply_image(img)
    resize_img_tensor = torch.as_tensor(resize_img.transpose(2, 0, 1)).to(device)
    input_image = sam_model_tune.preprocess(resize_img_tensor[None,:,:,:]) # (1, 3, 1024, 1024)
    #input_image2 = sam_model_tune_baseline.preprocess(resize_img_tensor[None,:,:,:]) # (1, 3, 1024, 1024)
    with torch.no_grad():
        image_embedding = sam_model_tune.image_encoder(input_image.to(device)) # (1, 256, 64, 64)
        # convert box to 1024x1024 grid
        bbox = sam_trans.apply_boxes(bbox_initial, (H, W))
        box_torch = torch.as_tensor(bbox, dtype=torch.float, device=device)
        if len(box_torch.shape) == 2:
            box_torch = box_torch[:, None, :] # (B, 1, 4)
        
        sparse_embeddings, dense_embeddings = sam_model_tune.prompt_encoder(
            points=None,
            boxes=box_torch,
            masks=None,
        )
        medsam_seg_prob, _ = sam_model_tune.mask_decoder(
            image_embeddings=image_embedding.to(device), # (B, 256, 64, 64)
            image_pe=sam_model_tune.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
            multimask_output=False,
            )
        medsam_seg_prob = torch.sigmoid(medsam_seg_prob)
        # convert soft mask to hard mask
        medsam_seg_prob = medsam_seg_prob.cpu().numpy().squeeze()
        medsam_seg = (medsam_seg_prob > 0.5).astype(np.uint8)
        medsam_segs.append(medsam_seg)


    with torch.no_grad():
        image_embedding = sam_model_tune2.image_encoder(input_image.to(device)) # (1, 256, 64, 64)
        # convert box to 1024x1024 grid
        bbox = sam_trans.apply_boxes(bbox_initial, (H, W))
        box_torch = torch.as_tensor(bbox, dtype=torch.float, device=device)
        if len(box_torch.shape) == 2:
            box_torch = box_torch[:, None, :] # (B, 1, 4)
        
        sparse_embeddings, dense_embeddings = sam_model_tune2.prompt_encoder(
            points=None,
            boxes=box_torch,
            masks=None,
        )
        medsam_seg_prob, _ = sam_model_tune2.mask_decoder(
            image_embeddings=image_embedding.to(device), # (B, 256, 64, 64)
            image_pe=sam_model_tune2.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
            multimask_output=False,
            )
        medsam_seg_prob = torch.sigmoid(medsam_seg_prob)
        # convert soft mask to hard mask
        medsam_seg_prob = medsam_seg_prob.cpu().numpy().squeeze()
        medsam_seg = (medsam_seg_prob > 0.5).astype(np.uint8)
        medsam_segs_l.append(medsam_seg)

    with torch.no_grad():
        image_embedding = sam_model_tune_baseline.image_encoder(input_image.to(device)) # (1, 256, 64, 64)
        # convert box to 1024x1024 grid
        bbox = sam_trans.apply_boxes(bbox_initial, (H, W))
        box_torch = torch.as_tensor(bbox, dtype=torch.float, device=device)
        if len(box_torch.shape) == 2:
            box_torch = box_torch[:, None, :] # (B, 1, 4)
        
        sparse_embeddings, dense_embeddings = sam_model_tune_baseline.prompt_encoder(
            points=None,
            boxes=box_torch,
            masks=None,
        )
        medsam_seg_prob2, _ = sam_model_tune_baseline.mask_decoder(
            image_embeddings=image_embedding.to(device), # (B, 256, 64, 64)
            image_pe=sam_model_tune_baseline.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
            multimask_output=False,
            )
        medsam_seg_prob2 = torch.sigmoid(medsam_seg_prob2)
        # convert soft mask to hard mask
        medsam_seg_prob2 = medsam_seg_prob2.cpu().numpy().squeeze()
        ori_sam_seg = (medsam_seg_prob2 > 0.5).astype(np.uint8)
        ori_sam_segs.append(ori_sam_seg)

#%% compute the DSC score
medsam_segs = np.stack(medsam_segs, axis=0)
medsam_dsc = compute_dice_coefficient(gts>0, medsam_segs>0)
print('MedSAM DSC: {:.4f}'.format(medsam_dsc))

medsam_segs_l = np.stack(medsam_segs_l, axis=0)
medsam_dsc_l = compute_dice_coefficient(gts>0, medsam_segs_l>0)
print('MedSAM_l DSC: {:.4f}'.format(medsam_dsc_l))

ori_sam_segs = np.stack(ori_sam_segs, axis=0)
medsam_dsc_baseline = compute_dice_coefficient(gts>0, ori_sam_segs>0)
print('MedSAM_baseline DSC: {:.4f}'.format(medsam_dsc_baseline))



def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([251/255, 252/255, 30/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='blue', facecolor=(0,0,0,0), lw=2))    


img_id = int(imgs.shape[0]/2)  # np.random.randint(imgs.shape[0])
_, axs = plt.subplots(1, 4, figsize=(25, 25))
axs[0].imshow(imgs[img_id])
show_mask(gts[img_id], axs[0])
# show_box(box_np[img_id], axs[0])
# axs[0].set_title('Mask with Tuned Model', fontsize=20)
axs[0].axis('off')

axs[1].imshow(imgs[img_id])
show_mask(ori_sam_segs[img_id], axs[1])
show_box(bboxes[img_id], axs[1])
# add text to image to show dice score
axs[1].text(0.5, 0.5, 'MedSAM DSC: {:.4f}'.format(medsam_dsc_baseline), fontsize=20, horizontalalignment='left', verticalalignment='top', color='yellow')
# axs[1].set_title('Mask with Untuned Model', fontsize=20)
axs[1].axis('off')

axs[2].imshow(imgs[img_id])
show_mask(medsam_segs[img_id], axs[2])
show_box(bboxes[img_id], axs[2])
# add text to image to show dice score
axs[2].text(0.5, 0.5, 'MedSAM_lora DSC: {:.4f}'.format(medsam_dsc), fontsize=20, horizontalalignment='left', verticalalignment='top', color='yellow')
# axs[2].set_title('Ground Truth', fontsize=20)
axs[2].axis('off')

axs[3].imshow(imgs[img_id])
show_mask(medsam_segs_l[img_id], axs[3])
show_box(bboxes[img_id], axs[3])
# add text to image to show dice score
axs[3].text(0.5, 0.5, 'MedSAM_1x1_3x3 DSC: {:.4f}'.format(medsam_dsc_l), fontsize=20, horizontalalignment='left', verticalalignment='top', color='yellow')
# axs[2].set_title('Ground Truth', fontsize=20)
axs[3].axis('off')


plt.show()  
plt.subplots_adjust(wspace=0.01, hspace=0)
# save plot
plt.savefig('plot.png')
# plt.savefig(join(model_save_path, test_npzs[npz_idx].split('.npz')[0] + str(img_id).zfill(3) + '.png'), bbox_inches='tight', dpi=300)
plt.close()