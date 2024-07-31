

import torch
import numpy as np
import matplotlib.pyplot as plt

def extract_patches_by_mask(image, mask):
    """
    Extracts patches from an image based on a mask and returns them in a dictionary.

    Args:
        image: Input image tensor of shape (c, h, w) with c channels, h height, and w width.
        mask: Tensor of shape (k, h, w) with values 0 or 1, where k is the number of masks.

    Returns:
        A dictionary where keys are cluster indices (k values) and values are lists of patch tensors.
    """

    patch_size = 16

    patches_by_cluster = {}
    for k, m in enumerate(mask):  # Iterate through each mask and get its index
        patches = []
        for i in range(0, image.shape[1], patch_size):
            for j in range(0, image.shape[2], patch_size):
                patch = image[:, i:i+patch_size, j:j+patch_size] * m[i:i+patch_size, j:j+patch_size]
                if patch.sum() > 0:  # Check if the patch has any non-zero values
                    patches.append(patch)
        if patches:  # Add the patches only if they are not empty
            patches_by_cluster[k] = patches

    return patches_by_cluster


def combine_patches_in_an_image(patches_by_cluster, brightness_factor):
    patch_size = 16
    cluster_to_image = {}
    for k, patch_list in patches_by_cluster.items():
        n_patches = int(np.ceil(len(patch_list)**0.5))
        new_image = torch.ones(3, patch_size*n_patches, patch_size*n_patches)
        i = 0
        j = 0
        for p in patch_list:
            new_image[:, i:i+16, j:j+16] = p[1:4, :, :].flip(0) * brightness_factor
            i += 16
            if i == patch_size*n_patches:
                i = 0
                j += 16
        cluster_to_image[k] = new_image

    return cluster_to_image

def combine_clusters_in_an_image(cluster_to_image):
    patch_size = 16
    cluster_to_len = {k: v.shape[-1] for k, v in cluster_to_image.items()}
    lengths = list(cluster_to_len.values())
    lengths.sort(reverse=True)
    width = sum(lengths[:5]) + patch_size*4
    height = lengths[0] + lengths[5] + 16

    used_clusters = []
    img_sorted = []
    for l in lengths:
        for k, v in cluster_to_image.items():
            if v.shape[-1] == l and k not in used_clusters:
                used_clusters.append(k)
                img_sorted.append(v)

    i = 0
    j = 0
    new_image = torch.ones(3, width, height)
    for count, img in enumerate(img_sorted):
        side = img.shape[-1]
        new_image[:, i:i+side, j:j+side] = img
        i += side + patch_size
        if count == 4:
            i = 0
            j += lengths[0] + patch_size

    return new_image


brightness_factor=2
colour_map = {
    0: (15,82,186),
    1: (80,200,120),
    2: (128,0,128),
    3: (224,17,95),
    4: (0,0,128),
    5: (145, 149, 246),
    6: (249, 240, 122),
    7: (251, 136, 180),
    8: (244,96,54),
    9: (49,73,94),
    10: (138,145,188),
    11: (137,147,124)
}



image_index = 46

img_0 = batch[image_index]
img_0_np = np.flip((img_0.to('cpu').numpy() * 255)[1:4, :, :], 0)

img_0_np_bright = (img_0_np * brightness_factor).clip(0, 255).astype(int)

plt.imshow(np.transpose(img_0_np_bright, (1, 2, 0)))
plt.show()

mask = masks[image_index]
mask_index = mask.to('cpu').argmax(0)

img_0_np_bright_copy = img_0_np_bright.copy()
img_rgb_first_channel = img_0_np_bright[0, :, :].copy()
img_rgb_second_channel = img_0_np_bright[1, :, :].copy()
img_rgb_third_channel = img_0_np_bright[2, :, :].copy()

for cluster_id, rgb_values in colour_map.items():
    bool_mask = (mask_index == cluster_id)
    img_rgb_first_channel[bool_mask] = rgb_values[0]
    img_rgb_second_channel[bool_mask] = rgb_values[1]
    img_rgb_third_channel[bool_mask] = rgb_values[2]

coloured_mask = np.concatenate([
    img_rgb_first_channel.reshape(-1, 448, 448),
    img_rgb_second_channel.reshape(-1, 448, 448),
    img_rgb_third_channel.reshape(-1, 448, 448)
], 0)

# Combine the bright image with the coloured mask
coloured_image = ((img_0_np_bright_copy + coloured_mask) / 2).astype(int)

plt.imshow(np.transpose(coloured_image, (1, 2, 0)))
plt.show()


patches_by_cluster = extract_patches_by_mask(batch[image_index], masks[image_index])
cluster_to_image = combine_patches_in_an_image(patches_by_cluster, brightness_factor)

new_image = combine_clusters_in_an_image(cluster_to_image)
plt.imshow(np.transpose(new_image, (1, 2, 0)))
plt.show()

