import numpy as np
import cv2
from PIL import Image
import random
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.models import feature_extraction
import torch.nn.functional as F

import torch
import torchvision
import torchvision.transforms.v2 as v2
import kornia

torch.autograd.set_detect_anomaly(True)


class Model(nn.Module):

    def __init__(self):
        super().__init__()

        vgg_full = torchvision.models.vgg19(pretrained=False).features
        self.vgg = nn.Sequential()
        for i_layer in range(24):
            self.vgg.add_module(str(i_layer), vgg_full[i_layer])

        self.vgg.add_module(str(i_layer), nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1))
        self.vgg.add_module(str(i_layer + 1), nn.ReLU())
        self.vgg.add_module(str(i_layer + 2), nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1))
        self.vgg.add_module(str(i_layer + 3), nn.ReLU())

        self.pre_upscale_layer = nn.Conv2d(128, 128, kernel_size=15, stride=1, padding=7)
        self.post_upscale_layer = nn.Conv2d(128, 1, kernel_size=15, stride=1, padding=7)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.vgg(x)

        x = self.pre_upscale_layer(x)

        x = F.interpolate(x, (224, 224))

        x = self.post_upscale_layer(x)

        x = F.sigmoid(x)

        # n_batch = x.size(0)
        # x = F.softmax(x.reshape(n_batch, -1), dim=-1).reshape(n_batch, 1, 224, 224)

        return x


def main():
    pil_imgs = [
        Image.open("../../img/000000.left.jpg"),
        Image.open("../../img/000001.left.jpg"),
    ]

    train_set = [
        v2.ToTensor()(img) for img in pil_imgs
    ]

    crop_transform = v2.Resize(size=(224, 224))
    normalize_transform = v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    n_epochs = 100
    n_warps = 2

    warp_range = 5

    model = Model()

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

    for epoch_i in range(n_epochs):
        for img in train_set:
            imgs = img.unsqueeze(0).expand(n_warps, -1, -1, -1)
            imgs = crop_transform(imgs)

            warp_imgs = torch.zeros_like(imgs)

            masks = torch.ones((imgs.size(0), 1, imgs.size(2), imgs.size(3)))
            warp_masks = torch.zeros_like(masks)

            warps = []

            for warp_i in range(n_warps):
                img_np = imgs[warp_i].permute(1, 2, 0).numpy()

                width = img_np.shape[1]
                height = img_np.shape[0]

                old_points = np.array([
                    [width // 4, height // 4],
                    [3 * width // 4, height // 4],
                    [3 * width // 4, 3 * height // 4],
                    [width // 4, 3 * height // 4],
                ])

                new_points = old_points + np.random.randint(-warp_range, warp_range, size=old_points.shape)

                old_points = torch.Tensor(old_points).unsqueeze(0)
                new_points = torch.Tensor(new_points).unsqueeze(0)

                M = kornia.geometry.get_perspective_transform(old_points, new_points)

                warps.append((old_points, new_points))

                warp_imgs[warp_i] = kornia.geometry.warp_perspective(imgs[warp_i].unsqueeze(0), M, dsize=(imgs[warp_i].size(1), imgs[warp_i].size(2))).squeeze(0)
                warp_masks[warp_i] = kornia.geometry.warp_perspective(masks[warp_i].unsqueeze(0), M, dsize=(masks[warp_i].size(1), masks[warp_i].size(2))).squeeze(0)

            warp_imgs = normalize_transform(warp_imgs)

            optimizer.zero_grad()

            warp_interest = model(warp_imgs)

            interest = torch.zeros_like(warp_interest)

            unwarp_masks = torch.zeros_like(warp_masks)

            for warp_i in range(n_warps):
                old_points, new_points = warps[warp_i]

                M = kornia.geometry.get_perspective_transform(new_points, old_points)

                interest[warp_i] = kornia.geometry.warp_perspective(warp_interest[warp_i].unsqueeze(0), M, dsize=(warp_interest[warp_i].size(1), warp_interest[warp_i].size(2))).squeeze(0)
                unwarp_masks[warp_i] = kornia.geometry.warp_perspective(warp_masks[warp_i].unsqueeze(0), M, dsize=(warp_masks[warp_i].size(1), warp_masks[warp_i].size(2))).squeeze(0)

            plt.figure()
            plt.imshow(torch.where(torch.isclose(unwarp_masks[0, 0], torch.ones_like(unwarp_masks[0, 0])), interest[0, 0].detach(), torch.nan))
            plt.figure()
            plt.imshow(torch.where(torch.isclose(unwarp_masks[1, 0], torch.ones_like(unwarp_masks[1, 0])), interest[1, 0].detach(), torch.nan))

            loss_map = torch.where(
                torch.isclose(unwarp_masks[0], torch.ones_like(unwarp_masks[0])) & torch.isclose(unwarp_masks[1], torch.ones_like(unwarp_masks[1])),
                F.mse_loss(interest[0], interest[1], reduction="none"),
                torch.nan
            )

            plt.figure()
            plt.imshow(loss_map[0].detach())

            plt.show()

            loss = loss_map.nansum()

            print(loss)

            loss.backward()
            optimizer.step()


if __name__ == "__main__":
    main()