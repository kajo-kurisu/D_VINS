# %BANNER_BEGIN%
# ---------------------------------------------------------------------
# %COPYRIGHT_BEGIN%
#
#  Magic Leap, Inc. ("COMPANY") CONFIDENTIAL
#
#  Unpublished Copyright (c) 2020
#  Magic Leap, Inc., All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains the property
# of COMPANY. The intellectual and technical concepts contained herein
# are proprietary to COMPANY and may be covered by U.S. and Foreign
# Patents, patents in process, and are protected by trade secret or
# copyright law.  Dissemination of this information or reproduction of
# this material is strictly forbidden unless prior written permission is
# obtained from COMPANY.  Access to the source code contained herein is
# hereby forbidden to anyone except current COMPANY employees, managers
# or contractors who have executed Confidentiality and Non-disclosure
# agreements explicitly covering such access.
#
# The copyright notice above does not evidence any actual or intended
# publication or disclosure  of  this source code, which includes
# information that is confidential and/or proprietary, and is a trade
# secret, of  COMPANY.   ANY REPRODUCTION, MODIFICATION, DISTRIBUTION,
# PUBLIC  PERFORMANCE, OR PUBLIC DISPLAY OF OR THROUGH USE  OF THIS
# SOURCE CODE  WITHOUT THE EXPRESS WRITTEN CONSENT OF COMPANY IS
# STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE LAWS AND
# INTERNATIONAL TREATIES.  THE RECEIPT OR POSSESSION OF  THIS SOURCE
# CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS
# TO REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE,
# USE, OR SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
#
# %COPYRIGHT_END%
# ----------------------------------------------------------------------
# %AUTHORS_BEGIN%
#
#  Originating Authors: Paul-Edouard Sarlin
#
# %AUTHORS_END%
# --------------------------------------------------------------------*/
# %BANNER_END%

# Adapted by Remi Pautrat, Philipp Lindenberger
# Adapted by Fabio Milentiansen Sim

from typing import Tuple

import torch
from torch import nn


def simple_nms(scores, nms_radius: int):
    """Fast Non-maximum suppression to remove nearby points"""
    assert nms_radius >= 0

    def max_pool(x):
        return torch.nn.functional.max_pool2d(
            x, kernel_size=nms_radius * 2 + 1, stride=1, padding=nms_radius
        )

    scores = scores[None]  # max_pool bug
    zeros = torch.zeros_like(scores)
    max_mask = scores == max_pool(scores)
    for _ in range(2):
        supp_mask = max_pool(max_mask.float()) > 0
        supp_scores = torch.where(supp_mask, zeros, scores)
        new_max_mask = supp_scores == max_pool(supp_scores)
        max_mask = max_mask | (new_max_mask & (~supp_mask))
    return torch.where(max_mask, scores, zeros)[0]


@torch.jit.script_if_tracing
def top_k_keypoints(
    keypoints: torch.Tensor, scores: torch.Tensor, k: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    if k >= keypoints.shape[0]:
        return keypoints, scores
    else:
        scores, indices = torch.topk(scores, k, dim=0, sorted=True)
        return keypoints[indices], scores


def sample_descriptors(keypoints, descriptors, s: int = 8):
    """Interpolate descriptors at keypoint locations"""
    b, c, h, w = descriptors.shape
    keypoints = keypoints - s / 2 + 0.5
    keypoints_x = torch.div(keypoints[..., 0], (w * s - s / 2 - 0.5))
    keypoints_y = torch.div(keypoints[..., 1], (h * s - s / 2 - 0.5))
    keypoints = torch.stack((keypoints_x, keypoints_y), dim=-1)

    keypoints = keypoints * 2 - 1  # normalize to (-1, 1)
    descriptors = torch.nn.functional.grid_sample(
        descriptors, keypoints.view(b, 1, -1, 2), mode="bilinear", align_corners=True
    )
    descriptors = torch.nn.functional.normalize(
        descriptors.reshape(b, c, -1), p=2, dim=1
    )
    return descriptors


class SuperPoint(nn.Module):
    """SuperPoint Convolutional Detector and Descriptor

    SuperPoint: Self-Supervised Interest Point Detection and
    Description. Daniel DeTone, Tomasz Malisiewicz, and Andrew
    Rabinovich. In CVPRW, 2019. https://arxiv.org/abs/1712.07629

    """

    default_config = {
        "descriptor_dim": 256,
        "nms_radius": 4,
        "max_num_keypoints": None,
        "detection_threshold": 0.0005,
        "remove_borders": 4,
    }

    def __init__(self, **conf):
        super().__init__()
        self.conf = {**self.default_config, **conf}

        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5 = 64, 64, 128, 128, 256

        self.conv1a = nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
        self.conv2a = nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        self.conv3a = nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
        self.conv4a = nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.conv4b = nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)

        self.convPa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convPb = nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0)

        self.convDa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convDb = nn.Conv2d(
            c5, self.conf["descriptor_dim"], kernel_size=1, stride=1, padding=0
        )

        url = "https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/superpoint_v1.pth"
        ##!!!!!!!!!!!!!!!!!!!!!!!!replace with your path!!!!!!!!!!!!!!!!!!!!
        self.load_state_dict(torch.load('/home/sy/sy/Mix_ws/src/mixvpr/model/sp+sg/superpoint_v1.pth'))

        mk = self.conf["max_num_keypoints"]
        if mk is not None and mk <= 0:
            raise ValueError('"max_num_keypoints" must be positive or None')

        print("Loaded SuperPoint model")

    def forward(
        self,
        image: torch.Tensor,  # (1, 1, H, W)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute keypoints, scores, descriptors for image"""
        # Shared Encoder
        x = self.relu(self.conv1a(image))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))

        # Compute the dense keypoint scores
        cPa = self.relu(self.convPa(x))
        scores = self.convPb(cPa)
        scores = torch.nn.functional.softmax(scores, 1)[:, :-1]
        _, _, h, w = scores.shape
        scores = scores.permute(0, 2, 3, 1).reshape(1, h, w, 8, 8)
        scores = scores.permute(0, 1, 3, 2, 4).reshape(1, h * 8, w * 8)
        scores = simple_nms(scores, self.conf["nms_radius"])

        # scores.shape == (B, H, W)

        # Discard keypoints near the image borders
        if pad := self.conf["remove_borders"]:
            scores[:, :pad] = -1
            scores[:, :, :pad] = -1
            scores[:, -pad:] = -1
            scores[:, :, -pad:] = -1

        # Below this, B > 1 is not supported as each image can have a different number of keypoints.

        # Extract keypoints
        best_kp = torch.where(scores > self.conf["detection_threshold"])
        scores = scores[best_kp]

        keypoints = torch.stack(best_kp[1:3], dim=-1)
        # keypoints.shape == (N, 2)

        # scores.shape == (N)

        # Keep the k keypoints with highest score
        if self.conf["max_num_keypoints"] is not None:
            keypoints, scores = top_k_keypoints(
                keypoints, scores, self.conf["max_num_keypoints"]
            )

        # Convert (h, w) to (x, y)
        keypoints = torch.flip(keypoints, (1,))

        # keypoints.shape == (N, 2)

        # Compute the dense descriptors
        cDa = self.relu(self.convDa(x))
        descriptors = self.convDb(cDa)
        descriptors = torch.nn.functional.normalize(descriptors, p=2, dim=1)

        # Extract descriptors
        descriptors = sample_descriptors(keypoints, descriptors, 8).permute(0, 2, 1)

        # Insert artificial batch dimension
        return (
            keypoints[None],  # (1, N, 2)
            scores[None],  # (1, N)
            descriptors,  # (1, N, desc_dim)
        )
