from pathlib import Path
import torch
from torch import nn

def simple_nms(scores, nms_radius: int):
    """ Fast Non-maximum suppression to remove nearby points """
    assert(nms_radius >= 0)

    # def max_pool(x):
    #     return torch.nn.functional.max_pool2d(
    #         x, kernel_size=nms_radius*2+1, stride=1, padding=nms_radius)

    zeros = torch.zeros_like(scores)
    max_mask = scores == torch.nn.functional.max_pool2d(scores, kernel_size=nms_radius*2+1, stride=1, padding=nms_radius)
    for _ in range(2):
        # supp_mask = max_pool(max_mask.float()) > 0
        supp_mask = torch.nn.functional.max_pool2d(max_mask.float(), kernel_size=nms_radius*2+1, stride=1, padding=nms_radius) > 0
        supp_scores = torch.where(supp_mask, zeros, scores)
        new_max_mask = supp_scores == torch.nn.functional.max_pool2d(supp_scores, kernel_size=nms_radius*2+1, stride=1, padding=nms_radius)
        max_mask = max_mask | (new_max_mask & (~supp_mask))
    return torch.where(max_mask, scores, zeros)


def remove_borders(keypoints, scores, border: int, height: int, width: int):
    """ Removes keypoints too close to the border """
    mask_h = (keypoints[:, 0] >= border) & (keypoints[:, 0] < (height - border))
    mask_w = (keypoints[:, 1] >= border) & (keypoints[:, 1] < (width - border))
    mask = mask_h & mask_w
    return keypoints[mask], scores[mask]


def top_k_keypoints(keypoints, scores, k: int):
    if k >= len(keypoints):
        return keypoints, scores
    scores, indices = torch.topk(scores, k, dim=0)
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


class UltraPoint(nn.Module):

    def __init__(self, descriptor_dim=256, nms_radius=4, keypoint_threshold=0.0005, max_keypoints=-1, remove_borders=4):
        super().__init__()
        # self.config = config
        self.descriptor_dim = descriptor_dim
        self.nms_radius = nms_radius
        self.keypoint_threshold = keypoint_threshold
        self.max_keypoints = max_keypoints
        self.remove_borders = remove_borders

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
            c5, 256,
            kernel_size=1, stride=1, padding=0)


        path = Path(__file__).parent / 'weights/superpoint_v1.pth'
        
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!REPLACE IT WITH YOUR SP PATH!!!!!!!!!!!!!!!!!!!!!!!!!!
        self.load_state_dict(torch.load('/home/sy/sy/Mix_ws/src/mixvpr/model/sp+sg/superpoint_v1.pth'))

        mk = self.max_keypoints
        if mk == 0 or mk < -1:
            raise ValueError('\"max_keypoints\" must be positive or \"-1\"')

        print('Loaded SuperPoint model')

    def forward(self, image, keypoints):
        """ Compute scores, descriptors for image and keypoints"""
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

        # Unused in project so it is set to 0, dont use it further!!!!!!!!!!
        scores = torch.zeros(keypoints.shape[1])

        # Compute the dense descriptors
        cDa = self.relu(self.convDa(x))
        descriptors = self.convDb(cDa)
        descriptors = torch.nn.functional.normalize(descriptors, p=2, dim=1)

        # Extract descriptors
        descriptors = sample_descriptors(keypoints, descriptors, 8).permute(0, 2, 1)

        return scores, descriptors

