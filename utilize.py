import torch.nn.functional as F
from torch import nn


def dice_coef(output, target):
    smooth = 1e-5

    output = (output > 0.5).float().detach().cpu().numpy()
    target = target.detach().cpu().numpy()

    intersection = (output * target).sum()

    return (2. * intersection + smooth) / \
        (output.sum() + target.sum() + smooth)


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predict, target):
        bce = F.binary_cross_entropy_with_logits(predict, target)

        smooth = 1e-5
        predict_flat = predict.view(-1)
        target_flat = target.view(-1)
        intersection = (predict_flat * target_flat).sum()
        union = predict_flat.sum() + target_flat.sum()
        dice_score = (2.0 * intersection + smooth) / (union + smooth)
        dice_loss = 1.0 - dice_score

        return 0.5 * bce + dice_loss