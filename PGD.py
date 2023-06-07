import torch
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms

from model import UNet
from dataset import MyDataset

device = torch.device('cuda:1')

# create model
print(f'=> creating model UNet')
model = UNet(1, 3).to(device)
model.load_state_dict(torch.load(f'models/model.pth'))


def pgd_attack(model, image, label, eps=0.3, alpha=2 / 255, iters=40):
    model.eval()

    image = image.float().to(device)
    label = label.float().to(device)

    adversarial_image = image.clone()

    optimizer = torch.optim.SGD([adversarial_image], lr=alpha)

    for _ in range(iters):
        image.requires_grad = True
        output = model(adversarial_image)
        loss = F.cross_entropy(output, label)

        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        diff = adversarial_image - image
        diff = torch.clamp(diff, -eps, eps)

        adversarial_image = torch.clamp(image + diff, 0, 1)

    return adversarial_image


if __name__ == '__main__':
    val_dataset = MyDataset([31, 32, 35, 27], is_train=False)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False)

    for i, data in enumerate(val_loader):
        input, target, img_id = data
        input = input.to(device)
        target = torch.where(target == 0, torch.tensor(1), torch.tensor(0))

        img_PGD = pgd_attack(model, input, target)[0]
        # img_PGD = img_PGD.clamp(0, 1)
        converter = transforms.ToPILImage()
        img = converter(img_PGD)
        img.save(f'PGD_img/{img_id}_PGD.jpg')
