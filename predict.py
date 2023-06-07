import os
import torch
import numpy as np
import torchvision.transforms as transforms

from PIL import Image
from sklearn.model_selection import train_test_split

from model import UNet
from dataset import MyDataset
from utilize import dice_coef


device = torch.device('cuda:1')
img_id = list(range(21, 41))
_, val_id = train_test_split(img_id, test_size=0.2, random_state=41)
print(val_id)
val_dataset = MyDataset(val_id, is_train=False)
val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
    drop_last=False)
# create model
print(f'=> creating model UNet')
model = UNet(1, 3).to(device)


def predict(output, target, save_path, img_id):
    dice = dice_coef(output, target)
    predict = np.array((output > 0.5).float().detach().cpu())[0][0]
    predict[np.where(predict != 0)] = 255
    predict_img = Image.fromarray(predict).resize((565, 584)).convert(mode='L')
    predict_img.save(f'{save_path}/{img_id}_predict_UNet.jpg')
    return dice


model.load_state_dict(torch.load(f'models/model.pth'))
with torch.no_grad():
    # dice_avg = 0
    # for i, data in enumerate(val_loader):
    #     input, target, img_id = data
    #     input = input.to(device)
    #     output = model(input)
    #     dice = predict(output, target, 'predict')
    #     print(f'{img_id} dice is {dice: .4f}')
    #     dice_avg += dice
    # print(f'avg dice is {dice_avg / 4: .4f}')

    dice_avg = 0
    transform_img = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.Normalize(0.5, 1)
    ])
    transform_label = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.NEAREST)
    ])
    for name_id in [27, 31, 32, 35]:
        img_path = f'PGD_img/{name_id}_PGD.jpg'
        img = Image.open(img_path)
        img = transform_img(img).unsqueeze(0).float().to(device)
        target_path = f'DRIVE/training/1st_manual/{name_id}_manual1.gif'
        target = Image.open(target_path)
        target = transform_label(target).unsqueeze(0).float().to(device)
        output = model(img)
        dice = predict(output, target, 'PGD_predict', name_id)
        print(f'{img_id} dice is {dice: .4f}')
        dice_avg += dice
    print(f'avg dice is {dice_avg / 4: .4f}')
