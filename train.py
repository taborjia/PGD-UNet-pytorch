import torch
from torch import optim
from torch.optim import lr_scheduler
from sklearn.model_selection import train_test_split

from dataset import MyDataset
from model import UNet
from utilize import BCEDiceLoss, dice_coef

use_cuda = True
device = torch.device("cuda:1" if use_cuda else "cpu")

# ===========================================================
# prepare dataset
img_id = list(range(21, 41))
train_id, val_id = train_test_split(img_id, test_size=0.2, random_state=41)
train_dataset = MyDataset(train_id, is_train=True)
val_dataset = MyDataset(val_id, is_train=False)
batch_size = 4
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True)
val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
    drop_last=False)
# create model
print(f'=> creating model UNet')
model = UNet(1, 3).to(device)


# =======================================================================
# train step
def train(lr, epochs):
    # ***********************
    # define loss function (criterion)
    criterion = BCEDiceLoss().to(device)

    # ***********************
    params = filter(lambda p: p.requires_grad, model.parameters())
    # 创建出优化器optimizer
    optimizer = optim.SGD(params, lr=lr)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    # ***********************
    best_dice = 0
    for epoch in range(epochs):
        model.train()
        loss_avg = 0
        dice_avg = 0
        for index, data in enumerate(train_loader):
            inputs, targets, _ = data
            inputs = inputs.to(device)
            targets = targets.to(device)
            # compute output
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            # compute gradient and do optimizing step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            dice = dice_coef(outputs, targets)

            loss_avg += loss.detach()
            dice_avg += dice
        print(f'epoch:{epoch}, loss:{loss_avg / batch_size:.4f}, train_dice:{dice_avg / batch_size:.4f}')

        scheduler.step()
        dice_val = validate()
        print(f'val_dice:{dice_val:.4f}')

        if dice_val > best_dice:
            best_dice = dice_val
            torch.save(model.state_dict(), f'models/model.pth')
            print(f'=> save best model')


# =======================================================================
# validate step
def validate():
    model.eval()
    with torch.no_grad():
        dice_avg = 0
        for input, target, _ in val_loader:
            input = input.to(device)
            target = target.to(device)
            # compute output
            output = model(input)
            dice = dice_coef(output, target)
            dice_avg += dice
        dice_avg /= len(val_loader)
    return dice_avg


if __name__ == '__main__':
    train(0.01, 200)
