import argparse
import json
import time
from dataset import EchoData, dataset_Con, enhance
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, ConcatDataset
from losses import *
from tqdm import tqdm

from EchoNet.uniformer import uniformer_small
from EchoNet.regressor import get_shallow_mlp_head
from EchoNet import uniformer
from sklearn.neighbors import KernelDensity
from utils import seed_everything

def train_one_epoch(model, regressor, device, optimizer, optimizer_reg, epoch, criterion, scheduler, train_loader, rates, names2idx,dataset):
    # for train
    torch.autograd.set_detect_anomaly(True) 
    total_loss = 0.0
    model.train()
    regressor.train()
    criterion_mse = torch.nn.L1Loss()
    lambd = np.random.beta(0.2, 0.2)
    
    for batch_idx, batch in tqdm(enumerate(train_loader)):
    
        views1, views2 = batch
        #mixup
        all_idx = list(names2idx.values())
        filenames = views1['filename']
        index1 = [names2idx[i] for i in filenames]
        index2 = [np.random.choice(all_idx,
                            p = rates[i] )for i in index1]
        # without aug
        X1 = views1['video'].to(device)
        Y1 = views1['label'].to(device)
        
        x2 = []
        y2 = []
        for i in index2:
            item = dataset[i]
            x2.append(item['video'])
            y2.append(item['label'])
        
        X2 = torch.stack(x2,dim=0).to(device)
        Y2 = torch.stack(y2,dim=0).to(device)
        
        mixup_Y_without_aug = Y1 * lambd + Y2 * (1 - lambd)
        mixup_X_without_aug = X1 * lambd + X2 * (1 - lambd)
        # aug
        X11 = views2['video'].to(device)
        
        X22 = enhance(X2)
        
        mixup_Y_aug = Y1 * lambd + Y2 * (1 - lambd)
        mixup_X_aug = X11 * lambd + X22 * (1 - lambd)
        
        images = torch.cat([mixup_X_without_aug, mixup_X_aug], dim=0)
        labels = mixup_Y_aug.unsqueeze(1)
        bsz = labels.shape[0]
        
        
        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        
        _, features = model(images)
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        
        loss = criterion(features, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        features = features.detach()
        y_preds = regressor(torch.cat((features[:,0], features[:,1]), dim=0))
        loss_reg = criterion_mse(y_preds, labels.repeat(2, 1))
        
        optimizer_reg.zero_grad()
        loss_reg.backward()
        optimizer_reg.step()

       
        total_loss += loss_reg.item()
        if batch_idx % 10 == 0:
            print(f':epoch: {epoch}  batch_idx: {batch_idx}  batch loss: {loss_reg.item()}')

    # torch.save(model, f'cnn/cnn_{epoch}.pt')
    avg_loss = total_loss / len(train_loader)
    print(f"epoch : {epoch} mAE:{avg_loss}")

    scheduler.step()
    return avg_loss



def test_one_epoch(model, regressor,device, criterion,epoch, validation_loader):
    # for test
    model.eval()
    regressor.eval()
    predict_total_loss = 0.0
    print('validate : --------')
    for batch_idx, item in tqdm(enumerate(validation_loader)):
        data = item['video']
        target = item['label']
        data, target = data.to(device), target.to(device)
        output, features = model(data)
        outputs = regressor(features)
        # print(outputs)
        # output = output.squeeze(1)
        outputs = outputs.squeeze(1)
        loss = criterion(outputs, target)
        predict_total_loss += loss.item()
    avg_loss_val = predict_total_loss / len(validation_loader)
    print(f'epoch:{epoch} avg loss:{avg_loss_val}')

    return avg_loss_val


def my_collate_fn(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    data = torch.stack(data)
    target = torch.tensor(target)
    target = target.unsqueeze(1)
    a, b, c, d, e = data.shape
    data = data.view(-1, c, d, e)
    return data, target


def make_json():
    from datetime import datetime
    import csv
    time = datetime.now()
    time = time.strftime('%Y_%m_%d_%H_%M_%S')
    json_file = 'result.json'
    result = {}
    try:
        with open(json_file, 'r+') as f:
            dic = json.load(f)
    except:
        dic = {}
    reader = csv.reader(open('codebook.csv', 'r+'))
    next(reader)
    result['predict'] = {}
    result['result'] = {}
    for line in reader:
        if line[-1] == 'validation':
            predict = {}
            predict['predict'] = None
            predict['truth'] = float(line[-2])
            result['predict'][line[0]] = predict
        else:
            pass
    result['result']['mAE'] = None
    result['result']['mSE'] = None
    result['result']['R2'] = None
    dic[time] = result
    # with open('result.json', 'w+') as f:
    #     json.dump(dic, f)
    return time, dic


def main():
    parser = argparse.ArgumentParser()

    # train parser
    parser.add_argument('--batch_size', type=int, help='batch_size', default=16)
    parser.add_argument('--epoch_nums', type=int, help='epoch nums', default=45)
    parser.add_argument('--adam_learning_rate', type=float, help='learning rate', default=0.0003)
    parser.add_argument('--SGD_learning_rate', type=float, help='SGD learning rate', default=0.02)
    parser.add_argument('--momentum', type=float, help='SGD momentum', default=0.8)
    parser.add_argument('--optim', type=str, help='optimizer', choices=['SGD', 'Adam'], default='Adam')
    parser.add_argument('--criterion', type=str, help=' loss ', choices=['L1', 'MSE'], default='L1')
    parser.add_argument('--scheduler', type=str, help='lr scheduler', default='StepLR')
    parser.add_argument('--step_num', type=int, help='step num', default=15)
    parser.add_argument('--gamma', type=float, help='step gammer', default=0.1)
    parser.add_argument('--num_workers', type=int, help='cpu workers', default=15)
    parser.add_argument('--collate_fn', type=bool, help='work om collate_fn', default=False)
    parser.add_argument('--criterion1', type=str, help='loss for train',
                        choices=['L1', 'MSE', 'Focal L1', 'Focal MSE', 'other'], default='other')
    parser.add_argument('--random_seed', type=bool, help='if use random seed', default=True)
    parser.add_argument('--seed', type=int, help='random seed', default=42)
    # dataset parser
    parser.add_argument('--train_file_path', type=str, help='train files path', default='train_video')

    parser.add_argument('--codebook_path', type=str, help='codebook path', default='codebook.csv')

    # Contrastive_Loss Parameters
    parser.add_argument('--rnc', type=bool, default=True, help='use Contrastive - loss')
    parser.add_argument('--temp', type=float, default=2, help='temperature')
    parser.add_argument('--label_diff', type=str, default='l1', choices=['l1'], help='label distance function')
    parser.add_argument('--feature_sim', type=str, default='l2', choices=['l2'], help='feature similarity function')
   

    # parser.add_argument()
    args = parser.parse_args()
    print(args)
    # 加载训练集
    seed_everything(args.seed)
    print("seed:", args.seed)
    train_temp = EchoData("train_video", split='train')
    train = dataset_Con("train_video", split='train')
    val = EchoData("train_video", split='val')
    test = EchoData("train_video", split='test')
    
    # 计算C-mixup rates
    labels = train.get_labels()
    filenames = train.get_filenames()
    print(max(labels))
    print(min(labels))
    import numpy as np
    print(np.mean(np.array(labels)))
    # filenames 与index 对应的字典
    name2idx = {}
    for idx, i in enumerate(filenames):
        name2idx[i] = idx
    
    # print(name2idx)
    rates = get_mixup_rate(np.array(labels))
    # print(rates)
    
    print(f"train length : {len(train)}")
    print(f"val length : {len(val)}")
    print(f"test length : {len(test)}")

    # cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dataLoader

    if args.collate_fn:
        my_collate = my_collate_fn()
    else:
        my_collate = None

    train_loader = DataLoader(dataset=train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,collate_fn=my_collate, drop_last=True)
    
    val_loader = DataLoader(dataset=val, batch_size=args.batch_size, shuffle=False,
                                   num_workers=args.num_workers, collate_fn=my_collate)
    test_loader = DataLoader(dataset=test, batch_size=1, shuffle=False, num_workers=args.num_workers,
                             collate_fn=my_collate)

    # model
    print('buliding model-----')
    checkpoint_pth = 'save/L1SG_uniformer_small_ep_60_lr_0.0001_d_0.1_wd_0.0001_bsz_16_aug_True/best.pth'
    checkpoint = torch.load(checkpoint_pth)['model']

    # model = uniformer_small()
    model = uniformer("uniformer_small", checkpoint)
    # model.load_state_dict(checkpoint, strict=False)
    model = model.to(device)

    regressor = get_shallow_mlp_head()
    regressor = regressor.to(device)

    
     # train loss
    if args.criterion1 == 'L1':
        criterion1 = weighted_l1_loss
    elif args.criterion1 == 'MSE':
        criterion1 = weighted_mse_loss
    elif args.criterion1 == 'Focal_L1':
        criterion1 = weighted_focal_l1_loss
    elif args.criterion1 == 'Focal_MSE':
        criterion1 = weighted_focal_mse_loss
    else:
        if args.rnc:
            criterion1 = ContrastiveLoss_weights(temperature=args.temp, label_diff=args.label_diff,feature_sim=args.feature_sim)
            print(f"use contrastive loss")
        else:
            print("criterion error")
        # test loss
    if args.criterion == 'L1':
        criterion = nn.L1Loss()
    else:
        criterion = nn.MSELoss()

    # optim
    if args.optim == 'Adam':
        optimizer = optim.AdamW(model.parameters(), lr=args.adam_learning_rate)
        optimizer_reg = optim.AdamW(regressor.parameters(), lr=args.adam_learning_rate)
    elif args.optim == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.SGD_learning_rate, momentum=args.momentum)
    else:
        optimizer = None

    # optim step
    if args.scheduler == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_num, gamma=args.gamma)
    else:
        scheduler = None
        pass

    # reweight default=False
    weighted = False
    if args.process_type == 'c':
        weighted = True

    # best result  ---- used to print and save best model
    best_result = 9999

    for epoch in range(args.epoch_nums):
        # for train
        train_loss = train_one_epoch(model, regressor,device, optimizer, optimizer_reg, epoch, criterion1, scheduler, train_loader, rates, name2idx,train_temp)
        # for val
        avg_loss_val = test_one_epoch(model, regressor, device, criterion, epoch, val_loader)
        if avg_loss_val < best_result:
            best_result = avg_loss_val
            best_train_loss = train_loss
            best_epoch = epoch
            best_model = model
            best_regressor = regressor 
        print(f'best epoch: {best_epoch} : val loss: {best_result} train loss: {best_train_loss}')


    predict_total_loss = 0.0
    model.eval()
    regressor.eval()
    seed_everything(args.seed)
    print('testing : --------')

    time, test_result = make_json()
    result = test_result[time]['predict']
    evaluation = test_result[time]['result']

    with torch.no_grad():
        for batch_idx, item in tqdm(enumerate(test_loader)):
            data = item['video'].cuda()
            target = item['label'].cuda()
            output, features = model(data)
            outputs = regressor(features)
            # output = output.squeeze(1)
            outputs = outputs.squeeze(1)
            loss = criterion(outputs, target)
            predict_total_loss += loss.item()

            filename = item['filename'][0]
            result[filename]['predict'] = float(outputs[0])
        avg_loss_test = predict_total_loss / len(test_loader)
        print(f'test loss:{avg_loss_test}')
    
    checkpoint_dict = {
    'val_result' : best_result, 
    'train_result' : best_train_loss,
    'test_result' : avg_loss_test,
    'epoch' : best_epoch,
    'model' : best_model.state_dict(),
    'regressor' : best_regressor.state_dict(),
    }
    
    torch.save(checkpoint_dict, 'save/checkpoint.pth')

    predicts = []
    truths = []
    for k, v in result.items():
        predicts.append(v['predict'])
        truths.append(v['truth'])
    y_pred = np.array(predicts)
    y_true = np.array(truths)
    mAE = np.mean(np.abs(y_true - y_pred))
    mSE = np.mean((y_true - y_pred) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    evaluation['mAE'] = mAE
    evaluation['mSE'] = mSE
    evaluation['R2'] = r2
    print(evaluation)

    test_result[time]['predict'] = result
    test_result[time]['result'] = evaluation
    with open('result.json', 'w+') as f:
        json.dump(test_result, f)


def get_mixup_rate(sample_list, mixtype='kde', kde_type='gaussian', bandwidth=50.0, device='cuda'):
    mix_idx = []
    if len(sample_list.shape) == 1:
        sample_list = sample_list[:, np.newaxis]
    with tqdm(total=len(sample_list)) as pbar:
        # pbar.set_description('Cal Sample Rate ')
        for i in range(len(sample_list)):
            if mixtype == 'kde':
                data_i = sample_list[i]
                # xi = x_list[i]
                data_i = data_i[:, np.newaxis]
                kd = KernelDensity(kernel=kde_type, bandwidth=bandwidth).fit(data_i)
                each_rate = np.exp(kd.score_samples(sample_list))
                each_rate /= np.sum(each_rate)  # norm
            else:  # random
                each_rate = np.ones(sample_list.shape[0]) * 1.0 / sample_list.shape[0]

            mix_idx.append(each_rate)
            pbar.set_postfix_str(
                'rate: max: {:.5f}, min: {:.8f}, std: {:.5f}, mean: {:.5f}'.format(max(each_rate), min(each_rate),
                                                                                   np.std(each_rate),
                                                                                   np.mean(each_rate)))
    return np.array(mix_idx)


if __name__ == '__main__':
    main()
