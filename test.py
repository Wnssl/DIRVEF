import torch
import numpy as np
import json


def make_json():
    from datetime import datetime
    import csv
    time = datetime.now()
    time = time.strftime('%Y_%m_%d_%H_%M_%S')
    json_file = 'externer_result.json'

    result = {}
    try:
        with open(json_file, 'r+') as f:
            dic = json.load(f)
    except:
        dic = {}

    reader = csv.reader(open('FileList.csv', 'r+'))
    next(reader)
    result['predict'] = {}
    result['result'] = {}
    for line in reader:
        if line[-1] == 'test':
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


if __name__ == '__main__':
    # cuda
    from tqdm import tqdm

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("正在构建数据集")
    from test.test_dataset import test_dataset
    from dataset import EchoData
    from torch.utils.data import DataLoader, random_split, ConcatDataset

    test = test_dataset("D:\\pre_test", codebook_path="FileList.csv")
    # test = EchoData("train_video", split="test")
    test_loader = DataLoader(dataset=test, batch_size=1, shuffle=False, num_workers=15)

    print('开始测试 : --------')

    checkpoint_pth = 'checkpoint4.028.pth'
    model_dict = torch.load(checkpoint_pth)['model']
    from EchoNet.uniformer import uniformer_small
    from EchoNet.regressor import get_shallow_mlp_head

    model = uniformer_small()
    model.load_state_dict(model_dict, strict=False)
    model = model.to(device)

    regressor = get_shallow_mlp_head()
    regressor = regressor.to(device)

    regressor_dict = torch.load(checkpoint_pth)['regressor']
    regressor.load_state_dict(regressor_dict)
    # 生成测试文件
    predict_total_loss = 0.0
    model.eval()
    regressor.eval()
    # seed_everything(args.seed)

    time, test_result = make_json()

    result = test_result[time]['predict']
    evaluation = test_result[time]['result']
    criterion = torch.nn.L1Loss()
    with torch.no_grad():
        for batch_idx, item in tqdm(enumerate(test_loader)):
            # 将输入数据和标签移动到GPU
            data = item['video'].cuda()
            target = item['label'].cuda()
            # print(outputs)
            output, features = model(data)
            outputs = regressor(features)
            output = output.squeeze(1)
            outputs = outputs.squeeze(1)

            print(f"{outputs} : {target}")

            loss = criterion(outputs, target)
            predict_total_loss += loss.item()
            # print(item)

            # output = outputs.cpu().numpy()
            filename = item['filename'][0].split("/")[1]
            result[filename]['predict'] = float(outputs[0])
        avg_loss_test = predict_total_loss / len(test_loader)
        print(f'avg loss:{avg_loss_test}')
    # print(result)
    # print(len(result.keys()))

    #     checkpoint_dict = {
    #     'val_result' : best_result,
    #     'train_result' : best_train_loss,
    #     'test_result' : avg_loss_test,
    #     'epoch' : best_epoch,
    #     'model' : best_model.state_dict(),
    #     'regressor' : best_regressor.state_dict(),
    #     }

    #     torch.save(checkpoint_dict, 'checkpoint.pth')

    #     print("正在进行评估")

    # 读取预测值
    predicts = []
    truths = []
    for k, v in result.items():
        predicts.append(v['predict'])
        truths.append(v['truth'])
    y_pred = np.array(predicts)
    y_true = np.array(truths)
    mAE = np.mean(np.abs(y_true - y_pred))
    mSE = np.mean((y_true - y_pred) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)  # 残差平方和
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)  # 总平方和
    r2 = 1 - (ss_res / ss_tot)
    evaluation['mAE'] = mAE
    evaluation['mSE'] = mSE
    evaluation['R2'] = r2
    print(evaluation)

    test_result[time]['predict'] = result
    test_result[time]['result'] = evaluation
    # print(test_result)
    with open('externer_result.json', 'w+') as f:
        json.dump(test_result, f)
