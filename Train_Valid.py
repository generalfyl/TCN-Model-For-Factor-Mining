import time
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
from Self_Defined_Loss import CorrLoss
from torch.optim.lr_scheduler import StepLR


def train(model, args, device, train_loader, valid_loader):
    start_time = time.time()  # 计算起始时间
    # model = model.to(device)
    loss_function = CorrLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    epochs = args.epochs
    model.train()  # 训练模式
    results_train_loss, results_valid_loss = [], []
    result_valid_IC_average = []
    result_valid_CORR_IJ = []
    # scheduler = StepLR(optimizer, step_size=1, gamma=0.1)
    # best_valid_loss = float('inf')
    # early_stop_counter = 0
    # patience = 30

    for i in tqdm(range(epochs)):
        loss = 0
        flag = 0
        for seq, labels in train_loader:
            flag += 1
            model.train()
            seq = seq.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            y_pred = model(seq)

            single_loss = loss_function(y_pred, labels, model)

            single_loss.backward()

            optimizer.step()
            loss += single_loss.item()

        torch.save(model.state_dict(), f"save_model_{i + 1}.pth")
        # results_train_loss.append(loss / len(train_loader))
        valid_loss, CORR_IJ, IC_AVERAGE = valid(model, valid_loader, i + 1)  # IC_all, IC_average
        """动态学习率"""
        # scheduler.step()
        result_valid_IC_average.append(IC_AVERAGE)
        result_valid_CORR_IJ.append(CORR_IJ)
        train_loss = loss / len(train_loader)
        results_train_loss.append(train_loss)
        results_valid_loss.append(valid_loss)
        tqdm.write(f"\t Epoch {i + 1} / {epochs},train Loss: {train_loss},valid Loss:{valid_loss}")
        """检测是否早停"""
        # if valid_loss < best_valid_loss:
        #     best_valid_loss = valid_loss
        #     early_stop_counter = 0
        # else:
        #     early_stop_counter += 1
        #     if early_stop_counter >= patience:
        #         print("Early Stop")
        #         break
        time.sleep(0.1)

    """画loss下降图"""
    plt.figure(figsize=(10, 5))
    plt.plot(results_train_loss, label='Train Loss')
    plt.plot(results_valid_loss, label='Valid Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('train_valid_loss.png')
    plt.show()
    """画因子间相关系数图"""
    plt.figure(figsize=(10, 5))
    plt.plot(result_valid_CORR_IJ, label='Corr_i_j')
    plt.xlabel('Epochs')
    plt.ylabel('Corr_i_j')
    plt.title('因子间相关系数')
    plt.legend()
    plt.savefig('Corr_i_j.png')
    plt.show()
    """画合成因子的IC图"""
    plt.figure(figsize=(10, 5))
    plt.plot(result_valid_IC_average, label='IC_average')
    plt.xlabel('Epochs')
    plt.ylabel('IC_average')
    plt.title('IC_average')
    plt.legend()
    plt.savefig('IC_average.png')
    plt.show()
    # 保存模型

    print(f">>>>>>>>>>>>>>>>>>>>>>模型已保存,用时:{(time.time() - start_time) / 60:.4f} min<<<<<<<<<<<<<<<<<<")


def valid(model, valid_loader, round_of_epoch):
    TCN_model = model
    # 加载模型进行预测
    TCN_model.load_state_dict(torch.load(f"save_model_{round_of_epoch}.pth"))
    TCN_model.eval()  # 评估模式
    loss_function = CorrLoss()
    loss = 0
    Corr_I_J = 0
    IC_Average = 0
    for seq, labels in valid_loader:
        seq = seq.cuda()
        labels = labels.cuda()
        pred = TCN_model(seq)
        single_loss, Corr_i_j, IC_average = loss_function(pred, labels, TCN_model)  # 计算loss
        loss += single_loss.item()
        IC_Average += IC_average.item()
        Corr_I_J += Corr_i_j.item()

    """返回：1.loss 2.因子间相关系数 3.均值合成因子IC"""
    return loss / len(valid_loader), Corr_I_J/len(valid_loader), IC_Average/len(valid_loader)