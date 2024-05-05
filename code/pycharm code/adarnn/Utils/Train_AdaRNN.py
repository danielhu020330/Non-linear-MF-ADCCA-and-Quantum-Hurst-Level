import os
import time
import pynvml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm
from Models.NoamScheduler import NoamScheduler
from Utils.DataPreprocessor import Preprocessor
from Utils.InfoLogger import Logger
from Utils.ParamsHandler import Handler
from Models.LSTM import LSTM
from Models.GRU import GRU
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from Models.AdaRNN import AdaRNN
from Models.SHARNN import SHARNN
#from Models.Transformer import TransformerModel
from Models.Transformer1 import TransformerModel
from Models.LeeOscillator import LeeOscillator
from Models.Chao import ChaoticLSTM
dfmse = []
dfmae = []
dfr2 = []

# Get the hyperparameters.
Cfg = Handler.Parser(Handler.Generator(paramsDir='./LSTMparams.txt'))
# Get the current time.
currentTime = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())

# Check the directory.
if not os.path.exists(Cfg.modelDir):
    os.mkdir(Cfg.modelDir)
if not os.path.exists(Cfg.logDir):
    os.mkdir(Cfg.logDir)
if not os.path.exists(f'{Cfg.logDir}//ChaoticExtractor//{currentTime}'):
    if not os.path.exists(f'{Cfg.logDir}//ChaoticExtractor'):
        os.mkdir(f'{Cfg.logDir}//ChaoticExtractor')
    if not os.path.exists(f'{Cfg.logDir}//OutputQuery'):
        os.mkdir(f'{Cfg.logDir}//OutputQuery')
    if not os.path.exists(f'{Cfg.logDir}//NoamScheduler'):
        os.mkdir(f'{Cfg.logDir}//NoamScheduler')
    os.mkdir(f'{Cfg.logDir}//ChaoticExtractor//{currentTime}')
    os.mkdir(f'{Cfg.logDir}//OutputQuery//{currentTime}')
    os.mkdir(f'{Cfg.logDir}//NoamScheduler//{currentTime}')
if not os.path.exists(Cfg.dataDir):
    os.mkdir(Cfg.dataDir)

# Fix the training devices and random seed.
if torch.cuda.is_available():
    np.random.seed(Cfg.seed)
    torch.cuda.manual_seed(Cfg.seed)
    if Cfg.GPUID > -1:
        torch.cuda.set_device(Cfg.GPUID)
        # Get the GPU logger.
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(Cfg.GPUID)
    device = 'cuda'
else:
    np.random.seed(Cfg.seed)
    torch.manual_seed(Cfg.seed)
    device = 'cpu'


# Set the class to encapsulate the functions.
class Trainer():
    '''
        This class is used to encapsulate all the functions which are used to train the model.\n
        This class contains two parts:\n
            - 'Trainer' is used to do the training.
            - 'Evaluator' is used to do the evaluating.
    '''

    # Set the function to train the model.
    def Trainer(model, MSE, MAE, R2, optim, trainSet, devSet, epoch, epoches, device, scheduler=None, eval=True):
        '''
            This function is used to train the model.\n
            Params:\n
                - model: The neural network model.
                - MSE: The MSE function.
                - MAE: The MAE function.
                - R2: The R2 function.
                - optim: The optimizer.
                - trainSet: The training dataset.
                - devSet: The evaluating dataset.
                - epoch: The current training epoch.
                - epoches: The total training epoches.
                - device: The device setting.
                - scheduler: The learning rate scheduler.
                - eval: The boolean value to indicate whether doing the test during the training.
        '''
        # Initialize the training loss and accuracy.
        trainMSE = []
        trainMAE = []
        trainR2 = []
        trainAccv1 = []
        trainAccv2 = []
        trainAccv3 = []
        trainAccv4 = []
        # Initialize the learning rate list.
        lrList = []
        # Set the training loading bar.
        with tqdm(total=len(trainSet), desc=f'Epoch {epoch + 1}/{epoches}', unit='batch', dynamic_ncols=True) as pbars:
            # Get the training data.
            for i, (data, label) in enumerate(trainSet):
                # Send the data into corresponding device.
                data = Variable(data).to(device)
                label = Variable(label).to(device)

                # print("data")
                # print(data.shape)
                # print("label")
                # print(label.shape)

                # Compute the prediction.
                prediction= model(data)

                # print("pre")
                # print(prediction.shape)
                # print("label")
                # print(label.shape)

                # Compute the M+SE, MAE, R2.
                mse = MSE(prediction, label.unsqueeze(1))
                mae = MAE(prediction.cpu().detach(), label.cpu().detach())
                r2 = R2(prediction.cpu().detach(), label.cpu().detach())


                # Store the cost.
                trainMSE.append(mse.item())
                trainMAE.append(mae.item())
                trainR2.append(r2.item())

                # Check whether apply the inner learning rate scheduler.
                if scheduler is not None:
                    scheduler.step()
                # Store the learning rate.
                lrList.append(optim.state_dict()['param_groups'][0]['lr'])
                # Clear the previous gradient.
                optim.zero_grad()
                # Compute the backward.
                mse.backward()
                # Update the parameters.
                optim.step()
                # Compute the accuracy.
                accuracyv1 = ((torch.abs(prediction - label) < Cfg.AccBoundv1).sum(dim=1).float() / prediction.shape[1])
                accuracyv1 = accuracyv1.sum().float() / len(accuracyv1)
                accuracyv2 = ((torch.abs(prediction - label) < Cfg.AccBoundv2).sum(dim=1).float() / prediction.shape[1])
                accuracyv2 = accuracyv2.sum().float() / len(accuracyv2)
                accuracyv3 = ((torch.abs(prediction - label) < Cfg.AccBoundv3).sum(dim=1).float() / prediction.shape[1])
                accuracyv3 = accuracyv3.sum().float() / len(accuracyv3)
                accuracyv4 = ((torch.abs(prediction - label) < Cfg.AccBoundv4).sum(dim=1).float() / prediction.shape[1])
                accuracyv4 = accuracyv4.sum().float() / len(accuracyv4)
                # Store the accuracy.
                trainAccv1.append(accuracyv1.item())
                trainAccv2.append(accuracyv2.item())
                trainAccv3.append(accuracyv3.item())
                trainAccv4.append(accuracyv4.item())
                # Update the loading bar.
                pbars.update(1)
                # Update the training info.
                pbars.set_postfix_str(' - Train MSE %.7f - Train MAE %.7f - Train R2 %.7f - Train Acc [%.4f, %.4f, %.4f, %.4f]' % (
                np.mean(trainMSE), np.mean(trainMAE), np.mean(trainR2), np.mean(trainAccv1), np.mean(trainAccv2), np.mean(trainAccv3), np.mean(trainAccv4)))
        # Close the loading bar.
        pbars.close()
        # Check whether do the evaluation.

        if eval == True:
            # Print the hint for evaluation.
            print('Evaluating...', end=' ')
            # Evaluate the model.
            evalMSE, evalMAE, evalR2, evalAccv1, evalAccv2, evalAccv3, evalAccv4 = Trainer.Evaluator(model.eval(), MSE, MAE, R2, devSet, device)

            dfmse.append(evalMSE)
            dfmae.append(evalMAE)
            dfr2.append(evalR2)

            mseloss = pd.DataFrame(data=dfmse, index=None)
            maeloss = pd.DataFrame(data=dfmae, index=None)
            r2loss = pd.DataFrame(data=dfr2, index=None)
            mseloss.to_csv("./loss/lstm_mse.csv",)
            maeloss.to_csv("./loss/lstm_mae.csv", )
            r2loss.to_csv("./loss/lstm_r2.csv", )


            # Print the evaluating result.
            print('- Eval MSE %.7f - Eval MAE %.7f - Eval R2 %.7f - Eval Acc [%.4f, %.4f, %.4f, %.4f]' % (
            evalMSE, evalMAE, evalR2, evalAccv1, evalAccv2, evalAccv3, evalAccv4), end=' ')
            # Return the training result.
            return model.train(), lrList, np.mean(trainMSE), np.mean(trainMAE), np.mean(trainR2), [np.mean(trainAccv1)/2, np.mean(trainAccv2)/2,
                                                               np.mean(trainAccv3)/2, np.mean(trainAccv4)/2], evalMSE, evalMAE, evalR2,[
                evalAccv1/2, evalAccv2/2, evalAccv3/2, evalAccv4/2]
        # Return the training result.
        return model.train(), lrList, np.mean(trainMSE), np.mean(trainMAE), np.mean(trainR2), np.mean(trainAcc), None, None

        # Set the function to evaluate the model.

    def Evaluator(model, MSE, MAE, R2, devSet, device):
        '''
            This function is used to evaluate the model.\n
            Params:\n
                - model: The nerual network model.
                - MSE: The MSE function.
                - MAE: The MAE function.
                - R2: The R2 function.
                - devSet: The evaluating dataset.
        '''
        # Initialize the evaluating loss and accuracy.
        evalMSE = []
        evalMAE = []
        evalR2 = []
        evalAccv1 = []
        evalAccv2 = []
        evalAccv3 = []
        evalAccv4 = []
        # Get the evaluating data.
        for i, (data, label) in enumerate(devSet):
            # Send the evaluating data into corresponding device.
            data = Variable(data).to(device)
            label = Variable(label).to(device)
            # Evaluate the model.
            prediction = model(data)
            # Compute the loss.
            mse = MSE(prediction, label.unsqueeze(1))
            mae = MAE(prediction.cpu().detach(), label.cpu().detach())
            r2 = R2(label.cpu().detach(), prediction.cpu().detach())
            # Store the loss.
            evalMSE.append(mse.item())
            evalMAE.append(mae.item())
            evalR2.append(r2.item())
            # Compute the accuracy.
            accuracyv1 = ((torch.abs(prediction - label) < Cfg.AccBoundv1).sum(dim=1).float() / prediction.shape[1])
            accuracyv1 = accuracyv1.sum().float() / len(accuracyv1)
            accuracyv2 = ((torch.abs(prediction - label) < Cfg.AccBoundv2).sum(dim=1).float() / prediction.shape[1])
            accuracyv2 = accuracyv2.sum().float() / len(accuracyv2)
            accuracyv3 = ((torch.abs(prediction - label) < Cfg.AccBoundv3).sum(dim=1).float() / prediction.shape[1])
            accuracyv3 = accuracyv3.sum().float() / len(accuracyv3)
            accuracyv4 = ((torch.abs(prediction - label) < Cfg.AccBoundv4).sum(dim=1).float() / prediction.shape[1])
            accuracyv4 = accuracyv4.sum().float() / len(accuracyv4)
            # Store the accuracy.
            evalAccv1.append(accuracyv1.item())
            evalAccv2.append(accuracyv2.item())
            evalAccv3.append(accuracyv3.item())
            evalAccv4.append(accuracyv4.item())


        # Return the evaluating result.
        return np.mean(evalMSE), np.mean(evalMAE), np.mean(evalR2), np.mean(evalAccv1)/2, np.mean(evalAccv2)/2, np.mean(evalAccv3)/2, np.mean(evalAccv4)/2



# Train the model.
if __name__ == "__main__":
    # Initialize the visdom server.
    vis = Logger.VisConfigurator(currentTime=currentTime, visName=f'{currentTime}')
    # Initialize the logger.
    logger = Logger.LogConfigurator(logDir=Cfg.logDir, filename=f"{currentTime}.txt")
    # Log the hyperparameters.
    logger.info('\n' + Handler.Displayer(Cfg))
    # Get the data.
    trainSet, devSet = Preprocessor.TrainData(dataDir=Cfg.dataDir, batchSize=Cfg.batchSize,
                                              trainPercent=Cfg.trainPercent)
    # Create the model.

    # model = AdaRNN(Cfg.input, Cfg.hidden, Cfg.layer, Cfg.labels, 0.2)
    # model = AdaRNN(Cfg.input, Cfg.hidden, Cfg.layer, Cfg.labels, )
    model = SHARNN(Cfg.input, Cfg.hidden, Cfg.layer, Cfg.labels)
    # model = TransformerModel(Cfg.input, Cfg.hidden, Cfg.labels, Cfg.head, Cfg.layer)
    # model = TransformerModel(input_size=Cfg.input,hidden_size=Cfg.hidden,output_size=Cfg.labels,nhead=Cfg.head,num_layers=Cfg.layer, block_size=Cfg.blocks)  # Add the block_size parameter
    # Lee = LeeOscillator()
    # model = ChaoticLSTM(Cfg.input, Cfg.hidden,Lee=Lee, chaotic=True, bidirection=False,)
    # Send the model to the corresponding device.
    model = model.to(device)
    # Create the loss function.
    MSE = nn.MSELoss()
    MAE = mean_absolute_error
    R2 = r2_score
    # Create the optimizer.
    optimizer = optim.Adam(model.parameters(), lr=Cfg.learningRate, betas=(Cfg.beta1, Cfg.beta2), eps=Cfg.epsilon)
    # optimizer = optim.RMSprop(model.parameters(), lr = Cfg.learningRate, weight_decay = Cfg.weightDecay, momentum = Cfg.momentum)
    # optimizer = optim.SGD(model.parameters(), lr = Cfg.learningRate, momentum = Cfg.momentum, weight_decay = Cfg.weightDecay)
    # Create the learning rate decay.
    scheduler = NoamScheduler(optimizer, Cfg.warmUp, Cfg.dModel)
    # Create the learning rates storer.
    lrs = []
    # Train the model.
    t1 = time.time()
    for epoch in range(Cfg.epoches):
        # Train the model.
        model, lrList, trainMSE, trainMAE, trainR2, trainAcc, evalMSE, evalMAE, evalR2, evalAcc = Trainer.Trainer(
            model=model, MSE=MSE, MAE=MAE, R2=R2, optim=optimizer,
            trainSet=trainSet, devSet=devSet,
            epoch=epoch, epoches=Cfg.epoches,
            device=device, scheduler=scheduler,
            eval=True)
        # Get the current learning rates.
        lrs.extend(lrList)
        # Store the learning rates.
        with open(f'{Cfg.logDir}//NoamScheduler//{currentTime}//learningRates.txt', 'w') as file:
            file.write(str(lrs))
        # Log the training result.
        if Cfg.GPUID > -1:
            # Compute the memory usage.
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle).used / 1024 / 1024
            print('- Memory %.4f/%.4f MB' % (memory, pynvml.nvmlDeviceGetMemoryInfo(handle).total / 1024 / 1024))
        else:
            print(' ')
        if evalMSE == None:
            if Cfg.GPUID > -1:
                logger.info(
                    'Epoch [%d/%d] -> Training: MSE [%.7f] - MAE [%.7f] - R2 [%.7f] - Acc [%.4f, %.4f, %.4f, %.4f] || lr: [%.10f] || Memory: [%.4f/%.4f] MB' % (
                        epoch + 1, Cfg.epoches, trainMSE, trainMAE, trainR2, trainAcc[0], trainAcc[1], trainAcc[2],
                        trainAcc[3],
                        optimizer.state_dict()['param_groups'][0]['lr'], memory,
                        pynvml.nvmlDeviceGetMemoryInfo(handle).total / 1024 / 1024))
            else:
                logger.info(
                    'Epoch [%d/%d] -> Training: MSE [%.7f] - MAE [%.7f] - R2 [%.7f] - Acc [%.4f, %.4f, %.4f, %.4f] || lr: [%.10f]' % (
                        epoch + 1, Cfg.epoches, trainMSE, trainMAE, trainR2, trainAcc[0], trainAcc[1], trainAcc[2],
                        trainAcc[3],
                        optimizer.state_dict()['param_groups'][0]['lr']))
        else:
            if Cfg.GPUID > -1:
                logger.info(
                    'Epoch [%d/%d] -> Training: MSE [%.7f] - MAE [%.7f] - R2 [%.7f] - Acc [%.4f, %.4f, %.4f, %.4f] || Evaluating: Training: MSE [%.7f] - MAE [%.7f] - R2 [%.7f] - Acc [%.4f, %.4f, %.4f, %.4f] || lr: [%.10f] || Memory: [%.4f/%.4f] MB' % (
                        epoch + 1, Cfg.epoches, trainMSE, trainMAE, trainR2, trainAcc[0], trainAcc[1], trainAcc[2],
                        trainAcc[3], evalMSE, evalMAE, evalR2,
                        evalAcc[0], evalAcc[1], evalAcc[2], evalAcc[3],
                        optimizer.state_dict()['param_groups'][0]['lr'],
                        memory, pynvml.nvmlDeviceGetMemoryInfo(handle).total / 1024 / 1024))
            else:
                logger.info(
                    'Epoch [%d/%d] -> Training: MSE [%.7f] - MAE [%.7f] - R2 [%.7f] - Acc [%.4f, %.4f, %.4f, %.4f] || Evaluating: MSE [%.7f] - MAE [%.7f] - R2 [%.7f] - Acc [%.4f, %.4f, %.4f, %.4f] || lr: [%.10f]' % (
                        epoch + 1, Cfg.epoches, trainMSE, trainMAE, trainR2, trainAcc[0], trainAcc[1], trainAcc[2],
                        trainAcc[3], evalMSE, evalMAE, evalR2,
                        evalAcc[0], evalAcc[1], evalAcc[2], evalAcc[3],
                        optimizer.state_dict()['param_groups'][0]['lr']))
        Logger.VisDrawer(vis=vis, epoch=epoch + 1, trainMSE=trainMSE, trainMAE=trainMAE, trainR2=trainR2,
                         evalMSE=evalMSE, evalMAE=evalMAE, evalR2=evalR2, trainAccv1=trainAcc[0],
                         trainAccv2=trainAcc[1], trainAccv3=trainAcc[2], trainAccv4=trainAcc[3],
                         evalAccv1=evalAcc[0],
                         evalAccv2=evalAcc[1], evalAccv3=evalAcc[2], evalAccv4=evalAcc[3])
        # Save the model.
        torch.save(model.state_dict(), Cfg.modelDir + f'/{currentTime}.pt')
        logger.info('Model Saved')

        t2 = time.time()
        times = t2 - t1
        print('Time taken: %d seconds' % times)
    # Close the visdom server.
    Logger.VisSaver(vis, visName=f'{currentTime}')
    mse1 = evalMSE
    mae1 = evalMAE
    r21 = evalR2

    trainSet, devSet = Preprocessor.TrainData(dataDir=Cfg.dataDir1, batchSize=Cfg.batchSize,
                                              trainPercent=Cfg.trainPercent)
    # Create the model.

    #model = AdaRNN(Cfg.input1, Cfg.hidden, Cfg.layer, Cfg.labels, 0.2)
    # model = AdaRNN(Cfg.input, Cfg.hidden, Cfg.layer, Cfg.labels, )
    model = SHARNN(Cfg.input1, Cfg.hidden, Cfg.layer, Cfg.labels)
    # model = TransformerModel(Cfg.input, Cfg.hidden, Cfg.labels, Cfg.head, Cfg.layer)
    # model = TransformerModel(input_size=Cfg.input,hidden_size=Cfg.hidden,output_size=Cfg.labels,nhead=Cfg.head,num_layers=Cfg.layer, block_size=Cfg.blocks)  # Add the block_size parameter
    # Lee = LeeOscillator()
    # model = ChaoticLSTM(Cfg.input, Cfg.hidden,Lee=Lee, chaotic=True, bidirection=False,)
    # Send the model to the corresponding device.
    model = model.to(device)
    # Create the loss function.
    MSE = nn.MSELoss()
    MAE = mean_absolute_error
    R2 = r2_score
    # Create the optimizer.
    optimizer = optim.Adam(model.parameters(), lr=Cfg.learningRate, betas=(Cfg.beta1, Cfg.beta2), eps=Cfg.epsilon)
    # optimizer = optim.RMSprop(model.parameters(), lr = Cfg.learningRate, weight_decay = Cfg.weightDecay, momentum = Cfg.momentum)
    # optimizer = optim.SGD(model.parameters(), lr = Cfg.learningRate, momentum = Cfg.momentum, weight_decay = Cfg.weightDecay)
    # Create the learning rate decay.
    scheduler = NoamScheduler(optimizer, Cfg.warmUp, Cfg.dModel)
    # Create the learning rates storer.
    lrs = []
    # Train the model.
    t1 = time.time()
    for epoch in range(Cfg.epoches):
        # Train the model.
        model, lrList, trainMSE, trainMAE, trainR2, trainAcc, evalMSE, evalMAE, evalR2, evalAcc = Trainer.Trainer(
            model=model, MSE=MSE, MAE=MAE, R2=R2, optim=optimizer,
            trainSet=trainSet, devSet=devSet,
            epoch=epoch, epoches=Cfg.epoches,
            device=device, scheduler=scheduler,
            eval=True)
        # Get the current learning rates.
        lrs.extend(lrList)
        # Store the learning rates.
        with open(f'{Cfg.logDir}//NoamScheduler//{currentTime}//learningRates.txt', 'w') as file:
            file.write(str(lrs))
        # Log the training result.
        if Cfg.GPUID > -1:
            # Compute the memory usage.
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle).used / 1024 / 1024
            print('- Memory %.4f/%.4f MB' % (memory, pynvml.nvmlDeviceGetMemoryInfo(handle).total / 1024 / 1024))
        else:
            print(' ')
        if evalMSE == None:
            if Cfg.GPUID > -1:
                logger.info(
                    'Epoch [%d/%d] -> Training: MSE [%.7f] - MAE [%.7f] - R2 [%.7f] - Acc [%.4f, %.4f, %.4f, %.4f] || lr: [%.10f] || Memory: [%.4f/%.4f] MB' % (
                        epoch + 1, Cfg.epoches, trainMSE, trainMAE, trainR2, trainAcc[0], trainAcc[1], trainAcc[2],
                        trainAcc[3],
                        optimizer.state_dict()['param_groups'][0]['lr'], memory,
                        pynvml.nvmlDeviceGetMemoryInfo(handle).total / 1024 / 1024))
            else:
                logger.info(
                    'Epoch [%d/%d] -> Training: MSE [%.7f] - MAE [%.7f] - R2 [%.7f] - Acc [%.4f, %.4f, %.4f, %.4f] || lr: [%.10f]' % (
                        epoch + 1, Cfg.epoches, trainMSE, trainMAE, trainR2, trainAcc[0], trainAcc[1], trainAcc[2],
                        trainAcc[3],
                        optimizer.state_dict()['param_groups'][0]['lr']))
        else:
            if Cfg.GPUID > -1:
                logger.info(
                    'Epoch [%d/%d] -> Training: MSE [%.7f] - MAE [%.7f] - R2 [%.7f] - Acc [%.4f, %.4f, %.4f, %.4f] || Evaluating: Training: MSE [%.7f] - MAE [%.7f] - R2 [%.7f] - Acc [%.4f, %.4f, %.4f, %.4f] || lr: [%.10f] || Memory: [%.4f/%.4f] MB' % (
                        epoch + 1, Cfg.epoches, trainMSE, trainMAE, trainR2, trainAcc[0], trainAcc[1], trainAcc[2],
                        trainAcc[3], evalMSE, evalMAE, evalR2,
                        evalAcc[0], evalAcc[1], evalAcc[2], evalAcc[3],
                        optimizer.state_dict()['param_groups'][0]['lr'],
                        memory, pynvml.nvmlDeviceGetMemoryInfo(handle).total / 1024 / 1024))
            else:
                logger.info(
                    'Epoch [%d/%d] -> Training: MSE [%.7f] - MAE [%.7f] - R2 [%.7f] - Acc [%.4f, %.4f, %.4f, %.4f] || Evaluating: MSE [%.7f] - MAE [%.7f] - R2 [%.7f] - Acc [%.4f, %.4f, %.4f, %.4f] || lr: [%.10f]' % (
                        epoch + 1, Cfg.epoches, trainMSE, trainMAE, trainR2, trainAcc[0], trainAcc[1], trainAcc[2],
                        trainAcc[3], evalMSE, evalMAE, evalR2,
                        evalAcc[0], evalAcc[1], evalAcc[2], evalAcc[3],
                        optimizer.state_dict()['param_groups'][0]['lr']))
        Logger.VisDrawer(vis=vis, epoch=epoch + 1, trainMSE=trainMSE, trainMAE=trainMAE, trainR2=trainR2,
                         evalMSE=evalMSE, evalMAE=evalMAE, evalR2=evalR2, trainAccv1=trainAcc[0],
                         trainAccv2=trainAcc[1], trainAccv3=trainAcc[2], trainAccv4=trainAcc[3],
                         evalAccv1=evalAcc[0],
                         evalAccv2=evalAcc[1], evalAccv3=evalAcc[2], evalAccv4=evalAcc[3])
        # Save the model.
        torch.save(model.state_dict(), Cfg.modelDir + f'/{currentTime}.pt')
        logger.info('Model Saved')

        t2 = time.time()
        times = t2 - t1
        print('Time taken: %d seconds' % times)
    # Close the visdom server.
    Logger.VisSaver(vis, visName=f'{currentTime}')
    mse2 = evalMSE
    mae2 = evalMAE
    r22 = evalR2

    a1 = 1
    b1 = 1
    c1 = 1
    a = mse1 / mse2
    b = mae1 / mae2
    c = r21 / r22
    days = Cfg.Days
    ba=Cfg.batchSize

    with open('4.txt', 'a') as f:
        f.write(str(mse1) + "\n")
        f.write(str(mse2) + "\n")
        f.write(str(a) + "\n")
        f.write(str(mae1) + "\n")
        f.write(str(mae2) + "\n")
        f.write(str(b) + "\n")
        f.write(str(r21) + "\n")
        f.write(str(r22) + "\n")
        f.write(str(c) + "\n")
        f.write(str(days) + "\n")
        f.write(str(ba) + "\n")
        f.write("\n")










    from matplotlib.pyplot import MultipleLocator

    plt.figure(figsize=(8, 6), dpi=120)
    plt.plot(dfmse, color='b')
    plt.xlabel('epoch', fontsize=14)
    # plt.title('Lstm EvalMse', fontsize=14)
    plt.title('AdaRNN EvalMse', fontsize=14)
    plt.legend(['EvalMse'])
    plt.savefig('./loss/lstm_mse.png')
    plt.show()

    plt.figure(figsize=(8, 6), dpi=120)
    plt.plot(dfmae, color='b')
    plt.xlabel('epoch', fontsize=14)
    # plt.title('Lstm EvalMae', fontsize=14)
    plt.title('AdaRNN EvalMae', fontsize=14)
    plt.legend(['EvalMae'])
    plt.savefig('./loss/lstm_mae.png')
    plt.show()

    plt.figure(figsize=(8, 6), dpi=120)
    plt.plot(dfr2, color='b')
    plt.xlabel('epoch', fontsize=14)
    # plt.title('Lstm EvalR2', fontsize=14)
    plt.title('AdaRNN EvalR2', fontsize=14)
    plt.legend(['EvalR2'])
    plt.savefig('./loss/lstm_r2.png')
    plt.show()
