import json

from evaluations import get_binary_metrics,roc_from_scratch
import torch
import dgl

from train.time.model.nestEGCN import nestGCN,nestEGCN,nestEGCNs
from DGLData.NestDataset import NestDataset
import numpy as np
from dgl.dataloading import GraphDataLoader
from sklearn.metrics import roc_curve, auc
rounds = 30
EPOCH = 500

trainset = NestDataset(file_graph_path='../../DGLData/raw_data/nest_new/file_time_nest.json',
                       func_graph_path="../../DGLData/raw_data/nest_new/func_time_nest.json",
                       save_dir="../../DGLData/cache/reen_cd.bin",
                       name='time', force_reload=True, ratio=0.85
                       )
testset = NestDataset(file_graph_path='../../DGLData/raw_data/nest_new/file_time_nest.json',
                      func_graph_path="../../DGLData/raw_data/nest_new/func_time_nest.json",
                      save_dir="../../DGLData/cache/reen_cd.bin", type='test',
                      name='time', force_reload=True, ratio=0.85
                      )
trainloader = GraphDataLoader(trainset,batch_size=32,drop_last=False,shuffle = False)
testloader = GraphDataLoader(testset,batch_size=16,drop_last=False,shuffle = False)


for round in range(rounds):
    MODEL = 'nestEGCN'
    sa = True
    if MODEL == 'nestEGCN':
        if sa == False:
            model = nestEGCN().cuda().train()
        else:
            model = nestEGCNs().cuda().train()
    if MODEL == 'nestGCN':
        model = nestGCN().cuda().train()

    opt = torch.optim.SGD(model.parameters(),lr=0.0001,momentum=0.0005)    #MRNGCN 0.005  #MRNGCNwosa 0.0015  #NGCN 0.0001
    criterion = torch.nn.CrossEntropyLoss()
    train_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    print('N trainable parameters:', np.sum([p.numel() for p in train_params]))
    print(model)

    # schedule = torch.optim.lr_scheduler.StepLR(optimizer, step_size=VALIDATE_EVERY * 10, gamma=0.6)

    best_loss = 100
    train_loss, train_acc = 0, 0
    flag = False
    for epoch in range(EPOCH):

        predicts = np.array([])
        ys = np.array([])
        total_loss = 0
        idx_file = 0
        idx_func = 0

        # train 每次送入 batchsize个file和label列表
        for batched_file_graph, labels in trainloader:
            lab = []
            num_func = 0    
            for i in range(len(labels)):
                num_func += trainset.num_funcs[idx_file]
                l = trainset.labels[idx_file] #函数的label
                if type(l) == int:
                    lab.append(l)
                else:
                    lab.extend(l)
                idx_file += 1
            ####最后num_fuc 就是一个batch file 的fuction 数,lab是一个batch file对应的标签
            batched_func_graph = dgl.batch(trainset.func_graphs[idx_func:idx_func+num_func]).to('cuda:0') #取对应的num_fuc个函数图
            idx_func = idx_func + num_func
            feats = batched_func_graph.ndata['n_attr'].cuda()
            e_feats = batched_func_graph.edata['e_attr'].cuda()
            e_feats = e_feats.view(-1,1).long().cuda()  ######
            feats = feats.view(-1,1).long().cuda()  #####?
            # prepare labels
            labels = torch.tensor(lab).long().cuda()
            if MODEL == 'nestEGCN':
                logits = model(batched_file_graph, batched_func_graph, feats,e_feats)
            else:
                logits = model(batched_file_graph,batched_func_graph,feats)
            loss = criterion(logits,labels)
            total_loss += loss.item()
            opt.zero_grad()
            loss.backward()
            opt.step()
            probability = torch.nn.functional.softmax(logits, dim=1).cuda().cpu()  # 计算softmax，即该图片属于各类的概率
            predict_dim1 = torch.argmax(probability, dim=1).detach().cpu().numpy()
            # acc, recall, prec, f_beta = get_binary_metrics(pred_y=predict_dim1, true_y=labels.cpu().numpy())
            #
            # with open('train_epoch_results.txt','a+') as f:
            #     f.write(f"train: epoch:{epoch}, acc: {acc}, recall: {recall}, precision: {prec}, f_beta: {f_beta}\n")
            # predict_dim11 = predict_dim1.cuda().data.cpu().numpy()
            predicts = np.concatenate((predicts, predict_dim1))
            ys = np.concatenate((ys,labels.cpu().numpy())) ####

        # train_result
        acc, recall, prec, f_beta = get_binary_metrics(pred_y=predicts, true_y=ys)
        if recall==0 and prec ==0:
            break

        print(
            f"train: epoch:{epoch}, loss: {total_loss}, acc: {acc}, recall: {recall}, precision: {prec}, f_beta: {f_beta}")
        if total_loss <= best_loss:
            best_loss = total_loss
            flag = True
        #test
        with torch.no_grad():
            r = 0
            prob = np.array([])
            predicts = np.array([])
            ys = np.array([])
            total_loss = 0
            idx_file = 0
            idx_func = 0
            bads = []
            for batched_file_graph, labels in testloader:
                r += 1
                #print(r)
                lab = []
                num_func = 0
                for i in range(len(labels)):
                    num_func += testset.num_funcs[idx_file]
                    l = testset.labels[idx_file]
                    if type(l) == int:
                        lab.append(l)
                    else:
                        lab.extend(l)
                    idx_file += 1
                batched_func_graph = dgl.batch(testset.func_graphs[idx_func:idx_func+num_func]).to('cuda:0')
                idx_func = idx_func + num_func

                feats = batched_func_graph.ndata['n_attr'].cuda()
                e_feats = batched_func_graph.edata['e_attr'].cuda()
                e_feats = e_feats.view(-1,1).long().cuda()
                feats = feats.view(-1,1).long().cuda()
                # prepare labels
                labels = torch.tensor(lab).long().cuda()
                if MODEL == 'nestEGCN':
                    logits = model(batched_file_graph, batched_func_graph, feats,e_feats)
                else:
                    logits = model(batched_file_graph,batched_func_graph,feats)
                loss = criterion(logits,labels)
                total_loss += loss.item()

                probability = torch.nn.functional.softmax(logits, dim=1).cuda().cpu()  # 计算softmax，即该图片属于各类的概率

                pred_max = probability[:,1].detach().cpu().numpy()
                predict_dim1 = torch.argmax(probability, dim=1).detach().cpu().numpy()

                # predict_dim11 = predict_dim1.cuda().data.cpu().numpy()
                acc, recall, prec, f_beta = get_binary_metrics(pred_y=predict_dim1, true_y=labels.cpu().numpy())
                if acc<0.5 or recall<0.5 or prec<0.5:
                    bads.append(r)
                # with open('test_epoch_results.txt', 'a+') as f:
                #     f.write(f"r:{r}, acc: {acc}, recall: {recall}, precision: {prec}, f_beta: {f_beta}\n")
                predicts = np.concatenate((predicts, predict_dim1))
                ys = np.concatenate((ys,labels.cpu().numpy()))
                # print('probability', probability, probability[:,1].shape)
                # print('pred_max', pred_max, pred_max.shape)
                # print('predicts', predicts, predicts.shape)
                # print('ys', ys, ys.shape)
                prob = np.concatenate((prob, pred_max))
            # print('predicts', prob)
            # print('ys', ys)
            acc, recall, prec, f_beta = get_binary_metrics(pred_y=predicts, true_y=ys)

            print(
                f"test: epoch:{epoch}, loss: {total_loss}, acc: {acc}, recall: {recall}, precision: {prec}, f_beta: {f_beta}")
            # print(bads)
