from dgl.data import DGLDataset
import json
import dgl
import torch
import os
from dgl import save_graphs, load_graphs


class MyDataset(DGLDataset):
    """ 用于在DGL中自定义图数据集的模板：

    Parameters
    ----------
    path_v : str, 有漏洞的data。
    path_no_v : str, 不带漏洞的data。
    name: 使用数据集的类型
    type:训练或测试
    ratio:训练/测试的比例
    rounds:数据增强复制轮次
    balance:将多余的非脆弱性文件删除，保存脆弱性函数等量的文件（未使用）
    copy:是否数据增强
    url:
    raw_dir : str
        指定下载数据的存储目录或已下载数据的存储目录。默认: ~/.dgl/
    save_dir : str
        处理完成的数据集的保存目录。默认：raw_dir指定的值
    force_reload : bool
        是否重新导入数据集。默认：False
    verbose : bool
        是否打印进度信息。
    """

    def __init__(self, path_v, path_no_v, name, type = 'train',ratio=0.8,rounds=20,balance=False,copy=False,url=None, raw_dir=None, save_dir=None, force_reload=True,
                 verbose=False):
       
        self.balance = balance
        self.copy = copy
        self.path_no_v = path_no_v
        self.path_v = path_v
        self.type = type
        self.ratio = ratio
        self.graphs = []
        self.labels = []
        self.rounds = rounds

        # super(path_no_v, path_v).__init__(path_no_v=path_no_v,
        #                                   path_v=path_v,
        #                                   name='dataset_name',
        #                                   url=url,
        #                                   raw_dir=raw_dir,
        #                                   save_dir=save_dir,
        #                                   force_reload=force_reload,
        #                                   verbose=verbose)
        # super().__init__(name, url, raw_dir, save_dir, force_reload, verbose)
        super().__init__(name, url, raw_dir, save_dir, force_reload, verbose)


    def download(self):
        # 将原始数据下载到本地磁盘
        pass

    def process(self):
        print("loading data")
        with open(self.path_v,'r') as f:
            vul_data = json.load(f)
            for i in range(len(vul_data)):
                if self.type == 'test':
                    i = len(vul_data) - 1 - i
                    if i<= self.ratio * len(vul_data):
                        break
                if self.type == 'train':
                    if i>= self.ratio * len(vul_data):
                        break
                data = vul_data[i]
                g = dgl.graph((data[0], data[1]))
                # attrs = data[2]
                # att = numpy
                g.ndata['n_attr'] = torch.Tensor(data[2])
                g.edata['e_attr'] = torch.Tensor(data[3])
                #g = dgl.add_self_loop(g)
                g = g.to('cuda:0')
                self.graphs.append(g)
                self.labels.append(1)

                #copy
                if self.copy:
                    for n in range(self.rounds):
                        self.graphs.append(g)
                        self.labels.append(1)

            l_v = len(self.graphs)
            print(self.type + '/v:' + str(l_v))
        with open(self.path_no_v, 'r') as f:
            no_vul_data = json.load(f)
            l = 0
            i=0
            for i in range(len(no_vul_data)):
                if self.balance and i >= l_v:
                    break
                if self.type == 'test':
                    i = len(no_vul_data) - 1 - i
                    if i <= self.ratio * len(no_vul_data):
                        break
                if self.type == 'train':
                    if i >= self.ratio * len(no_vul_data):
                        break
                try:
                    data = no_vul_data[i]
                    g = dgl.graph((data[0], data[1]))
                    # t = torch.Tensor()
                    g.ndata['n_attr'] = torch.Tensor(data[2])
                    g.edata['e_attr'] = torch.Tensor(data[3])
                    #g = dgl.add_self_loop(g)
                    g = g.to('cuda:0')
                    self.graphs.append(g)
                    self.labels.append(0)
                except:
                    print("error data")
        l_no_v = len(self.graphs) - l_v
        print(self.type + '/no_v:' + str(l_no_v))
        print("successfully load data")

    def __getitem__(self, idx):
        # 通过idx得到与之对应的一个样本
        return self.graphs[idx],self.labels[idx]

    def __len__(self):
        # 数据样本的数量
        # l = 0
        # with open(self.path_v,'r') as f:
        #     vul_data = json.load(f)
        #     l += len(vul_data)
        # with open(self.path_v, 'r') as f:
        #     no_vul_data = json.load(f)
        #     l += len(no_vul_data)
        return len(self.graphs)

    def save(self):
        # 将处理后的数据保存至 `self.save_path` _dgl_graph.bin
        # graph_path = self.save_dir
        # labels = torch.tensor(self.labels)
        # save_graphs(graph_path, self.graphs, {'labels': labels})
        # 在Python字典里保存其他信息
        # info_path = os.path.join(self.save_path, self.mode + '_info.pkl')
        # save_info(info_path, {'num_classes': self.num_classes})
        pass
    def load(self):
        # 从 `self.save_path` 导入处理后的数据
        graph_path = self.save_dir
        self.graphs, label_dict = load_graphs(graph_path)
        for g in self.graphs:
            g.to("cuda:0")
        self.labels = label_dict['labels']

    def has_cache(self):
        # 检查在 `self.save_path` 中是否存有处理后的数据
        # graph_path = os.path.join(self.save_path)
        # info_path = os.path.join(self.save_path, self.mode + '_info.pkl')
        # return os.path.exists(self.save_dir)
        return False
