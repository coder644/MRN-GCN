from dgl.data import DGLDataset
import json
import dgl
import torch
import os
from dgl import save_graphs, load_graphs


class NestDataset(DGLDataset):
    """ 用于在DGL中自定义图数据集的模板：

    Parameters
    ----------
    file_graph_path : str, 文件级图。
    func_graph_path : str, 函数级图。
    name: 使用数据集的类型
    type:训练或测试
    ratio:训练/测试的比例
    
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

    def __init__(self, file_graph_path, func_graph_path, name, type = 'train',ratio=1,url=None, raw_dir=None, save_dir=None, force_reload=True,
                 verbose=False):
        self.file_graph_path = file_graph_path
        self.func_graph_path = func_graph_path
        self.type = type
        self.ratio = ratio
        self.file_graphs = []
        self.func_graphs = []
        self.num_funcs = []
        self.labels = []

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
        f = open(self.file_graph_path,'r')
        ff = open(self.func_graph_path,'r')

        if self.type == 'train':
            # generate file_g_list
            file_data = json.load(f)
            print("number of files:",len(file_data))
            num_train = int(self.ratio * len(file_data))
            num_funcs = 0
            for i in range(num_train):
                data = file_data[i]

                #num_funs***********************************
                self.num_funcs.append(data[0])
                self.labels.append(data[1])
                num_funcs += len(data[1])
                if(data[0]!=len(data[1])):
                    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!++++' + str(i))
                g = dgl.graph((data[2], data[3]))
                g = g.to('cuda:0')
                self.file_graphs.append(g)
            print(num_funcs)
            # generate func_g_list
            func_data = json.load(ff)
            # print(num_funcs,len(func_data))
            for i in range(num_funcs):
                data = func_data[i]
                g = dgl.graph((data[0], data[1]))
                g.ndata['n_attr'] = torch.Tensor(data[2])
                g.edata['e_attr'] = torch.Tensor(data[3])
                # g = dgl.add_self_loop(g)
                g = g.to('cuda:0')
                self.func_graphs.append(g)
            # print(num_funcs,len(self.func_graphs))
        elif self.type == 'test':
            #skip train data
            file_data = json.load(f)
            num_train = int(self.ratio * len(file_data))
            func_begin = 0
            num_funs = 0
            for i in range(num_train):
                data = file_data[i]
                func_begin += len(data[1])

            # generate file_g_list
            for i in range(num_train,len(file_data)):
                data = file_data[i]
                self.num_funcs.append(data[0])
                self.labels.append(data[1])
                num_funs += len(data[1])
                if(data[0]!=len(data[1])):
                    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!++++' + str(i))
                g = dgl.graph((data[2], data[3]))
                g = g.to('cuda:0')
                self.file_graphs.append(g)
                # print(len(g.nodes()),data[0])

            # generate func_g_list
            func_data = json.load(ff)
            for i in range(func_begin,len(func_data)):
                data = func_data[i]
                g = dgl.graph((data[0], data[1]))
                g.ndata['n_attr'] = torch.Tensor(data[2])
                g.edata['e_attr'] = torch.Tensor(data[3])
                # g = dgl.add_self_loop(g)
                g = g.to('cuda:0')
                self.func_graphs.append(g)
            # n_nodes = 0
            # for g in self.file_graphs:
            #     n_nodes += len(g.nodes())
            # print(n_nodes,len(self.func_graphs))
        print("successfully load data")
        # print(len(self.file_graphs))


    def __getitem__(self, idx):
        # 通过idx得到与之对应的一个样本, label with no meaning
        return self.file_graphs[idx],1

    def __len__(self):
        # 数据样本的数量
        # l = 0
        # with open(self.path_v,'r') as f:
        #     vul_data = json.load(f)
        #     l += len(vul_data)
        # with open(self.path_v, 'r') as f:
        #     no_vul_data = json.load(f)
        #     l += len(no_vul_data)
        return len(self.file_graphs)

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
