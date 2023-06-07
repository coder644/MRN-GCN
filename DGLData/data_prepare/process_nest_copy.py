import os
import numpy as np
import json
import dgl
import torch


def copy_nodes(g, idx, rounds):
    ins = g.in_edges(idx)
    outs = g.out_edges(idx)

    for i in range(rounds):
        g.add_nodes(1)
        n_id = g.num_nodes() - 1
        g.add_edges(ins[0], torch.full((len(ins[0]),), n_id, dtype=torch.int64))
        g.add_edges(torch.full((len(outs[1]),), n_id, dtype=torch.int64), outs[1])


class VertexNode(object):
    # 图的顶点
    def __init__(self, data, token, index):
        self.data = data
        self.token = token
        self.index = index

    def __str__(self):
        return self.data


class EdgeNode(object):
    # 图的顶点
    def __init__(self, eType, token, index, dst):
        self.eType = eType
        self.token = token
        self.index = index
        self.dst = dst
        assert (type(dst) == VertexNode)

    def __str__(self):
        return self.eType


keys = ['ENTRY',
        'num',
        'str']
tokens = {"ENTRY": 0,
          "num": 1,
          "str": 2}

e_keys = ['Sequential']
e_tokens = {"Sequential": 0}

nums = []
strs = []
token_ind = 3
edge_token_ind = 1
ENTRY = VertexNode('enter', 0, 0)
index = 1
e_index = 0
graph = {ENTRY: []}

expression = []
statement = []


def getNodeToken(keyword):
    global token_ind
    global keys
    global tokens
    if keyword[0:3] in ['num', 'str']:
        return tokens[keyword[0:3]]
    elif keyword in keys:
        return tokens[keyword]
    else:
        keys.append(keyword)
        tokens[keyword] = token_ind
        token_ind += 1
        return tokens[keyword]


def getEdgeToken(keyword):
    global edge_token_ind
    global e_keys
    global e_tokens
    if keyword in e_keys:
        return e_tokens[keyword]
    else:
        e_keys.append(keyword)
        e_tokens[keyword] = edge_token_ind
        edge_token_ind += 1
        return e_tokens[keyword]


def addEdge(start, e_type, e_token, dst):
    global e_index
    new_edge = EdgeNode(e_type, e_token, e_index, dst)
    graph[start].append(new_edge)
    e_index += 1


def addNode(start, data, token, e_type='Sequential', e_token=0):
    global index
    nnode = VertexNode(data, token, index)
    '''
    TODO
    '''
    # 加入边
    addEdge(start, e_type, e_token, nnode)
    # graph[start].append(nnode)

    # 加入新节点
    graph[nnode] = []
    index += 1
    return start, nnode


def StateVariableDeclaration(start, end, b):
    if b['variables'] != []:
        #         token = getNodeToken('variables')
        #         start1,end1 = addNode(end0,'variables',token)  # start1 == end0  end1 was the new node just generated
        for item in b['variables']:
            processVariables(start, end, item)
        if b['initialValue'] != None:
            processInitialValue(start, end, b)


def EventDefinition(start, end, b):
    if '}' in b['type']:
        print(3, b['type'])
    if b['parameters'] != None and b['parameters']['parameters'] != []:
        processParameters(start, end, b['parameters'])


def FunctionDefinition(start, end, b):
    # 添加FunctionDefination
    processStr(start, end, b, 'name')
    data = b['type']
    token = getNodeToken(data)
    start, end = addNode(end, data, token)

    # 处理入参
    if b['parameters']['parameters'] != []:
        processParameters(start, end, b['parameters'])

    # 处理返回值
    if b['returnParameters'] != [] and b['returnParameters'] != None:
        processParameters(start, end, b['returnParameters'])

    # 处理函数体
    if b['body'] != []:
        e_type = 'body'
        processBodyblockstatement(start, end, b['body'], e_type)


# OK
def ModifierDefinition(start, end, b):
    if b['parameters'] != [] and b['parameters']['parameters'] != []:
        processParameters(start, end, b['parameters'])
    if b['body'] != []:
        processBodyblockstatement(start, end, b['body'])


def UsingForDeclaration(start, end, b):
    #     print(b)
    if 'typeName' in b.keys() and b['typeName'] != '*':
        processTypeName(start, end, b['typeName'])
    token = getNodeToken(b['libraryName'])
    addNode(end, b['libraryName'], token)
    if '}' in b['libraryName']:
        print(9, b['libraryName'])


def StructDefinition(start, end, b):
    processStr(start, end, b, 'name')

    if b['members'] != [] and b['members'] != None:
        start1, end1 = start, end
        for item in b['members']:
            start1, end1 = processVariables(start1, end1, item)


def EnumDefinition(start, end, b):
    processStr(start, end, b, 'name')
    if b['members'] != [] and b['members'] != None:
        start1, end1 = start, end
        for item in b['members']:
            start1, end1 = processVariables(start1, end1, item)


# Done
def processVariables(start, end, b):
    '''
    TODO：
    这里要将具体的TypeName加入到边中
    已完成。
    '''

    # 变量的类型信息在typeName.name中
    if 'typeName' in b.keys() and b['typeName'] != '*':
        processTypeName(start, end, b['typeName'])

    if 'expression' in b.keys():
        if b['expression'] != None:
            processExpression(start, end, b['expression'])
    return start, end


# Done
def processTypeName(start, end, b):
    if 'name' in b.keys():
        # 获取边信息
        e_type = b['name']
        e_token = getEdgeToken(e_type)

        # 获取节点信息
        data, token = processStr(start, end, b, 'name')

        # 加入节点
        start0, end0 = addNode(end, data, token, e_type=e_type, e_token=e_token)


# Done
def processParameters(start, end, b):
    # 加入ParameterList
    token = getNodeToken(b['type'])
    start0, end0 = addNode(end, b['type'], token)
    for item in b['parameters']:
        if '}' in item['type']:
            print(26, ':', item['type'])
        processVariables(start0, end0, item)


# Done
def processStr(start, end, b, key):
    if key in b.keys():
        temp = 'str'
        if b[key] not in strs:
            strs.append(b[key])
            temp = 'str' + str(len(strs) + 1)
        else:
            temp = 'str' + str(strs.index(b[key]) + 1)

        token = 2  # num == 1, str == 2
        # 加入变量名的节点，toDelete
        return temp, token


def processNum(start, end, b, key):
    if key in b.keys():
        temp = 'num'
        if b[key] not in nums:
            nums.append(b[key])
            temp = 'num' + str(len(nums) + 1)
        else:
            temp = 'num' + str(nums.index(b[key]) + 1)

        token = 1  # num == 1, str == 2
        addNode(end, temp, token)


def processInitialValue(start, end, b):
    token = getNodeToken('initialValue')
    start0, end0 = addNode(end, 'initialValue', token)

    token = getNodeToken(b['initialValue']['type'])
    addNode(end0, b['initialValue']['type'], token)

    if 'name' in b['initialValue'].keys():
        temp = 'str'
        if b['initialValue']['name'] not in strs:
            strs.append(b['initialValue']['name'])
            temp = 'str' + str(len(strs) + 1)
        else:
            temp = 'str' + str(strs.index(b['initialValue']['name']) + 1)

        token = 2  # num == 1, str == 2
        addNode(end0, temp, token)

    elif 'number' in b['initialValue'].keys():
        temp = 'num'
        if b['initialValue']['number'] not in nums:
            nums.append(b['initialValue']['number'])
            temp = 'num' + str(len(nums) + 1)
        else:
            temp = 'num' + str(nums.index(b['initialValue']['number']) + 1)

        token = 1  # num == 1, str == 2
        addNode(end0, temp, token)


# 修改
def processBodyblockstatement(start, end, b, e_type = 'Sequential'):
    # 加入Block节点:ContractDefination -> Block/Expression......
    start0, end0 = start, end
    # token = getNodeToken(b['type'])
    # start0, end0 = addNode(end, b['type'], token)

    if b['type'] == 'InLineAssemblyStatement':
        for item in b['body']['operations']:
            processExpression(start0, end0, item)

    elif b['type'] == 'VariableDeclarationStatement':
        if b['variables'] != None and b['variables'] != []:
            start1, end1 = start0, end0
            for item in b['variables']:
                start1, end1 = processVariables(start1, end1, item)
        if b['initialValue'] != None:
            processInitialValue(start0, end0, b)

    elif b['type'] == 'ExpressionStatement':
        token = getNodeToken(b['type'])
        e_token = getEdgeToken(e_type)
        start0, end0 = addNode(end, b['type'], token, e_type, e_token)
        if b['expression'] != None:
            processExpression(start0, end0, b['expression'])

    elif b['type'] == 'IfStatement':
        token = getNodeToken('condition')
        start1, end1 = addNode(end0, 'condition', token)
        start2, end2 = start1, end1
        processExpression(start1, end1, b['condition'])
        if b['TrueBody'] != None and b['TrueBody'] != ';':
            token = getNodeToken('TrueBody')
            start2, end2 = addNode(end1, 'TrueBody', token)
            processBodyblockstatement(start2, end2, b['TrueBody'])

        if b['FalseBody'] != None and b['FalseBody'] != ';':
            token = getNodeToken('FalseBody')
            start3, end3 = addNode(end2, 'FalseBody', token)
            processBodyblockstatement(start3, end3, b['FalseBody'])

    elif b['type'] == 'EmitStatement':
        processExpression(start0, end0, b['eventCall'])

    elif b['type'] == 'Identifier':
        processExpression(start0, end0, b)

    elif b['type'] == 'BooleanLiteral':
        processExpression(start0, end0, b)

    elif b['type'] == 'IndexAccess':
        processExpression(start0, end0, b)

    elif b['type'] == 'UnaryOperation':
        processExpression(start0, end0, b)

    elif b['type'] == 'FunctionCall':
        processExpression(start0, end0, b)

    elif b['type'] == 'MemberAccess':
        processExpression(start0, end0, b['expression'])

    elif b['type'] == 'ForStatement':
        start1, end1 = start0, end0
        start2, end2 = start0, end0
        start3, end3 = start0, end0
        if b['initExpression'] != None:
            token = getNodeToken('initExpression')
            start1, end1 = addNode(end0, 'initExpression', token)
            processBodyblockstatement(start1, end1, b['initExpression'])
        if b['conditionExpression'] != None:
            token = getNodeToken('conditionExpression')
            start2, end2 = addNode(end1, 'conditionExpression', token)
            processExpression(start2, end2, b['conditionExpression'])
        if b['loopExpression'] != None:
            token = getNodeToken('loopExpression')
            start3, end3 = addNode(end2, 'loopExpression', token)
            processBodyblockstatement(start3, end3, b['loopExpression'])

        if b['body'] != []:
            #             print(b['body'])
            #             print(b['body']['type'])
            #             print(b['body'],'*********')
            token = getNodeToken('body')
            start4, end4 = addNode(end3, 'body', token)
            processBodyblockstatement(start4, end4, b['body'])
    #             b = b['body']['statements']
    #             tt = []
    #             for item in b:
    #                 if item != None and item != ';':
    #                     tt = processBodyblockstatement(item)
    #                 res.extend(tt)

    elif b['type'] == 'WhileStatement':
        token = getNodeToken('condition')
        start1, end1 = addNode(end0, 'condition', token)
        processExpression(start1, end1, b['condition'])
        if b['body'] != []:
            token = getNodeToken('body')
            start2, end2 = addNode(end1, 'body', token)
            processBodyblockstatement(start2, end2, b['body'])

    elif b['type'] == 'TupleExpression':
        start1, end1 = start0, end0
        for item in b['components']:
            if item != None:
                # 增加 processExpression reutrn start,end.
                start1, end1 = processExpression(start1, end1, item)

    elif b['type'] == 'Conditional':
        token = getNodeToken('condition')
        start1, end1 = addNode(end0, 'condition', token)
        processExpression(start1, end1, b['condition'])

        if b['TrueExpression'] != None:
            token = getNodeToken('TrueExpression')
            start2, end2 = addNode(end1, 'TrueExpression', token)
            processExpression(start2, end2, b['TrueExpression'])

        if b['FalseExpression'] != None:
            token = getNodeToken('FalseExpression')
            start3, end3 = addNode(end2, 'FalseExpression', token)
            processExpression(start3, end3, b['FalseExpression'])

    elif b['type'] == 'NumberLiteral':
        processNum(start, end, b, 'number')

    elif b['type'] == 'StringLiteral':
        processStr(start, end, b, 'value')

    elif b['type'] == 'DoWhileStatement':
        token = getNodeToken('condition')
        start1, end1 = addNode(end0, 'condition', token)
        processExpression(start1, end1, b['condition'])
        if b['body'] != []:
            token = getNodeToken('body')
            start2, end2 = addNode(end1, 'body', token)
            processBodyblockstatement(start2, end2, b['body'])

    elif b['type'] == 'Block':
        if b['statements'] != [] and b['statements'] != None:
            b = b['statements']
            start1, end1 = start0, end0
            for item in b:
                if item != None and item != ';':
                    start1, end1 = processBodyblockstatement(start1, end1, item)
    return start0, end0


# 修改
def processExpression(start, end, b, e_type='Sequential'):
    start0, end0 = start, end

    if b['type'] == 'AssemblyExpression':
        if 'functionName' in b.keys():
            processStr(start0, end0, b, 'functionName')
        if 'arguments' in b.keys() and b['arguments'] != []:
            token = getNodeToken('arguments')
            start1, end1 = addNode(end0, 'arguments', token)
            for item in b['arguments']:
                #                 if item['type'] == 'AssemblyExpression':
                start1, end1 = processExpression(start1, end1, item)

    elif b['type'] == 'AssemblyLocalDefinition':
        if b['names'] != []:
            token = getNodeToken('names')
            start1, end1 = addNode(end0, 'names', token)
            for item in b['names']:
                processExpression(start1, end1, item)
        if b['expression'] != None:
            processExpression(start0, end0, b['expression'])

    elif b['type'] == 'AssemblySwitch':
        processExpression(start0, end0, b['expression'])

        processStr(start0, end0, b, 'functionName')

        if 'arguments' in b.keys() and b['arguments'] != []:
            token = getNodeToken('arguments')
            start1, end1 = addNode(end0, 'arguments', token)
            for item in b['arguments']:
                start1, end1 = processExpression(start1, end1, item)
            start1, end1 = start0, end0
            for item in b['cases']:
                start1, end1 = processExpression(start1, end1, item)

    elif b['type'] == 'AssemblyCase':
        if b['block'] != None:
            processExpression(start0, end0, b['block'])
        if b['value'] != None:
            processExpression(start0, end0, b['value'])

    elif b['type'] == 'AssemblyBlock':
        if b['operations'] != None:
            start1, end1 = start0, end0
            for item in b['operations']:
                start1, end1 = processExpression(start1, end1, item)

    elif b['type'] == 'AssemblyAssignment':
        if b['names'] != []:
            start1, end1 = start0, end0
            for item in b['names']:
                if item != None:
                    start1, end1 = processExpression(start1, end1, item)

    # Done
    elif b['type'] == 'BinaryOperation':
        token = getNodeToken(b['type'])
        e_token = getEdgeToken(e_type)
        start0, end0 = addNode(end, b['type'], token, e_type, e_token)
        # 处理operator
        e_type = 'operator'
        e_token = getEdgeToken(e_type)
        token = getNodeToken(b['operator'])
        start1, end1 = addNode(end0, b['operator'], token, e_type=e_type, e_token=e_token)
        if '}' in b['operator']:
            print(16, ':', b['operator'])

        # 处理left
        e_type = 'left'
        processExpression(start0, end0, b['left'], e_type)

        # 处理right
        e_type = 'right'
        processExpression(start0, end0, b['right'], e_type)

    # Done
    elif b['type'] == 'Identifier':
        data, token = processStr(start, end, b, 'name')
        e_token = getEdgeToken(e_type)
        # 这里加入bianryOperation --(left/right)--> 变量名
        addNode(end0, data, token, e_type, e_token)


    elif b['type'] == 'MemberAccess':
        e_token = getEdgeToken(e_type)
        #         processExpression(start0,end0,b['expression'])
        data, token = processStr(start0, end0, b, 'memberName')
        start1, end1 = addNode(end0,data,token,e_type,e_token)
        e_type = b['type']
        processExpression(start1, end1, b['expression'], e_type)


    # 修改
    elif b['type'] == 'FunctionCall':
        e_type = 'FunctionCall'
        processExpression(start0, end0, b['expression'], e_type)
        if b['names'] is not None and b['names'] != []:
            token = getNodeToken('names')
            start1, end1 = addNode(end0, 'names', token)
            #             print('hello',b['names'])
            for item in b['names']:
                if type(item) != str:
                    processExpression(start1, end1, item)

        if 'typeName' in b.keys() and b['typeName'] != None:
            processTypeName(start0, end0, b['typeName'])

        if b['arguments'] is not None and b['arguments'] != []:
            # token = getNodeToken('arguments')
            # start1, end1 = addNode(end0, 'arguments', token)
            e_type = 'arguments'
            for item in b['arguments']:
                start1, end1 = processExpression(start0, end0, item,e_type)

    elif b['type'] == 'ElementaryTypeNameExpression':
        if 'typeName' in b.keys() and b['typeName'] != '*':
            processTypeName(start0, end0, b['typeName'])

    elif b['type'] == 'BooleanLiteral':
        if b['value'] not in booleanLiteral:
            booleanLiteral.append(b['value'])
        token = getNodeToken(str(b['value']))
        addNode(end, str(b['value']), token)

    elif b['type'] == 'UnaryOperation':
        token = getNodeToken(b['operator'])
        start1, end1 = addNode(end, b['operator'], token)
        if '}' in b['operator']:
            print(21, ':', b['operator'])
        processExpression(start1, end1, b['subExpression'])

    elif b['type'] == 'IndexAccess':
        token = getNodeToken('base')
        start1, end1 = addNode(end0, 'base', token)
        processExpression(start1, end1, b['base'])

        token = getNodeToken('index')
        start2, end2 = addNode(end1, 'index', token)
        processExpression(start2, end2, b['index'])

    elif b['type'] == 'NumberLiteral':
        processNum(start, end, b, 'number')

    elif b['type'] == 'TupleExpression':
        start1, end1 = start0, end0
        for item in b['components']:
            if item != None:
                start1, end1 = processBodyblockstatement(start1, end1, item)

    elif b['type'] == 'StringLiteral':
        processNum(start, end, b, 'value')

    elif b['type'] == 'DecimalNumber':
        processNum(start, end, b, 'value')

    return start0, end0


def getParseResult(file):
    if os.path.getsize(file) != 0:
        f = open(file)
        jsonfile = json.load(f)
        return jsonfile


Contractsubkeywords = ['StateVariableDeclaration', 'EventDefinition', 'FunctionDefinition', 'ModifierDefinition',
                       'UsingForDeclaration', 'StructDefinition', 'EnumDefinition']


def processJsonFile(parseresult):
    if parseresult['children'] != []:
        if parseresult['children'][0] != None:
            start1 = ENTRY
            end1 = ENTRY
            for i in range(1, len(parseresult['children'])):
                if parseresult['children'][i] != None:
                    a = parseresult['children'][i]
                    token = getNodeToken(a['type'])
                    start1, end1 = addNode(end1, a['type'], token)
                    if 'subNodes' in a.keys():
                        b = a['subNodes']
                        for item in b:
                            if 'name' in item.keys() and 'type' not in item.keys():
                                t = item['name']
                            elif 'type' in item.keys():
                                t = item['type']
                            if t == 'StateVariableDeclaration':
                                StateVariableDeclaration(start1, end1, item)
                            elif t == 'EventDefinition':
                                EventDefinition(start1, end1, item)
                            elif t == 'FunctionDefinition':
                                FunctionDefinition(start1, end1, item)
                            elif t == 'ModifierDefinition':
                                ModifierDefinition(start1, end1, item)
                            elif t == 'UsingForDeclaration':
                                UsingForDeclaration(start1, end1, item)
                            elif t == 'StructDefinition':
                                StructDefinition(start1, end1, item)
                            elif t == 'EnumDefinition':
                                EnumDefinition(start1, end1, item)
    return graph


def vertexnode2dict(node):
    t = {'data': '', 'token': '', 'index': ''}
    t['data'] = node.data
    t['token'] = node.token
    t['index'] = node.index
    return str(t)


def write2file(file, graph):
    path = './graph_vlunerable/'
    f = open(path + file, 'w')
    dic = {}
    for node in graph.keys():
        li = graph[node]
        t = vertexnode2dict(node)
        dic[t] = []
        for item in li:
            temp = vertexnode2dict(item)
            dic[t].append(temp)
    j = json.dumps(dic)
    f.write(j)
    f.close()


functioncall = []
booleanLiteral = []


# main
def getIndex(file, path):
    t = open(path + file.split('+')[0] + '.txt', 'r')
    length = int(t.readline().split(' ')[0])
    np.loadtxt(t, max_rows=length)
    nodes = t.readline().split(',')
    nodes[-1] = nodes[-1][:-1]
    node_index = {node: i for node, i in zip(nodes, range(length))}
    t.close()
    return node_index


def processMain(file):
    global graph
    global index
    global nums
    global strs
    global e_index
    nums = []
    strs = []
    index = 1
    e_index = 0
    graph = {ENTRY: []}
    parseresult = getParseResult(file)
    try:
        processJsonFile(parseresult)
    except:
        return False
    # for graph, file in zip(Graph, files1):
    #     f = open(path3 + file.split('.')[0] + '.txt', 'w')
    #     pFile = []
    #     for item in graph.keys():
    #         pFile.append(item.token)
    #         for li in graph[item]:
    #             pFile.append(li.token)
    #     #     print(pFile.shape)
    #     np.savetxt(f, np.array(pFile), fmt='%d')
    #     f.close()

    return graph


def GenerateData(path,data_path):
    i = 0
    graphs = []
    data = []

    files = os.listdir(path)
    for file in files:
        i = i + 1
        if i % 100 == 0:
            print(i)
        graph = processMain(path + file)
        if(graph is False):
            continue
        src = []
        dst = []
        n_attr = []
        e_attr = []
        for node in graph.keys():
            n_attr.append(node.token)
            if (graph[node] != []):
                for edge in graph[node]:
                    src.append(node.index)
                    dst.append(edge.dst.index)
                    e_attr.append(edge.token)
        graph_data = [src, dst, n_attr, e_attr]
        data.append(graph_data)
        graphs.append(graph)

    # with open(data_path, 'w+') as f:
    #     json.dump(data, f)




def generate_data(global_graph_path,f_v_path,f_no_v_path,global_save_path,f_save_path,rounds):
    global_graphs = os.listdir(global_graph_path)

    #data to fill
    file_graph_list = []
    func_data_list = []

    # number of files
    ngs = len(global_graphs)
    total_nums = 0
    n = 0
    # n_fun = 0

    for ggraph in global_graphs:
        n+=1
        # ggraph = "0x0a6f2d9d82baef6dd0f3a2f4679287680aad1fc0.txt"
        # open a single file
        with open(global_graph_path + ggraph , 'r') as f:

            #edges
            file_src,file_dst = [],[]
            #包含的函数个数
            func_num = int(f.readline())
            num_f_in_this_file = func_num
            #标签列表和函数名
            label = np.loadtxt(f,max_rows=func_num).tolist()
            func_names = f.readline().strip().split(',')
            #脆弱函数列表
            v_idx = []
            for i in range(func_num):
                if(label[i] == 1):
                    v_idx.append(i)
            #没有脆弱性函数
            if len(v_idx)==0:
                f.close()
                continue
            graph = True

            #临界矩阵
            adj = np.loadtxt(f)
            for i in range(adj.shape[0]):
                for j in range(adj.shape[1]):
                    if adj[i][j] == 1:
                    #if adj[i][j] == 1 and i != j:
                            file_src.append(i)
                            file_dst.append(j)

            # 在文件中复制脆弱函数
            for idx in v_idx:
                for i in range(rounds):
                    label.append(1.0)
                    file_src.append(func_num)
                    file_dst.append(func_num)
                    func_num += 1

            #文件数据生成完成
            file_graph = [func_num,label,file_src,file_dst]


            # 将文件中的函数加入临时列表
            func_in_this_file = []
            for i in range(len(func_names)):

                #获取函数图
                fn = func_names[i]
                if label[i] == 0:
                    f_path = f_no_v_path
                else:
                    f_path = f_v_path
                f_name = ggraph[:-4] + '+' + fn + ".sol.json"
                graph = processMain(f_path + f_name)
                if (graph is False):
                    print('----------------------------------')
                    break


                src = []
                dst = []
                n_attr = []
                e_attr = []
                for node in graph.keys():
                    n_attr.append(node.token)
                    if (graph[node] != []):
                        for edge in graph[node]:
                            src.append(node.index)
                            dst.append(edge.dst.index)
                            e_attr.append(edge.token)

                # 将该函数数据加入临时列表
                graph_data = [src, dst, n_attr, e_attr]
                if (len(src) == 0):
                    graph = False
                    print('----------------------------------')
                    break
                func_in_this_file.append(graph_data)

            try:
                assert num_f_in_this_file == len(func_in_this_file)
            except:
                print('difference in nums of funcs')
                continue

            # 复制脆弱函数
            for idx in v_idx:
                for r in range(rounds):
                    func_in_this_file.append(func_in_this_file[idx])
            if graph is False:
                print('----------------------------------')
                continue
            # try:
            #     assert file_graph[0] == len(func_in_this_file)
            # except:
            #     print("different func nums ")

            #添加至最终列表
            file_graph_list.append(file_graph)
            for g in func_in_this_file:
                func_data_list.append(g)
            total_nums += file_graph[0]

            if(n == 50000):
                break
            print(n,ngs,'()', total_nums, len(func_data_list))

    print(len(file_graph_list))


    #save data
    with open(global_save_path, 'w+') as f:
        json.dump(file_graph_list, f)

    with open(f_save_path, 'w+') as f:
        json.dump(func_data_list, f)


if __name__ == '__main__':
    print("")
    # reentrancy 10
    # global_graph_path = "/media/bmw_lab/2eb7452f-ae82-4707-82ed-218aabcd7aaf/bmw_lab/franda/Func_level_Nest_GCN/data/func_Reentrancy/GlobalGraph/"
    # v_f_path = "/media/bmw_lab/2eb7452f-ae82-4707-82ed-218aabcd7aaf/bmw_lab/franda/Func_level_Nest_GCN/data/func_Reentrancy/Ast_v/"
    # no_v_f_path = "/media/bmw_lab/2eb7452f-ae82-4707-82ed-218aabcd7aaf/bmw_lab/franda/Func_level_Nest_GCN/data/func_Reentrancy/Ast_no_v/"
    # global_save_path = "../raw_data/nest_new/file_reen_nest.json"
    # f_save_path = "../raw_data/nest_new/func_reen_nest.json"

    #timestamp_dependency 20
    # global_graph_path = "/media/bmw_lab/2eb7452f-ae82-4707-82ed-218aabcd7aaf/bmw_lab/franda/Func_level_Nest_GCN/data/func_timestamp/GlobalGraph/"
    # v_f_path = "/media/bmw_lab/2eb7452f-ae82-4707-82ed-218aabcd7aaf/bmw_lab/franda/Func_level_Nest_GCN/data/func_timestamp/Ast_v/"
    # no_v_f_path = "/media/bmw_lab/2eb7452f-ae82-4707-82ed-218aabcd7aaf/bmw_lab/franda/Func_level_Nest_GCN/data/func_timestamp/Ast_no_v/"
    # global_save_path = "../raw_data/nest_new/file_time_nest10.json"
    # f_save_path = "../raw_data/nest_new/func_time_nest10.json"

    # un_checked_calls
    # global_graph_path = "/media/bmw_lab/2eb7452f-ae82-4707-82ed-218aabcd7aaf/bmw_lab/franda/Func_level_Nest_GCN/data/func_Unchecked_low_Calls/GlobalGraph/"
    # v_f_path = "/media/bmw_lab/2eb7452f-ae82-4707-82ed-218aabcd7aaf/bmw_lab/franda/Func_level_Nest_GCN/data/func_Unchecked_low_Calls/Ast_v/"
    # no_v_f_path = "/media/bmw_lab/2eb7452f-ae82-4707-82ed-218aabcd7aaf/bmw_lab/franda/Func_level_Nest_GCN/data/func_Unchecked_low_Calls/Ast_no_v/"
    # global_save_path = "../raw_data/nest/file_call_nest.json"
    # f_save_path = "../raw_data/nest/func_call_nest.json"

    # ari 5
    global_graph_path = "/media/bmw_lab/2eb7452f-ae82-4707-82ed-218aabcd7aaf/bmw_lab/franda/Func_level_Nest_GCN/data/func_Arithmetic/GlobalGraph/"
    v_f_path = "/media/bmw_lab/2eb7452f-ae82-4707-82ed-218aabcd7aaf/bmw_lab/franda/Func_level_Nest_GCN/data/func_Arithmetic/AST_vulnerable/"
    no_v_f_path = "/media/bmw_lab/2eb7452f-ae82-4707-82ed-218aabcd7aaf/bmw_lab/franda/Func_level_Nest_GCN/data/func_Arithmetic/AST_normal/"
    global_save_path = "../raw_data/nest_new/file_ari_nest.json"
    f_save_path = "../raw_data/nest_new/func_ari_nest.json"
    generate_data(global_graph_path, v_f_path, no_v_f_path, global_save_path, f_save_path,10)