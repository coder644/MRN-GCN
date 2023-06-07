# from BasicNodes import VertexNode, EdgeNode
import numpy as np
import os
import json


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

    print("")
    with open(data_path, 'w+') as f:
        json.dump(data, f)


if __name__ == '__main__':
    # gs = processMain("testASTReen.json")
    path = "/media/bmw_lab/2eb7452f-ae82-4707-82ed-218aabcd7aaf/bmw_lab/franda/Data Process/Vulnerable_File/Unchecked_Low_Calls/AST/"
    data_path = "../raw_data/data.json"

    reen_path = "/media/bmw_lab/2eb7452f-ae82-4707-82ed-218aabcd7aaf/bmw_lab/franda/Reentrancy/Reentrancy_data/AST_vulnerable/"
    no_reen_path = "/media/bmw_lab/2eb7452f-ae82-4707-82ed-218aabcd7aaf/bmw_lab/franda/Reentrancy/Reentrancy_data/AST_normal/"

    time_path = "/media/bmw_lab/2eb7452f-ae82-4707-82ed-218aabcd7aaf/bmw_lab/franda/TimeStamp/CCG/AST_vulnerable/"
    no_time_path = "/media/bmw_lab/2eb7452f-ae82-4707-82ed-218aabcd7aaf/bmw_lab/franda/TimeStamp/CCG/AST/"

    call_path = "/media/bmw_lab/2eb7452f-ae82-4707-82ed-218aabcd7aaf/bmw_lab/franda/Unchecked_low_Calls/Unchecked_low_Calls_data/AST_vulnerable/"
    no_call_path = "/media/bmw_lab/2eb7452f-ae82-4707-82ed-218aabcd7aaf/bmw_lab/franda/Unchecked_low_Calls/Unchecked_low_Calls_data/AST_normal/"

    ari_path = "/media/bmw_lab/2eb7452f-ae82-4707-82ed-218aabcd7aaf/bmw_lab/franda/Arithmetic/CCG/AST_vulnerable/"
    no_ari_path = "/media/bmw_lab/2eb7452f-ae82-4707-82ed-218aabcd7aaf/bmw_lab/franda/Arithmetic/CCG/AST/"

    # GenerateData(reen_path, "../raw_data/reen.json")
    # GenerateData(no_reen_path, "../raw_data/no_reen.json")

    # GenerateData(time_path, "../raw_data/time.json")
    # GenerateData(no_time_path, "../raw_data/no_time.json") #71 #35

    # GenerateData(call_path, "../raw_data/call.json")    #62 #28
    # GenerateData(no_call_path, "../raw_data/no_call.json")

    GenerateData(ari_path, "../raw_data/ari.json")    #143 #94
    GenerateData(no_ari_path, "../raw_data/no_ari.json")

    print(keys)
    print(len(keys))
    print(e_keys)
    print(len(e_keys))
    # with open(data_path,'r') as f:
    #     data = json.load(f)

    # print(data)