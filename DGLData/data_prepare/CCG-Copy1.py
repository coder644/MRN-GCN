
import os
import numpy as np
import json
import pprint
import ast


# In[2]:


# reentrancy = os.listdir('./Vulnerable_File/Reentrancy/GlobalGraph/')
# time = os.listdir('./Vulnerable_File/Time_Stamp_Dependency/GlobalGraph/')
# unchecked = os.listdir('./Vulnerable_File/Unchecked_Low_Calls/GlobalGraph/')

# arithmatic_global_file = os.listdir('./Vulnerable_File/Arithmatic/GlobalGraph/')[:1000]
# a_global_file = os.listdir('./Normal_File/GlobalGraph/')[:1000]
# r_global_file = os.listdir('./Normal_File/GlobalGraph/')[:len(reentrancy)]
# t_global_file = os.listdir('./Normal_File/GlobalGraph/')[:len(time)]
# u_global_file = os.listdir('./Normal_File/GlobalGraph/')[:len(unchecked)]

# arithmatic_global_file = [item.split('.')[0] for item in arithmatic_global_file]
# a_global_file = [item.split('.')[0] for item in a_global_file]
# r_global_file = [item.split('.')[0] for item in r_global_file]
# t_global_file = [item.split('.')[0] for item in t_global_file]
# u_global_file = [item.split('.')[0] for item in u_global_file]
# len(r_global_file),len(t_global_file),len(u_global_file)


# In[3]:


path_r_v = './func_Reentrancy/Ast_v/'
path_r_no_v = './func_Reentrancy/Ast_no_v//'
files_r_v = os.listdir(path_r_v)
files_r_no_v = os.listdir(path_r_no_v)
len(files_r_v),len(files_r_no_v)


# In[ ]:


# r_files_normal = [item for item in files_normal if item.split('+')[0] in r_global_file]
# t_files_normal = [item for item in files_normal if item.split('+')[0] in t_global_file]
# u_files_normal = [item for item in files_normal if item.split('+')[0] in u_global_file]
# a_files_normal = [item for item in files_normal if item.split('+')[0] in a_global_file]
# a_files = [item for item in files_arithmetic if item.split('+')[0] in arithmatic_global_file]

# len(r_files_normal),len(t_files_normal),len(u_files_normal)


# In[ ]:


expression = []
statement = []


# In[ ]:


class VertexNode(object):
    #图的顶点
    def __init__(self,data,token,index):
        self.data = data
        self.token = token
        self.index = index
    def __str__(self):
        return self.data
# ENTRY = VertexNode('enter',0,0)
# index = 1
# graph = {ENTRY:[]}


# In[ ]:


# keys = []
# tokens = {'ENTRY':0,'num':1,'str':2}
# nums = []
# strs = []
# token_ind = 3
# ENTRY = VertexNode('enter',0,0)
# index = 1
# graph = {ENTRY:[]}


# In[ ]:


def getNodeToken(keyword):
    global token_ind
    global keys
    global tokens
    if keyword[0:3] in ['num','str']:
        return tokens[keyword[0:3]]
    elif keyword in keys:
        return tokens[keyword]
    else:
        keys.append(keyword)
        tokens[keyword] = token_ind
        token_ind += 1
        return tokens[keyword]

def addNode(start,data,token):
    global index
    nnode = VertexNode(data,token,index)
    graph[start].append(nnode)
    graph[nnode] = []
    index += 1
    return start,nnode

def addEdge(start,end):
    graph[start].append(end)

def addDataFlow(g):
    nodes = list(g.keys())
    for node1,i in zip(nodes,range(len(nodes))):
        key = node1.data[:3]
        if key == 'num' or key == 'str':
            for node2 in nodes[i+1:]:
                if node1.data == node2.data:
                    addEdge(node1,node2)


# In[ ]:


test = {'type': 'FunctionDefinition',
 'name': 'Ownable',
 'parameters': {'type': 'ParameterList', 'parameters': []},
 'returnParameters': [],
 'body': {'type': 'Block',
  'statements': [{'type': 'ExpressionStatement',
    'expression': {'type': 'BinaryOperation',
     'operator': '=',
     'left': {'type': 'Identifier', 'name': 'owner'},
     'right': {'type': 'MemberAccess',
      'expression': {'type': 'Identifier', 'name': 'msg'},
      'memberName': 'sender'}}}]},
 'visibility': 'public',
 'modifiers': [],
 'isConstructor': True,
 'stateMutability': None}

# FunctionDefinition(ENTRY,ENTRY,test)
# for item in graph.keys():
#     print(item,end=' ')
#     if graph[item] != []:
#         for i in graph[item]:
#             print(i,end=' ')
#     print('\n')


# In[ ]:


def StateVariableDeclaration(start,end,b):
    if b['variables'] != []:
#         token = getNodeToken('variables')
#         start1,end1 = addNode(end0,'variables',token)  # start1 == end0  end1 was the new node just generated
        for item in b['variables']:
            processVariables(start,end,item)
        if b['initialValue'] != None:
            processInitialValue(start,end,b)
#             token = getNodeToken('initialValue')
#             start3,end3 = addNode(end0,'initialValue',token)
            
#             token = getNodeToken(b['initialValue']['type'])
#             start4,end4 = addNode(end3,b['initialValue']['type'],token)
            
#             if 'name' in b['initialValue'].keys():
#                 temp = 'str'
#                 if b['initialValue']['name'] not in strs:
#                     strs.append(b['initialValue']['name'])
#                     temp = 'str'+str(len(strs)+1)
#                 else:
#                     temp = 'str'+str(strs.index(b['initialValue']['name'])+1)
                
#                 token = 2  #num == 1, str == 2
#                 addNode(end4,temp,token)
                    
#             elif 'number' in b['initialValue'].keys():
#                 temp = 'num'
#                 if b['initialValue']['number'] not in nums:
#                     nums.append(b['initialValue']['number'])
#                     temp = 'num'+str(len(nums)+1)
#                 else:
#                     temp = 'num'+str(nums.index(b['initialValue']['number'])+1)
                
#                 token = 1  #num == 1, str == 2
#                 addNode(end4,temp,token)
                
#             token = getNodeToken(b['initialValue']['type'])
#             addNode(end3,b['initialValue']['type'],token)
            
#             if '}' in b['initialValue']['type']:
#                 print(2,b['initialValue']['type'])

def EventDefinition(start,end,b):
    if '}' in b['type']:
        print(3,b['type'])
    if b['parameters'] != None and b['parameters']['parameters'] != []:
        processParameters(start,end,b['parameters'])
#         res.append(b['parameters']['type'])
#         if '}' in b['parameters']['type']:
#             print(4,b['parameters']['type'])
#         tt = processParameters(b['parameters']['parameters'])

def FunctionDefinition(start,end,b):
 
    processStr(start,end,b,'name')
    
    if b['parameters']['parameters'] != []:
#         res.append(b['parameters']['type'])
#         if '}' in b['parameters']['type']:
#             print(5,b['parameters']['type'])
        processParameters(start,end,b['parameters'])
        
    if b['returnParameters'] != [] and b['returnParameters']!=None:
#         res.append(b['returnParameters']['type'])
#         if '}' in b['returnParameters']['type']:
#             print(6,b['returnParameters']['type'])
        processParameters(start,end,b['returnParameters'])

    if b['body'] != []:
        processBodyblockstatement(start,end,b['body'])


def ModifierDefinition(start,end,b):
 
    if b['parameters'] != [] and b['parameters']['parameters'] != []:
        processParameters(start,end,b['parameters'])
    if b['body'] != []:
        processBodyblockstatement(start,end,b['body'])

def UsingForDeclaration(start,end,b):

#     print(b)
    if 'typeName' in b.keys() and  b['typeName'] != '*':
        processTypeName(start,end,b['typeName'])
    token = getNodeToken(b['libraryName'])
    addNode(end,b['libraryName'],token)
    if '}' in b['libraryName']:
        print(9,b['libraryName'])

def StructDefinition(start,end,b):
    processStr(start,end,b,'name')
    
    if b['members'] != [] and b['members'] != None: 
        start1,end1 = start,end
        for item in b['members']:
            start1,end1 = processVariables(start1,end1,item)

def EnumDefinition(start,end,b):

    processStr(start,end,b,'name')
    if b['members'] != [] and b['members'] != None: 
        start1,end1 = start,end
        for item in b['members']:
            start1,end1 = processVariables(start1,end1,item)
#             res.append(item['type'])
#             if '}' in item['type']:
#                 print(12,':',item['type'])


# In[ ]:


def processVariables(start,end,b):
    token = getNodeToken('variables')
    start0,end0 = addNode(end,'variables',token)
    
    token = getNodeToken(b['type'])
    addNode(end0,b['type'],token)
        
    processStr(start0,end0,b,'name')
#     if 'name' in b.keys():
#         temp = 'str'
#         if b['name'] not in strs:
#             strs.append(b['name'])
#             temp = 'str'+str(len(strs)+1)
#         else:
#             temp = 'str'+str(strs.index(b['name'])+1)
                
#         token = 2  #num == 1, str == 2
#         addNode(end0,temp,token)
             
    if 'typeName' in b.keys() and  b['typeName'] != '*':
#         print(b['typeName'])
#         token = getNodeToken('typeName')
#         start2,end2 = addNode(end,'typeName',token)
        processTypeName(start0,end0,b['typeName'])
    
    if 'expression' in b.keys():
        if b['expression'] != None:
    #         token = getNodeToken('expression')
    #         start1,end1 = addNode(end,'expression',token)
            processExpression(start0,end0,b['expression'])
    return start0,end0
    
    
def processTypeName(start,end,b):
    token = getNodeToken('typeName')
    start0,end0 = addNode(end,'typeName',token)
                        
def processParameters(start,end,b):
    token = getNodeToken(b['type'])
    start0,end0 = addNode(end,b['type'],token)
    for item in b['parameters']:
        if '}' in item['type']:
            print(26,':',item['type'])
        processVariables(start0,end0,item)
        
def processStr(start,end,b,key):
    if key in b.keys():
        temp = 'str'
        if b[key] not in strs:
            strs.append(b[key])
            temp = 'str'+str(len(strs)+1)
        else:
            temp = 'str'+str(strs.index(b[key])+1)
                
        token = 2  #num == 1, str == 2
        addNode(end,temp,token)

def processNum(start,end,b,key):
    if key in b.keys():
        temp = 'num'
        if b[key] not in nums:
            nums.append(b[key])
            temp = 'num'+str(len(nums)+1)
        else:
            temp = 'num'+str(nums.index(b[key])+1)
                
        token = 1  #num == 1, str == 2
        addNode(end,temp,token)

def processInitialValue(start,end,b):
    token = getNodeToken('initialValue')
    start0,end0 = addNode(end,'initialValue',token)
            
    token = getNodeToken(b['initialValue']['type'])
    addNode(end0,b['initialValue']['type'],token)
    
    if 'name' in b['initialValue'].keys():
        temp = 'str'
        if b['initialValue']['name'] not in strs:
            strs.append(b['initialValue']['name'])
            temp = 'str'+str(len(strs)+1)
        else:
            temp = 'str'+str(strs.index(b['initialValue']['name'])+1)
                
        token = 2  #num == 1, str == 2
        addNode(end0,temp,token)
                    
    elif 'number' in b['initialValue'].keys():
        temp = 'num'
        if b['initialValue']['number'] not in nums:
            nums.append(b['initialValue']['number'])
            temp = 'num'+str(len(nums)+1)
        else:
            temp = 'num'+str(nums.index(b['initialValue']['number'])+1)
                
        token = 1  #num == 1, str == 2
        addNode(end0,temp,token)
        
def processBodyblockstatement(start,end,b):
    token = getNodeToken(b['type'])
    start0,end0 = addNode(end,b['type'],token)
    
#     if '}' in b['type']:
#         print(34,':',b['type'])
    
    if b['type'] == 'InLineAssemblyStatement':
        for item in b['body']['operations']:
            processExpression(start0,end0,item)
            
    elif b['type'] == 'VariableDeclarationStatement':
        if b['variables'] != None and b['variables'] != []:
            start1,end1 = start0,end0
            for item in b['variables']:
                start1,end1 = processVariables(start1,end1,item)
        if b['initialValue'] != None:
            processInitialValue(start0,end0,b)
            
    elif b['type'] == 'ExpressionStatement':
        if b['expression'] != None:
            processExpression(start0,end0,b['expression'])
        
    elif b['type'] == 'IfStatement':
        token = getNodeToken('condition')
        start1,end1 = addNode(end0,'condition',token)
        start2,end2 = start1,end1
        processExpression(start1,end1,b['condition'])
        if b['TrueBody'] != None and b['TrueBody'] != ';':
            token = getNodeToken('TrueBody')
            start2,end2 = addNode(end1,'TrueBody',token)
            processBodyblockstatement(start2,end2,b['TrueBody'])
            
        if b['FalseBody'] != None and b['FalseBody'] != ';':
            token = getNodeToken('FalseBody')
            start3,end3 = addNode(end2,'FalseBody',token)
            processBodyblockstatement(start3,end3,b['FalseBody'])
            
    elif b['type'] == 'EmitStatement':
        processExpression(start0,end0,b['eventCall'])
        
    elif b['type'] == 'Identifier':
        processExpression(start0,end0,b)
        
    elif b['type'] == 'BooleanLiteral':
        processExpression(start0,end0,b)
        
    elif b['type'] == 'IndexAccess':
        processExpression(start0,end0,b)
    
    elif b['type'] == 'UnaryOperation':
        processExpression(start0,end0,b)
        
    elif b['type'] == 'FunctionCall':
        processExpression(start0,end0,b)
        
    elif b['type'] == 'MemberAccess':
        processExpression(start0,end0,b['expression'])
        
    elif b['type'] == 'ForStatement':
        start1,end1 = start0,end0
        start2,end2 = start0,end0
        start3,end3 = start0,end0
        if b['initExpression'] != None:
            token = getNodeToken('initExpression')
            start1,end1 = addNode(end0,'initExpression',token)
            processBodyblockstatement(start1,end1,b['initExpression'])
        if b['conditionExpression'] != None:
            token = getNodeToken('conditionExpression')
            start2,end2 = addNode(end1,'conditionExpression',token)
            processExpression(start2,end2,b['conditionExpression'])
        if b['loopExpression'] != None:
            token = getNodeToken('loopExpression')
            start3,end3 = addNode(end2,'loopExpression',token)
            processBodyblockstatement(start3,end3,b['loopExpression'])
            
        if b['body'] != []:
#             print(b['body'])
#             print(b['body']['type'])
#             print(b['body'],'*********')
            token = getNodeToken('body')
            start4,end4 = addNode(end3,'body',token)
            processBodyblockstatement(start4,end4,b['body'])
#             b = b['body']['statements']
#             tt = []
#             for item in b:
#                 if item != None and item != ';':
#                     tt = processBodyblockstatement(item)
#                 res.extend(tt)
                
    elif b['type'] == 'WhileStatement':
        token = getNodeToken('condition')
        start1,end1 = addNode(end0,'condition',token)
        processExpression(start1,end1,b['condition'])
        if b['body'] != []:
            token = getNodeToken('body')
            start2,end2 = addNode(end1,'body',token)
            processBodyblockstatement(start2,end2,b['body'])
                
    elif b['type'] == 'TupleExpression':
        start1,end1 = start0,end0
        for item in b['components']:
            if item != None:
                #增加 processExpression reutrn start,end.
                start1,end1 = processExpression(start1,end1,item)
            
    elif b['type'] == 'Conditional':
        token = getNodeToken('condition')
        start1,end1 = addNode(end0,'condition',token)
        processExpression(start1,end1,b['condition'])
        
        if b['TrueExpression'] != None:
            token = getNodeToken('TrueExpression')
            start2,end2 = addNode(end1,'TrueExpression',token)
            processExpression(start2,end2,b['TrueExpression'])
            
        if b['FalseExpression'] != None:
            token = getNodeToken('FalseExpression')
            start3,end3 = addNode(end2,'FalseExpression',token)
            processExpression(start3,end3,b['FalseExpression'])
            
    elif b['type'] == 'NumberLiteral':
        processNum(start,end,b,'number')
        
    elif b['type'] == 'StringLiteral':
        processStr(start,end,b,'value')
    
    elif b['type'] == 'DoWhileStatement':
        token = getNodeToken('condition')
        start1,end1 = addNode(end0,'condition',token)
        processExpression(start1,end1,b['condition'])
        if b['body'] != []:
            token = getNodeToken('body')
            start2,end2 = addNode(end1,'body',token)
            processBodyblockstatement(start2,end2,b['body'])
#             b = b['body']['statements']
#             tt = []
#             for item in b:
#                 if item != None and item != ';':
#                     tt = processBodyblockstatement(item)
#                 res.extend(tt)
    elif b['type'] == 'Block':
        if b['statements'] != [] and b['statements'] != None:
            b = b['statements']
            start1,end1 = start0,end0
            for item in b:
                if item != None and item != ';':
                    start1,end1 = processBodyblockstatement(start1,end1,item)
    return start0,end0

def processExpression(start,end,b):
    token = getNodeToken(b['type'])
    start0,end0 = addNode(end,b['type'],token)
#     if '}' in b['type']:
#         print(13,':',b['type'])
    if b['type'] == 'AssemblyExpression':
        if 'functionName' in b.keys():
            processStr(start0,end0,b,'functionName')
        if  'arguments' in b.keys() and b['arguments'] != []:
            token = getNodeToken('arguments')
            start1,end1 = addNode(end0,'arguments',token)
            for item in b['arguments']:
#                 if item['type'] == 'AssemblyExpression':
                start1,end1 = processExpression(start1,end1,item)
                    
    elif b['type'] == 'AssemblyLocalDefinition':
        if b['names'] != []:
            token = getNodeToken('names')
            start1,end1 = addNode(end0,'names',token)
            for item in b['names']:
                processExpression(start1,end1,item)
        if b['expression'] != None:
            processExpression(start0,end0,b['expression'])
        
    elif b['type'] == 'AssemblySwitch':
        processExpression(start0,end0,b['expression'])
        
        processStr(start0,end0,b,'functionName')
        
        if 'arguments' in b.keys() and b['arguments'] != []:
            token = getNodeToken('arguments')
            start1,end1 = addNode(end0,'arguments',token)
            for item in b['arguments']:
                start1,end1 = processExpression(start1,end1,item)
            start1,end1 = start0,end0
            for item in b['cases']:
                start1,end1 = processExpression(start1,end1,item)
            
    elif b['type'] == 'AssemblyCase':
        if b['block'] != None:
            processExpression(start0,end0,b['block'])
        if b['value'] != None:
            processExpression(start0,end0,b['value'])
            
    elif b['type'] == 'AssemblyBlock':
        if b['operations'] != None:
            start1,end1 = start0,end0
            for item in b['operations']:
                start1,end1 = processExpression(start1,end1,item)
                
    elif b['type'] == 'AssemblyAssignment':
        if b['names'] != []:
            start1,end1 = start0,end0
            for item in b['names']:
                if item != None:
                    start1,end1 = processExpression(start1,end1,item)
                    
    elif b['type'] == 'BinaryOperation':
        token = getNodeToken(b['operator'])
        start1,end1 = addNode(end0,b['operator'],token)
        if '}' in b['operator']:
            print(16,':',b['operator'])
            
        token = getNodeToken('left')
        start2,end2 = addNode(end1,'left',token)
        processExpression(start2,end2,b['left'])
        
        token = getNodeToken('right')
        start3,end3 = addNode(end1,'right',token)
        processExpression(start3,end3,b['right'])
                    
    elif b['type'] == 'Identifier':
        processStr(start,end,b,'name')
        
    elif b['type'] == 'MemberAccess':
#         processExpression(start0,end0,b['expression'])
        processStr(start0,end0,b,'memberName')
        
    elif b['type'] == 'FunctionCall':
        processExpression(start0,end0,b['expression'])
        if b['names'] != None:
            token = getNodeToken('names')
            start1,end1 = addNode(end0,'names',token)
#             print('hello',b['names'])
            for item in b['names']:
                if type(item) != str:
                    processExpression(start1,end1,item)
                    

        if 'typeName' in b.keys() and b['typeName'] != None:
            processTypeName(start0,end0,b['typeName'])
        
        if b['arguments'] != None:
            token = getNodeToken('arguments')
            start1,end1 = addNode(end0,'arguments',token)
            for item in b['arguments']:
                start1,end1 = processExpression(start1,end1,item)
        
    elif b['type'] == 'ElementaryTypeNameExpression':
        if 'typeName' in b.keys() and  b['typeName'] != '*':
            processTypeName(start0,end0,b['typeName'])
        
    elif b['type'] == 'BooleanLiteral':
        if b['value'] not in booleanLiteral:
            booleanLiteral.append(b['value'])
        token = getNodeToken(str(b['value']))
        addNode(end,str(b['value']),token)
    
    elif b['type'] == 'UnaryOperation':
        token = getNodeToken(b['operator'])
        start1,end1 = addNode(end,b['operator'],token)
        if '}' in b['operator']:
            print(21,':',b['operator'])
        processExpression(start1,end1,b['subExpression'])
        
    elif b['type'] == 'IndexAccess':
        token = getNodeToken('base')
        start1,end1 = addNode(end0,'base',token)
        processExpression(start1,end1,b['base'])
        
        token = getNodeToken('index')
        start2,end2 = addNode(end1,'index',token)
        processExpression(start2,end2,b['index'])
        
    elif b['type'] == 'NumberLiteral':
        processNum(start,end,b,'number')
        
    elif b['type'] == 'TupleExpression':
        start1,end1 = start0,end0
        for item in b['components']:
            if item != None:
                start1,end1 = processBodyblockstatement(start1,end1,item)
        
    elif b['type'] == 'StringLiteral':   
        processNum(start,end,b,'value')
    
    elif b['type'] == 'DecimalNumber':
        processNum(start,end,b,'value')
         
    return start0,end0


# In[ ]:


def getParseResult(path,file):
    if os.path.getsize(path+file) != 0:
        f = open(path+file)
        jsonfile = json.load(f)
        return jsonfile

Contractsubkeywords = ['StateVariableDeclaration','EventDefinition','FunctionDefinition','ModifierDefinition','UsingForDeclaration','StructDefinition','EnumDefinition']
def processJsonFile(parseresult):
    if parseresult['children'] != []:
        if parseresult['children'][0] != None:
            start1 = ENTRY
            end1 = ENTRY
            for i in range(1,len(parseresult['children'])):
                if parseresult['children'][i] != None:
                    a = parseresult['children'][i]
                    token = getNodeToken(a['type'])
                    start1,end1 = addNode(end1,a['type'],token)
                    if 'subNodes' in a.keys():
                        b = a['subNodes']
                        for item in b:
                            if 'name' in item.keys() and 'type' not in item.keys():
                                t = item['name']
                            elif 'type' in item.keys():
                                t = item['type']
                            if t == 'StateVariableDeclaration':
                                StateVariableDeclaration(start1,end1,item)
                            elif t == 'EventDefinition':
                                EventDefinition(start1,end1,item)
                            elif t == 'FunctionDefinition':
                                FunctionDefinition(start1,end1,item)
                            elif t == 'ModifierDefinition':
                                ModifierDefinition(start1,end1,item)
                            elif t == 'UsingForDeclaration':
                                UsingForDeclaration(start1,end1,item)
                            elif t == 'StructDefinition':
                                StructDefinition(start1,end1,item)
                            elif t == 'EnumDefinition':
                                EnumDefinition(start1,end1,item)
    addDataFlow(graph)
    return graph
            


# In[ ]:


keys = ['ENTRY',
 'num',
 'str']
tokens = {"ENTRY": 0,
  "num": 1,
  "str": 2}
nums = []
strs = []
token_ind = 3
ENTRY = VertexNode('enter',0,0)
index = 1
graph = {ENTRY:[]}


# In[ ]:


def vertexnode2dict(node):
    t = {'data':'','token':'','index':''}
    t['data'] = node.data
    t['token'] = node.token
    t['index'] = node.index
    return str(t)

def write2file(file,graph):
    path = './graph_vlunerable/'
    f = open(path+file,'w')
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


# In[4]:


functioncall = []
booleanLiteral = []


# In[17]:


def processMain(files1,files2,path1,path2,path3):
    i = 1
    Graph = []
    Graph1 = []
    global graph
    global index
    global nums
    global strs
    for file in files1:
        nums = []
        strs = []
        index = 1
        i = i+1
        graph = {ENTRY:[]}
        if os.path.getsize(path1+file) != 0:
            parseresult = getParseResult(path1,file)
            processJsonFile(parseresult)     
            Graph.append(graph)
    for file in files2:
        nums = []
        strs = []
        index = 1
        graph = {ENTRY:[]}
        if os.path.getsize(path2+file) != 0:
            parseresult = getParseResult(path2,file)
            processJsonFile(parseresult)            
            Graph1.append(graph)
            
    
    return Graph,Graph1


# In[18]:


Graph,Graph1 = processMain(files_r_no_v,files_r_v,path_r_no_v,path_r_v,'./func_Reentrancy/graph_series/')  #no_v and v
# Graph,Graph1 = processMain(r_files_normal,files_renntrancy,'./Normal_File/AST/','./Vulnerable_File/Reentrancy/AST/','./Vulnerable_File/Reentrancy/graph_series/')  #no_v and v
# Graph,Graph1 = processMain(t_files_normal,files_time,'./Normal_File/AST/','./Vulnerable_File/Time_Stamp_Dependency/AST/','./Vulnerable_File/Time_Stamp_Dependency/graph_series/')  #no_v and v
# Graph,Graph1 = processMain(u_files_normal,files_unchecked,'./Normal_File/AST/','./Vulnerable_File/Unchecked_Low_Calls/AST/','./Vulnerable_File/Unchecked_Low_Calls/graph_series/')  #no_v and v


# In[19]:


# 存储 tokens
# f = open('./nodes-reentrancy.json','w')
# f.write(json.dumps(tokens))
# f.close()
# li = list(Graph[0].keys())
# print(li[0].token)


# In[5]:


parseresult = getParseResult(path_r_v,files_r_v[0])
a = processJsonFile(parseresult)


# In[20]:


# 获取features
word_index = np.loadtxt('./word_index_reentrancy.txt',dtype= np.int32)
word_vectors = np.loadtxt('./word_vectors_reentrancy.txt')
featues_vec = np.zeros((113,100))
for i,j in zip(word_index,range(len(word_index))):
    featues_vec[i] = word_vectors[j]


# In[21]:


def getIndex(file,path):
    t =  open(path+file.split('+')[0]+'.txt','r')
    length = int(t.readline().split(' ')[0])
    np.loadtxt(t,max_rows=length)
    nodes = t.readline().split(',')
    nodes[-1] = nodes[-1][:-1]
    node_index = {node:i for node,i in zip(nodes,range(length))}
    t.close()
    return node_index


# In[ ]:


for graph,file in zip(Graph,files_r_no_v):
    directory = file.split('+')[0]
    func_name = file.split('.')[0].split('+')[1]
    f =  open('../../Func_level_GCN/data/graph_no_v/'+directory+'+'+func_name+'.txt','a')
    length = len(graph)
    f.write(str(length))
    f.write(' 0')
    f.write('\n')
    for item in graph.keys():
        pf = []
#         pf.append(item.index) # node id
        pf.append(len(graph[item]))  #相邻节点数量
        for li in graph[item]:
            pf.append(li.index)  #相邻边 id
        pf.extend(featues_vec[item.token])  #该节点属性
        for i in pf:
            f.write(str(i))
            f.write(' ')
        f.write('\n')
    f.close()


# In[ ]:


for graph,file in zip(Graph1,files_r_v):
    directory = file.split('+')[0]
    func_name = file.split('.')[0].split('+')[1]
    f =  open('../../Func_level_GCN/data/graph_v/'+directory+'+'+func_name+'.txt','a')
    length = len(graph)
    f.write(str(length))
    f.write(' 1')
    f.write('\n')
    for item in graph.keys():
        pf = []
#         pf.append(item.index) # node id
        pf.append(len(graph[item]))  #相邻节点数量
        for li in graph[item]:
            pf.append(li.index)  #相邻边 id
        pf.extend(featues_vec[item.token])  #该节点属性
        for i in pf:
            f.write(str(i))
            f.write(' ')
        f.write('\n')
    f.close()

