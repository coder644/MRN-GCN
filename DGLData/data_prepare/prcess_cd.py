
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



# path_r_v = './func_Reentrancy/Ast_v/'
# path_r_no_v = './func_Reentrancy/Ast_no_v//'
# files_r_v = os.listdir(path_r_v)
# files_r_no_v = os.listdir(path_r_no_v)
# len(files_r_v),len(files_r_no_v)



expression = []
statement = []


class VertexNode(object):
    #图的顶点
    def __init__(self,data,token,index):
        self.data = data
        self.token = token
        self.index = index
    def __str__(self):
        return self.data


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


def getParseResult(file):
    if os.path.getsize(file) != 0:
        f = open(file)
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



functioncall = []
booleanLiteral = []

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

# Graph,Graph1 = processMain(files_r_no_v,files_r_v,path_r_no_v,path_r_v,'./func_Reentrancy/graph_series/')  #no_v and v
# Graph,Graph1 = processMain(r_files_normal,files_renntrancy,'./Normal_File/AST/','./Vulnerable_File/Reentrancy/AST/','./Vulnerable_File/Reentrancy/graph_series/')  #no_v and v
# Graph,Graph1 = processMain(t_files_normal,files_time,'./Normal_File/AST/','./Vulnerable_File/Time_Stamp_Dependency/AST/','./Vulnerable_File/Time_Stamp_Dependency/graph_series/')  #no_v and v
# Graph,Graph1 = processMain(u_files_normal,files_unchecked,'./Normal_File/AST/','./Vulnerable_File/Unchecked_Low_Calls/AST/','./Vulnerable_File/Unchecked_Low_Calls/graph_series/')  #no_v and v


# parseresult = getParseResult(path_r_v,files_r_v[0])
# a = processJsonFile(parseresult)

word_index = np.loadtxt('/media/bmw_lab/2eb7452f-ae82-4707-82ed-218aabcd7aaf/bmw_lab/franda/Func_level_GCN/data/word_index_reentrancy.txt',dtype= np.int32)
word_vectors = np.loadtxt('/media/bmw_lab/2eb7452f-ae82-4707-82ed-218aabcd7aaf/bmw_lab/franda/Func_level_GCN/data/word_vectors_reentrancy.txt')
featues_vec = np.zeros((113,100))
for i,j in zip(word_index,range(len(word_index))):
    featues_vec[i] = word_vectors[j]

def getIndex(file,path):
    t =  open(path+file.split('+')[0]+'.txt','r')
    length = int(t.readline().split(' ')[0])
    np.loadtxt(t,max_rows=length)
    nodes = t.readline().split(',')
    nodes[-1] = nodes[-1][:-1]
    node_index = {node:i for node,i in zip(nodes,range(length))}
    t.close()
    return node_index


def processMain(file):
    global graph
    global index
    global nums
    global strs
    nums = []
    strs = []
    index = 1
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


def GenerateData(path,data_path,save_data=True):
    i = 0
    graphs = []
    data = []

    files = os.listdir(path)
    for file in files:
        i = i + 1
        if i % 100 == 0:
            print(i)
        try:
            graph = processMain(path + file)
            if(graph is False):
                continue
            src = []
            dst = []
            n_attr = []
            # e_attr = []
            for node in graph.keys():
                n_attr.append(featues_vec[node.token].tolist())
                if (graph[node] != []):
                    for edge in graph[node]:
                        src.append(node.index)
                        dst.append(edge.index)
                        # e_attr.append(edge.token)
            # graph_data = [src, dst, n_attr, e_attr]
            graph_data = [src, dst, n_attr]
            data.append(graph_data)
        except:
            continue
        # graphs.append(graph)

    with open(data_path, 'w+') as f:
        json.dump(data, f)

if __name__ == '__main__':
    # gs = processMain("testASTReen.json")
    path = "/media/bmw_lab/2eb7452f-ae82-4707-82ed-218aabcd7aaf/bmw_lab/franda/Data Process/Vulnerable_File/Unchecked_Low_Calls/AST/"
    reen_path = "/media/bmw_lab/2eb7452f-ae82-4707-82ed-218aabcd7aaf/bmw_lab/franda/Reentrancy/Reentrancy_data/AST_vulnerable/"
    no_reen_path = "/media/bmw_lab/2eb7452f-ae82-4707-82ed-218aabcd7aaf/bmw_lab/franda/Reentrancy/Reentrancy_data/AST_normal/"
    data_path = "../raw_data/data.json"
    GenerateData(reen_path, "../raw_data/reen_cd.json")
    GenerateData(no_reen_path, "../raw_data/no_reen_cd.json")
