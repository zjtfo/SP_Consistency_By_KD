import json
import csv
from numpy.random.mtrand import exponential
import pandas as pd
import numpy as np
import operator


def get_program_seq(program):
    seq = []
    for item in program:
        func = item['function']
        inputs = item['inputs']
        args = ''
        for input in inputs:
            args += ' <arg> ' + input
        seq.append(func + args)
    seq = ' <func> '.join(seq)
    return seq

   

new_tokens = ['<func>', '<arg>']

functions_27 = ['FindAll', 'Find', 'FilterConcept', 'FilterStr', 'FilterNum', 'FilterYear', 'FilterDate', 'QFilterStr', 'QFilterNum', 'QFilterYear', 'QFilterDate', 
'Relate', 'And', 'Or', 'QueryName', 'Count', 'QueryAttr', 'QueryAttrUnderCondition', 'QueryRelation', 'SelectBetween', 'SelectAmong', 'VerifyStr', 'VerifyNum', 
'VerifyYear', 'VerifyDate', 'QueryAttrQualifier', 'QueryRelationQualifier']


# 构造的负例是将program中函数都随机替换掉（1:5）
# （并且如果program里的一些函数有很强的相关性，可以考虑替换的时候增大替换为比较相关的函数的概率）
def get_program_replace_seq(program):
    original = str(get_program_seq(program))
    # print(original)
    # FindAll <func> FilterStr <arg> TOID <arg> 4000000074573917 <func> FilterConcept <arg> town <func> FindAll <func> FilterStr <arg> OS grid reference <arg> SP8778 <func> FilterConcept <arg> town <func> And <func> What
    
    seg = []
    func_list = []
    for i in original.split(' <func> '):
        seg.append(i)
        idx = i.find('<')
        # print(idx)
        if idx == -1:
            func_list.append(i.replace(" ", ""))
        else:
            func_list.append(i[:idx].replace(" ", ""))
    print(func_list)
    # ['FindAll', 'FilterStr ', 'FilterConcept ', 'FindAll', 'FilterStr ', 'FilterConcept ', 'And', 'What']
    # print(seg)
    # ['FindAll', 'FilterStr <arg> TOID <arg> 4000000074573917', 'FilterConcept <arg> town', 'FindAll', 'FilterStr <arg> OS grid reference <arg> SP8778', 'FilterConcept <arg> town', 'And', 'What']

    F = ['FindAll', 'Find']
    FI = ['FilterConcept', 'FilterStr', 'FilterNum', 'FilterYear', 'FilterDate']
    QFI = ['QFilterStr', 'QFilterNum', 'QFilterYear', 'QFilterDate']
    Q = ['QueryName', 'QueryAttr', 'QueryAttrUnderCondition', 'QueryRelation']
    S = ['SelectBetween', 'SelectAmong']
    V = ['VerifyStr', 'VerifyNum', 'VerifyYear', 'VerifyDate']
    QQ = ['QueryAttrQualifier', 'QueryRelationQualifier']
    AO = ['And', 'Or']
    RC= ['Relate', 'Count']

    
    result = []
    rep_pro = []
    # while(True):
    for i in range(50):
        length = len(func_list)
        l = 0
        rep = []
        for k in func_list:
            if k in F:
                if l != length:
                    index = np.random.randint(0,2)
                    l += 1
                    rep.append(F[index])
                else:
                    break
            elif k in FI:
                if l != length:
                    index = np.random.randint(0,5)
                    l += 1
                    rep.append(FI[index])
                else:
                    break
            elif k in QFI:
                if l != length:
                    index = np.random.randint(0,4)
                    l += 1
                    rep.append(QFI[index])
                else:
                    break
            elif k in Q:
                if l != length:
                    index = np.random.randint(0,4)
                    l += 1
                    rep.append(Q[index])
                else:
                    break
            elif k in S:
                if l != length:
                    index = np.random.randint(0,2)
                    l += 1
                    rep.append(S[index])
                else:
                    break
            elif k in V:
                if l != length:
                    index = np.random.randint(0,4)
                    l += 1
                    rep.append(V[index])
                else:
                    break
            elif k in QQ:
                if l != length:
                    index = np.random.randint(0,2)
                    l += 1
                    rep.append(QQ[index])
                else:
                    break
            elif k in AO:
                if l != length:
                    index = np.random.randint(0,2)
                    l += 1
                    rep.append(AO[index])
                else:
                    break
            elif k in RC:
                if l != length:
                    index = np.random.randint(0,27)
                    l += 1
                    rep.append(functions_27[index])
                else:
                    break
            else:
                rep.append(k)
        
        if rep not in rep_pro and len(rep_pro) < 5:
            # print(operator.eq(rep, func_list))
            if operator.eq(rep, func_list) == False:
                print(rep)
                rep_pro.append(rep)
                # five += 1

                now = ""
                # print(seg)
                # ['FindAll', 'FilterStr <arg> TOID <arg> 4000000074573917', 'FilterConcept <arg> town', 'FindAll', 'FilterStr <arg> OS grid reference <arg> SP8778', 'FilterConcept <arg> town', 'And', 'What']
                for i in range(len(seg)):
                    idx = seg[i].find('<')
                    if idx == -1:
                        tep = seg[i].replace(seg[i], rep[i])
                    else:
                        tep = seg[i].replace(seg[i][:idx], rep[i])
                    if i != len(seg) - 1:
                        now += tep + " <func> "
                    else:
                        now += tep
            
                result.append(now)
        else:
            pass
    
    return result


if __name__ == '__main__':
    res = []
    val_data = {}
    with open('./dataset/train.json','r',encoding='utf8') as fp:
        val_data = json.load(fp)

    
    for i in range(len(val_data)):
        print("第" + str(i + 1) + "条数据")
        seq_program = str(get_program_seq(val_data[i]['program']))
        seq_sparql = val_data[i]['sparql']
        res.append([seq_program + "[SEP]" + seq_sparql,"1"])
        seq_program_now = get_program_replace_seq(val_data[i]['program'])
        for i in seq_program_now:
            res.append([i + "[SEP]" + seq_sparql,"0"])
        


    name = ["comment", "sentiment"]
    test = pd.DataFrame(columns=name,data=res)
    test.to_csv('./Dynamic_Weight_Assignment_Model/train.csv',index=False,encoding='utf8')

    # 测试函数get_program_replace_seq
    # with open('./KQA-Pro-v1.0/val.json','r',encoding='utf8') as fp:
    #     val_data = json.load(fp)

    # seq_program_now = get_program_replace_seq(val_data[0]['program'])
    # print(seq_program_now)
