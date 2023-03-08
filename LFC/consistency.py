import os
import json


# for Bart_SPARQL
S = open("./KQAPro_ckpt/eval_result/sparql/eval_result.txt", "r", encoding='utf-8')
datas_1 = S.readlines()
pre_result_1 = []
for i in datas_1:
    pre_result_1.append(i.replace("\n", ""))
S.close()
print(len(pre_result_1))  #

# for Bart_Program
P = open("./KQAPro_ckpt/eval_result/kopl/eval_result.txt", "r", encoding='utf-8')
datas_2 = P.readlines()
pre_result_2 = []
for i in datas_2:
    pre_result_2.append(i.replace("\n", ""))
P.close()
print(len(pre_result_2))  #

# find same
same = []
P_right_S_wrong = []
S_right_P_wrong = []
P_wrong_S_wrong = []
special = []
n = len(pre_result_1)

seven = []

for i in range(n):
    if pre_result_1[i] != "*" and pre_result_2[i] != "*":
        if pre_result_1[i] == pre_result_2[i]:
            same.append(pre_result_1[i])
            seven.append(i)
        else:  # 特殊在答案为时间类型上，一个为比如一个为1979-01-01、一个为1979，我认为是正确的
            special.append(i + 1)
            seven.append(i)
    elif pre_result_1[i] != "*" and pre_result_2[i] == "*":
        S_right_P_wrong.append(pre_result_1[i])
    elif pre_result_1[i] == "*" and pre_result_2[i] != "*":
        P_right_S_wrong.append(pre_result_2[i])
    elif pre_result_1[i] == "*" and pre_result_2[i] == "*":
        P_wrong_S_wrong.append("*")

# (Bart_Program)find the number of right answer except *
BP = 0
for i in pre_result_1:
    if i != "*":
        BP += 1
    else:
        pass

# (Bart_SPARQL)find the number of right answer except *
BS = 0
for i in pre_result_2:
    if i != "*":
        BS += 1
    else:
        pass


print(len(seven))



gt_folder, pred_fn = './', './KQAPro_ckpt/predict_result/program/predict.txt'

gt_fn = os.path.join(gt_folder, 'test_answer.json')
gt = json.load(open(gt_fn))
pred = [x.strip() for x in open(pred_fn).readlines()] # one prediction per line

# to compute zero-shot accuracy
train_set = json.load(open(os.path.join('./dataset', 'train.json')))
train_answer_set = set(x['answer'] for x in train_set)

labels = ['overall', 'multihop', 'qualifier', 'comparison', 'logical', 'count', 'verify', 'zero-shot']
total = {k:0 for k in labels}
correct = {k:0 for k in labels}
info = []
for i in range(len(pred)):
    
    cur_labels = ['overall']
    functions = [f['function'] for f in gt[i]['program']]

    for f in functions:
        if f in {'Relate'} or f.startswith('Filter'):
            cur_labels.append('multihop')
            break
    for f in functions:
        if f in {'QFilterStr', 'QFilterNum', 'QFilterYear', 'QFilterDate', 'QueryAttrUnderCondition', 'QueryAttrQualifier', 'QueryRelationQualifier'}:
            cur_labels.append('qualifier')
            break
    for f in functions:
        if f in {'SelectBetween','SelectAmong'}:
            cur_labels.append('comparison')
            break
    for f in functions:
        if f in {'And', 'Or'}:
            cur_labels.append('logical')
            break
    for f in functions:
        if f in {'Count'}:
            cur_labels.append('count')
            break
    for f in functions:
        if f in {'VerifyStr','VerifyNum','VerifyYear','VerifyDate'}:
            cur_labels.append('verify')
            break
    
    answer = gt[i]['answer']
    if answer not in train_answer_set:
        cur_labels.append('zero-shot')

    info.append(cur_labels[1:])

print(len(info))

seven_pro = []
for i in seven:
    seven_pro.append(info[i])
print(len(seven_pro))

f_7 = open('./7.txt', "w", encoding='utf-8')
for tmp in seven_pro:
    f_7.write(str(tmp) + '\n')

mul = 0
qua = 0
com = 0
log = 0
cou = 0
ver = 0
zer = 0
for tmp in seven_pro:
    for i in tmp:
        if i == "multihop":
            mul += 1
        elif i == "qualifier":
            qua += 1
        elif i == "comparison":
            com += 1
        elif i == "logical":
            log += 1
        elif i == "count":
            cou += 1
        elif i == "verify":
            ver += 1
        elif i == "zero-shot":
            zer += 1
print(mul)
print(qua)
print(com)
print(log)
print(cou)
print(ver)
print(zer)
# 7279
# 2159
# 1974
# 2819
# 1075
# 1283
# 1302


f = open('./consistency.txt', "a", encoding='utf-8')
description = "论文中提供的finetune的sparql模型 VS 论文中提供的finetune的program模型"
f.write(description + '\n')
f.write("****************" + '\n')
f.write("测试的问题总数：" + str(n) + '\n')
f.write("Bart_Program中答案预测正确的总个数：" + str(BP) + '\n')
f.write("Bart_SPARQL中答案预测正确的总个数：" + str(BS) + '\n')
f.write("Bart_Program和Bart_SPARQL中答案预测正确且一致的个数：" + str(len(same) + len(special)) + '\n')
f.write("一致性：" + str(len(same) / len(pre_result_1)) + '\n')
f.write("mul：" + str(mul / len(pre_result_1)) + '\n')
f.write("qua：" + str(qua / len(pre_result_1)) + '\n')
f.write("com：" + str(com / len(pre_result_1)) + '\n')
f.write("log：" + str(log / len(pre_result_1)) + '\n')
f.write("cou：" + str(cou / len(pre_result_1)) + '\n')
f.write("ver：" + str(ver / len(pre_result_1)) + '\n')
f.write("zer：" + str(zer / len(pre_result_1)) + '\n')
f.write("Bart_Program中答案预测正确但Bart_SPARQL中答案预测错误的个数：" + str(len(P_right_S_wrong)) + '\n')
f.write("Bart_Program中答案预测错误但Bart_SPARQL中答案预测正确的个数：" + str(len(S_right_P_wrong)) + '\n')
f.write("Bart_Program中答案预测错误并且Bart_SPARQL中答案预测错误的个数：" + str(len(P_wrong_S_wrong)) + '\n')
f.write("****************" + '\n')
f.write('\n')
f.close()



