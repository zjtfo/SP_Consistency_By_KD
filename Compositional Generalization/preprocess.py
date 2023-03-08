import json
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
from tqdm import tqdm

def main():
    data = json.load(open('/data/zj_sfa/KQAPro_New/CompGen/KQAPro.json'))
    data = sorted(data, key = lambda x:len(x['program']))
    print('size', len(data))
    print('min length', len(data[0]['program']))
    print('max length', len(data[-1]['program']))
    num = (int)(0.9 * len(data))
    train = data[:num]
    test = data[num:]
    random.shuffle(train)
    random.shuffle(test)
    num = (int)(len(test)/2)
    # print(num)
    # print(test[0:num])
    valid = test[:num]
    test = test[num:]
    print(len(train))
    print(len(test))
    print(len(valid))
    counter = Counter([len(x['program']) for x in train])
    print(counter)
    counter = Counter([len(x['program']) for x in test])
    print(counter)
    counter = Counter([len(x['program']) for x in valid])
    print(counter)

    func = {}
    for item in tqdm(train):
        for f in item['program']:
            func[f['function']] = 1
    for item in tqdm(test):
        for f in item['program']:
            if f['function'] not in func:
                print(item)
    for item in tqdm(valid):
        for f in item['program']:
            if f['function'] not in func:
                print(item)

    # print(x[0])
    # print(x[-1])
    # print([item['function'] for item in x[-1]['program']])
    # counter = Counter([len(x['program']) for x in data])
    # plt.bar(counter.keys(), counter.values())
    # plt.xlabel('program length')
    # plt.ylabel('number')
    # plt.title('The distribution of program length')
    # plt.savefig('programlength.png',dpi=200)

    # sum = [0] * 16
    # for x in range(1, 16):
    #     sum[x] = sum[x-1] + counter[x]
    # ratio = []
    # for x in sum:
    #     ratio.append(x / len(data))
    # print(ratio)

    # train = []
    # test = []
    # for item in data:
    #     if len(item['program']) <= 7:
    #         train.append(item)
    #     else:
    #         test.append(item)
    # random.shuffle(test)
    # num = (int)(len(test)/2)
    # # print(num)
    # # print(test[0:num])
    # valid = test[:num]
    # test = test[num:]
    # random.shuffle(train)
    print('size', len(train))
    print('size', len(valid))
    print('size', len(test))
    json.dump(train, open('train.json', 'w'))
    json.dump(test, open('test.json', 'w'))
    json.dump(valid, open('val.json', 'w'))

if __name__ == "__main__":
    main()
