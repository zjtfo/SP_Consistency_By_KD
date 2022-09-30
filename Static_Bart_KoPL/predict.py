import os
import torch
import torch.optim as optim
import torch.nn as nn
import argparse
import shutil
import json
from tqdm import tqdm
from datetime import date
from misc import MetricLogger, seed_everything, ProgressBar
from load_kb import DataForSPARQL
from data import DataLoader
from transformers import BartConfig, BartForConditionalGeneration, BartTokenizer
import torch.optim as optim
import logging
import time
from lr_scheduler import get_linear_schedule_with_warmup
import re
from executor_rule import RuleExecutor
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s')
logFormatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
rootLogger = logging.getLogger()
import warnings
warnings.simplefilter("ignore") # hide warnings that caused by invalid sparql query


def post_process(text):
    pattern = re.compile(r'".*?"')
    nes = []
    for item in pattern.finditer(text):
        nes.append((item.group(), item.span()))
    pos = [0]
    for name, span in nes:
        pos += [span[0], span[1]]
    pos.append(len(text))
    assert len(pos) % 2 == 0
    assert len(pos) / 2 == len(nes) + 1
    chunks = [text[pos[i]: pos[i+1]] for i in range(0, len(pos), 2)]
    for i in range(len(chunks)):
        chunks[i] = chunks[i].replace('?', ' ?').replace('.', ' .')
    bingo = ''
    for i in range(len(chunks) - 1):
        bingo += chunks[i] + nes[i][0]
    bingo += chunks[-1]
    return bingo

def vis(args, kb, model, data, device, tokenizer):
    while True:
        # text = 'Who is the father of Tony?'
        # text = 'Donald Trump married Tony, where is the place?'
        text = input('Input your question:')
        with torch.no_grad():
            input_ids = tokenizer.batch_encode_plus([text], max_length = 512, pad_to_max_length = True, return_tensors="pt", truncation = True)
            source_ids = input_ids['input_ids'].to(device)
            outputs = model.generate(
                input_ids=source_ids,
                max_length = 500,
            )
            outputs = [tokenizer.decode(output_id, skip_special_tokens = True, clean_up_tokenization_spaces = True) for output_id in outputs]
            outputs = [post_process(output) for output in outputs]
            print(outputs[0])

def predict(args, kb, model, data, device, tokenizer, executor):
    model.eval()
    count, correct = 0, 0
    pattern = re.compile(r'(.*?)\((.*?)\)')
    # torch.no_grad() 是一个上下文管理器，被该语句 wrap 起来的部分将不会track 梯度
    # 详解https://blog.csdn.net/weixin_46559271/article/details/105658654
    with torch.no_grad():
        all_outputs = []
        for batch in tqdm(data, total=len(data)):
            batch = batch[:3]
            source_ids, source_mask, choices = [x.to(device) for x in batch]
            outputs = model.generate(
                input_ids=source_ids,
                max_length = 500,
            )

            all_outputs.extend(outputs.cpu().numpy())
            # break
        
        outputs = [tokenizer.decode(output_id, skip_special_tokens = True, clean_up_tokenization_spaces = True) for output_id in all_outputs]
        # questions = [tokenizer.decode(source_id, skip_special_tokens = True, clean_up_tokenization_spaces = True) for source_id in all_answers]
        with open(os.path.join(args.save_dir, 'predict.txt'), 'w') as f:
            for output in tqdm(outputs):
                chunks = output.split('<b>')
                func_list = []
                inputs_list = []
                for chunk in chunks:
                    # print(chunk)
                    res = pattern.findall(chunk)
                    # print(res)
                    if len(res) == 0:
                        continue
                    res = res[0]
                    func, inputs = res[0], res[1]
                    if inputs == '':
                        inputs = []
                    else:
                        inputs = inputs.split('<c>')
                    
                    func_list.append(func)
                    inputs_list.append(inputs)
                ans = executor.forward(func_list, inputs_list, ignore_error = True)
                if ans == None:
                    ans = 'no'
                f.write(ans + '\n')


# def validate(args, kb, model, data, device, tokenizer, executor):
#     model.eval()
#     count, correct = 0, 0
#     pattern = re.compile(r'(.*?)\((.*?)\)')
#     with torch.no_grad():
#         all_outputs = []
#         all_answers = []
#         for batch in tqdm(data, total=len(data)):
#             source_ids, source_mask, choices, target_ids, answer = [x.to(device) for x in batch]
#             outputs = model.generate(
#                 input_ids=source_ids,
#                 max_length = 500,
#             )

#             all_outputs.extend(outputs.cpu().numpy())
#             all_answers.extend(answer.cpu().numpy())
#             # break
#         # outputs是模型根据测试集decode出来的问题
#         outputs = [tokenizer.decode(output_id, skip_special_tokens = True, clean_up_tokenization_spaces = True) for output_id in all_outputs]
#         # given_answer是测试集中的答案
#         given_answer = [data.vocab['answer_idx_to_token'][a] for a in all_answers]
#         # questions = [tokenizer.decode(source_id, skip_special_tokens = True, clean_up_tokenization_spaces = True) for source_id in all_answers]
#         # total = []
#         for a, output in tqdm(zip(given_answer, outputs)):  # 下面主体操作是根据decode出来的问题，模型来生成答案，然后和测试集原有答案进行计算准确率
#             # print(output)
#             chunks = output.split('<b>')
#             func_list = []
#             inputs_list = []
#             for chunk in chunks:
#                 # print(chunk)
#                 res = pattern.findall(chunk)
#                 # print(res)
#                 if len(res) == 0:
#                     continue
#                 res = res[0]
#                 func, inputs = res[0], res[1]
#                 if inputs == '':
#                     inputs = []
#                 else:
#                     inputs = inputs.split('<c>')
                
#                 func_list.append(func)
#                 inputs_list.append(inputs)
#             ans = executor.forward(func_list, inputs_list, ignore_error = True)
#             if ans == None:
#                 ans = 'no'
#             if ans == a:
#                 correct += 1
#             count += 1
#         acc = correct / count
#         logging.info('acc: {}'.format(acc))

#         return acc


# for 单机多卡
def validate(args, kb, model, data, tokenizer, executor):
    model.eval()
    count, correct = 0, 0
    pattern = re.compile(r'(.*?)\((.*?)\)')
    with torch.no_grad():
        all_outputs = []
        all_answers = []
        for batch in tqdm(data, total=len(data)):
            source_ids, source_mask, choices, target_ids, answer = [x.cuda() for x in batch]
            outputs = model.module.generate(
                input_ids=source_ids,
                max_length = 500,
            )

            all_outputs.extend(outputs.cpu().numpy())
            all_answers.extend(answer.cpu().numpy())
            # break
        # outputs是模型根据测试集decode出来的问题
        outputs = [tokenizer.decode(output_id, skip_special_tokens = True, clean_up_tokenization_spaces = True) for output_id in all_outputs]
        # given_answer是测试集中的答案
        given_answer = [data.vocab['answer_idx_to_token'][a] for a in all_answers]
        # questions = [tokenizer.decode(source_id, skip_special_tokens = True, clean_up_tokenization_spaces = True) for source_id in all_answers]
        # total = []
        for a, output in tqdm(zip(given_answer, outputs)):  # 下面主体操作是根据decode出来的问题，模型来生成答案，然后和测试集原有答案进行计算准确率
            # print(output)
            chunks = output.split('<b>')
            func_list = []
            inputs_list = []
            for chunk in chunks:
                # print(chunk)
                res = pattern.findall(chunk)
                # print(res)
                if len(res) == 0:
                    continue
                res = res[0]
                func, inputs = res[0], res[1]
                if inputs == '':
                    inputs = []
                else:
                    inputs = inputs.split('<c>')
                
                func_list.append(func)
                inputs_list.append(inputs)
            ans = executor.forward(func_list, inputs_list, ignore_error = True)
            if ans == None:
                ans = 'no'
            if ans == a:
                correct += 1
            count += 1
        acc = correct / count
        logging.info('acc: {}'.format(acc))

        return acc


def train(args):
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(args.gpu if args.use_cuda else "cpu")

    logging.info("Create train_loader and val_loader.........")
    vocab_json = os.path.join(args.input_dir, 'vocab.json')
    train_pt = os.path.join(args.input_dir, 'train.pt')
    val_pt = os.path.join(args.input_dir, 'test.pt')
    train_loader = DataLoader(vocab_json, train_pt, args.batch_size, training=True)
    val_loader = DataLoader(vocab_json, val_pt, args.batch_size)
    vocab = train_loader.vocab
    kb = DataForSPARQL(os.path.join(args.input_dir, 'kb.json'))
    logging.info("Create model.........")
    config_class, model_class, tokenizer_class = (BartConfig, BartForConditionalGeneration, BartTokenizer)
    tokenizer = tokenizer_class.from_pretrained(args.ckpt)
    model = model_class.from_pretrained(args.ckpt)
    model = model.to(device)
    logging.info(model)
    rule_executor = RuleExecutor(vocab, os.path.join(args.input_dir, 'kb.json'))
    # validate(args, kb, model, val_loader, device, tokenizer, rule_executor)
    predict(args, kb, model, val_loader, device, tokenizer, rule_executor)

    # vis(args, kb, model, val_loader, device, tokenizer)
def main():
    parser = argparse.ArgumentParser()
    # input and output
    parser.add_argument('--input_dir', default='./Bart_Program/preprocessed_data')  # 调试的时候路径用一个点，在下方终端Bart_Program下运行用两个点，对于这个项目要想统一，看should_read.md文件
    parser.add_argument('--save_dir', default='./static_bart_kopl/output/predict_result', help='path to save checkpoints and logs')
    
    parser.add_argument('--ckpt',  default='./static_bart_kopl/output/checkpoint/checkpoint-best')

    # training parameters
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--seed', type=int, default=666, help='random seed')

    # validating parameters
    # parser.add_argument('--num_return_sequences', default=1, type=int)
    # parser.add_argument('--top_p', default=)
    # model hyperparameters
    parser.add_argument('--dim_hidden', default=1024, type=int)
    parser.add_argument('--alpha', default = 1e-4, type = float)

    parser.add_argument('--use_cuda', type=bool, default=True)  
    parser.add_argument('--gpu', type=int, default=0)
    os.environ["CUDA_VISIBLE_DEVICES"]="3"

    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    time_ = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
    fileHandler = logging.FileHandler(os.path.join(args.save_dir, '{}.predict.log'.format(time_)))
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)
    # args display
    for k, v in vars(args).items():
        logging.info(k+':'+str(v))

    seed_everything(666)

    train(args)


if __name__ == '__main__':
    main()

