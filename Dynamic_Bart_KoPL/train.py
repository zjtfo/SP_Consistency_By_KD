port os
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import argparse
import shutil
import json
from tqdm import tqdm
from datetime import date
from misc import MetricLogger, seed_everything, ProgressBar
from load_kb import DataForSPARQL
from data import DataLoader
from transformers import BartConfig, BartForConditionalGeneration, BartTokenizer
from transformers import BertTokenizer,BertModel
import re
# from .sparql_engine import get_sparql_answer
import torch.optim as optim
import logging
import time
from lr_scheduler import get_linear_schedule_with_warmup
from predict import validate
from executor_rule import RuleExecutor
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s')
logFormatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
rootLogger = logging.getLogger()
import warnings
warnings.simplefilter("ignore") # hide warnings that caused by invalid sparql query

# 分类器
class bert_lstm(nn.Module):
    def __init__(self, bertpath, hidden_dim, output_size, n_layers, bidirectional=True, drop_prob=0.5):
        super(bert_lstm, self).__init__()
 
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        
        #Bert ----------------重点，bert模型需要嵌入到自定义模型里面
        self.bert=BertModel.from_pretrained(bertpath)
        for param in self.bert.parameters():
            param.requires_grad = True
        
        # LSTM layers
        self.lstm = nn.LSTM(768, hidden_dim, n_layers, batch_first=True,bidirectional=bidirectional)
        
        # dropout layer
        self.dropout = nn.Dropout(drop_prob)
        
        # linear and sigmoid layers
        if bidirectional:
            self.fc = nn.Linear(hidden_dim*2, output_size)
        else:
            self.fc = nn.Linear(hidden_dim, output_size)
          
        #self.sig = nn.Sigmoid()
 
    def forward(self, x, hidden):
        batch_size = x.size(0)
        #生成bert字向量
        x=self.bert(x)[0]     #bert 字向量
        
        # lstm_out
        #x = x.float()
        lstm_out, (hidden_last,cn_last) = self.lstm(x, hidden)
        #print(lstm_out.shape)   #[32,100,768]
        #print(hidden_last.shape)   #[4, 32, 384]
        #print(cn_last.shape)    #[4, 32, 384]
        
        #修改 双向的需要单独处理
        if self.bidirectional:
            #正向最后一层，最后一个时刻
            hidden_last_L=hidden_last[-2]
            #print(hidden_last_L.shape)  #[32, 384]
            #反向最后一层，最后一个时刻
            hidden_last_R=hidden_last[-1]
            #print(hidden_last_R.shape)   #[32, 384]
            #进行拼接
            hidden_last_out=torch.cat([hidden_last_L,hidden_last_R],dim=-1)
            #print(hidden_last_out.shape,'hidden_last_out')   #[32, 768]
        else:
            hidden_last_out=hidden_last[-1]   #[32, 384]
            
            
        # dropout and fully-connected layer
        out = self.dropout(hidden_last_out)
        #print(out.shape)    #[32,768]
        out = self.fc(out)
        
        return out
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        
        number = 1
        if self.bidirectional:
            number = 2
        
        if (torch.cuda.is_available()):
            hidden = (weight.new(self.n_layers*number, batch_size, self.hidden_dim).zero_().float().cuda(),
                      weight.new(self.n_layers*number, batch_size, self.hidden_dim).zero_().float().cuda()
                     )
        else:
            hidden = (weight.new(self.n_layers*number, batch_size, self.hidden_dim).zero_().float(),
                      weight.new(self.n_layers*number, batch_size, self.hidden_dim).zero_().float()
                     )
        
        return hidden


class ModelConfig:
    batch_size = 32
    output_size = 2
    hidden_dim = 384   #768/2
    n_layers = 2
    lr = 2e-5
    bidirectional = True  #这里为True，为双向LSTM
    # training params
    epochs = 10
    # batch_size=50
    print_every = 100
    clip=5 # gradient clipping
    use_cuda = torch.cuda.is_available()
    # bert_path = 'bert-base-chinese' #预训练bert路径
    # save_path = 'bert_bilstm.pth' #模型保存路径
    bert_path = './Program_SPARQL_Classification/bert-base-cased'
    save_path = './Program_SPARQL_Classification/bert_bilstm_kqa_3(2_epoch).pth'


# 剔除标点符号,\xa0 空格
def pretreatment(comments):
    result_comments=[]
    punctuation='。，？！：%&~（）、；“”&|,.?!:%&~();""'
    for comment in comments:
        comment= ''.join([c for c in comment if c not in punctuation])
        comment= ''.join(comment.split())   #\xa0
        result_comments.append(comment)
    
    return result_comments


def classifier_predict(test_comment_list, net, net_tokenizer):
    result_comments = pretreatment(test_comment_list)   # 预处理去掉标点符号
    # 转换为字id
    result_comments_id = net_tokenizer(result_comments,
                                    padding=True,
                                    truncation=True,
                                    max_length=500,
                                    return_tensors='pt')
    tokenizer_id = result_comments_id['input_ids']
    # print(tokenizer_id.shape)
    inputs = tokenizer_id
    batch_size = inputs.size(0)
    # batch_size = 32
    # initialize hidden state
    h = net.init_hidden(batch_size)

    if(torch.cuda.is_available()):
        inputs = inputs.cuda()
    
    net.eval()
    with torch.no_grad():
        # get the output from the model
        output = net(inputs, h)
        output = torch.nn.Softmax(dim=1)(output)
        # print(output.shape)  # torch.Size([1, 2])
        pred = torch.max(output, 1)[1]
        # print(pred.shape)  # torch.Size([1])
        # print(pred)  # tensor([1], device='cuda:0')
        # printing output value, before rounding
        # print('预测概率为: {:.6f}'.format(torch.max(output, 1)[0].item()))
        # if(pred.item()==1):
        #     print("预测结果为:正向")
        #     return "yes"
        # else:
        #     print("预测结果为:负向")
        #     return "no"
        return pred.sum().int()


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



def train(args):
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = torch.device(args.gpu if args.use_cuda else "cpu")

    logging.info("Create train_loader and val_loader.........")
    vocab_json = os.path.join(args.input_dir, 'vocab.json')
    train_pt = os.path.join(args.input_dir, 'train.pt')
    val_pt = os.path.join(args.input_dir, 'val.pt')
    train_loader = DataLoader(vocab_json, train_pt, args.batch_size, training=True)
    val_loader = DataLoader(vocab_json, val_pt, 64)  # DataLoader去data.py里面找，默认的training是False

    vocab = train_loader.vocab
    kb = DataForSPARQL(os.path.join(args.input_dir, 'kb.json'))
    rule_executor = RuleExecutor(vocab, os.path.join(args.input_dir, 'kb.json'))
    logging.info("Create model.........")
    config_class, model_class, tokenizer_class = (BartConfig, BartForConditionalGeneration, BartTokenizer)
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    model = model_class.from_pretrained(args.model_name_or_path)
    # model = model.to(device)
    model = model.cuda()
    model = nn.DataParallel(model)

    # KD
    teacher_tokenizer = tokenizer_class.from_pretrained(args.teacher_ckpt)
    teacher_model = model_class.from_pretrained(args.teacher_ckpt)
    # teacher_model = teacher_model.to(device)
    teacher_model = teacher_model.cuda()
    teacher_model = nn.DataParallel(teacher_model)

    # 转换模型
    transfer_tokenizer = tokenizer_class.from_pretrained(args.transfer_ckpt)
    transfer_model = model_class.from_pretrained(args.transfer_ckpt)
    # transfer_model = transfer_model.to(device)
    transfer_model = transfer_model.cuda()
    transfer_model = nn.DataParallel(transfer_model)

    # 分类器
    config = ModelConfig()
    net = bert_lstm(config.bert_path, 
                config.hidden_dim, 
                config.output_size,
                config.n_layers, 
                config.bidirectional)
    net.load_state_dict(torch.load(config.save_path))
    net.cuda()
    net_tokenizer = BertTokenizer.from_pretrained(config.bert_path)

    logging.info(model)
    t_total = len(train_loader) // args.gradient_accumulation_steps * args.num_train_epochs    # 准备优化器和时间表(线性预热和衰减)
    no_decay = ["bias", "LayerNorm.weight"]
    bart_param_optimizer = list(model.named_parameters())
    optimizer_grouped_parameters = [
        {'params': [p for n, p in bart_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay, 'lr': args.learning_rate},
        {'params': [p for n, p in bart_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         'lr': args.learning_rate}
    ]
    args.warmup_steps = int(t_total * args.warmup_proportion)  # total为147475
    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=t_total)
    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
            os.path.join(args.model_name_or_path, "scheduler.pt")):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    # Train!
        logging.info("***** Running training *****")
        logging.info("  Num examples = %d", len(train_loader.dataset))
        logging.info("  Num Epochs = %d", args.num_train_epochs)
        logging.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
        logging.info("  Total optimization steps = %d", t_total)

    global_step = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path) and "checkpoint" in args.model_name_or_path:
        # set global_step to gobal_step of last saved checkpoint from model path
        global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
        epochs_trained = global_step // (len(train_loader) // args.gradient_accumulation_steps)
        steps_trained_in_current_epoch = global_step % (len(train_loader) // args.gradient_accumulation_steps)
        logging.info("  Continuing training from checkpoint, will skip to saved global_step")
        logging.info("  Continuing training from epoch %d", epochs_trained)
        logging.info("  Continuing training from global step %d", global_step)
        logging.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
    logging.info('Checking...')
    logging.info("===================Dev==================")
    # validate(args, kb, model, val_loader, device, tokenizer, rule_executor)
    validate(args, kb, model, val_loader, tokenizer, rule_executor)
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    prefix = 25984
    acc = 0.0
    for _ in range(int(args.num_train_epochs)):
        pbar = ProgressBar(n_total=len(train_loader), desc='Training')
        for step, batch in enumerate(train_loader):
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue
            model.train()
            batch = tuple(t.cuda() for t in batch)
            pad_token_id = tokenizer.pad_token_id
            source_ids, source_mask, y = batch[0], batch[1], batch[-2]
            y_ids = y[:, :-1].contiguous()
            lm_labels = y[:, 1:].clone()
            lm_labels[y[:, 1:] == pad_token_id] = -100

            inputs = {
                "input_ids": source_ids.cuda(),
                "attention_mask": source_mask.cuda(),
                "decoder_input_ids": y_ids.cuda(),
                "labels": lm_labels.cuda(),  # 因为报错，后来发现将lm_labels改成labels就可以
                "output_hidden_states": True,
            }


            # 生成的sparql
            outputs_s = teacher_model.module.generate(
                input_ids=source_ids,
                max_length = 500,
            )
            outputs_sparql = [teacher_tokenizer.decode(output_id, skip_special_tokens = True, clean_up_tokenization_spaces = True) for output_id in outputs_s]
            outputs_sparql = [post_process(output) for output in outputs_sparql]
            # print(outputs_sparql[0])
            # print(len(outputs_sparql))  # 16，就是等于batch_size大小

            
            # 生成的program
            outputs_p = model.module.generate(
                input_ids=y_ids,
                max_length = 500,
            )
            outputs_program = [tokenizer.decode(output_id, skip_special_tokens = True, clean_up_tokenization_spaces = True) for output_id in outputs_p]
            outputs_program = [post_process(output) for output in outputs_program]
            # print(outputs_program[0])
            # print(len(outputs_program))

            
            # 送到训练好的分类器
            
            test_comments = []

            for i in range(len(outputs_sparql)):
                test_comments.append(str(outputs_program[i] + '[SEP]' + outputs_sparql[i].replace("\"", "'")))

            sim_count = classifier_predict(test_comments, net, net_tokenizer)

            print("sim_count->", sim_count)
            print("蒸馏权重->", sim_count / len(outputs_sparql))
            kd_weight = sim_count / len(outputs_sparql)


            # sparql to kopl
            sparql_2_kopl = transfer_model.module.generate(
                input_ids=outputs_s,
                max_length = 500,
            )

            transfer_pad_token_id = transfer_tokenizer.pad_token_id
            transfer_y_ids= sparql_2_kopl[:, :-1].contiguous()
            transfer_lm_labels = sparql_2_kopl[:, 1:].clone()
            transfer_lm_labels[sparql_2_kopl[:, 1:] == transfer_pad_token_id] = -100

            transfer_inputs = {
                "input_ids": source_ids.cuda(),
                "attention_mask": source_mask.cuda(),
                "decoder_input_ids": transfer_y_ids.cuda(),
                "labels": transfer_lm_labels.cuda(),  # 因为报错，后来发现将lm_labels改成labels就可以
            }
            transfer_loss_ce = model(**transfer_inputs)[0]


            outputs = model(**inputs)
            loss_ce = outputs[0]

            
            
            loss = (1 - kd_weight) * loss_ce.mean() + kd_weight * transfer_loss_ce.mean()
            # loss = (1 - kd_weight) * loss_ce.mean() + kd_weight * transfer_loss_ce.mean() + kd_weight * rep_loss
            
            # loss = args.ce_weight * loss_ce.mean() + args.kd_weight * transfer_loss_ce.mean()
            # loss = args.ce_weight * loss_ce.mean() + args.kd_weight * transfer_loss_ce.mean() + args.kd_rep_weight * rep_loss

            # break

            loss.backward()
            pbar(step, {'loss': loss.item()})
            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
            if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                logging.info("===================Dev==================")
                # cur_acc = validate(args, kb, model, val_loader, device, tokenizer, rule_executor)
                cur_acc = validate(args, kb, model, val_loader, tokenizer, rule_executor)
                if cur_acc >= acc:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format("best"))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logging.info("Saving model checkpoint to %s", output_dir)
                    tokenizer.save_vocabulary(output_dir)
                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    logging.info("Saving optimizer and scheduler states to %s", output_dir)
                    
                    acc = cur_acc

        # break
        logging.info("\n")
        # if 'cuda' in str(device):
        torch.cuda.empty_cache()
    return global_step, tr_loss / global_step


def main():
    parser = argparse.ArgumentParser()
    # input and output
    parser.add_argument('--input_dir', default='./Bart_Program/preprocessed_data')
    parser.add_argument('--output_dir', default='./dynamic_bart_kopl/output/checkpoint')

    parser.add_argument('--save_dir', default='./dynamic_bart_kopl/log', help='path to save checkpoints and logs')
    parser.add_argument('--model_name_or_path', default='./bart-base')
    
    parser.add_argument('--ckpt')

    # training parameters
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--seed', type=int, default=666, help='random seed')
    parser.add_argument('--learning_rate', default=3e-5, type = float)
    parser.add_argument('--num_train_epochs', default=25, type = int)
    parser.add_argument('--save_steps', default=448, type = int)
    parser.add_argument('--logging_steps', default=448, type = int)
    parser.add_argument('--warmup_proportion', default=0.1, type = float,
                        help="Proportion of training to perform linear learning rate warmup for,E.g., 0.1 = 10% of training.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.", )
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    
    # KD
    parser.add_argument("--ce-weight", type=float, default=0.2)
    parser.add_argument("--kd-weight", type=float, default=0.8)
    parser.add_argument("--T", type=float, default=6.0)
    parser.add_argument("--teacher-ckpt", default='./KQAPro_ckpt/sparql_ckpt')
    parser.add_argument("--kd_rep_weight", type=float, default=0.8)

    # transfer_based
    parser.add_argument("--transfer-ckpt", default='./sparql_2_kopl/output/checkpoint/checkpoint-best')

    # validating parameters
    # parser.add_argument('--num_return_sequences', default=1, type=int)
    # parser.add_argument('--top_p', default=)
    # model hyperparameters
    parser.add_argument('--dim_hidden', default=1024, type=int)
    parser.add_argument('--alpha', default = 1e-4, type = float)

    

    args = parser.parse_args()

    if not os.path.exists(args.save_dir):  # 接下来几行都是为了生成log日志
        os.makedirs(args.save_dir)
    time_ = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
    fileHandler = logging.FileHandler(os.path.join(args.save_dir, '{}.log'.format(time_)))
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)
    # args display
    for k, v in vars(args).items():  # vars(args)将args传递的参数从namespace转换为dict
        logging.info(k+':'+str(v))  # 在控制台进行打印

    seed_everything(666)

    train(args)


if __name__ == '__main__':
    main()
