import os
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
# from sparql_engine import get_sparql_answer
import torch.optim as optim
import logging
import time
from lr_scheduler import get_linear_schedule_with_warmup
from predict_s import validate, post_process
# from PytorchBertBiLSTMClassify import predict
from executor_rule import RuleExecutor
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s')
logFormatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
rootLogger = logging.getLogger()
import warnings
warnings.simplefilter("ignore") # hide warnings that caused by invalid sparql query




def train(args):
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = torch.device(args.gpu if args.use_cuda else "cpu")

    logging.info("Create train_loader and val_loader.........")
    vocab_json = os.path.join(args.input_dir, 'vocab.json')
    train_pt = os.path.join(args.input_dir, 'train.pt')
    val_pt = os.path.join(args.input_dir, 'val.pt')
    train_loader = DataLoader(vocab_json, train_pt, args.batch_size, training=True)
    val_loader = DataLoader(vocab_json, val_pt, args.batch_size)  # DataLoader去data.py里面找，默认的training是False

    vocab = train_loader.vocab
    kb = DataForSPARQL(os.path.join(args.input_dir, 'kb.json'))
    
    logging.info("Create model.........")
    # rule_executor = RuleExecutor(vocab, os.path.join(args.input_dir, 'kb.json'))
    config_class, model_class, tokenizer_class = (BartConfig, BartForConditionalGeneration, BartTokenizer)
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    model = model_class.from_pretrained(args.model_name_or_path)
    model = model.cuda()
    model = nn.DataParallel(model)

    # KD
    teacher_tokenizer = tokenizer_class.from_pretrained(args.teacher_ckpt)
    teacher_model = model_class.from_pretrained(args.teacher_ckpt)
    teacher_model = teacher_model.cuda()
    teacher_model = nn.DataParallel(teacher_model)
    
    
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
    args.warmup_steps = int(t_total * args.warmup_proportion)
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
    # evaluate(args, model, val_loader, device)
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
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

            
            outputs = model(**inputs)
            loss_ce = outputs[0]

            student_reps = outputs["encoder_hidden_states"]

            # KD
            with torch.no_grad():
                teacher_reps = teacher_model(**inputs)["encoder_hidden_states"]

            new_teacher_reps = teacher_reps
            new_student_reps = student_reps

            rep_loss = 0.0
            for student_rep, teacher_rep in zip(new_student_reps, new_teacher_reps):
                    tmp_loss = F.mse_loss(F.normalize(student_rep, p=2, dim=1), F.normalize(teacher_rep, p=2, dim=1))
                    rep_loss += tmp_loss
            
            s_probs = F.log_softmax(outputs[0] / args.T, dim=-1)
            t_probs = F.softmax(t_logits / args.T, dim=-1)
            loss_kd = F.kl_div(s_probs, t_probs, reduction='batchmean') * args.T * args.T
            loss = args.ce_weight * loss_ce.mean() + args.kd_weight * loss_kd
            # loss = (1 - kd_weight) * loss_ce.mean() + kd_weight * loss_kd
            # loss = (1 - kd_weight) * loss_ce.mean() + kd_weight * rep_loss
            # loss = (1 - kd_weight) * loss_ce.mean() + kd_weight * loss_kd + kd_weight * rep_loss
            

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
                cur_acc = validate(args, kb, model, val_loader, tokenizer)
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
            
           
        logging.info("\n")
        # if 'cuda' in str(device):
            # torch.cuda.empty_cache()
        torch.cuda.empty_cache()
    return global_step, tr_loss / global_step
        


def main():
    parser = argparse.ArgumentParser()
    # input and output
    parser.add_argument('--input_dir', default='./static_bart_sparql/preprocessed_data')
    parser.add_argument('--output_dir', default='./static_bart_sparql/output/checkpoint')

    parser.add_argument('--save_dir', default='./static_bart_sparql/log', help='path to save checkpoints and logs')
    
    parser.add_argument('--model_name_or_path', default='./KQAPro_ckpt/sparql_ckpt')
    parser.add_argument('--ckpt')

    # training parameters
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--seed', type=int, default=666, help='random seed')
    parser.add_argument('--learning_rate', default=3e-5, type = float)
    parser.add_argument('--num_train_epochs', default=25, type = int)
    parser.add_argument('--save_steps', default=448, type = int)
    parser.add_argument('--logging_steps', default=10000, type = int)
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
    parser.add_argument("--teacher-ckpt", default='./ckpt_kd_result/program/output_kd_2/checkpoint/checkpoint-best')
    parser.add_argument("--kd_rep_weight", type=float, default=0.8)

    # validating parameters
    # parser.add_argument('--num_return_sequences', default=1, type=int)
    # parser.add_argument('--top_p', default=)
    # model hyperparameters
    parser.add_argument('--dim_hidden', default=1024, type=int)
    parser.add_argument('--alpha', default = 1e-4, type = float)

    
    parser.add_argument('--use_cuda', type=bool, default=True)  
    
    os.environ["CUDA_VISIBLE_DEVICES"]="1, 3"

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

