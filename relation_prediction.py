import sys
import torch
import numpy as np
from torch.utils.data._utils.collate import default_convert
from torch.utils.data import RandomSampler, DataLoader, SequentialSampler, TensorDataset, random_split
from tqdm import tqdm

from colorama import Fore
import os
import random
from utils import load_count_dict, load_cycle, negative_relation_sampling, load_text, load_paths, load_triplets_with_pos_tails, \
    myConvert,reshape_relation_prediction_ranking_data,element_wise_cos,cal_metrics,CosineEmbeddingLoss
from torch.cuda.amp import GradScaler, autocast
from transformers import set_seed
from models import SentenceTransformer
import torch.nn.functional as F
import argparse
from torch.optim import lr_scheduler
from collections import defaultdict

parser = argparse.ArgumentParser(description='Relation Prediction')

parser.add_argument('--device', type=str, default='cuda:0',
                    help='CUDA device or CPU')
parser.add_argument('--epochs', type=int, default=30,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=1, metavar='N',
                    help='batch size')
parser.add_argument('--dataset', type=str, default='FB15k-237-subset',
                    help='name of the dataset')
parser.add_argument('--path_dir', type=str, default=None,
                    help='location of extracted paths for each triplet')
parser.add_argument('--text_dir', type=str, default=None,
                    help='location of relation and entity texts')
parser.add_argument('--model_load_file', type=str, default=None,
                    help='location to load pretrained cycle model')
parser.add_argument('--model_save_dir', type=str, default=None,
                    help='location to save model')
parser.add_argument('--learning_rate', type=float, default=1e-5,
                    help='learning rate')
parser.add_argument('--model', type=str, default='sentence-transformers/all-mpnet-base-v2',
                    help='sentence transformer model name on Hugging Face website (huggingface.co)')
parser.add_argument('--tokenizer', type=str, default='sentence-transformers/all-mpnet-base-v2',
                    help='tokenizer name on Hugging Face website (huggingface.co)')
parser.add_argument('--train_sample_num', type=int, default=-1,
                    help='number of training samples randomly sampled, use -1 for all data')
parser.add_argument('--valid_sample_num', type=int, default=-1,
                    help='number of validating samples randomly sampled, use -1 for all data')
parser.add_argument('--max_path_num', type=int, default=3,
                    help='number of paths loaded for each triplet')
parser.add_argument('--neg_sample_num_train', type=int, default=5,
                    help='number of negative training samples')
parser.add_argument('--neg_sample_num_valid', type=int, default=5,
                    help='number of negative validating samples')
parser.add_argument('--neg_sample_num_test', type=int, default=50,
                    help='number of negative testing samples')
parser.add_argument('--mode', type=str, default='head',
                    help='whether head or tail is fixed')
parser.add_argument('--seed', type=int, default=42,
                    help='random seed')
parser.add_argument('--suffix', type=str, default="_full",
                    help='suffix of the train file name')
parser.add_argument('--do_train',action='store_true', default=True,
                    help='whether train or not')
parser.add_argument('--do_test',action='store_true', default=True,
                    help='whether test or not')
parser.add_argument('--output_dir', type=str, default=None,
                    help='location to output test results')
parser.add_argument('--text_file', type=str, default='GoogleWikipedia',
                    help='long text to be loaded')
parser.add_argument('--text_length', type=int, default=0,
                    help='long text length')
args = parser.parse_args()


print(args)
set_seed(args.seed)
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'
device = torch.device(args.device)

if args.path_dir is None:
    path_dir = os.path.join("data/relation_prediction_path_data/", args.dataset, f'ranking_{args.mode}{args.suffix}/')
else:
    path_dir=args.path_dir

if args.text_dir is None:
    text_dir = os.path.join("data/data", args.dataset)
else:
    text_dir=args.text_dir

if args.model_load_file is None:
    model_load_file=os.path.join(f"save/{args.dataset}{args.suffix}",f"relation_prediction_{args.mode}/best_val.pth")
else:
    model_load_file=args.model_load_file

if args.model_save_dir is None:
    model_save_dir=os.path.join(f"save/{args.dataset}{args.suffix}", f"relation_prediction_{args.mode}/")
else:
    model_save_dir=args.model_save_dir
if args.output_dir is None:
    output_dir=os.path.join(f"output/{args.dataset}{args.suffix}", f"relation_prediction_{args.mode}/")
else:
    output_dir=args.output_dir


all_pos_tails = defaultdict(lambda: defaultdict(list))
train_triplets = load_triplets_with_pos_tails(os.path.join(path_dir, "ranking_train.txt"),args.mode,all_pos_tails,args.neg_sample_num_train)
train_paths = load_paths(os.path.join(path_dir, "relation_paths_train.txt"),
                         os.path.join(path_dir, "entity_paths_train.txt"), len(train_triplets),args.max_path_num,args.mode)
valid_triplets = load_triplets_with_pos_tails(os.path.join(path_dir, "ranking_valid.txt"),args.mode,all_pos_tails,args.neg_sample_num_valid)
valid_paths = load_paths(os.path.join(path_dir, "relation_paths_valid.txt"),
                         os.path.join(path_dir, "entity_paths_valid.txt"), len(valid_triplets),args.max_path_num,args.mode)
ranking_triplets = load_triplets_with_pos_tails(os.path.join(path_dir, "ranking_test.txt"),args.mode,all_pos_tails,args.neg_sample_num_test)
ranking_paths = load_paths(os.path.join(path_dir, "relation_paths_test.txt"),
                        os.path.join(path_dir, "entity_paths_test.txt"), len(ranking_triplets),args.max_path_num,args.mode)

# train_rules = load_rules(os.path.join(path_dir, "relation_rules_train.txt"),
#                          os.path.join(path_dir, "rules_heads_train.txt"), len(train_triplets),args.max_path_num)
# valid_rules = load_rules(os.path.join(path_dir, "relation_rules_valid.txt"),
#                          os.path.join(path_dir, "rules_heads_valid.txt"), len(valid_triplets),args.max_path_num)
# ranking_rules = load_rules(os.path.join(path_dir, "relation_rules_test.txt"),
#                         os.path.join(path_dir, "rules_heads_test.txt"), len(ranking_triplets),args.max_path_num)

# train_merged=merge_rules_and_paths(train_paths,train_rules,args.neg_sample_num_train,args.max_path_num)
# valid_merged=merge_rules_and_paths(valid_paths,valid_rules,args.neg_sample_num_valid,args.max_path_num)
# ranking_merged=merge_rules_and_paths(ranking_paths,ranking_rules,args.neg_sample_num_test,args.max_path_num)

text,relation_texts = load_text(text_dir,args.text_file,args.text_length)
all_dict = {**text['entity'], **text['relation']}

valid_pos_tails=[all_pos_tails[t[0]][t[1]] for t in valid_triplets[::args.neg_sample_num_valid]]
ranking_pos_tails=[all_pos_tails[t[0]][t[1]] for t in ranking_triplets[::args.neg_sample_num_test]]

train_triplets,train_paths,train_labels,_=reshape_relation_prediction_ranking_data(train_triplets,train_paths,args.neg_sample_num_train,all_dict)
valid_triplets,valid_paths,valid_labels,_=reshape_relation_prediction_ranking_data(valid_triplets,valid_paths,args.neg_sample_num_valid,all_dict)
ranking_triplets,ranking_paths,ranking_labels,ranking_indexes=reshape_relation_prediction_ranking_data(ranking_triplets,ranking_paths,args.neg_sample_num_test,all_dict)




train_data = list(zip(train_triplets, train_paths,train_labels))
valid_data = list(zip(valid_triplets, valid_paths,valid_labels,valid_pos_tails))
if args.train_sample_num==-1:
    train_sampler = RandomSampler(train_data)
else:
    train_sampler=RandomSampler(train_data,replacement=True,num_samples=args.train_sample_num)
train_data_loader = DataLoader(train_data, sampler=train_sampler, batch_size=args.batch_size, collate_fn=myConvert)

if args.valid_sample_num==-1:
    valid_sampler = RandomSampler(valid_data)
else:
    valid_sampler = RandomSampler(valid_data,replacement=True,num_samples=args.valid_sample_num)
valid_data_loader = DataLoader(valid_data, sampler=valid_sampler, batch_size=args.batch_size, collate_fn=myConvert)

ranking_data = list(zip(ranking_triplets, ranking_paths,ranking_labels,ranking_pos_tails))
ranking_sampler = SequentialSampler(ranking_data)
ranking_data_loader = DataLoader(ranking_data, sampler=ranking_sampler, batch_size=args.batch_size, collate_fn=myConvert)

scaler = GradScaler()




model=SentenceTransformer(tokenizer_name=args.tokenizer,model_name=args.model,device=device)

# model=torch.nn.DataParallel(model,device_ids=[0,1])
model.to(device)

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'gamma', 'beta']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.0}
]
optimizer = torch.optim.AdamW(lr=args.learning_rate, betas=(0.9, 0.98), eps=1e-9, params=optimizer_grouped_parameters)
scheduler=lr_scheduler.StepLR(optimizer,step_size=5,gamma=0.5)
# criterion=torch.nn.CrossEntropyLoss()
criterion=CosineEmbeddingLoss
def train():
    best_val_acc = 0
    for epoch in range(1, args.epochs + 1):
        # ============================================ TRAINING ============================================================
        print(f"Training epoch {epoch}")
        training_pbar = tqdm(total=args.train_sample_num if args.train_sample_num>0 else len(train_data),
                             position=0, leave=True,
                             file=sys.stdout, bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.WHITE, Fore.RESET))
        model.train()
        tr_loss = 0
        nb_tr_steps = 0
        for step, batch in enumerate(train_data_loader):
            sentence1, sentence2, targets = batch
            # sentence1 = [["; ".join([all_dict[er] for er in st]) + " [SEP]" for st in st1] for st1 in
            #                      sentence1]
            # sentence2 = [[["; ".join([all_dict[er] for er in s]) + " [SEP]" for s in st] for st in st2] for st2 in
            #                   sentence2]
            # sentence1 = [[f'{all_dict[st1[1]]} ; {all_dict[st1[0]]} ; {all_dict[st1[2]]} [SEP]' for st1 in st2] for st2 in
            #              sentence1]
            # sentence2 = [
            #     [[f'{"; ".join([all_dict[r] for r in st[1]])} ; {all_dict[st[0]]} ; {all_dict[st[2]]} [SEP]' for st in st1]
            #      for st1 in st2] for st2 in sentence2]
            sentence1 = [
                [
                    f'{text["entity"][st1[0]]}: {text["entitylong"][st1[0]]} [SEP] {text["relation"][st1[1]]} [SEP] {text["entity"][st1[2]]}: {text["entitylong"][st1[2]]} [SEP]'
                    for st1 in st2] for st2 in sentence1]
            sentence2 = [[[
                              f'{text["entity"][s[0]]}: {text["entitylong"][s[0]]} [SEP] {" [SEP] ".join([all_dict[er] for er in s[1:-1]])} [SEP] {text["entity"][s[-1]]}:{text["entitylong"][s[-1]]} [SEP]'
                              for s in st] for
                          st in st2] for st2 in sentence2]
            targets = torch.tensor(targets).to(device)
            optimizer.zero_grad()
            outputs = []
            with autocast():
                for s1, s2, tgt in zip(sentence1, sentence2, targets):
                    embed1 = model(s1).unsqueeze(1)
                    embed2 = torch.stack([model(s) for s in s2])
                    sim = torch.cosine_similarity(embed1, embed2, dim=2)
                    # sim = torch.stack(
                    #     [torch.max(torch.stack([torch.cosine_similarity(e1, e2, dim=0) for e2 in emb2])) for
                    #      e1, emb2 in
                    #      zip(embed1, embed2)])
                    outputs.append(sim)
                outputs = torch.stack(outputs)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            tr_loss += loss.item()
            nb_tr_steps += 1
            training_pbar.update(len(targets))
        training_pbar.close()
        scheduler.step()
        print(f"Learning rate={optimizer.param_groups[0]['lr']}\nTraining loss={tr_loss / nb_tr_steps:.4f}")
        test()
        if epoch % 3 == 0:
            valid_acc=validate()
            if valid_acc > best_val_acc:
                best_val_acc = valid_acc
                if not os.path.exists(model_save_dir):
                    os.makedirs(model_save_dir)
                torch.save(model.state_dict(), os.path.join(model_save_dir,'best_val.pth'))

    print(f"Best Validation Accuracy: {best_val_acc}")
def validate():
    valid_pbar = tqdm(total=args.valid_sample_num if args.valid_sample_num>0 else len(valid_data),
                     position=0, leave=True,
                     file=sys.stdout, bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.BLUE, Fore.RESET))
    model.eval()
    metrics = np.array([0., 0., 0., 0., 0.])  # MR, MRR, Hit@1, Hit@3, Hit@10
    nb_valid_steps = 0
    for batch in valid_data_loader:
        sentence1, sentence2, targets,pos_tails = batch
        pos_mask = [[s[2] in pt and i != tgt for i, s in enumerate(st1)] for st1, pt, tgt in
                    zip(sentence1, pos_tails, targets)]
        # sentence1 = [["; ".join([all_dict[er] for er in st]) + " [SEP]" for st in st1] for st1 in
        #              sentence1]
        # sentence2 = [[["; ".join([all_dict[er] for er in s]) + " [SEP]" for s in st] for st in st2] for st2 in
        #              sentence2]

        sentence1 = [
            [
                f'{text["entity"][st1[0]]}: {text["entitylong"][st1[0]]} [SEP] {text["relation"][st1[1]]} [SEP] {text["entity"][st1[2]]}: {text["entitylong"][st1[2]]} [SEP]'
                for st1 in st2] for st2 in sentence1]
        sentence2 = [[[
                          f'{text["entity"][s[0]]}: {text["entitylong"][s[0]]} [SEP] {" [SEP] ".join([all_dict[er] for er in s[1:-1]])} [SEP] {text["entity"][s[-1]]}:{text["entitylong"][s[-1]]} [SEP]'
                          for s in st] for
                      st in st2] for st2 in sentence2]

        targets = torch.tensor(targets).to(device)
        optimizer.zero_grad()
        outputs = []
        with autocast():
            with torch.no_grad():
                for s1, s2, tgt in zip(sentence1, sentence2, targets):
                    embed1 = model(s1).unsqueeze(1)
                    embed2 = torch.stack([model(s) for s in s2])

                    sim,_ = torch.max(torch.cosine_similarity(embed1,embed2,dim=2),dim=1)
                    # sim = torch.stack(
                    #     [torch.max(torch.stack([torch.cosine_similarity(e1, e2, dim=0) for e2 in emb2])) for
                    #      e1, emb2 in
                    #      zip(embed1, embed2)])
                    outputs.append(sim)
                outputs = torch.stack(outputs)
        metrics += cal_metrics(outputs.cpu().numpy(), targets.cpu().numpy(),pos_mask)
        nb_valid_steps +=1
        valid_pbar.update(len(targets))
    valid_pbar.close()
    metrics = metrics / nb_valid_steps
    print(f"MR: {metrics[0]}, MRR: {metrics[1]}, Hit@1: {metrics[2]}, Hit@3: {metrics[3]}, Hit@10: {metrics[4]}")
    return metrics[2]
def test():
    ranking_pbar = tqdm(total=len(ranking_triplets),
                        position=0, leave=True,
                        file=sys.stdout, bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.GREEN, Fore.RESET))
    model.eval()
    metrics=np.array([0.,0.,0.,0.,0.]) # MR, MRR, Hit@1, Hit@3, Hit@10
    nb_ranking_steps = 0
    scores=[]
    ranking_positions=[]
    for batch in list(ranking_data_loader):
        sentence1, sentence2, targets,pos_tails = batch
        pos_mask=[[s[2] in pt and i!=tgt for i,s in enumerate(st1)] for st1,pt,tgt in zip(sentence1,pos_tails,targets)]
        # sentence1 = [["; ".join([all_dict[er] for er in st]) + " [SEP]" for st in st1] for st1 in
        #              sentence1]
        # sentence2 = [[["; ".join([all_dict[er] for er in s]) + " [SEP]" for s in st] for st in st2] for st2 in
        #              sentence2]
        # sentence1 = [[f'{all_dict[st1[1]]} ; {all_dict[st1[0]]} ; {all_dict[st1[2]]} [SEP]' for st1 in st2] for st2 in
        #              sentence1]
        # sentence2 = [
        #     [[f'{"; ".join([all_dict[r] for r in st[1]])} ; {all_dict[st[0]]} ; {all_dict[st[2]]} [SEP]' for st in st1]
        #      for st1 in st2] for st2 in sentence2]

        sentence1 = [
            [f'{text["entity"][st1[0]]}: {text["entitylong"][st1[0]]} [SEP] {text["relation"][st1[1]]} [SEP] {text["entity"][st1[2]]}: {text["entitylong"][st1[2]]} [SEP]'
             for st1 in st2] for st2 in sentence1]
        sentence2 = [[[f'{text["entity"][s[0]]}: {text["entitylong"][s[0]]} [SEP] {" [SEP] ".join([all_dict[er] for er in s[1:-1]])} [SEP] {text["entity"][s[-1]]}:{text["entitylong"][s[-1]]} [SEP]' for s in st] for
                      st in st2] for st2 in sentence2]

        targets = torch.tensor(targets).to(device)
        optimizer.zero_grad()
        outputs = []
        with autocast():
            with torch.no_grad():
                for s1, s2, tgt in zip(sentence1, sentence2, targets):
                    embed1 = model(s1).unsqueeze(1)
                    embed2 = torch.stack([model(s) for s in s2])
                    sim,_ = torch.max(torch.cosine_similarity(embed1,embed2,dim=2),dim=1)
                    # sim = torch.stack(
                    #     [torch.max(torch.stack([torch.cosine_similarity(e1, e2, dim=0) for e2 in emb2])) for
                    #      e1, emb2 in
                    #      zip(embed1, embed2)])
                    outputs.append(sim)
                outputs = torch.stack(outputs)
        metrics+=cal_metrics(outputs.cpu().numpy(),targets.cpu().numpy(),pos_mask)
        batch_scores = np.array(outputs.cpu().numpy())
        batch_positions = np.argsort(-batch_scores, axis=1)
        scores.append(batch_scores)
        ranking_positions.append(batch_positions)
        nb_ranking_steps += 1
        ranking_pbar.update(len(targets))

    scores=np.concatenate(scores)
    ordered_scores=np.array([scores[i][list(ranking_indexes[i])] for i in range(len(scores))])
    ordered_positions=np.argsort(-ordered_scores, axis=1)
    false_predicted_triplets=[]
    false_positions=[]
    for i in range(len(ordered_positions)):
        if ordered_positions[i][0]!=0:
            false_predicted_triplets.append(ranking_triplets[i][ranking_labels[i]])
            false_positions.append(ordered_positions[i])
    false_positions=np.array(false_positions)
    np.savetxt(os.path.join(output_dir,"false_positions.txt"), false_positions,fmt="%d", delimiter="\t")

    with open(os.path.join(output_dir,"false_triplets.txt"),'w') as f:
        for triplet in false_predicted_triplets:
            f.write("\t".join(triplet)+"\n")
        f.close()
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    np.savetxt(os.path.join(output_dir,"indexes.txt"),ordered_positions,fmt="%d", delimiter="\t")
    np.savetxt(os.path.join(output_dir,"scores.txt"), ordered_scores,fmt="%.5f", delimiter="\t")
    metrics=metrics/nb_ranking_steps
    ranking_pbar.close()
    print(f"MR: {metrics[0]}, MRR: {metrics[1]}, Hit@1: {metrics[2]}, Hit@3: {metrics[3]}, Hit@10: {metrics[4]}")

if args.do_train:
    try:
        test()
        train()
    except KeyboardInterrupt:
        print("Receive keyboard interrupt, start testing:")
        model.load_state_dict(torch.load(model_load_file, map_location=device))
        test()
if args.do_test:
    # model=torch.nn.DataParallel(model,device_ids=[0,1])
    model.load_state_dict(torch.load(model_load_file, map_location=device))
    test()

