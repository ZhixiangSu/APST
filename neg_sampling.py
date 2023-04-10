import networkx as nx
from collections import defaultdict
import numpy as np
import random
from tqdm import tqdm
import sys
from colorama import Fore
import os
import argparse

parser = argparse.ArgumentParser(description='Negative Sampling')
parser.add_argument('--dataset', type=str, default='FB15k-237-subset',
                    help='name of the dataset')
parser.add_argument('--suffix', type=str, default='_full',
                    help='suffix of the train file name')
parser.add_argument('--output_dir', type=str, default=None,
                    help='directory to store output')
parser.add_argument('--data_dir', type=str, default=None,
                    help='directory to load data')
parser.add_argument('--finding_mode', type=str, default='head',
                    help='whether head, relation or tail is fixed')
parser.add_argument('--training_mode', type=str, default='train',
                    help='whether train, valid or test')
parser.add_argument('--neg_num', type=int, default=5,
                    help='number of negative samples')
parser.add_argument('--triplet_sample_num', type=int, default=-1,
                    help='number of samples from all positive triplets, set -1 for all')
parser.add_argument('--seed', type=int, default=42,
                    help='random seed')
args = parser.parse_args()
print(args)
random.seed(args.seed)
G=nx.DiGraph()
relation_count=defaultdict(int)
train_triplets=[]
if args.training_mode=='train':
    file_name=f"{args.training_mode}{args.suffix}.txt"
else:
    file_name=f"{args.training_mode}.txt"
if args.data_dir is None:
    data_dir=os.path.join("data/data/", args.dataset)
else:
    data_dir=args.data_dir
if args.output_dir is None:
    output_dir=f'data/relation_prediction_path_data/{args.dataset}/ranking_{args.finding_mode}{args.suffix}/'
else:
    output_dir=args.output_dir




entities=[]
relation=[]
with open(os.path.join(data_dir,file_name),encoding='utf-8') as f:
    for line in f:
        h,r,t=line.split()
        train_triplets.append([h,r,t])
        entities.append(h)
        entities.append(t)
        relation.append(r)
        G.add_edge(h,t,relation=r)
        G.add_edge(t,h,relation=str("{"+r+"}^-1"))
        # relation_count[r]+=1
        # relation_count["{"+r+"}^-1"]+=1
if args.triplet_sample_num!=-1:
    train_triplets=random.sample(train_triplets,args.triplet_sample_num)
pbar=tqdm(total=len(train_triplets), desc=args.training_mode,
                 position=0, leave=True,
                 file=sys.stdout, bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.BLUE, Fore.RESET))
neg_triplets=[]
if args.finding_mode=='head':
    for t in train_triplets:
        neg_triplets.append(t)
        neg_tails=set(entities) - set(G[t[0]])
        neg_tails=random.sample(neg_tails,args.neg_num-1)
        for n in neg_tails:
            neg_triplets.append([t[0],t[1],n])
        pbar.update(1)

if args.finding_mode=='tail':
    for t in train_triplets:
        neg_triplets.append(t)
        neg_tails = set(entities) - set(G[t[2]])
        neg_tails = random.sample(neg_tails, args.neg_num - 1)
        for n in neg_tails:
            neg_triplets.append([n,t[1],t[2]])
        pbar.update(1)
if args.finding_mode=='relation':
    for t in train_triplets:
        neg_triplets.append(t)
        neg_tails=set(relation) - set(G[t[0]][t[2]]['relation'])
        neg_tails=random.sample(neg_tails,args.neg_num-1)
        for n in neg_tails:
            neg_triplets.append([t[0],r,t[2]])
        pbar.update(1)
pbar.close()
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
# with open(os.path.join(output_dir,f'entities.txt'),'w',encoding='utf-8') as f:
#     for e in entities:
#         f.write(e+"\n")
with open(os.path.join(output_dir,f'ranking_{args.training_mode}.txt'),'w',encoding='utf-8') as f:
    for line in neg_triplets:
        f.write("\t".join(line)+"\n")
