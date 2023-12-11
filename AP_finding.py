import networkx as nx
from collections import defaultdict
import numpy as np
import random
from tqdm import tqdm
import sys
from colorama import Fore
import os
import argparse
import copy
from torch_geometric.utils import remove_self_loops
from torch_sparse import SparseTensor
import torch
from utils import inverse_relation
import pickle



parser = argparse.ArgumentParser(description='Rule Finding for Relation Prediction')
parser.add_argument('--dataset', type=str, default='FB15k-237-subset',
                    help='name of the dataset')
parser.add_argument('--suffix', type=str, default='_full',
                    help='suffix of the train file name')
parser.add_argument('--finding_mode', type=str, default='head',
                    help='whether head, relation or tail is fixed')
parser.add_argument('--training_mode', type=str, default='train',
                    help='whether train, valid, test or interpret')
parser.add_argument('--data_dir', type=str, default=None,
                    help='directory to load data')
parser.add_argument('--output_dir', type=str, default=None,
                    help='directory to store output')
parser.add_argument('--ranking_dataset', type=str, default=None,
                    help='dataset to load ranking files')
parser.add_argument('--npaths_ranking', type=int, default=3,
                    help='number of paths for each triplet')
parser.add_argument('--neg_sample_num', type=int, default=50,
                    help='negative sample number')
args = parser.parse_args()
allow_flexible=False
max_path_num=3
min_group_rules_num=max_path_num*3
rule_dir=os.path.join('data/relation_prediction_path_data/', args.dataset)



print(args)
device='cpu'
G_all = nx.DiGraph()
relation_count = defaultdict(int)
train_triplets = []
ranking_triplets = []
if args.data_dir is None:
    data_dir = os.path.join("data/data/", args.dataset)
else:
    data_dir = args.data_dir

if args.output_dir is None:
    output_dir = os.path.join('data/relation_prediction_path_data/', args.dataset,
                                  f"ranking_{args.finding_mode}{args.suffix}/")
else:
    output_dir = args.output_dir
if args.ranking_dataset is None:
    ranking_dataset = os.path.join("data/relation_prediction_path_data/", args.dataset,
                                       f"ranking_{args.finding_mode}{args.suffix}/ranking_{args.training_mode}.txt")
else:
    ranking_dataset = args.ranking_dataset

if args.dataset.split("-")[-1] == "inductive" and args.training_mode != 'train':
    graph = "inductive_graph.txt"
else:
    graph = "train_full.txt"
G = defaultdict(lambda: nx.DiGraph())
entities2id = {}
relation_sparse_matrix = defaultdict(lambda: {'v': defaultdict(lambda: defaultdict(int)), 'x': [], 'y': []})
with open(os.path.join(data_dir, graph), encoding='utf-8') as f:
    for line in f:
        h, r, t = line.split()
        G_all.add_edge(h, t, relation=r)
        G_all.add_edge(t, h, relation=str("{" + r + "}^-1"))
        G[r].add_edge(h, t)
        G["{" + r + "}^-1"].add_edge(t, h)
        if h not in entities2id:
            entities2id[h] = len(entities2id)
        if t not in entities2id:
            entities2id[t] = len(entities2id)

        if relation_sparse_matrix[r]['v'][entities2id[h]][entities2id[t]] == 0:
            relation_sparse_matrix[r]['x'].append(entities2id[h])
            relation_sparse_matrix[r]['y'].append(entities2id[t])
        relation_sparse_matrix[r]['v'][entities2id[h]][entities2id[t]] += 1

        if relation_sparse_matrix["{" + r + "}^-1"]['v'][entities2id[t]][entities2id[h]] == 0:
            relation_sparse_matrix["{" + r + "}^-1"]['x'].append(entities2id[t])
            relation_sparse_matrix["{" + r + "}^-1"]['y'].append(entities2id[h])
        relation_sparse_matrix["{" + r + "}^-1"]['v'][entities2id[t]][entities2id[h]] += 1

        relation_count[r] += 1
        relation_count["{" + r + "}^-1"] += 1
    for rsm in relation_sparse_matrix.values():
        rsm['values'] = []
        for x, y in zip(rsm['x'], rsm['y']):
            rsm['values'].append(rsm['v'][x][y])
id2entities = {entities2id[k]: k for k in entities2id}
relation_sparse_matrix = {r: torch.sparse_coo_tensor(
    torch.tensor([relation_sparse_matrix[r]['x'], relation_sparse_matrix[r]['y']]),
    torch.tensor(relation_sparse_matrix[r]['values'], dtype=torch.float),
    (len(entities2id), len(entities2id))
).to(device) for r in relation_sparse_matrix}

with open(ranking_dataset,encoding='utf-8') as f:
    for c,line in enumerate(f):
        if c % args.neg_sample_num==0:
            ranking_triplets.append([])
        h,r,t=line.split()
        if args.finding_mode=='head':
            ranking_triplets[-1].append([h,r,t])
        elif args.finding_mode=='tail':
            ranking_triplets[-1].append([t, inverse_relation(r), h])

def remove_pos_triplet(pos,rsm):
    if pos[1] in rsm:
        sp=rsm[pos[1]].coalesce()
        is_pos=torch.logical_and(sp.indices()[0]==entities2id[pos[0]],sp.indices()[1]==entities2id[pos[2]])
        idcs=sp.indices()[:,is_pos==False]
        vals=sp.values()[is_pos==False]
        rsm[pos[1]]=torch.sparse_coo_tensor(idcs, vals, (len(entities2id), len(entities2id))).coalesce()

        sp2=rsm[inverse_relation(pos[1])].coalesce()
        is_pos_inverse=torch.logical_and(sp2.indices()[0]==entities2id[pos[2]],sp2.indices()[1]==entities2id[pos[0]])
        idcs2 = sp2.indices()[:, is_pos_inverse == False]
        vals2 = sp2.values()[is_pos_inverse == False]
        rsm[inverse_relation(pos[1])]=torch.sparse_coo_tensor(idcs2, vals2, (len(entities2id), len(entities2id))).coalesce()
    return rsm
def get_head_entities(Ap,id2entities,tail):
    Ap=Ap.coalesce()
    idcs=Ap.indices()[:,Ap.indices()[1]==tail]
    entities=set(id2entities[idcs[0][i].item()] for i in range(idcs.shape[1]))
    return entities

strict_logical_rules = pickle.load(open(os.path.join(rule_dir,f"strict_logical_rules0.pkl"),'rb'))
ranking_rules=[]
ranking_head_entities=[]
def path_finding(path,pos_removed_rsm,triplets,ranking_rules,ranking_head_entities,mode):
    Ap = pos_removed_rsm[path[0]]
    for pi in range(1, len(path)):
        Ap = torch.sparse.mm(Ap, pos_removed_rsm[path[pi]])
    tcandidates = torch.sparse.sum(Ap, dim=0)
    entity_candidates = set(
        id2entities[tcandidates.indices()[0][i].item()] for i in range(tcandidates.indices().shape[1]))
    path_head_entities = {e: get_head_entities(Ap, id2entities, entities2id[e]) for e in entity_candidates}
    for ti, triplet in enumerate(triplets):
        if triplet[2] in entity_candidates and len(ranking_rules[-1][ti])<max_path_num:
            ranking_rules[-1][ti].append([mode]+path)
            ranking_head_entities[-1][ti].append(path_head_entities[triplet[2]])

pbar=tqdm(total=len(ranking_triplets), desc='Rule Finding',
                 position=0, leave=True,
                 file=sys.stdout, bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.BLUE, Fore.RESET))
for triplets in ranking_triplets:
    ranking_rules.append([[] for i in range(args.neg_sample_num)])
    ranking_head_entities.append([[] for i in range(args.neg_sample_num)])
    pos=triplets[0]
    neg=triplets[1:]
    pos_removed_rsm=copy.deepcopy(relation_sparse_matrix)
    if args.training_mode=='train':
        pos_removed_rsm=remove_pos_triplet(pos,pos_removed_rsm)

    for path,_ in strict_logical_rules[pos[1]]:
        if sum(len(rules) for rules in ranking_rules[-1])>=min_group_rules_num:
            break
        path_finding(path,pos_removed_rsm,triplets,ranking_rules,ranking_head_entities,'strict')
        pbar.update(1)
pbar.close()
with open(os.path.join(output_dir,f"relation_rules_{args.training_mode}.txt"), "w", encoding='utf-8') as f1,\
        open(os.path.join(output_dir,f"rules_heads_{args.training_mode}.txt"), "w", encoding='utf-8') as f2:
    for i in range(len(ranking_rules)):
        for path_group,head_entities in zip(ranking_rules[i],ranking_head_entities[i]):
            f1.write(str(len(path_group)) + "\n")
            f2.write(str(len(head_entities)) + "\n")
            for path,heads in zip(path_group,head_entities):
                for r in path:
                    f1.write(r+ "\t")
                for h in heads:
                    f2.write(h+ "\t")
                f1.write("\n")
                f2.write("\n")
    f1.close()
    f2.close()
