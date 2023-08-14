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
from functools import partial



parser = argparse.ArgumentParser(description='Path Finding for Relation Prediction')
parser.add_argument('--dataset', type=str, default='FB15k-237-subset',
                    help='name of the dataset')
parser.add_argument('--data_dir', type=str, default=None,
                    help='directory to load data')
parser.add_argument('--output_dir', type=str, default=None,
                    help='directory to store output')
parser.add_argument('--npaths_ranking', type=int, default=3,
                    help='number of paths for each triplet')
parser.add_argument('--recall_threshold', type=float, default=0.5,
                    help='recall threshold')
parser.add_argument('--accuracy_threshold', type=float, default=0.5,
                    help='accuracy threshold')
parser.add_argument('--min_search_depth', type=int, default=2,
                    help='min search depth')
parser.add_argument('--max_search_depth', type=int, default=2,
                    help='max search depth')

args = parser.parse_args()
allow_flexible=False

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
    output_dir = os.path.join('data/relation_prediction_path_data/', args.dataset)
else:
    output_dir = args.output_dir

if args.dataset.split("-")[-1] == "inductive":
    graph = "inductive_graph.txt"
else:
    graph = "train_full.txt"
G = defaultdict(lambda: nx.DiGraph())
entities2id = {}
relation_sparse_matrix = defaultdict(lambda: {'v': defaultdict(lambda: defaultdict(int)), 'x': [], 'y': []})
with open(os.path.join(data_dir, graph), encoding='utf-8') as f:
    for line in f:
        h, r, t = line.split()
        train_triplets.append([h, r, t])
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

A_all = [torch.sparse_coo_tensor(torch.empty([1, 0]), torch.empty([0, len(entities2id)]),
                                 (len(entities2id), len(entities2id))).to(device)]
for r in relation_sparse_matrix:
    A_all[0] = A_all[0] + relation_sparse_matrix[r]
for i in range(1,args.max_search_depth):
    A_all.append(torch.sparse.mm(A_all[-1],A_all[0])+A_all[0])


def remove_self_loop(sp):
    sp=sp.coalesce()
    is_loop = sp.indices()[1] == sp.indices()[0]
    idcs = sp.indices()[:, is_loop == False]
    vals = sp.values()[is_loop == False]
    return torch.sparse_coo_tensor(idcs, vals, (len(entities2id), len(entities2id))).coalesce()


def cal_metrics(Ap, rsm):
    Ap=Ap.coalesce()
    tmp1 = torch.sparse.sum(rsm, dim=0)
    tcandidates = torch.sparse_coo_tensor(torch.cat([torch.zeros_like(tmp1.indices()), tmp1.indices()]),
                                          torch.ones_like(tmp1.values()), (1, len(entities2id)))
    tmp2=torch.sparse.sum(Ap,dim=0)
    pcandidates = torch.sparse_coo_tensor(torch.cat([tmp2.indices(), torch.zeros_like(tmp2.indices())]), torch.ones_like(tmp2.values()),(len(entities2id),1))
    support = torch.sparse.mm(tcandidates,pcandidates)
    recall = support / torch.sparse.sum(tcandidates)
    tmp4 = torch.sparse_coo_tensor(Ap.indices(), torch.ones_like(Ap.values()), Ap.shape)
    accuracy = support / torch.sparse.sum(tmp4)
    return recall.item(), accuracy.item(),pcandidates
def cal_frequency(Ap,rsm,Ap_all):
    tmp1 = torch.sparse.sum(rsm, dim=0)
    hcandidates = torch.sparse_coo_tensor(torch.cat([tmp1.indices(),torch.zeros_like(tmp1.indices())]),
                                          torch.ones_like(tmp1.values()), ( len(entities2id),1))
    tmp2=torch.sparse.mm(Ap,hcandidates)
    tmp3=torch.sparse.mm(Ap_all,hcandidates)
    frequency=torch.sparse.sum(tmp2)/torch.sparse.sum(tmp3)
    return frequency.item()




def rule_generating(relation_sparse_matrix, A, path, strict_logical_rules,frequency, thresholds):
    for rp in relation_sparse_matrix:
        Ap = torch.sparse.mm(A, relation_sparse_matrix[rp])
        Ap = remove_self_loop(Ap)
        if Ap._nnz() != 0 and cal_frequency(Ap,relation_sparse_matrix[rp],A_all[len(path)-1])> frequency:
            if args.min_search_depth <= len(path + [rp]):
                for r in relation_sparse_matrix:
                    if r==rp:
                        continue
                    recall, accuracy,pcandidates = cal_metrics(Ap, relation_sparse_matrix[r])
                    for i,threshold in enumerate(thresholds):
                        if accuracy>threshold[0] and recall>threshold[1]:
                            strict_logical_rules[i][r].append([path + [rp], [accuracy, recall]])
            if args.max_search_depth > len(path + [rp]):
                rule_generating(relation_sparse_matrix, Ap, path + [rp],strict_logical_rules,frequency,thresholds)
# def rule_matching(rsm,strict_logical_rules,thresholds):
#     matched_rules=defaultdict(partial(defaultdict,list))
#     pbar=tqdm(total=len(rsm.keys()), desc='Rule Matching',
#                      position=0, leave=True,
#                      file=sys.stdout, bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.BLUE, Fore.RESET))
#     for r in rsm:
#         for path1,pcandidates1 in strict_logical_rules[r]:
#             for path2,pcandidates2 in strict_logical_rules[inverse_relation(r)]:
#                 tmp1=torch.sparse.mm(pcandidates2.T,rsm[r])
#                 tmp2=torch.sparse.mm(rsm[r],pcandidates1)
#                 match_num=torch.sparse.mm(tmp1,pcandidates1)
#                 match_rate1=match_num/torch.sparse.sum(tmp1)
#                 match_rate2 = match_num / torch.sparse.sum(tmp2)
#                 if match_rate1.item()>thresholds:
#                     matched_rules[r][tuple(path2)].append(path1)
#                 if match_rate2.item()>thresholds:
#                     matched_rules[inverse_relation(r)][tuple(path1)].append(path2)
#         pbar.update(1)
#     pbar.close()
#     return matched_rules

# thresholds=[[args.recall_threshold,args.accuracy_threshold]]
thresholds=[[args.accuracy_threshold,args.recall_threshold]]

strict_logical_rules = [defaultdict(list) for i in range(len(thresholds))]
# rules_accuracy=defaultdict(list)
# rules_recall= defaultdict(list)
# rules_frequency=defaultdict(list)
pbar=tqdm(total=len(relation_sparse_matrix.keys()), desc='Rule Generating',
                 position=0, leave=True,
                 file=sys.stdout, bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.BLUE, Fore.RESET))
with torch.no_grad():
    for r1 in relation_sparse_matrix:
        if args.min_search_depth <= len([r1]):
            for r in relation_sparse_matrix:
                if r == r1:
                    continue
                recall, accuracy, pcandidates = cal_metrics(relation_sparse_matrix[r1], relation_sparse_matrix[r])
                for i, threshold in enumerate(thresholds):
                    if accuracy > threshold[0] and recall > threshold[1]:
                        strict_logical_rules[i][r].append([[r1], [accuracy, recall]])
        if args.max_search_depth > len([r1]):
            rule_generating(relation_sparse_matrix, relation_sparse_matrix[r1], [r1],
                            strict_logical_rules,0,thresholds)
        pbar.update(1)
pbar.close()
# for r in strict_logical_rules:
#     strict_logical_rules[r]=sorted(strict_logical_rules[r],key=lambda x:(-x[1][0],-x[1][1]))
# strict_logical_rules=pickle.load(open(os.path.join(output_dir,f"strict_logical_rules.pkl"),'rb'))
# matched_rules=rule_matching(relation_sparse_matrix,strict_logical_rules,0.9)

for i in range(len(thresholds)):
    with open(os.path.join(output_dir,f"strict_logical_rules{i}.pkl"), "wb") as f:
        pickle.dump(strict_logical_rules[i],f)

# with open(os.path.join(output_dir,f"flexible_logical_rules.pkl"), "wb") as f:
#     pickle.dump(flexible_logical_rules,f)
# with open(os.path.join(output_dir,f"matched_rules.pkl"), "wb") as f:
#     pickle.dump(matched_rules,f)



# with open(os.path.join(output_dir,f"rules_recall.pkl"), "wb") as f:
#     pickle.dump(rules_recall,f)
# with open(os.path.join(output_dir,f"rules_accuracy.pkl"), "wb") as f:
#     pickle.dump(rules_accuracy,f)
