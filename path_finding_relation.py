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

# parser = argparse.ArgumentParser(description='Path Finding for Relation Prediction')
# parser.add_argument('--dataset', type=str, default='FB15k-237-subset',
#                     help='name of the dataset')
# parser.add_argument('--suffix', type=str, default='_full',
#                     help='suffix of the train file name')
# parser.add_argument('--finding_mode', type=str, default='tail',
#                     help='whether head, relation or tail is fixed')
# parser.add_argument('--training_mode', type=str, default='interpret',
#                     help='whether train, valid, test or interpret')
# parser.add_argument('--data_dir', type=str, default=None,
#                     help='directory to load data')
# parser.add_argument('--output_dir', type=str, default=None,
#                     help='directory to store output')
# parser.add_argument('--train_dataset', type=str, default=None,
#                     help='dataset to load train graph')
# parser.add_argument('--ranking_dataset', type=str, default=None,
#                     help='dataset to load ranking files')
# parser.add_argument('--npaths_ranking', type=int, default=100,
#                     help='number of paths for each triplet')
# parser.add_argument('--support_threshold', type=float, default=0,
#                     help='threshold of path filtering function')
# parser.add_argument('--support_type', type=int, default=2,
#                     help='choice of path filtering function. 0: none, 1: relation path coverage 2: relation path confidence  ')
# parser.add_argument('--search_depth', type=int, default=8,
#                     help='search depth')


parser = argparse.ArgumentParser(description='Path Finding for Relation Prediction')
parser.add_argument('--dataset', type=str, default='FB15k-237-subset',
                    help='name of the dataset')
parser.add_argument('--suffix', type=str, default='_full',
                    help='suffix of the train file name')
parser.add_argument('--finding_mode', type=str, default='head',
                    help='whether head, relation or tail is fixed')
parser.add_argument('--training_mode', type=str, default='test',
                    help='whether train, valid, test or interpret')
parser.add_argument('--data_dir', type=str, default=None,
                    help='directory to load data')
parser.add_argument('--output_dir', type=str, default=None,
                    help='directory to store output')
parser.add_argument('--train_dataset', type=str, default=None,
                    help='dataset to load train graph')
parser.add_argument('--ranking_dataset', type=str, default=None,
                    help='dataset to load ranking files')
parser.add_argument('--npaths_ranking', type=int, default=3,
                    help='number of paths for each triplet')
parser.add_argument('--support_threshold', type=float, default=1e-5,
                    help='partial completeness assumption confidence')
parser.add_argument('--support_type', type=int, default=1,
                    help='choice of path filtering function. 0: none, 1: relation path coverage 2: relation path confidence  ')
parser.add_argument('--search_depth', type=int, default=5,
                    help='search depth')

args = parser.parse_args()
print(args)

G=nx.DiGraph()
relation_count=defaultdict(int)
train_triplets=[]
ranking_triplets=[]
if args.data_dir is None:
    data_dir=os.path.join("data/data/", args.dataset)
else:
    data_dir=args.data_dir

if args.output_dir is None:
    if args.training_mode == 'interpret':
        output_dir = os.path.join('data/relation_prediction_path_data/', args.dataset,
                                  f"interpret/")
    else:
        output_dir=os.path.join('data/relation_prediction_path_data/', args.dataset, f"ranking_{args.finding_mode}{args.suffix}/")
else:
    output_dir=args.output_dir
if args.ranking_dataset is None:
    if args.training_mode == 'interpret':
        ranking_dataset = os.path.join("data/relation_prediction_path_data/", args.dataset,
                                       f"interpret/interpret.txt")
    else:
        ranking_dataset = os.path.join("data/relation_prediction_path_data/", args.dataset,
                                   f"ranking_{args.finding_mode}{args.suffix}/ranking_{args.training_mode}.txt")
else:
    ranking_dataset=args.ranking_dataset

if args.dataset.split("-")[-1] == "inductive" and args.training_mode!='train':
    graph="inductive_graph.txt"
else:
    graph="train_full.txt"
def supportType(num,mode):
    if num==0:
        return lambda x:np.inf
    elif num==1:
        if mode=='head':
            return pathCoverageHead
        elif mode=='tail':
            return pathCoverageTail
    elif num==2:
        if mode=='head':
            return pathConfidenceHead
        elif mode=='tail':
            return pathConfidenceTail
with open(os.path.join(data_dir,graph),encoding='utf-8') as f:
    for line in f:
        h,r,t=line.split()
        train_triplets.append([h,r,t])
        G.add_edge(h,t,relation=r)
        G.add_edge(t,h,relation=str("{"+r+"}^-1"))
        relation_count[r]+=1
        relation_count["{"+r+"}^-1"]+=1

with open(ranking_dataset,encoding='utf-8') as f:
    for c,line in enumerate(f):
        h,r,t=line.split()
        ranking_triplets.append([h,r,t])
def findPathsHead(G,triplets,paths,num_path):
    global cur_paths
    pbar=tqdm(total=len(triplets), desc=args.training_mode,
                 position=0, leave=True,
                 file=sys.stdout, bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.BLUE, Fore.RESET))
    for line in triplets:
        visited = defaultdict(int)
        previous=defaultdict(int)
        h,r,t=line
        cur_paths = 0
        queue = []
        queue.append([h,0])
        visited[h] = 1
        paths.append([])
        while len(queue)!=0:
            u,dpth=queue[0]
            queue.pop(0)
            if u not in G.nodes or dpth>=args.search_depth:
                continue
            for v in sorted(G[u],key=lambda x:relation_count[G[u][x]['relation']]):
                if v == t:
                    if u == h and args.training_mode=='train':
                        continue
                    else:
                        p=generatePathHead(previous,u,h,t)
                        support = supportType(args.support_type,'head')(p)
                        if support >= args.support_threshold:
                            paths[-1].append(p)
                elif visited[v]==0:
                    queue.append([v,dpth+1])
                    visited[v]=1
                    previous[v]=u
                if len(paths[-1]) >= num_path:
                    break
            if len(paths[-1]) >= num_path:
                break
        pbar.update(1)
    pbar.close()


def generatePathHead(prev,u,h,t):
    p=[t,u]
    if u==h:
        p.reverse()
        return p
    while prev[u]!=h:
        u = prev[u]
        p.append(u)
    p.append(h)
    p.reverse()
    return p

def pathCoverageHead(path):
    relations=[G[path[i]][path[i+1]]['relation'] for i in range(len(path)-1)]
    path_num=defaultdict(dict)
    path_num[path[0]][0]=1
    count=0
    support=0
    for i in range(len(relations)):
        new_dict=copy.deepcopy(path_num)
        for entity in path_num.keys():
            for v in G[entity]:
                for dpth in path_num[entity].keys():
                    if dpth+1 not in new_dict[v]:
                        new_dict[v][dpth+1]=path_num[entity][dpth]
                    else:
                        new_dict[v][dpth + 1] += path_num[entity][dpth]
        path_num=copy.deepcopy(new_dict)
    for entity in path_num.keys():
        if entity==path[-1]:
            support+=path_num[entity][len(relations)]
        if len(relations) in path_num[entity]:
            count+=path_num[entity][len(relations)]
    return support/count
def pathConfidenceHead(path):
    relations=[G[path[i]][path[i+1]]['relation'] for i in range(len(path)-1)]
    queue=[]
    queue.append([path[0],0])
    count=0
    support=0
    while len(queue)!=0:
        entity,dpth=queue[0]
        queue.pop(0)
        if dpth==len(relations):
            if entity==path[-1]:
                support+=1
            count+=1
            continue
        for v in G[entity]:
            if G[entity][v]['relation']==relations[dpth]:
                queue.append([v,dpth+1])
    return support/count



def findPathsTail(G,triplets,paths,num_path):
    global cur_paths
    pbar=tqdm(total=len(triplets), desc=args.training_mode,
                 position=0, leave=True,
                 file=sys.stdout, bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.BLUE, Fore.RESET))
    for line in triplets:
        visited = defaultdict(int)
        next=defaultdict(int)
        h,r,t=line
        cur_paths = 0
        queue = []
        queue.append([t,0])
        visited[t] = 1
        paths.append([])
        while len(queue)!=0:
            u,dpth=queue[0]
            queue.pop(0)
            if u not in G.nodes or dpth>=args.search_depth:
                continue
            for v in sorted(G[u],key=lambda x:relation_count[G[x][u]['relation']]):
                if v == h:
                    if u == t and args.training_mode=='train':
                        continue
                    else:
                        p=generatePathTail(next,u,h,t)
                        support = supportType(args.support_type,'tail')(p)
                        if support >= args.support_threshold:
                            paths[-1].append(p)
                elif visited[v]==0:
                    queue.append([v,dpth+1])
                    visited[v]=1
                    next[v]=u
                if len(paths[-1]) >= num_path:
                    break
            if len(paths[-1]) >= num_path:
                break
        pbar.update(1)
    pbar.close()


def generatePathTail(nxt,u,h,t):
    p=[h,u]
    if u==t:
        return p
    while nxt[u]!=t:
        u = nxt[u]
        p.append(u)
    p.append(t)
    return p
def pathCoverageTail(path):
    relations = [G[path[i]][path[i + 1]]['relation'] for i in range(len(path) - 1)]
    path_num = defaultdict(dict)
    path_num[path[-1]][0] = 1
    count = 0
    support = 0
    for i in range(len(relations)):
        new_dict = copy.deepcopy(path_num)
        for entity in path_num.keys():
            for v in G[entity]:
                for dpth in path_num[entity].keys():
                    if dpth + 1 not in new_dict[v]:
                        new_dict[v][dpth + 1] = path_num[entity][dpth]
                    else:
                        new_dict[v][dpth + 1] += path_num[entity][dpth]
        path_num = copy.deepcopy(new_dict)
    for entity in path_num.keys():
        if entity == path[0]:
            support += path_num[entity][len(relations)]
        if len(relations) in path_num[entity]:
            count += path_num[entity][len(relations)]
    return support / count
def pathConfidenceTail(path):
    relations=[G[path[i]][path[i+1]]['relation'] for i in range(len(path)-1)]
    queue=[]
    queue.append([path[-1],0])
    count=0
    support=0
    while len(queue)!=0:
        entity,dpth=queue[0]
        queue.pop(0)
        if dpth==len(relations):
            if entity==path[0]:
                support+=1
            count+=1
            continue
        for v in G[entity]:
            if G[v][entity]['relation']==relations[-1-dpth]:
                queue.append([v,dpth+1])
    return support/count
ranking_paths=[]
if args.finding_mode=='head':
    findPathsHead(G,ranking_triplets,ranking_paths,args.npaths_ranking)
else:
    findPathsTail(G, ranking_triplets, ranking_paths, args.npaths_ranking)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
with open(os.path.join(output_dir,f"entity_paths_{args.training_mode}.txt"), "w", encoding='utf-8') as f:
    for path_group in ranking_paths:
        f.write(str(len(path_group))+"\n")
        for path in path_group:
            for entity in path:
                f.write(entity+"\t")
            f.write("\n")


with open(os.path.join(output_dir,f"relation_paths_{args.training_mode}.txt"), "w", encoding='utf-8') as f:
    for path_group in ranking_paths:
        f.write(str(len(path_group)) + "\n")
        for path in path_group:
            for i in range(len(path)-1):
                f.write(G[path[i]][path[i+1]]['relation'] + "\t")
            f.write("\n")




