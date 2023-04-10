set -v
dataset="FB15k-237-subset"
suffix="_full"
finding_mode="head"
device="cuda:0"
seed=42
relation_prediction_lr=1e-5
search_depth=5
support_type=2
support_threshold=5e-3


# "negative sampling"
python neg_sampling.py --dataset $dataset --suffix $suffix --finding_mode $finding_mode --training_mode train --neg_num 5 --seed $seed
python neg_sampling.py --dataset $dataset --suffix $suffix --finding_mode $finding_mode --training_mode valid --neg_num 5 --seed $seed
# For fair comparison, we use the test ranking file provided by BERTRL(https://github.com/zhw12/BERTRL/tree/master/data)
cp data/data/${dataset}/ranking_${finding_mode}.txt data/relation_prediction_path_data/${dataset}/ranking_${finding_mode}${suffix}/ranking_test.txt

# "path finding"
python path_finding_relation.py --dataset $dataset --suffix $suffix --finding_mode $finding_mode --training_mode train --npaths_ranking 3 --support_threshold $support_threshold --search_depth $search_depth --support_type $support_type
python path_finding_relation.py --dataset $dataset --suffix $suffix --finding_mode $finding_mode --training_mode valid --npaths_ranking 3 --support_threshold $support_threshold --search_depth $search_depth --support_type $support_type
python path_finding_relation.py --dataset $dataset --suffix $suffix --finding_mode $finding_mode --training_mode test --npaths_ranking 3 --support_threshold $support_threshold --search_depth $search_depth --support_type $support_type

# "relation prediction"
python relation_prediction.py --device $device --epochs 30 --batch_size 1 --dataset $dataset --learning_rate $relation_prediction_lr --neg_sample_num_train 5 --neg_sample_num_valid 5 --neg_sample_num_test 50 --mode $finding_mode --seed $seed --suffix $suffix --do_train --do_test