set -v
dataset="FB15k-237-subset"
suffix="_full"
finding_mode="head"
device="cuda:1"
seed=42
relation_prediction_lr=1e-5


# "relation prediction"
python relation_prediction.py --device $device --epochs 30 --batch_size 1 --dataset $dataset --learning_rate $relation_prediction_lr --neg_sample_num_train 5 --neg_sample_num_valid 5 --neg_sample_num_test 50 --mode $finding_mode --seed $seed --suffix $suffix --do_test
