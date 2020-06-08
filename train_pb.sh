ulimit -n 4096 
CUDA_VISIBLE_DEVICES=1 python main_pb.py --env-name "push_ball_origin" --model_dir '/push_ball' \
--save-interval 200 --algo ppo --use-gae --lr 2e-3 --clip-param 0.2 --value-loss-coef 0.5  \
--num-processes 700 --num-steps 80 --num-mini-batch 1 --log-interval 1 --entropy-coef 0.01 \
--adv_num 2 --good_num 1 --landmark_num 1 --ppo-epoch 25 --gae-lambda 0.98 --seed 1 --num-env-steps 300000000 \
--share_policy --use-linear-lr-decay --use_attention
