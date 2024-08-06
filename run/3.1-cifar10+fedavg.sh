#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/

part_strategy="iid"; part_strategy_list=("iid" "labeldir0.3" "labelcnt0.3")
model="resnet10"; dataset="cifar10"; 
# lr=0.1; momentum=0; l2=0; rounds=200; epochs=10; batch_size=64; save_round=0; num_per_round=10; num_client=30; val_ratio=0.0;
lr=0.1; momentum=0; l2=0; rounds=200; epochs=10; batch_size=64; save_round=0; num_per_round=10; num_client=100; val_ratio=0.0;
com_type="fedavg";

gpu=1
gpu_clients=(1 1 1 1 1 1 1 1 1 1) 
ip="0.0.0.0:11310"
lrs=(1.0 0.1 0.01)
for c in 1; do # fors
    lr=${lrs[${c}]}
    for b in 0 1 2; do  # part_strategy
        part_strategy=${part_strategy_list[${b}]}
        dir="../log/${dataset}+${num_client}/${dataset}+${part_strategy}/${model}+${com_type}+${lr}/"
        name="${dataset}+${part_strategy}+${model}+${com_type}"
        python ../train/server.py \
        --com_type ${com_type} \
        --model ${model} --dataset ${dataset} --lr ${lr} --momentum ${momentum} --l2 ${l2} \
        --rounds ${rounds} --epochs ${epochs} --batch_size ${batch_size} --save_round ${save_round} \
        --num_per_round ${num_per_round} --num_client ${num_client} \
        --gpu ${gpu} --ip ${ip} --log_dir ${dir} --log_name ${name} &

        for a in $(seq 0 $(($num_per_round-1))); do # client
            python ../train/client.py \
            --com_type ${com_type} \
            --model ${model} --dataset ${dataset} --part_strategy ${part_strategy} --num_client ${num_client} --id ${a} --val_ratio ${val_ratio} \
            --gpu ${gpu_clients[${a}]} --ip ${ip} --log_dir ${dir} --log_name ${name} &
        done
        trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM 
        wait
    done 
done
