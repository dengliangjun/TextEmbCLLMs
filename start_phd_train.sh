cd /opt/workspace/skyopen
DIR=$(cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)

netif=lo
export GLOO_SOCKET_IFNAME=${netif}
export NCCL_SOCKET_IFNAME=${netif}
export MODEL_NAME=phd2
export CUDA_DEVICE_ORDER=PCI_BUS_ID

#which cpu can be used?
#export CUDA_VISIBLE_DEVICES=7

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1024
export TOKENIZERS_PARALLELISM=1

BASE_MODEL="/opt/models/llama3_chkpt"
chkpt_path="/data/train_chkpts/${MODEL_NAME}"
listen_port=7033
CHECKPOINT_STEPS=150
read -p "Set 【CheckPoint】 Steps【default=150】:" input
if [ -z "$input" ]; then
    input=500
    echo "Input is empty,default value=$input"
fi
CHECKPOINT_STEPS=$input

#training step, namely, iterations
TRAIN_STEPS=150
Last_Global_Steps=$(head -n 1 ${chkpt_path}/latest)
echo "last traning global steps is $Last_Global_Steps."
read -p "Set Train Total Steps【default=$Last_Global_Steps+1000】：" input
if [ -z "$input" ]; then
    echo "Train Total Steps is neccessary params for training, Exiting..."
    input=$(($Last_Global_Steps + 1000))
fi
TRAIN_STEPS=$input
#重启学习率学习
lr_restart=false
#不要加载checkpoint
load_chkpt=true
#忽略系统prompt提示词
ignore_system_prompt=false

read -p "load 【checkpoint】 from latest train disk?【default=true】：" input
if [ -z "$input" ]; then
    echo "defalt load checkpoint from disk!!!"
    input=true
    lr_restart=false
elif [ $input == true ]; then 
    lr_restart=false
elif [ $input == false ]; then 
    lr_restart=true
fi
load_chkpt=$input

read -p "Set Model lr_restart. true or false >>> default=$lr_restart:" input
if [ -z "$input" ]; then
    echo "lr_restart use default $lr_restart"
else
    lr_restart=$input
fi

log_backen=print
read -p "Set 【logger】: wandb/print（default:wandb）:" input
if [ -z "$input" ]; then
    echo "log backend default to wandb！！！"
    input=wandb
else
    input=print
fi
log_backend=$input

read -p "--ignore system prompt to train（default:false）:" input
if [ -z "$input" ]; then
    echo ">>>>>>use system prompt to train"
    input=false
elif [ $input == true ];then
    echo ">>>>>>【ignore】 system prompt to train！！！！"
else
    echo "input error, default >>>>>> use system prompt to train"
    input=false
fi
ignore_system_prompt=$input

echo "Model Training：CHECKPOINT_STEPS=【$CHECKPOINT_STEPS】, TRAIN_STEPS=【$TRAIN_STEPS】"
echo "Model Training：load_check_point=【$load_chkpt】,    lr_restart=【$lr_restart】，log_backend=【$log_backend】"

eval_steps=$(($CHECKPOINT_STEPS / 5))
#不做评估,因为是通用测试
eval_steps=0

#最后的步数
latest_step=0
#for train log
export WANDB_NAME=skyopen_${TRAIN_STEPS}
export WANDB_NOTES="phd_chen for federal learning part 1"
export WANDB_PROJECT=phd_federal
project_name=phd_federal

#批次6的时45G显存
batch_size=6
micro_batch_size=1
word_size=8
pp_size=8
dp_size=2
#是否打印样本到日志,0=不打印,n=打印间隔
print_sample_every_n=256

train_warmup_steps=$(($batch_size * 1))
warmup_steps=$(($batch_size * 1))
#2e-5
learning_rate=5e-6
optimizer_name=8bit-adam

#是否显示评估文本
show_pred_text=true
#使用offload无法使用checkpoint
#--use_offload
#测试dataloader的数据集
test_dataloader=false
debug_model=false

#对话式训练，只有要给system
chat_train_mode=true
#把多条数据组装形成一条数据训练，多个system
full_seq_train=true

DATASETS="\
/opt/workspace/skyopen/dataset/skyqa/skyqa.py:phd2:0.5\
"

#test dataset
EVALUATION_DATA="/opt/workspace/skyopen/dataset/train_eval/train_eval.py:eval_code"

echo -e "train mode:\n \
【1】.    5 x pp node x 1【A】pp-0-4 5,9,9,9 bsz = 10\n \
【2】.    5 x pp node x 1【B】pp-5-9 5,9,9,9 bsz = 10 \n \
【3】.    9 x pp node x 1\n \
【4】.    test script running\n \
【5】.    4 x pp node x 2 batch_size=6 8,12,12,0\n \
【6】.    4 x pp node x 2 batch_size=8 9,11,11,1 latest train\n \
【7】.    5 x pp node x 2 batch_size=10 5,9,9,9 bsz = 10 \n \
input choice:" 
read input
if [ -z "$input" ]; then
    echo ">>>>>>>using default choice 【7】"
    input=7
fi
train_method=$input

if [ $train_method == 1 ]; then
    batch_size=10
    pp_size=5
    dp_size=1
    listen_port=7033
elif [ $train_method == 2 ]; then
    batch_size=10
    pp_size=5
    dp_size=1
    listen_port=6033
elif [ $train_method == 3 ]; then
    pp_size=9
    dp_size=1
elif [ $train_method == 4 ]; then
    echo "【4】test script running"
    test_dataloader=true
    debug_model=true
elif [ $train_method == 5 ]; then
    batch_size=6
    pp_size=4
    dp_size=2
elif [ $train_method == 6 ]; then
    batch_size=6
    pp_size=4
    dp_size=2
elif [ $train_method == 7 ]; then
    batch_size=10
    pp_size=5
    dp_size=2
else
    echo "use default config file params to train"
    pp_size=4
    dp_size=2
fi
echo "training batch_size=$batch_size"

micro_batch_size=1
gradient_accumulate_step=$(($batch_size * 1))
word_size=$(($pp_size * $dp_size))
#evaluation
eval_num_batch=0

#true：训练步骤少于full_loss_train部分，计算全损失h+b。剩下部分,只计算b部分损失。
#false: 直接计算h+b部分
use_prefix=true
#无效参数了
full_loss_train=0.99

echo -e "word_size=${word_size},\nnode=${dp_size},\npipline=${pp_size},\ncheckpoint,\ngradient_accumulate_step=${gradient_accumulate_step}"


#model-type dist_llama3=meta-llama3 hf_skyllama=hf_llama3

ARGS="--model-name ${BASE_MODEL} \
--tokenizer-name ${BASE_MODEL} \
--project-name ${project_name} \
--model-type hf_skyllama \
--optimizer ${optimizer_name} \
--seed 42 \
--total-steps ${TRAIN_STEPS} \
--print-sample-every-n ${print_sample_every_n} \
--load-pretrained-model true \
--load-checkpoint ${load_chkpt} \
--full-seq-train ${full_seq_train} \
--ignore-system ${ignore_system_prompt} \
--chat-train-mode ${chat_train_mode} \
--train-log-backend ${log_backend} \
--evaluation-steps ${eval_steps} \
--evaluation-data "${EVALUATION_DATA}" \
--evaluation-num-batch ${eval_num_batch} \
--task-name "${DATASETS}" \
--checkpoint-path $chkpt_path \
--warmup-steps ${warmup_steps} --train-warmup-steps ${train_warmup_steps} \
--checkpoint-steps ${CHECKPOINT_STEPS} \
--lr ${learning_rate} --seq-length 8192 --batch-size ${batch_size} --micro-batch-size ${micro_batch_size} \
--gradient-accumulate-step ${gradient_accumulate_step} \
--dist-url tcp://127.0.0.1:${listen_port} \
--num-layers 4 --embedding-dim 4096 \
--world-size ${word_size} --pipeline-group-size ${pp_size} --data-group-size ${dp_size} \
--job-id 0 --net-interface ${netif} \
--fp16 \
--debug-model ${debug_model} \
--use-prefix ${use_prefix} \
--lr-restart ${lr_restart} \
--show-pred-text ${show_pred_text} \
--full-loss-train ${full_loss_train} \
--latest-step ${latest_step} \
--test-dataloader ${test_dataloader} \
--trace-postfix skycto \
--dp-backend nccl \
--dp-mode allreduce \
--pp-mode gpipe --profiling no-profiling"

#重置训练停止条件
echo {\"stopNext\":false} > ${DIR}/stop.json

if [ $train_method == 1 ]; then
    (\
    python ${DIR}/dist_skyopen_train.py $(echo ${ARGS}) --cuda-id 0 --rank 0 --layer-idx 0,4 \
        & \
    python ${DIR}/dist_skyopen_train.py $(echo ${ARGS}) --cuda-id 1 --rank 1 --layer-idx 5,13 \
        & \
    python ${DIR}/dist_skyopen_train.py $(echo ${ARGS}) --cuda-id 2 --rank 2 --layer-idx 14,22 \
        & \
    python ${DIR}/dist_skyopen_train.py $(echo ${ARGS}) --cuda-id 3 --rank 3 --layer-idx 23,31 \
        & \
    python ${DIR}/dist_skyopen_train.py $(echo ${ARGS}) --cuda-id 4 --rank 4 --layer-idx 32,31 \
        & \
    )

elif [ $train_method == 2 ]; then
    ( \
    python ${DIR}/dist_skyopen_train.py $(echo ${ARGS}) --cuda-id 5 --rank 0 --layer-idx 0,4 \
        & \
    python ${DIR}/dist_skyopen_train.py $(echo ${ARGS}) --cuda-id 6 --rank 1 --layer-idx 5,13 \
        & \
    python ${DIR}/dist_skyopen_train.py $(echo ${ARGS}) --cuda-id 7 --rank 2 --layer-idx 14,22 \
        & \
    python ${DIR}/dist_skyopen_train.py $(echo ${ARGS}) --cuda-id 8 --rank 3 --layer-idx 23,31 \
        & \
    python ${DIR}/dist_skyopen_train.py $(echo ${ARGS}) --cuda-id 9 --rank 4 --layer-idx 32,31 \
        & \
    )
elif [ $train_method == 3 ]; then
    (trap 'kill 0' SIGINT; \
    python ${DIR}/dist_skyopen_train.py $(echo ${ARGS}) --cuda-id 0 --rank 0 --layer-idx 0,1 \
        & \
    python ${DIR}/dist_skyopen_train.py $(echo ${ARGS}) --cuda-id 1 --rank 1 --layer-idx 2,5 \
        & \
    python ${DIR}/dist_skyopen_train.py $(echo ${ARGS}) --cuda-id 2 --rank 2 --layer-idx 6,9 \
        & \
    python ${DIR}/dist_skyopen_train.py $(echo ${ARGS}) --cuda-id 3 --rank 3 --layer-idx 10,13 \
        & \
    python ${DIR}/dist_skyopen_train.py $(echo ${ARGS}) --cuda-id 4 --rank 4 --layer-idx 14,17 \
        & \
    python ${DIR}/dist_skyopen_train.py $(echo ${ARGS}) --cuda-id 5 --rank 5 --layer-idx 18,21 \
        & \
    python ${DIR}/dist_skyopen_train.py $(echo ${ARGS}) --cuda-id 6 --rank 6 --layer-idx 22,25 \
        & \
    python ${DIR}/dist_skyopen_train.py $(echo ${ARGS}) --cuda-id 7 --rank 7 --layer-idx 26,29 \
        & \
    python ${DIR}/dist_skyopen_train.py $(echo ${ARGS}) --cuda-id 8 --rank 8 --layer-idx 30,31 \
        & \
    wait )
elif [ $train_method == 4 ]; then
    echo "test script running"
    (trap 'kill 0' SIGINT; \
    python ${DIR}/dist_skyopen_train.py $(echo ${ARGS}) --cuda-id 0 --rank 0 --layer-idx 0,15 \
        & \
    python ${DIR}/dist_skyopen_train.py $(echo ${ARGS}) --cuda-id 1 --rank 1 --layer-idx 16,31 \
        & \
    python ${DIR}/dist_skyopen_train.py $(echo ${ARGS}) --cuda-id 2 --rank 2 --layer-idx 32,31 \
        & \
    wait )
elif [ $train_method == 5 ]; then
    (trap 'kill 0' SIGINT; \
    python ${DIR}/dist_skyopen_train.py $(echo ${ARGS}) --cuda-id 0 --rank 0 --layer-idx 0,7 \
        & \
    python ${DIR}/dist_skyopen_train.py $(echo ${ARGS}) --cuda-id 1 --rank 1 --layer-idx 8,19 \
        & \
    python ${DIR}/dist_skyopen_train.py $(echo ${ARGS}) --cuda-id 2 --rank 2 --layer-idx 20,31 \
        & \
    python ${DIR}/dist_skyopen_train.py $(echo ${ARGS}) --cuda-id 3 --rank 3 --layer-idx 32,31 \
        & \
    python ${DIR}/dist_skyopen_train.py $(echo ${ARGS}) --cuda-id 4 --rank 4 --layer-idx 0,7 \
        & \
    python ${DIR}/dist_skyopen_train.py $(echo ${ARGS}) --cuda-id 5 --rank 5 --layer-idx 8,19 \
        & \
    python ${DIR}/dist_skyopen_train.py $(echo ${ARGS}) --cuda-id 6 --rank 6 --layer-idx 20,31 \
        & \
    python ${DIR}/dist_skyopen_train.py $(echo ${ARGS}) --cuda-id 7 --rank 7 --layer-idx 32,31 \
        & \
    wait )
elif [ $train_method == 6 ]; then
    (trap 'kill 0' SIGINT; \
    python ${DIR}/dist_skyopen_train.py $(echo ${ARGS}) --cuda-id 0 --rank 0 --layer-idx 0,8 \
        & \
    python ${DIR}/dist_skyopen_train.py $(echo ${ARGS}) --cuda-id 1 --rank 1 --layer-idx 9,19 \
        & \
    python ${DIR}/dist_skyopen_train.py $(echo ${ARGS}) --cuda-id 2 --rank 2 --layer-idx 20,30 \
        & \
    python ${DIR}/dist_skyopen_train.py $(echo ${ARGS}) --cuda-id 3 --rank 3 --layer-idx 31,31 \
        & \
    python ${DIR}/dist_skyopen_train.py $(echo ${ARGS}) --cuda-id 4 --rank 4 --layer-idx 0,8 \
        & \
    python ${DIR}/dist_skyopen_train.py $(echo ${ARGS}) --cuda-id 5 --rank 5 --layer-idx 9,19 \
        & \
    python ${DIR}/dist_skyopen_train.py $(echo ${ARGS}) --cuda-id 6 --rank 6 --layer-idx 20,30 \
        & \
    python ${DIR}/dist_skyopen_train.py $(echo ${ARGS}) --cuda-id 7 --rank 7 --layer-idx 31,31 \
        & \
    wait )
elif [ $train_method == 7 ]; then
    (trap 'kill 0' SIGINT; \
    python ${DIR}/dist_skyopen_train.py $(echo ${ARGS}) --cuda-id 0 --rank 0 --layer-idx 0,4 \
        & \
    python ${DIR}/dist_skyopen_train.py $(echo ${ARGS}) --cuda-id 1 --rank 1 --layer-idx 5,13 \
        & \
    python ${DIR}/dist_skyopen_train.py $(echo ${ARGS}) --cuda-id 2 --rank 2 --layer-idx 14,22 \
        & \
    python ${DIR}/dist_skyopen_train.py $(echo ${ARGS}) --cuda-id 3 --rank 3 --layer-idx 23,31 \
        & \
    python ${DIR}/dist_skyopen_train.py $(echo ${ARGS}) --cuda-id 4 --rank 4 --layer-idx 32,31 \
        & \
    python ${DIR}/dist_skyopen_train.py $(echo ${ARGS}) --cuda-id 5 --rank 5 --layer-idx 0,4 \
        & \
    python ${DIR}/dist_skyopen_train.py $(echo ${ARGS}) --cuda-id 6 --rank 6 --layer-idx 5,13 \
        & \
    python ${DIR}/dist_skyopen_train.py $(echo ${ARGS}) --cuda-id 7 --rank 7 --layer-idx 14,22 \
        & \
    python ${DIR}/dist_skyopen_train.py $(echo ${ARGS}) --cuda-id 8 --rank 8 --layer-idx 23,31 \
        & \
    python ${DIR}/dist_skyopen_train.py $(echo ${ARGS}) --cuda-id 9 --rank 9 --layer-idx 32,31 \
        & \
    wait )
else
    echo "Exiting"
fi