#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"

cd "$PROJECT_ROOT"

# 用法说明
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --nproc_per_node=<value>           Number of processes to run per node."
    echo "  --nnodes=<value>                   Total number of processes across all nodes."
    echo "  --out_path=<value>                 Directory to store model results."
    echo "  --s3_remote_dir_path=<value>       Directory of s3 bucket to store model results."
    echo "  --data_path=<value>                Directory of sc data."
    echo "  --emb_path=<value>                 Directory of emb data."
    echo "  --seq_len=<value>                  seq_len."
    echo "  --batch_size=<value>               batch_size."
    echo "  --epoch=<value>                    epoch."
    echo "  --compile=<value>                  compile."
    echo "  --num_of_used_data=<value>         num_of_used_data."
    echo "  --save_data_interval=<value>       save_data_interval."
    echo "  -h, --help                         Display this help message and exit."
    echo ""
    echo "Example:"
    echo "  $0 --init_method=tcp://127.0.0.1:23456 --nproc_per_node=4 --nnodes=8"
    echo ""
    exit 1
}

# 默认值
s3_remote_dir_path="s3://bucket-3028/public/SpeciesLLM-Training/"
data_path=./all_shuffled_data
emb_path=./macrogene_embeddings_1stage
out_path="hs_{hidden_size}_nh_{num_hidden_layers}_na_{num_attention_heads}_hdp_{hidden_dropout_prob}_lr_{learning_rate}_mlr_{min_lr}_wd_{weight_decay}_wr_{warmup_ratio}"
seq_len=862
batch_size=64
backend='hccl'
device_type=0
epoch=1
compile='false'
num_of_used_data=0
save_data_interval=5120000
warmup_ratio=0.05
learning_rate=0.0001
min_lr=6e-5
decay_lr=true
weight_decay=0.1
beta1=0.9
beta2=0.95
grad_clip=1.0
gradient_accumulation_steps=4
device="npu"
device_type="npu"
hidden_size=1280
num_hidden_layers=24
num_attention_heads=20
intermediate_size=5120
hidden_act="gelu"
hidden_dropout_prob=0.1
cell_hidden_size=128
attention_probs_dropout_prob=0.1
type_vocab_size=2
initializer_range=0.02
layer_norm_eps=1e-12
_attn_implementation="sdpa"
use_batch_labels=false
num_batch_labels=12028
use_species_labels=true
num_species_labels=11
use_tissue_labels=true
num_tissue_labels=154
use_seqmethod_labels=true
num_seqmethod_labels=28
use_disease_labels=true
num_disease_labels=143
use_age_labels=true
num_age_labels=5
use_sex_labels=true
num_sex_labels=3
cell_emb_style="cls"
chunk_size_feed_forward=0
explicit_zero_prob=true

# 解析命令行参数
for arg in "$@"; do
    case $arg in
        --nproc_per_node=*)
            nproc_per_node="${arg#*=}"
            shift
            ;;
        --nnodes=*)
            nnodes="${arg#*=}"
            shift
            ;;
        --out_path=*)
            out_path="${arg#*=}"
            shift
            ;;
        --s3_remote_dir_path=*)
            s3_remote_dir_path="${arg#*=}"
            shift
            ;;
        --data_path=*)
            data_path="${arg#*=}"
            shift
            ;;
        --emb_path=*)
            emb_path="${arg#*=}"
            shift
            ;;
        --seq_len=*)
            seq_len="${arg#*=}"
            shift
            ;;
        --batch_size=*)
            batch_size="${arg#*=}"
            shift
            ;;
        --epoch=*)
            epoch="${arg#*=}"
            shift
            ;;
        --compile=*)
            compile="${arg#*=}"
            shift
            ;;
        --num_of_used_data=*)
            num_of_used_data="${arg#*=}"
            shift
            ;;
        --save_data_interval=*)
            save_data_interval="${arg#*=}"
            shift
            ;;
        --s3_remote_dir_path=*)
            s3_remote_dir_path="${arg#*=}"
            shift
            ;;
        --learning_rate=*)
            learning_rate="${arg#*=}"
            shift
            ;;
        --min_lr=*)
            min_lr="${arg#*=}"
            shift
            ;;
        --decay_lr=*)
            decay_lr="${arg#*=}"
            shift
            ;;
        --warmup_ratio=*)
            warmup_ratio="${arg#*=}"
            shift
            ;;
        --weight_decay=*)
            weight_decay="${arg#*=}"
            shift
            ;;
        --beta1=*)
            beta1="${arg#*=}"
            shift
            ;;
        --beta2=*)
            beta2="${arg#*=}"
            shift
            ;;
        --grad_clip=*)
            grad_clip="${arg#*=}"
            shift
            ;;
        --gradient_accumulation_steps=*)
            gradient_accumulation_steps="${arg#*=}"
            shift
            ;;
        --device=*)
            device="${arg#*=}"
            shift
            ;;
        --device_type=*)
            device_type="${arg#*=}"
            shift
            ;;
        --hidden_size=*)
            hidden_size="${arg#*=}"
            shift
            ;;
        --num_hidden_layers=*)
            num_hidden_layers="${arg#*=}"
            shift
            ;;
        --num_attention_heads=*)
            num_attention_heads="${arg#*=}"
            shift
            ;;
        --intermediate_size=*)
            intermediate_size="${arg#*=}"
            shift
            ;;
        --hidden_act=*)
            hidden_act="${arg#*=}"
            shift
            ;;
        --hidden_dropout_prob=*)
            hidden_dropout_prob="${arg#*=}"
            shift
            ;;
        --cell_hidden_size=*)
            cell_hidden_size="${arg#*=}"
            shift
            ;;
        --attention_probs_dropout_prob=*)
            attention_probs_dropout_prob="${arg#*=}"
            shift
            ;;
        --type_vocab_size=*)
            type_vocab_size="${arg#*=}"
            shift
            ;;
        --initializer_range=*)
            initializer_range="${arg#*=}"
            shift
            ;;
        --layer_norm_eps=*)
            layer_norm_eps="${arg#*=}"
            shift
            ;;
        --_attn_implementation=*)
            _attn_implementation="${arg#*=}"
            shift
            ;;
        --use_batch_labels=*)
            use_batch_labels="${arg#*=}"
            shift
            ;;
        --num_batch_labels=*)
            num_batch_labels="${arg#*=}"
            shift
            ;;
        --use_species_labels=*)
            use_species_labels="${arg#*=}"
            shift
            ;;
        --num_species_labels=*)
            num_species_labels="${arg#*=}"
            shift
            ;;
        --use_tissue_labels=*)
            use_tissue_labels="${arg#*=}"
            shift
            ;;
        --num_tissue_labels=*)
            num_tissue_labels="${arg#*=}"
            shift
            ;;
        --use_seqmethod_labels=*)
            use_seqmethod_labels="${arg#*=}"
            shift
            ;;
        --num_seqmethod_labels=*)
            num_seqmethod_labels="${arg#*=}"
            shift
            ;;
        --use_disease_labels=*)
            use_disease_labels="${arg#*=}"
            shift
            ;;
        --num_disease_labels=*)
            num_disease_labels="${arg#*=}"
            shift
            ;;
        --use_age_labels=*)
            use_age_labels="${arg#*=}"
            shift
            ;;
        --num_age_labels=*)
            num_age_labels="${arg#*=}"
            shift
            ;;
        --use_sex_labels=*)
            use_sex_labels="${arg#*=}"
            shift
            ;;
        --num_sex_labels=*)
            num_sex_labels="${arg#*=}"
            shift
            ;;
        --cell_emb_style=*)
            cell_emb_style="${arg#*=}"
            shift
            ;;
        --chunk_size_feed_forward=*)
            chunk_size_feed_forward="${arg#*=}"
            shift
            ;;
        --explicit_zero_prob=*)
            explicit_zero_prob="${arg#*=}"
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $arg"
            usage
            ;;
    esac
done

export out_path="$out_path"
export s3_remote_dir_path="$s3_remote_dir_path"
export data_path="$data_path"
export emb_path="$emb_path"
export seq_len="$seq_len"
export batch_size="$batch_size"
export epoch="$epoch"
export compile="$compile"
export num_of_used_data="$num_of_used_data"
export save_data_interval="$save_data_interval"
export learning_rate="$learning_rate"
export min_lr="$min_lr"
export decay_lr="$decay_lr"
export warmup_ratio="$warmup_ratio"
export weight_decay="$weight_decay"
export beta1="$beta1"
export beta2="$beta2"
export grad_clip="$grad_clip"
export gradient_accumulation_steps="$gradient_accumulation_steps"
export device="$device"
export device_type="$device_type"

export hidden_size="$hidden_size"
export num_hidden_layers="$num_hidden_layers"
export num_attention_heads="$num_attention_heads"
export intermediate_size="$intermediate_size"
export hidden_act="$hidden_act"
export hidden_dropout_prob="$hidden_dropout_prob"
export cell_hidden_size="$cell_hidden_size"
export attention_probs_dropout_prob="$attention_probs_dropout_prob"
export type_vocab_size="$type_vocab_size"
export initializer_range="$initializer_range"
export layer_norm_eps="$layer_norm_eps"
export _attn_implementation="$_attn_implementation"
export use_batch_labels="$use_batch_labels"
export num_batch_labels="$num_batch_labels"
export use_species_labels="$use_species_labels"
export num_species_labels="$num_species_labels"
export use_tissue_labels="$use_tissue_labels"
export num_tissue_labels="$num_tissue_labels"
export use_seqmethod_labels="$use_seqmethod_labels"
export num_seqmethod_labels="$num_seqmethod_labels"
export use_disease_labels="$use_disease_labels"
export num_disease_labels="$num_disease_labels"
export use_age_labels="$use_age_labels"
export num_age_labels="$num_age_labels"
export use_sex_labels="$use_sex_labels"
export num_sex_labels="$num_sex_labels"
export cell_emb_style="$cell_emb_style"
export chunk_size_feed_forward="$chunk_size_feed_forward"
export explicit_zero_prob="$explicit_zero_prob"


# 注意这个方式是MA起分布式的方式对应的
# 这是根据MA后台给的参数：“ python -m torch.dis.....  --init_method "tcp://$(echo ${VC_WORKER_HOSTS} | cut -d "," -f 1):6666" --rank ${VC_TASK_INDEX} ”
# 其中--rank ${VC_TASK_INDEX}，这点与原生torch ddp一致，实际上就是node_rank
# 后续传入主训练脚本后，根据num_nodes、node_rank、nproc_per_node去算相应的值

# 计算节点相关参数 导出脚本传参
NNODES="$nnodes"
export NNODES="$NNODES"

NPROC_PER_NODE="$nproc_per_node"
export NPROC_PER_NODE="$NPROC_PER_NODE"

# 计算节点相关参数 导出环境变量
MASTER_HOST="$VC_WORKER_HOSTS"
MASTER_ADDR="${VC_WORKER_HOSTS%%,*}"
MASTER_PORT="12345"
NODE_RANK="$VC_TASK_INDEX"
export MASTER_HOST="$MASTER_HOST"
export MASTER_ADDR="$MASTER_ADDR"
export MASTER_PORT="$MASTER_PORT"
export NODE_RANK="$NODE_RANK"
export HCCL_CONNECT_TIMEOUT=7200
export HCCL_EXEC_TIMEOUT=7200
export HCCL_WHITELIST_DISABLE=1
export ASCEND_TOOLKIT_HOME=/usr/local/Ascend/ascend-toolkit/latest
export ASCEND_HOME_PATH=${ASCEND_TOOLKIT_HOME}

# 参数校验
if [[ -z "$MASTER_HOST" || -z "$MASTER_ADDR" || -z "$MASTER_PORT" || -z "$NNODES" || -z "$NODE_RANK" || -z "$NPROC_PER_NODE" ]]; then
    echo "Error: Required parameters are missing."
    usage
fi
echo "===================================================================================================================================="
echo "===================================================================================================================================="
echo "NNODES=$NNODES, NPROC_PER_NODE=$NPROC_PER_NODE, MASTER_HOST=$MASTER_HOST, MASTER_ADDR=$MASTER_ADDR, MASTER_PORT=$MASTER_PORT, NODE_RANK=$NODE_RANK"
#NNODES=2, NPROC_PER_NODE=8, MASTER_HOST=ma-job-a6b44790-ca10-4228-bc2d-0a05ccdd5420-worker-0.ma-job-a6b44790-ca10-4228-bc2d-0a05ccdd5420,ma-job-a6b44790-ca10-4228-bc2d-0a05ccdd5420-worker-1.ma-job-a6b44790-ca10-4228-bc2d-0a05ccdd5420, MASTER_ADDR=ma-job-a6b44790-ca10-4228-bc2d-0a05ccdd5420-worker-0.ma-job-a6b44790-ca10-4228-bc2d-0a05ccdd5420, MASTER_PORT=12345, NODE_RANK=0
echo "===================================================================================================================================="
echo "===================================================================================================================================="


## 定义复制命令
#cmd="cp -r /home/ma-user/modelarts/user-job-dir/SpeciesLLM /cache/SpeciesLLM"
#echo "$cmd"
## 执行复制命令
#eval "$cmd"
## 切换到目标目录
#cd "/cache/SpeciesLLM/" || exit 1
## 打印当前工作目录
#echo "Current working directory: $(pwd)"


# 定义 find_modearts_task_ID 函数，目的是找到本次任务中的本地项目路径，方便后续拷贝结果到S3桶上
find_modearts_task_ID() {
    local src=$1  # 源路径
    local dest=$2 # 目标路径
    local base_dir="/"  # 在根目录下查找
    local output_dir=""
    # 遍历根目录下的内容
    for item in $(ls "$base_dir"); do
        local item_path="$base_dir$item"
        # 检查是否是目录且以 "modelarts-job" 开头
        if [[ -d "$item_path" && "$item" == modelarts-job* ]]; then
            output_dir="${dest}/${item}"
            break
        fi
    done
    # 返回结果
    echo "$output_dir"
}
# 调用示例
full_s3_remote_dir_path=$(find_modearts_task_ID "$out_path" "$s3_remote_dir_path")
export full_s3_remote_dir_path="$full_s3_remote_dir_path"
echo "Output directory: $full_s3_remote_dir_path"

# 构建命令行参数
cmd_args="--data_path=$data_path \
    --num_of_used_data=$num_of_used_data \
    --emb_path=$emb_path \
    --seq_len=$seq_len \
    --out_path=$out_path \
    --batch_size=$batch_size \
    --epoch=$epoch \
    --gradient_accumulation_steps=$gradient_accumulation_steps \
    --learning_rate=$learning_rate \
    --min_lr=$min_lr \
    --decay_lr=$decay_lr \
    --warmup_ratio=$warmup_ratio \
    --weight_decay=$weight_decay \
    --save_data_interval=$save_data_interval \
    --beta1=$beta1 \
    --beta2=$beta2 \
    --grad_clip=$grad_clip \
    --compile=$compile \
    --backend=$backend \
    --device=$device \
    --device_type=$device_type \
    --s3_remote_dir_path=$full_s3_remote_dir_path \
    --hidden_size=$hidden_size \
    --num_hidden_layers=$num_hidden_layers \
    --num_attention_heads=$num_attention_heads \
    --intermediate_size=$intermediate_size \
    --hidden_act=$hidden_act \
    --hidden_dropout_prob=$hidden_dropout_prob \
    --cell_hidden_size=$cell_hidden_size \
    --attention_probs_dropout_prob=$attention_probs_dropout_prob \
    --type_vocab_size=$type_vocab_size \
    --initializer_range=$initializer_range \
    --layer_norm_eps=$layer_norm_eps \
    --_attn_implementation=$_attn_implementation \
    --use_batch_labels=$use_batch_labels \
    --num_batch_labels=$num_batch_labels \
    --use_species_labels=$use_species_labels \
    --num_species_labels=$num_species_labels \
    --use_tissue_labels=$use_tissue_labels \
    --num_tissue_labels=$num_tissue_labels \
    --use_seqmethod_labels=$use_seqmethod_labels \
    --num_seqmethod_labels=$num_seqmethod_labels \
    --use_disease_labels=$use_disease_labels \
    --num_disease_labels=$num_disease_labels \
    --use_age_labels=$use_age_labels \
    --num_age_labels=$num_age_labels \
    --use_sex_labels=$use_sex_labels \
    --num_sex_labels=$num_sex_labels \
    --cell_emb_style=$cell_emb_style \
    --chunk_size_feed_forward=$chunk_size_feed_forward \
    --explicit_zero_prob=$explicit_zero_prob"

# 构建 torchrun 命令
cmd="torchrun \
    --nproc_per_node=$NPROC_PER_NODE \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    train_MNodes_torchrun_mfu_preindexparquet.py $cmd_args"
# 打印并执行命令
echo "===================================================================================================================================="
echo "===================================================================================================================================="
echo "Executing command: $cmd"
echo "===================================================================================================================================="
echo "===================================================================================================================================="
