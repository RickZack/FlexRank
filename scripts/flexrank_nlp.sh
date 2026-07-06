HAS_MODEL=0
HAS_DATASET=0
for arg in "$@"; do
    case $arg in
        model=*) HAS_MODEL=1 ;;
        dataset=*) HAS_DATASET=1 ;;
    esac
done

if (( ! HAS_MODEL || ! HAS_DATASET )); then
    echo "Usage: bash scripts/flexrank_nlp.sh model=<model> dataset=<dataset> [key=value ...]"
    echo "Error: model and dataset are required"
    exit 1
fi

###### DISTRIBUTED ENV VARIABLES ########
echo "--- Configuration ---"
if command -v nvidia-smi &> /dev/null; then
    NUM_GPUS=$(nvidia-smi -L | wc -l)
else
    echo "No GPUs detected"
    exit 1
fi
MASTER_ADDR="127.0.0.1"
MASTER_PORT=29500
NUM_MACHINES=1
echo "Script directly launched on a single node"
echo "TOTAL GPUS: $NUM_GPUS"
echo "---------------------"

########## END OF DISTR VARIABLES ######

declare -A DATASETS_DICT=(
    [finewebedu_10bt]=1
)

declare -A MODELS_DICT=(
    [gpt2]=1
    [llama1b]=1
    [llama3b]=1
    [llama8b]=1
)

declare -A FREEZE_POLICY=(
    [true]="traindecomp"
    [false]="trainall"
)

declare -A SAMPLERS_DICT=(
    [predefined_models]="predefineduniform"
    [all_layer_independent]="allmodels"
    [single_layer_linear]="singlellinear"
)

declare -A DATASETS_NAMES
DATASETS_NAMES["finewebedu_10bt_gpt2"]="finewebedu10bt"
DATASETS_NAMES["finewebedu_10bt_llama1b"]="finewebedu10bt"
DATASETS_NAMES["finewebedu_10bt_llama3b"]="finewebedu10bt"
DATASETS_NAMES["finewebedu_10bt_llama8b"]="finewebedu10bt"

usage() {
    echo "Usage: bash scripts/flexrank_nlp.sh model=<model> dataset=<dataset> [key=value ...]"
    echo "Required:"
    echo "  model: one of ${!MODELS_DICT[*]}"
    echo "  dataset: one of ${!DATASETS_DICT[*]}"
}

#### HPARAMS SECTION ####

# MODEL AND DATASET - MUST BE SPECIFIED
MODEL=''
RAW_DATASET=''

# TRAINING HPARAMS
## OBJECTIVE - DEFAULT: PURE DISTILLATION
CE=0.0
KL=1.0
## OPTIM
LR=1e-3
MIN_LR=1e-4
WD=0
## BS
FULL_BS=512
TRAIN_DEVICE_BS=16
EVAL_DEVICE_BS=16
## OTHER
WARMUP_STEPS=715
STEPS=6000
EVAL_STEPS=500

# FLEXRANK HPARAMS
PROF_ALGO=dp
LAST_PROF_ALGO=nosearch
SAMPLER='predefined_models'
FREEZE_NON_DECOMPOSED=false
SVD_TYPE=datasvd # or svd

# LOGGING AND PERSISTENCE
PROJECT='FlexRank'
BASE_DIR='/tmp'

# ACCELERATE
ACCELERATE_CONFIG_FILE=ddp
##### END OF HPARAMS ####


#### OVERRIDING PARAMS FROM CMD LINE ####
CE_SET=0
KL_SET=0
EVAL_BS_SET=0

for arg in "$@"; do
    [[ $arg == *=* ]] || {
        echo "Error: '$arg' must be key=value"
        exit 1
    }

    key=${arg%%=*}
    value=${arg#*=}

    case $key in
        # ---- model / dataset ----
        model)
            if [[ -z "${MODELS_DICT[$value]}" ]]; then
                echo "Error: Invalid model '$value'"
                echo "Allowed models: ${!MODELS_DICT[*]}"
                exit 1
            fi
            MODEL=$value
            ;;
        dataset)
            if [[ -z "${DATASETS_DICT[$value]}" ]]; then
                echo "Error: Invalid dataset '$value'"
                echo "Allowed datasets: ${!DATASETS_DICT[*]}"
                exit 1
            fi
            RAW_DATASET=$value
            ;;

        # ---- loss weights ----
        # ---- loss weights (ce + kl = 1) ----
        ce)
            CE=$value
            CE_SET=1
            ;;
        kl)
            KL=$value
            KL_SET=1
            ;;

        # ---- optim ----
        lr)        LR=$value ;;
        min_lr)    MIN_LR=$value ;;
        wd)        WD=$value ;;
        full_bs)   FULL_BS=$value ;;

        # ---- batch size ----
        train_bs|bs)
            TRAIN_DEVICE_BS=$value
            if (( FULL_BS % TRAIN_DEVICE_BS != 0 )); then
                echo "Error: train_bs must divide FULL_BS ($FULL_BS)"
                exit 1
            fi
            GA_STEPS=$(( FULL_BS / ( TRAIN_DEVICE_BS * NUM_GPUS ) ))
            ;;
        eval_bs)
            EVAL_DEVICE_BS=$value
            EVAL_BS_SET=1 
            ;;
        # ---- training length ----
        steps)         STEPS=$value ;;
        warmup_steps)  WARMUP_STEPS=$value ;;

        # ---- flexrank ----
        svd_type)
            value=$(echo "$value" | tr '[:upper:]' '[:lower:]')        
            SVD_TYPE=$value
            ;;
        prof_algo)  PROF_ALGO=$value ;;
        sampler)    SAMPLER=$value ;;
        last_profile_algo) LAST_PROF_ALGO=$value ;;
        freeze_non_decomposed)
            value=$(echo "$value" | tr '[:upper:]' '[:lower:]')
            if [[ -z "${FREEZE_POLICY[$value]}" ]]; then
                echo "Error: Invalid choice '$value'"
                echo "Allowed choices: ${!FREEZE_POLICY[*]}"
                exit 1
            fi
            FREEZE_NON_DECOMPOSED=$value
            ;;

        # ---- logging / paths ----
        project)    PROJECT=$value ;;

        # ------- accelerate ------
        accelerate) ACCELERATE_CONFIG_FILE=$value ;;

        *)
            echo "Error: Unknown argument '$key'"
            exit 1
            ;;
    esac
done

if [[ -z "$MODEL" || -z "$RAW_DATASET" ]]; then
    echo "Error: model and dataset are required"
    usage
    exit 1
fi

# SET CE OR KL BASED ON THE OTHER OVERRIDDEN PARAMS
if (( CE_SET && KL_SET )); then
    echo "Error: Specify only one of ce or kl (the other is set automatically)"
    exit 1
fi

if (( CE_SET )); then
    KL=$(awk -v ce="$CE" 'BEGIN { printf "%.2f", 1 - ce }')
elif (( KL_SET )); then
    CE=$(awk -v kl="$KL" 'BEGIN { printf "%.2f", 1 - kl }')
fi


####### DEPENDENT VARIABLES #############
DATASET=${RAW_DATASET}_${MODEL}
if [[ -z "${DATASETS_NAMES[$DATASET]}" ]]; then
    echo "Error: Unsupported dataset/model combination: $DATASET"
    echo "Known combinations:"
    printf "  %s\n" "${!DATASETS_NAMES[@]}"
    exit 1
fi

if (( FULL_BS % ( TRAIN_DEVICE_BS * NUM_GPUS ) != 0 )); then
    echo "Error: train_bs*num_gpus must divide FULL_BS ($FULL_BS)"
    exit 1
fi

if (( ! EVAL_BS_SET )); then
    EVAL_DEVICE_BS=$TRAIN_DEVICE_BS
fi
GA_STEPS=$(( FULL_BS / ( TRAIN_DEVICE_BS * NUM_GPUS ) ))

# COMPOSING RUN NAME AND OUTPUT DIR
DCMP=${SVD_TYPE}-${FREEZE_POLICY[$FREEZE_NON_DECOMPOSED]}
P_STR=${PROF_ALGO}-${LAST_PROF_ALGO}
S_STR=${SAMPLERS_DICT[$SAMPLER]}
DS_STR=${DATASETS_NAMES[$DATASET]}
LOSS_STR=ce${CE}-kl${KL}
LR_STR=lr${LR}-minlr${MIN_LR}

# RUN_NAME FORMAT: <part0>_..._<partN>, <part> can contain '-'
# 0. svd or datasvd + whether non-decomp layers are frozen or not
# 1. profile algo **before** training
# 2. profile algo **after** training: usually None (submodels remain the one seached before training)
# 3. sampler: usually {uniform, dp}, can be None if we do not want to train for a finite number of 
#             submodels and want to search profiles afterwards
# 4. model
# 5. dataset
# 6. loss: combination of coefficients for ce and kl
# 7. lr: base and minimum lr, to be used with cosine annealing scheduler
#        (we do not consider this as hparam)

RUN_NAME=${DCMP}_${P_STR}_${S_STR}_${MODEL}_${DS_STR}_${LOSS_STR}_${LR_STR}
OUTPUT_DIR=$BASE_DIR/$RUN_NAME
####### END OF DEPENDENT VARIABLES ######
####### END OF OVERRIDING PARAMS ########

############ END OF ENV VARIABLES ######

############ LAUNCH COMMAND ############
# Define the launcher arguments (BEFORE the script)
LAUNCH_BASE="accelerate launch \
    --config_file ./flextrain/src/flextrain/config/accelerate/${ACCELERATE_CONFIG_FILE}.yaml \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --num_machines $NUM_MACHINES \
    --num_processes $NUM_GPUS"

# Define your script and its arguments (AFTER the script)
SCRIPT_ARGS="train.py \
    task=NLP \
    dataset=$DATASET \
    model=$MODEL \
    train.ce_loss_w=$CE \
    train.kl_loss_w=$KL \
    train.per_device_train_batch_size=$TRAIN_DEVICE_BS \
    train.per_device_eval_batch_size=$EVAL_DEVICE_BS \
    eval.per_device_train_batch_size=$TRAIN_DEVICE_BS \
    eval.per_device_eval_batch_size=$EVAL_DEVICE_BS \
    train.gradient_accumulation_steps=$GA_STEPS \
    train.output_dir=$OUTPUT_DIR \
    train.learning_rate=$LR \
    train.weight_decay=$WD \
    train.lr_scheduler_kwargs.min_lr=$MIN_LR \
    train.max_steps=$STEPS \
    train.warmup_steps=$WARMUP_STEPS \
    train.eval_steps=$EVAL_STEPS \
    train.save_steps=$EVAL_STEPS \
    decomposition=$SVD_TYPE \
    decomposition.freeze_non_decomposed=$FREEZE_NON_DECOMPOSED \
    sampler=$SAMPLER \
    profile_algo=$PROF_ALGO \
    profile_algo@last_profile_algo=$LAST_PROF_ALGO \
    logger.name=$RUN_NAME \
    logger.project=$PROJECT \
    logger.mode=disabled"

# Existance of this file is a flag for resuming the run
if [ -f "$OUTPUT_DIR/.restarted" ]; then
    SCRIPT_ARGS="$SCRIPT_ARGS \
                 load_flexrank_path=$OUTPUT_DIR \
                 resume_checkpoint_path=True"
fi

# LOCAL: machine_rank is always 0
echo "Launching locally..."
eval "$LAUNCH_BASE --machine_rank 0 $SCRIPT_ARGS"

######## END OF LAUNCH COMMAND #########
