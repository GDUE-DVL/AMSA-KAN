#!/bin/bash
# =============================================================================
# LAR-IQA 双分支模型训练脚本
# 针对四个数据集: ShanghaiTech Part A/B、UCF-QNRF、UCF_CC_50
# 用法: bash train_lariqa_model.sh [GPU_ID]
# =============================================================================

# 配置参数
GPU_ID=${1:-2}                   # 默认使用 GPU 2，可通过第一个参数修改
BACKBONE="mobilenet"            # 主干网络: mobilenet 或 resnet50
BATCH_SIZE=4                    # 批次大小
EPOCHS=100                      # 训练轮数
LEARNING_RATE=0.0001            # 学习率
HIDDEN_DIM=256                  # KAN隐藏层维度
KAN_LAYERS=7                    # KAN 层数 (可调)
KAN_GRID_SIZE=10                # KAN 样条网格大小 (可调)
KAN_SPLINE_TYPE="bspline"       # KAN 样条类型: bspline / poly
OUTPUT_ROOT="./experiments/lariqa_models"  # 输出根目录

# 激活 conda 环境
source $(conda info --base)/etc/profile.d/conda.sh
conda activate pytorch || echo "⚠️ 请确认 conda 环境名称是否为 pytorch"

# 清理 CUDA 缓存，确保最大可用内存
echo "清理 CUDA 缓存..."
python -c "import torch; torch.cuda.empty_cache()"

# 检查脚本位置
if [ ! -f "./scripts/train_kan_crowd_counter.py" ]; then
  echo "❌ 错误: 未找到训练脚本 ./scripts/train_kan_crowd_counter.py"
  echo "请在项目根目录下运行此脚本"
  exit 1
fi

# 确保输出目录存在
mkdir -p "${OUTPUT_ROOT}"

# 数据集配置
declare -A DATASETS=(
  # ["ShanghaiTech"]="./datasets/shanghai_tech/shanghai_tech" # 注释掉合并的条目
  ["ShanghaiTechA"]="./people_count/ShanghaiTech/part_A_final" # 使用 Part A 路径
  ["ShanghaiTechB"]="./people_count/ShanghaiTech/part_B_final" # 使用 Part B 路径
  ["UCF-QNRF"]="./datasets/ucf_qnrf"
  ["UCF-CC-50"]="./datasets/ucf_cc_50"  # 添加UCF_CC_50数据集
  #["UCF-QNRF-Fixed"]="./datasets/ucf_qnrf_fixed/ucf_qnrf_fixed"
)

# 循环训练所有数据集
for NAME in "${!DATASETS[@]}"; do
  BASE_NAME="${DATASETS[$NAME]}"
  # 假设 CSV 文件位于 BASE_NAME 目录下
  TRAIN_CSV="${BASE_NAME}/train.csv"
  VAL_CSV="${BASE_NAME}/val.csv"
  TEST_CSV="${BASE_NAME}/test.csv"
  OUTPUT_DIR="${OUTPUT_ROOT}/${NAME}" # 输出目录按数据集名称区分

  # 跳过如果CSV不存在
  if [ ! -f "${TRAIN_CSV}" ]; then
    echo "⚠️ 数据集 ${NAME} 的训练CSV不存在: ${TRAIN_CSV}"
    echo "跳过此数据集训练，请准备好训练数据后再试"
    continue
  fi

  echo "=================================================="
  echo "🚀 开始训练数据集: ${NAME}"
  echo "=================================================="
  echo "→ 训练CSV: ${TRAIN_CSV}"
  echo "→ 验证CSV: ${VAL_CSV}"
  echo "→ 测试CSV: ${TEST_CSV}"
  echo "→ 输出目录: ${OUTPUT_DIR}"
  echo "→ GPU ID: ${GPU_ID}"
  echo "→ 主干网络: ${BACKBONE}"
  echo "→ 批次大小: ${BATCH_SIZE}"
  echo "→ 训练轮数: ${EPOCHS}"

  # 创建输出目录
  mkdir -p "${OUTPUT_DIR}"

  # 构建训练命令并执行
  CUDA_VISIBLE_DEVICES=$GPU_ID python ./scripts/train_kan_crowd_counter.py \
    --model_type lariqa \
    --train_csv "${TRAIN_CSV}" \
    --val_csv "${VAL_CSV}" \
    --test_csv "${TEST_CSV}" \
    --batch_size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --learning_rate ${LEARNING_RATE} \
    --backbone ${BACKBONE} \
    --kan_hidden_dim ${HIDDEN_DIM} \
    --kan_layers ${KAN_LAYERS} \
    --kan_grid_size ${KAN_GRID_SIZE} \
    --kan_spline_type ${KAN_SPLINE_TYPE} \
    --output_dir "${OUTPUT_DIR}" \
    --random_flip \
    --color_jitter \
    --use_scheduler \
    --mixed_precision \
    --pretrained \
    --pin_memory \
    --num_workers 4

  # 检查训练是否成功
  if [ $? -ne 0 ]; then
    echo "❌ ${NAME} 训练失败，请检查日志了解详情"
  else
    echo "✅ ${NAME} 训练成功完成!"
  fi
done

echo "所有数据集训练完成!"

# 发送系统通知
if command -v notify-send &> /dev/null; then
  notify-send "LAR-IQA训练完成" "所有数据集模型训练已完成!"
fi

# 清理GPU缓存
python -c "import torch; torch.cuda.empty_cache()" 