backbones=(
    "sentence-transformers/all-MiniLM-L6-v2" 
    "sentence-transformers/stsb-xlm-r-multilingual" 
    "sentence-transformers/quora-distilbert-multilingual" 
    "sentence-transformers/use-cmlm-multilingual"
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    "sentence-transformers/multi-qa-mpnet-base-dot-v1"
)

backbone_names=(
    "all-MiniLM-L6-v2" 
    "stsb-xlm-r-multilingual" 
    "quora-distilbert-multilingual" 
    "use-cmlm-multilingual"
    "paraphrase-multilingual-mpnet-base-v2"
    "paraphrase-multilingual-MiniLM-L12-v2"
    "multi-qa-mpnet-base-dot-v1"
)

# for backbone in ${backbones[@]}; do
#     if [[ "$backbone" == "sentence-transformers/multi-qa-mpnet-base-dot-v1" ]]; then
#         python train_ret.py --batch_size 64 --epochs 5 --backbone_type $backbone
#     else
#         python train_ret.py --batch_size 128 --epochs 5 --backbone_type $backbone
#     fi
# done

# for backbone_name in ${backbone_names[@]}; do
#     python ../../kaggle_dataset.py --title "lecr_retriever_$backbone_name" --save_p "../../../models/retriever_sweep/$backbone_name"
# done

# for backbone_name in ${backbone_names[@]}; do
#     python make_train.py --model_p "../../../models/retriever_sweep/$backbone_name/ep5" --backbone_type $backbone_name
# done

# for backbone_name in ${backbone_names[@]}; do
#     python make_split.py --model_p "../../../models/retriever_sweep/$backbone_name/ep5" --train_p "../../../input/retriever_sweep_train/$backbone_name/"
# done

for backbone_name in ${backbone_names[@]}; do
    python train_reranker.py --model_save_p "../../../models/retriever_sweep/$backbone_name/" --train_p "../../../input/retriever_sweep_train/$backbone_name/"
done

