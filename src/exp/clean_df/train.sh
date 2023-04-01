# =======================================================================#
# Testing original df vs my cleaned + preprocessed + contextualized df.
# =======================================================================#

# python train.py --project_run_root "original"
# python train.py --project_run_root "cleaned" --batch_size 64 --train_p "../../../input/prep_cleaned_train_context_5fold.csv"

# =======================================================================#
# Testing different backbones for reranking.
# =======================================================================#

backbone = "paraphrase-MiniLM-L3-v2"
python train.py --save_root "../../../models/test_models/" --project "lecr_test_models" \
                --project_run_root $backbone \
                --backbone_type "sentence-transformers/$backbone" \
                --batch_size 128
python ../../kaggle_dataset.py --title "lecr_$backbone" --save_p "../../../models/test_models/$backbone"

backbone = "stsb-xlm-r-multilingual"
python train.py --save_root "../../../models/test_models/" --project "lecr_test_models" \
                --project_run_root $backbone \
                --backbone_type "sentence-transformers/$backbone" \
                --batch_size 128
python ../../kaggle_dataset.py --title "lecr_$backbone" --save_p "../../../models/test_models/$backbone"