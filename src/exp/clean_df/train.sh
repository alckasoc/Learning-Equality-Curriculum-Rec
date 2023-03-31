# python train.py --project_run_root "original"
# python train.py --project_run_root "cleaned" --batch_size 64 --train_p "../../../input/prep_cleaned_train_context_5fold.csv"

# python train.py --save_root "../../../models/test_models/" --project "lecr_test_models" \
#                 --project_run_root "paraphrase-MiniLM-L3-v2" \
#                 --backbone_type "sentence-transformers/paraphrase-MiniLM-L3-v2" \
#                 --batch_size 128
# python ../../kaggle_dataset.py --title "lecr_paraphrase-MiniLM-L3-v2" --save_p "../models/test_models/paraphrase-MiniLM-L3-v2"

python train.py --save_root "../../../models/test_models/" --project "lecr_test_models" \
                --project_run_root "stsb-xlm-r-multilingual" \
                --backbone_type "sentence-transformers/stsb-xlm-r-multilingual" \
                --batch_size 128
python ../../kaggle_dataset.py --title "lecr_stsb-xlm-r-multilingual" --save_p "../models/test_models/stsb-xlm-r-multilingual"