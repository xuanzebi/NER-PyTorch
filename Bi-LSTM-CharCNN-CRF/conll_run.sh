export CUDA_VISIBLE_DEVICES=2

python main.py --data_type='conll' \
               --data_path='../dataset/conll2003/' \
               --model_save_dir='./saved_models/conll_char_32/' \
               --tensorboard_dir='./saved_models/conll_char_32/runs/' \
               --num_train_epochs=100 \
               --batch_size=64 \
               --use_char=True \
               --freeze=True \
               --use_crf=True \
               --use_bieos=True \
               --use_highway=True \
               --use_number_norm=False \
               --dropout=0.25 \
               --dropoutlstm=0.25 \
               --pred_embed_path='' \
