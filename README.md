# Data Diversification: A Universally Elegant ...

## Pretrained Models


## Instruction To train WMT English-German

**Step 1**: Follow instruction from [Fairseq](https://github.com/pytorch/fairseq/tree/master/examples/translation) to create
the WMT'14 Dataset. 

Save the processed data as ``data_fairseq/translate_ende_wmt16_bpe32k``

Save the raw data (which contains the file train.tok.clean.bpe.32000.en) to ``raw_data/wmt_ende`` 

**Step 2**: copy the same data to ``data_fairseq/translate_deen_wmt16_bpe32k`` for De-En


**Step 3**: Train forward models
```bash

export seed_prefix=100
export problem=translate_ende_wmt16_bpe32k
export model_name=big_tfm_baseline_df3584_s${seed_prefix}
export data_dir=`pwd`/data_fairseq/$problem

for index in {1..3}
do
export model_dir=train_fairseq/${problem}/${model_name}/model_${index}
fairseq-train \
    ${data_dir} \
    -s en -t de \
    --arch transformer_vaswani_wmt_en_de_big --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 0.001 --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07 \
    --dropout 0.3 --attention-dropout 0.1 --weight-decay 0.0 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-update 43000 \
    --keep-last-epochs 10 \
    --save-dir ${model_dir} \
    --ddp-backend no_c10d \
    --seed ${seed_prefix}${index} \
    --max-tokens 3584 \
    --fp16 --update-freq 16 --log-interval 10000 --no-progress-bar
done
```


**Step 4**: Train backward models
```bash

export seed_prefix=101
export problem=translate_deen_wmt16_bpe32k
export model_name=big_tfm_baseline_df3584_s${seed_prefix}
export data_dir=`pwd`/data_fairseq/$problem

for index in {1..3}
do
export model_dir=train_fairseq/${problem}/${model_name}/model_${index}
fairseq-train \
    ${data_dir} \
    -s de -t en \
    --arch transformer_vaswani_wmt_en_de_big --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 0.001 --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07 \
    --dropout 0.3 --attention-dropout 0.1 --weight-decay 0.0 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-update 43000 \
    --keep-last-epochs 10 \
    --save-dir ${model_dir} \
    --ddp-backend no_c10d \
    --seed ${seed_prefix}${index} \
    --max-tokens 3584 \
    --fp16 --update-freq 16 --log-interval 10000 --no-progress-bar
done
```


**Step 5**: Inference forward models

```bash
export seed_prefix=100
export problem=translate_ende_wmt16_bpe32k
export model_name=big_tfm_baseline_df3584_s${seed_prefix}
export data_dir=`pwd`/data_fairseq/$problem
export beam=5
export lenpen=0.6
export round=1

for index in {1..3}
do
export model_dir=train_fairseq/${problem}/${model_name}/model_${index}
export best_file=$model_dir/checkpoint_best.pt
export gen_out=$model_dir/infer_train_b${beam}_lp${lenpen}
fairseq-generate ${data_dir} \
    -s en -t de \
    --path ${best_file} \
    --gen-subset train \
    --max-tokens ${infer_bsz} --beam ${beam}  --lenpen ${lenpen} | dd of=$gen_out && \
grep ^S ${gen_out} | cut -f2- > $gen_out.en && \
grep ^H ${gen_out} | cut -f3- > $gen_out.de && \
done

```


**Step 6**: Inference backward models

```bash
export seed_prefix=101
export problem=translate_deen_wmt16_bpe32k
export model_name=big_tfm_baseline_df3584_s${seed_prefix}
export data_dir=`pwd`/data_fairseq/$problem
export beam=5
export lenpen=0.6
export round=1

for index in {1..3}
do
export model_dir=train_fairseq/${problem}/${model_name}/model_${index}
export best_file=$model_dir/checkpoint_best.pt
export gen_out=$model_dir/infer_train_b${beam}_lp${lenpen}
fairseq-generate ${data_dir} \
    -s de -t en \
    --path ${best_file} \
    --gen-subset train \
    --max-tokens ${infer_bsz} --beam ${beam}  --lenpen ${lenpen} | dd of=$gen_out && \
grep ^S ${gen_out} | cut -f2- > $gen_out.de && \
grep ^H ${gen_out} | cut -f3- > $gen_out.en && \
done

```

**Step 7**: Merge and filter duplicates with the original dataset

```bash

export ori=raw_data/wmt_ende/train.tok.clean.bpe.32000
export bw_prefix=train_fairseq/translate_deen_wmt16_bpe32k/big_tfm_baseline_df3584_s101/model_
export fw_prefix=train_fairseq/translate_ende_wmt16_bpe32k/big_tfm_baseline_df3584_s100/model_
export prefix=
for i in {1..3}
do
  export prefix=$bw_prefix$i/infer_train_b5_lp0.6:$prefix
done
for i in {1..3}
do
  export prefix=$fw_prefix$i/infer_train_b5_lp0.6:$prefix
done

mkdir -p raw_data/aug_ende_wmt16_bpe32k_s3_r1
python -u combine_corpus.py --src en --tgt de --ori $ori --hypos $prefix --dir raw_data/aug_ende_wmt16_bpe32k_s3_r1 --out train

export out=data_fairseq/translate_ende_aug_b5_r1_s3_nodup_wmt16_bpe32k
cp -r data_fairseq/translate_ende_wmt16_bpe32k $out

fairseq-preprocess --source-lang en --target-lang de \
  --trainpref raw_data/aug_ende_wmt16_bpe32k_s3_r1/train \
  --destdir $out \
  --nwordssrc 0 --nwordstgt 0 \
  --workers 16 \
  --srcdict $out/dict.en.txt --tgtdict $out/dict.fr.txt

```

**Step 8**: Train final models

```bash

export seed_prefix=200
export problem=translate_ende_aug_b5_r1_s3_nodup_wmt16_bpe32k
export model_name=big_tfm_baseline_df3584_s${seed_prefix}
export data_dir=`pwd`/data_fairseq/$problem
export index=1
export model_dir=train_fairseq/${problem}/${model_name}/model_${index}
fairseq-train \
    ${data_dir} \
    -s en -t de \
    --arch transformer_vaswani_wmt_en_de_big --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 0.001 --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07 \
    --dropout 0.3 --attention-dropout 0.1 --weight-decay 0.0 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-update 43000 \
    --keep-last-epochs 10 \
    --save-dir ${model_dir} \
    --ddp-backend no_c10d \
    --seed ${seed_prefix}${index} \
    --max-tokens 3584 \
    --fp16 --update-freq 16 --log-interval 10000 --no-progress-bar

export avg_checkpoint=$model_dir/checkpoint_avg5.pt

# average checkpoints
python average_checkpoints.py \
        --inputs ${model_dir} \
        --num-epoch-checkpoints 5 \
        --checkpoint-upper-bound 10000 \
        --output ${avg_checkpoint}

export gen_out=$model_dir/infer.test.avg5.b5.lp0.6
export ref=${gen_out}.ref
export hypo=${gen_out}.hypo
export ref_atat=${ref}.atat
export hypo_atat=${hypo}.atat
export beam=5
export lenpen=0.6
echo "Finish generating averaged, start generating samples"
fairseq-generate ${data_dir} \
-s en -t de \
--gen-subset test \
--path ${avg_checkpoint} \
--max-tokens 2048 \
--beam ${beam}  \
--lenpen ${lenpen} \
--remove-bpe | dd of=${gen_out}
grep ^T ${gen_out} | cut -f2- > ${ref}
grep ^H ${gen_out} | cut -f3- > ${hypo}

perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' < ${hypo} > ${hypo_atat}
perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' < ${ref} > ${ref_atat}
echo "------ Score BLEU ------------"
$(which fairseq-score) --sys ${hypo_atat} --ref ${ref_atat}
```


## Citation

```bibtex
```


