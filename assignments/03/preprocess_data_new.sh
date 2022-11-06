#!/bin/bash
# -*- coding: utf-8 -*-

set -e

pwd=`dirname "$(readlink -f "$0")"`
base=$pwd/../..
src=fr
tgt=en
data=$base/data/$tgt-$src/

# change into base directory to ensure paths are valid
cd $base

# create preprocessed directory
mkdir -p $data/preprocessed/bpe/

# normalize and tokenize raw data
cat $data/raw/train.$src | perl moses_scripts/normalize-punctuation.perl -l $src | perl moses_scripts/tokenizer.perl -l $src -a -q > $data/preprocessed/bpe/train.$src.p
cat $data/raw/train.$tgt | perl moses_scripts/normalize-punctuation.perl -l $tgt | perl moses_scripts/tokenizer.perl -l $tgt -a -q > $data/preprocessed/bpe/train.$tgt.p

# train truecase models
perl moses_scripts/train-truecaser.perl --model $data/preprocessed/bpe/tm.$src --corpus $data/preprocessed/bpe/train.$src.p
perl moses_scripts/train-truecaser.perl --model $data/preprocessed/bpe/tm.$tgt --corpus $data/preprocessed/bpe/train.$tgt.p

# apply truecase models to splits
cat $data/preprocessed/bpe/train.$src.p | perl moses_scripts/truecase.perl --model $data/preprocessed/bpe/tm.$src > $data/preprocessed/bpe/train.$src.trc
cat $data/preprocessed/bpe/train.$tgt.p | perl moses_scripts/truecase.perl --model $data/preprocessed/bpe/tm.$tgt > $data/preprocessed/bpe/train.$tgt.trc

# Learn byte pair encoding on the concatenation of the training text, and get resulting vocabulary for each:
python subword_nmt/learn_joint_bpe_and_vocab.py --input $data/preprocessed/bpe/train.$src.trc $data/preprocessed/bpe/train.$tgt.trc -s 3800 -o $data/preprocessed/bpe/$src$tgt.bpe --write-vocabulary $data/preprocessed/bpe/train.vocab.$src $data/preprocessed/bpe/train.vocab.$tgt

# re-apply byte pair encoding with vocabulary filter:
python3 subword_nmt/apply_bpe.py -c $data/preprocessed/bpe/$src$tgt.bpe --vocabulary $data/preprocessed/bpe/train.vocab.$src < $data/preprocessed/bpe/train.$src.trc > $data/preprocessed/bpe/train.$src
python3 subword_nmt/apply_bpe.py -c $data/preprocessed/bpe/$src$tgt.bpe --vocabulary $data/preprocessed/bpe/train.vocab.$tgt < $data/preprocessed/bpe/train.$tgt.trc > $data/preprocessed/bpe/train.$tgt


# prepare remaining splits with learned models
for split in valid test tiny_train
do
    cat $data/raw/$split.$src | perl moses_scripts/normalize-punctuation.perl -l $src | perl moses_scripts/tokenizer.perl -l $src -a -q | perl moses_scripts/truecase.perl --model $data/preprocessed/bpe/tm.$src | python3 subword_nmt/apply_bpe.py -c $data/preprocessed/bpe/$src$tgt.bpe --vocabulary $data/preprocessed/bpe/train.vocab.$src > $data/preprocessed/bpe/$split.$src
    cat $data/raw/$split.$tgt | perl moses_scripts/normalize-punctuation.perl -l $tgt | perl moses_scripts/tokenizer.perl -l $tgt -a -q | perl moses_scripts/truecase.perl --model $data/preprocessed/bpe/tm.$tgt | python3 subword_nmt/apply_bpe.py -c $data/preprocessed/bpe/$src$tgt.bpe --vocabulary $data/preprocessed/bpe/train.vocab.$tgt > $data/preprocessed/bpe/$split.$tgt
done

# remove tmp files
rm $data/preprocessed/bpe/train.$src.p
rm $data/preprocessed/bpe/train.$tgt.p

# preprocess all files for model training
python preprocess.py --target-lang $tgt --source-lang $src --dest-dir $data/prepared/bpe --train-prefix $data/preprocessed/bpe/train --valid-prefix $data/preprocessed/bpe/valid --test-prefix $data/preprocessed/bpe/test --tiny-train-prefix $data/preprocessed/bpe/tiny_train --threshold-src 1 --threshold-tgt 1 --num-words-src 4000 --num-words-tgt 4000

echo "done!"