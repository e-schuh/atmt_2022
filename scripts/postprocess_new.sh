infile=$1
outfile=$2
lang=$3

## Add command to decode BPE: sed -r 's/(@@ )|(@@ ?$)//g'
cat $infile | sed -r 's/(@@ )|(@@ ?$)//g' | perl moses_scripts/detruecase.perl | perl moses_scripts/detokenizer.perl -q -l $lang > $outfile