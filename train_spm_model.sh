spm_train \
    --input ./mn_corpus/all.txt --model_prefix=30k-clean-mn --vocab_size=30000 \
    --pad_id=0 --unk_id=1 --eos_id=-1 --bos_id=-1 \
    --control_symbols=[CLS],[SEP],[MASK] \
    --user_defined_symbols="(,),\",-,.,–,£,€" \
    --shuffle_input_sentence=true --input_sentence_size=10000000 \
    --character_coverage=0.99995 --model_type=unigram

mv 30k-clean-mn* sp_models/
