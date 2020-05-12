basedir=$1
max_seq_len=$2

for path in $basedir/*.txt
    do
        file_name=${path##*/}
        new_name="$(cut -d'.' -f1 <<<"$file_name")"-maxseq-${max_seq_len}.tf_record
        # echo ${path}
        # echo ${file_name}
        echo from ${basedir}/${file_name}  --  to ${basedir}/${new_name}

        python3 -m albert.create_pretraining_data \
            --input_file=${basedir}/${file_name} \
            --output_file=${basedir}/${new_name} \
            --spm_model_file=./sp_models/30k-clean-mn.model \
            --vocab_file=./sp_models/30k-clean-mn.vocab \
            --do_lower_case=False \
            --max_seq_length=${max_seq_len}\
            --max_predictions_per_seq=20 \
            --masked_lm_prob=0.15 \
            --random_seed=12345 \
            --dupe_factor=5 \
            --do_whole_word_mask=False \
            --do_permutation=False \
            --favor_shorter_ngram=False \
            --random_next_sentence=False

    done
