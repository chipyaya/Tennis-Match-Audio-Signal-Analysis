declare -a flags=("player_flag" "hand_flag" "dis_flag" "serve_flag")
declare -a models=("knn" "nb" "rf" "svm-linear" "ridge")
declare -a modes=("mfcc" "mfcc-4sec" "mfcc-avg" "mfcc-delta" "mel" "lfcc-4sec" "mfcc-lfcc-4sec")

for flag in "${flags[@]}"
do
    for model in "${models[@]}"
    do
        for mode in "${modes[@]}"
        do
            echo "flag:" "$flag" ", model:" "$model" ", mode:" "$mode" 
            python3 ml.py --mode "$mode" --classifier "$model" --target "$flag"
        done
    done
done