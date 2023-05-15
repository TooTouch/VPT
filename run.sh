model_list="vit_small_patch16_224 vit_base_patch16_224 vit_small_patch16_224_in21k vit_base_patch16_224_in21k"
type_list="shallow deep"
datasets="CIFAR10 CIFAR100"

for d in $datasets
do
    for m in $model_list
    do
        echo "modelname: $m, dataset: $d"
        python main.py \
            --default_setting default_configs.yaml \
            --dataname $d \
            --modelname $m \
            --img_resize 224
        for t in $type_list
        do
            echo "modelname: $m-$t, dataset: $d"
            python main.py \
            --default_setting default_configs.yaml \
            --dataname $d \
            --modelname $m \
            --prompt_type $t \
            --img_resize 224
        done
    done
done


# ablation number of prompt tokens
model="vit_base_patch16_224"
datasets="CIFAR10 CIFAR100"
type_list="shallow deep"
tokens="1 5 10 50 100"

for d in $datasets
do
    for t in $type_list
    do
        for token in $tokens
        do
            echo "modelname: $model-$t, dataset: $d, prompt tokens: $token"
            python main.py \
            --default_setting default_configs.yaml \
            --dataname $d \
            --modelname $model \
            --prompt_type $t \
            --prompt_tokens $token \
            --img_resize 224 
        done
    done
done