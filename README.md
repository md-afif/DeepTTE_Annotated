# DeepTTE_Annotated

Original repository: [GitHub](https://github.com/UrbComp/DeepTTE)

This is an annotated version of DeepTTE for a more detailed walk-through of the model based on personal understanding. Code has been updated to run on Python 3.7.7 and PyTorch 1.3.1, with the deprecated functions replaced. Rest of the code is unchanged from original author. 

## Training
```
python main.py --task train --batch_size 10 --result_file ./result/deeptte.res --pooling_method attention --kernel_size 3 --alpha 0.1 --log_file train_log
```

## Testing 
```
python main.py --task test --weight_file ./saved_weights/weight --batch_size 10 --result_file ./result/deeptte.res --pooling_method attention --kernel_size 3 --alpha 0.1 --log_file test_log
```
