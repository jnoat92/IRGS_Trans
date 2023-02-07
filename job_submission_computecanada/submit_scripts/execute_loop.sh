#!/bin/bash

exp=0   # multi-stage-cnn
# exp=1   # multi-stage-transformer
# exp=2   # End-to-end loss-end-to-end
# exp=3   # End-to-end loss-end-to-end

# # train
for i in {0..20}
do
    echo "executing train in model $i"
    sbatch execute_train.sh $i 1 $exp
    sleep 5
done

# # # test
# for i in {0..20}
# do
#     echo "executing test in model $i"
#     sbatch execute_test.sh $i 0 $exp
#     sleep 5
# done
