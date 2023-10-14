#!/bin/bash

# exp=0   # multi-stage-cnn                   time 5hrs
# exp=1   # multi-stage-transformer         time 10 hrs
# exp=2   # End-to-end loss-end-to-end      time 20 hrs
# exp=3   # End-to-end loss-transformer

# exp=5   # multi-stage-loss-end-to-end     time 20 hrs

# # train
# for i in {0..34}
# do
#     echo "executing train in model $i"
#     sbatch execute_train.sh $i 1 $exp
#     sleep 5
# done

# # test
for i in {0..34}
do
    echo "executing test in model $i"
    sbatch execute_test.sh $i 0 $exp
    sleep 5
done
