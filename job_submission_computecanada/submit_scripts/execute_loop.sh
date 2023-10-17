#!/bin/bash

# exp=0   # multi-stage-cnn                 time 6hrs    --> 5.5 min per epoch (04:30:00 for 50 epochs)
# exp=2   # End-to-end loss-end-to-end      time 20 hrs  --> 0.85hrs per epoch (45:00:00 for 50 epochs)

# exp=3   # End-to-end loss-transformer     calculating time per epoch job: 41634895

# exp=1   # multi-stage-transformer         calculating time per epoch job: 41634896

# exp=4   # multi-stage-loss-end-to-end     time --> 0.85hrs per epoch (45:00:00 for 50 epochs)

# train
for i in {0..0}
do
    echo "executing train in model $i"
    sbatch execute_train.sh $i 1 $exp
    sleep 5
done

# # test
# for i in {0..34}
# do
#     echo "executing test in model $i"
#     sbatch execute_test.sh $i 0 $exp
#     sleep 5
# done
