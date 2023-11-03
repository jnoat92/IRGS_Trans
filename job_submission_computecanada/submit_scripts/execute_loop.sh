#!/bin/bash

# exp=1   # multi-stage-cnn                 time --> 5.5 min per epoch (04:30:00 for 50 epochs)     trained & tested!  beluga

# exp=2   # End-to-end loss-transformer     time --> 0.80hrs per epoch (40:00:00 for 50 epochs)     trained & tested!  beluga

# exp=3   # multi-stage-transformer         time --> 0.85hrs per epoch (45:00:00 for 50 epochs)     trained & tested!  beluga

# exp=4   # multi-stage-loss-end-to-end     time --> 0.85hrs per epoch (45:00:00 for 50 epochs)     trained & tested!  graham

# exp=5   # End-to-end loss-end-to-end      time --> 0.85hrs per epoch (45:00:00 for 50 epochs)      trained & tested! beluga



# # train
# for i in {0..28}
# do
#     echo "executing train in model $i"
#     sbatch execute_train.sh $i 1 $exp
#     sleep 5
# done

# test
for i in {0..34}
do
    echo "executing test in model $i"
    sbatch execute_test.sh $i 0 $exp
    sleep 5
done
