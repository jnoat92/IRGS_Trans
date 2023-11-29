#!/bin/bash

# exp=1   # multi-stage-cnn                 time --> 5.5 min per epoch (02:55:00 for 50 epochs)     training!  graham (def-dclausi)

# exp=2   # End-to-end loss-transformer     time --> 1.236hrs per epoch (65:00:00 for 50 epochs)    

# exp=3   # multi-stage-transformer         time --> 1.236hrs per epoch (65:00:00 for 50 epochs)     

# exp=4   # multi-stage-loss-end-to-end     time --> 1.236hrs per epoch (65:00:00 for 50 epochs)     

exp=5   # End-to-end loss-end-to-end      time --> 1.236hrs per epoch (65:00:00 for 50 epochs)    training!  graham (def-l44xu-ab)



# train
for i in {1..32}
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
