#!/bin/bash

exp=1   # multi-stage-cnn                 time --> 5.5 min per epoch (02:55:00 for 50 epochs)     trained & tested!  beluga (def-dclausi)

# exp=2   # End-to-end loss-transformer     time --> 1.236hrs per epoch (65:00:00 for 50 epochs)    trained & tested!  beluga (def-dclausi)

# exp=3   # multi-stage-transformer         time --> 1.236hrs per epoch (65:00:00 for 50 epochs)    trained & tested!  beluga (def-l44xu-ab)

# exp=4   # End-to-end loss-end-to-end      time --> 1.236hrs per epoch (65:00:00 for 50 epochs)    trained & tested!  beluga (def-l44xu-ab) 

# exp=5   # multi-stage-loss-end-to-end     time --> 1.236hrs per epoch (65:00:00 for 50 epochs)    trained & tested!  beluga (def-dclausi) 


# # train
# for i in {0..32}
# do
#     echo "executing train in model $i"
#     sbatch execute_train.sh $i 0 $exp
#     sleep 5
# done

# # test
# for i in {0..32}
# do
#     echo "executing test in model $i"
#     sbatch execute_test.sh $i 1 $exp
#     sleep 5
# done

# # buffer metrics
# for i in {0..32}
# do
#     echo "calculating buffer metrics in model $i"
#     sbatch buffer_job.sh $i 2 $exp
#     sleep 5
# done

# Combine results (only for protocols 4 and 5)
for i in {0..32}
do
    echo "calculating buffer metrics in model $i"
    sbatch combine_outputs_job.sh $i 3 $exp
    sleep 5
done
