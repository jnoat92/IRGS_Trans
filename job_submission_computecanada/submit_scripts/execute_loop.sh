#!/bin/bash

# exp=1   # multi-stage-cnn                 time --> 5.5 min per epoch (04:30:00 for 50 epochs)     trained & tested!

# exp=2   # End-to-end loss-transformer     time --> 0.80hrs per epoch (40:00:00 for 50 epochs)     def-l44xu-ab beluga7 12 17 22 29 30 32 33 34 

# exp=3   # multi-stage-transformer         time --> 0.85hrs per epoch (45:00:00 for 50 epochs)     def-l44xu-ab beluga [12-23]
                                                                                                #   def-dclausi  beluga [24-34]

# exp=4   # multi-stage-loss-end-to-end     time --> 0.85hrs per epoch (45:00:00 for 50 epochs)     trained & tested!

exp=5   # End-to-end loss-end-to-end      time --> 0.85hrs per epoch (45:00:00 for 50 epochs)       def-dclausi  graham [0-28]
                                                                                                #   def-l44xu-ab graham [30-34]



# train
for i in {0..28}
do
    echo "executing train in model $i"
    sbatch execute_train.sh $i 1 $exp
    sleep 5
done

# # test
# for i in {0..0}
# do
#     echo "executing test in model $i"
#     sbatch execute_test.sh $i 0 $exp
#     sleep 5
# done
