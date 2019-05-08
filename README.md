# NICE
This code implements NICE papper
List of links to pre-trained models:

DeepISP
--------
http://www.mediafire.com/file/15anb9x44nxkkke/best_checkpoint.pth.tar/file

ResNet18-CIFAR10
-----------------
http://www.mediafire.com/file/legt0epbrw8qii3/model_best.pth.tar/file

ResNet18-ImageNet
-----------------
http://www.mediafire.com/file/l5qbobd2mm5wry5/model_best.pth.tar/file

ResNet34-ImageNet
-----------------
http://www.mediafire.com/file/et7mvajxamm8sup/model_best.pth.tar/file

ResNet50-ImageNet
-----------------
http://www.mediafire.com/file/93f7s5h66d6n8z1/model_best.pth.tar/file

Running instructions
--------------------
--------------------

DeepISP
-------
python3 deep_isp_main.py --batch_size=16 --resume=<path to model> --quant=True --quant_bitwidth=4 --inject_noise=True --inject_act_noise=False --act_quant=True --act_bitwidth=8 --quant_epoch_step=6 --quant_start_stage=0 --epochs=500 --learning_rate=3e-5 --gpus 0 --set_gpu=True --stage_only_clamp=False

#### Params ####

*  ```--seed N``` - Random seed
* ```--start-epoch N``` - manual epoch number (useful on restarts)
* ```--epochs EPOCHS```       Number of epochs to train.
* ```--batch_size```  BATCH_SIZE
                          Number of epochs to train.
* ```--num_denoise_layers``` NUM_DENOISE_LAYERS
                          num of layers.
* ```--learning_rate```  LEARNING_RATE, 
* ```-lr``` LEARNING_RATE
                          The learning rate.
* ```--decay DECAY```
* ``` -d DECAY``` 
                          Weight decay (L2 penalty).
* ```--gpus GPUS``` List of GPUs used for training - e.g 0,1,3. Remove to run on CPU
* ```--datapath DATAPATH```   Path to MSR-Demosaicing dataset
* ```--resume RESUME```       Path to checkpoint file
* ```--out_dir OUT_DIR```     Path to save model and results
* ```--quant_epoch_step QUANT_EPOCH_STEP```
                          quant_bitwidth.
* ```--num_workers NUM_WORKERS```
                          Num of workers for data.
* ```--quant_start_stage QUANT_START_STAGE```
                          Num of workers for data.
* ```--inject_noise INJECT_NOISE```
                          use preproccesing for the grad
* ```--show_test_result SHOW_TEST_RESULT```
                          show figures of test result
* ```--quant QUANT```        use preproccesing for the grad
* ```--quant_bitwidth QUANT_BITWIDTH```
                          quant_bitwidth.
* ```--inject_act_noise INJECT_ACT_NOISE```
                          use preproccesing for the grad
* ```--act_quant ACT_QUANT```
                          use preproccesing for the grad
* ```--act_bitwidth ACT_BITWIDTH```
                          quant_bitwidth.
*  ```--step STEP```           amount of split the layer in quant.

* ```--set_gpu SET_GPU```  - show figures of test result (False to run on CPU)
* ```--adaptive_lr ADAPTIVE_LR```
                          show figures of test result
* ```--enable_decay ENABLE_DECAY```
                          decay_enable
* ```--weight_relu WEIGHT_RELU```
                          weight_relu
* ```--weight_grad_after_quant WEIGHT_GRAD_AFTER_QUANT```
                          weight_grad_after_quant
* ``` --random_inject_noise RANDOM_INJECT_NOISE```
                          random_inject_noise
* ```--stage_only_clamp STAGE_ONLY_CLAMP```
                          stage_only_clamp
*  ```--wrpn WRPN```           wrpn quantization

*  ```--copy_statistics COPY_STATISTICS```
                          copy_statistics
* ```--quant_decay QUANT_DECAY```
                          quant decay.
* ```--val_part VAL_PART```   quant decay.

ResNet18 CIFAR10
----------------
python3 main.py --model resnet --depth 18 --bitwidth <weight bitwidth> --act-bitwidth <activation bitwidth> --step 21 --gpus 0 --epochs 120 -b 256 --dataset cifar10 --start-from-zero --resume <path to model> --learning_rate=0.01 --quant_start_stage=0 --quant_epoch_step=3 --datapath <path to CIFAR10 dataset> --schedule 300

ResNet18 ImageNet
-----------------
python main.py --model resnet --depth 18 --bitwidth <weight bitwidth> --act-bitwidth <activation bitwidth> --step 21 --schedule 42 110 -lr 1e-4 --decay 4e-5 --gamma 0.93451921456 --gpus 0,1 --epochs 120 -b 128 --dataset imagenet --datapath <Path to ImageNet dataset> --resume <Path to model file> --quant_start_stage=0 --quant_epoch_step=2 --no-quant-edges --noise_mask 0.05 --act_stats_batch_size 64

ResNet34 ImageNet
-----------------
python main.py --model resnet --depth 34 --bitwidth <weight bitwidth> --act-bitwidth <activation bitwidth> --step 37 --schedule 73 110 -lr 1e-4 --decay 4e-5 --gamma 0.88296999554 --gpus 0,1 --epochs 120 -b 128 --dataset imagenet --datapath <Path to ImageNet dataset> --resume <Path to model file> --quant_start_stage=0 --quant_epoch_step=2 --no-quant-edges --noise_mask 0.05 --act_stats_batch_size 64

ResNet50 ImageNet
-----------------
python main.py --model resnet --depth 50 --bitwidth <weight bitwidth> --act-bitwidth <activation bitwidth> --step 37 --schedule 83 110 -lr 1e-4 --decay 4e-5 --gamma 0.843190929--gpus 0,1 --epochs 120 -b 128 --dataset imagenet --datapath <Path to ImageNet dataset> --resume <Path to model file> --quant_start_stage=0 --quant_epoch_step=1.5 --no-quant-edges --noise_mask 0.05 --act_stats_batch_size 64







