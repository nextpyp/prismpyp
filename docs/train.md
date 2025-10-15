## 🤖 Train the SimSiam model
   1. Download the ResNet50 pre-trained weights:
   ```bash
   mkdir -p pretrained_weights
   wget https://dl.fbaipublicfiles.com/simsiam/models/100ep/pretrain/checkpoint_0099.pth.tar -P pretrained_weights/
   ```

   2. To train the model on real-domain images, run:
   ```bash
   prismpyp train \
    --output-path output_dir/real \
    --metadata-path metadata_from_nextpyp \
    -a resnet50 \
    --epochs 100 \
    --batch-size 512 \
    --workers 1 \
    --dim 512 \
    --pred-dim 256 \
    --lr 0.05 \
    --resume pretrained_weights/checkpoint_0099.pth.tar \
    --multiprocessing-distributed \
    --dist-url 'tcp://localhost:10057' \
    --world-size 1 \
    --rank 0
   ```

   3. For Fourier-domain images, run:
   ```bash
   prismpyp train \
    --output-path output_dir/fft \
    --metadata-path metadata_from_nextpyp \
    -a resnet50 \
    --epochs 100 \
    --batch-size 512 \
    --workers 1 \
    --dim 512 \
    --pred-dim 256 \
    --lr 0.05 \
    --resume pretrained_weights/checkpoint_0099.pth.tar \
    --multiprocessing-distributed \
    --dist-url 'tcp://localhost:10058' \
    --world-size 1 \
    --rank 0 \
    --use-fft
   ```
  
  If you'd rather start from scratch or use your own pretrained model, omit the ```--resume``` flag or point it to your own ```.pth.tar``` checkpoint.
  
  During training, the per-batch loss and collapse level will be written to the terminal. At the end of training, the following will be available at the output path ```/path/to/output_dir``` specified in the command(s) above:

  * Model checkpoints (```.pth.tar```) for the epoch with the lowest loss (```checkpoints/model_best.pth.tar```), the epoch with the lowest collapse (```checkpoints/model_lowest_collapse.pth.tar```), and the last epoch (```checkpoints/model_last.pth.tar```)
  
  * Loss during training (```total_loss.webp```)
    * This plot is good to look at to confirm that training converged. If it has, then the loss line will stabilize. Lower (more negative) numbers are better.
  * Collapse level during training (```collapse_level.webp```)
    * This plot is good to look at to conform that the model has not learned nonsensical representations of image features. Higher numbers are better.
  * The settings for training (```training_config.yaml```)