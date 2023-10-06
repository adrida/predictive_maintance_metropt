# metro-anomaly-ECML2023

Repository to store code necessary to reproduce the experiments in the paper

## Trial to replicate results and reproduce training

Download dataset: https://zenodo.org/record/7766691

### Step 1: Generate processing data:

`python generate_chunks.py`

### Step 3: Run training:

python train_chunks.py \
-model LSTMDiscriminator_TCN \
-encoder TCN \
-decoder TCN \
-tcn_hidden 30 \
-tcn_layers 10 \
-tcn_kernel 3 \
-disc_hidden 32 \
-disc_layers 3 \
-epochs 150 \
-lr 1e-3 \
-disc_lr 1e-3 \
-use_discriminator


`python3 train_chunks.py -encoder TCN -decoder TCN -use_discriminator -model LSTMDiscriminator_TCN -epochs 150 -lr 1e-3 -batch_size 64 -feats analog`

### Step 4: Run failure detection expe:

Change file namnes with savec pickle reconstruction losses in folder `results`

run `python3 failure_detection.py` 

if it doesn't work try: `python3 failure_detection.py analog` 


usage: train_chunks.py [-h] [-lr LR] [-disc_lr DISC_LR] [-epochs EPOCHS] [-l2reg WEIGHT_DECAY] [-critic_iterations CRITIC_ITERATIONS] [-gradient_penalty GP_HYPERPARAM] [-WAEreg WAE_REGULARIZATION_TERM] [-dropout DROPOUT] [-embedding EMBEDDING] [-hidden HIDDEN_DIMS] [-n_layers LSTM_LAYERS] [-batch_size BATCH_SIZE] [-disc_hidden DISC_HIDDEN] [-disc_layers DISC_LAYERS] [-tcn_hidden TCN_HIDDEN] [-tcn_layers TCN_LAYERS] [-tcn_kernel TCN_KERNEL] [-sw SPARSITY_WEIGHT] [-sp SPARSITY_PARAMETER] [-att_heads NHEADS] [-feats {analog,digital,all,noflow}] [-SI SUCCESSIVE_ITERS] [-delta_worse DELTA_WORSE] [-delta_better DELTA_BETTER] -model

{lstm_ae,lstm_sae,multi_enc_sae,multi_enc_ae,lstm_all_layer_sae,diff_comp_sae,diff_comp_ae,GAN,SimpleDiscriminator,LSTMDiscriminator,ConvDiscriminator,tcn_ae,alt_lstm_ae,SimpleDiscriminator_TCN,LSTMDiscriminator_TCN,ConvDiscriminator_TCN} [-encoder {LSTM,TCN}] [-decoder {LSTM,TCN}] [-recons_error {dtw,mse}] [-dtw_local DTW_LOCAL_SIZE] [-separate_comp] [-init INIT_LOOP] [-end END_LOOP] [-force-training] [-sensor SENSOR] [-train_tensor TRAIN_TENSOR] [-test_tensor TEST_TENSOR] [-use_discriminator] train_chunks.py: error: argument -model: invalid choice: 'wagwan' (choose from 'lstm_ae', 'lstm_sae', 'multi_enc_sae', 'multi_enc_ae', 'lstm_all_layer_sae', 'diff_comp_sae', 'diff_comp_ae', 'GAN', 'SimpleDiscriminator', 'LSTMDiscriminator', 'ConvDiscriminator', 'tcn_ae', 'alt_lstm_ae', 'SimpleDiscriminator_TCN', 'LSTMDiscriminator_TCN', 'ConvDiscriminator_TCN')
               
                       
### Suggested configuration from the paper:
Considering the challenges faced by our proposed models to identify failures without false alarms when using compressor cycles as the input data, we change our focus towards detecting failures in the input stream by considering chunks of data spanning 30 minutes. As before, we train our models on the first month of data, which is assumed to be a normal period, and test on the following two months. We use the same optimizer to train the models, Adam [14], and batch sizes of 64. After hyperparameter tuning, the following architectures yielded the top F1 scores:

– LSTM autoencoder : both encoder and decoder are composed of 5 layers of stacked LSTMs, each layer with 4 hidden units. Trained for 150 epochs with a learning rate of 1 × 10−3

– TCN autoencoder : both encoder and decoder are composed of 8 convolution layers, each with 6 hidden units and a kernel size of 7 that leads to receptive field size of ≈ 1500, a value slightly smaller than the length of each data chunk. We use a 4-dimensional latent space. Trained for 100 epochs with a learning rate of 1 × 10−3

– WAE-GAN : both encoder and decoder are TCNs with 10 convolution layers, each layer with 30 hidden units and a kernel size of 3 (resulting in a receptive field size of ≈ 2000). The discriminator network has a different architecture from the encoder and decoder, it is composed of 3 LSTM layers with 32 hidden units. Trained for 150 epochs, the encoder, decoder and discriminator optimizers with a learning rate of 1 × 10−3

Figure 3 shows the probability of failure outputted by each model over time for the test set. In this case, instead of a data point representing a compressor cycle, it represents an interval 5 minutes long to simulate the synchronization rate between our servers and the data arriving from the sensors installed in the APU of the train. 
We observe that the WAE-GAN model is able to identify the two failures at least two hours before the LPS signal is active and is able to do so without generating any false alarm (achieving a perfect F1 score). On the other hand, the TCN autoencoder is also able to detect both failures early but generates two false alarms (F1 of 0.67). The LSTM autoencoder is able to detect both failures without generating a false alarm, but is unable to detect the first failure before the LPS signal
