
# How to 

This markdown file is not a report but only a documentation on how to make everything work. \
This work was only tested with the following configuration:

- OS : Windows 10
- CUDA version : 11.4
- Torch version : 1.10.1
- Python version : 3.9.7
- GPU : GTX 1660Ti (Laptop)

## Installing requirements

Make sure to create a virtual environment and run 
```
pip install -r requirements.txt
```


## (Optional beacause provided) Converting dataset_dev.json and dataset_test.json to the required format

At first, I looked into [Nvidia NeMo Joint Intent Slot Classifier](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/nlp/joint_intent_slot.html) in order to learn more about the subject with a turnkey solution. I liked their data format so I ended up using it for the whole project. 
I provided you with two files ```dict.slots.csv``` and ```dict.intents.csv``` that I wrote by hand given the informations you gave me. 
(PS : we could easily write a parser to obtain those files but : - it is unclear that every intent/slot pair is present in the dataset - to be completely honest if I did now intents and slots would not be in the same order and I would have to retrain !)

Simply run 
```
python to_nemo_dataset_converter.py
```

Four files will then be written into ```./data_dir ```:
- train.tsv
- train_slots.tsv
- test.tsv
- test_slots.tsv

## (Optional because provided) Using GloVe to create vectors.txt for embedding

Embedding layers were initialized with GloVe vectors trained on the dev/train dataset. \

If you are on Linux, you can open ```create_vectorstxt_from_GloVe.ipynb``` and execute the cells. Make sure to modify ```GloVe/demo.sh``` as explained in the notebook. However this is purely optional since the file ```vectors.txt``` is provided here. 

As I was on Windows and I did not have a C compiler ready, I used Google Colab for this part. 

## Run trainings 

Several models are made available with this work :
- One BiLSTM joint Intent/Slot classifier 
- Two separate BiLSTM Intent and Slot Classifier (count as one Intent/Slot model in the end)
- One Encoder/Decoder model with attention (taken from [github](https://github.com/pengshuang/Joint-Slot-Filling), with slight modifications)

For simplicity reasons, and because models differ quite a lot. Each model has its own training file. The first two that I wrote from scratch are of course very much alike, the last one is quite different in its command line arguments etc.. 

### Training for BiLSTM Joint Intent Slot classification

Here are the command lines I personnaly ran to produce all my results : 
```
//Classic training, 50 epochs, cross-entropy loss for both intent and slots classif

cd src

python train.py --epochs 50 --exp_name joint_training_ce_loss 

// 50 epochs, focal_loss for slots classif to tackle class imbalance

python train.py --epochs 50 --exp_name joint_training_focal_loss --focal_loss

// 50 epochs, dice_loss in order to try to maximise f1 score
python train.py --epochs 50 --exp_name joint_training_dice_loss --dice_loss 
 ```

 Best and last models checkpoints are saved into ```checkpoints/{exp_name}/``` 

 ### Training for BiLSTM Separate Intent Slot classification

Same kind of commands as before : 

 ```
//Classic training, 50 epochs, cross-entropy loss for both intent and slots classif

cd src

python train_separate.py --epochs 50 --exp_name separate_training_ce_loss 

// 50 epochs, focal_loss for slots classif to tackle class imbalance

python train_separate.py --epochs 50 --exp_name separate_training_focal_loss --focal_loss

// 50 epochs, dice_loss in order to try to maximise f1 score
python train_separate.py --epochs 50 --exp_name separate_training_dice_loss --dice_loss 
 ```

 Models are also saved in ```checkpoints/{exp_name}/``` 

### Training for Attention based model, from Github

One can run : 
```
python github_joint_slot_train.py --num_epochs 50

```

Models checkpoints are this time saved inside directory ```models```

### Disclaimer
Training command line argument allow to modify models hyperparameters such as embedding dimension, hidden layers dimension etc... 
As those hyperparameters have to be the same when we will load models during evaluation and demo, I advised not to modify the default values because for the sake of simplicity here (in order not to have a ton of command line argument to handle), some of these hyperparameters are hard-coded in evaluation/demo scripts.

## Running evaluation on test set

This will print several metrics as well as save plots in ```plots/{exp_name}```

Joint BiLSTM : 

```
python evaluate.py --ckpt_path ./checkpoints/joint_training_ce_loss/best.pt --exp_name joint_training_ce_loss

python evaluate.py --ckpt_path ./checkpoints/joint_training_focal_loss/best_focal_loss.pt --exp_name joint_training_focal_loss

python evaluate.py --ckpt_path ./checkpoints/joint_training_dice_loss/best.pt --exp_name joint_training_dice_loss

```

Separate BiLSTM : 

```
python evaluate_separate.py --intent_ckpt_path ./checkpoints/separate_training_ce_loss/best_intent.pt --slots_ckpt_path ./checkpoints/separate_training_ce_loss/best_slots.pt --exp_name separate_training_ce_loss

python evaluate_separate.py --intent_ckpt_path ./checkpoints/separate_training_focal_loss/best_intent.pt --slots_ckpt_path ./checkpoints/separate_training_focal_loss/best_slots_focal_loss.pt --exp_name separate_training_focal_loss

python evaluate_separate.py --intent_ckpt_path ./checkpoints/separate_training_dice_loss/best_intent.pt --slots_ckpt_path ./checkpoints/separate_training_dice_loss/best_slots.pt --exp_name separate_training_dice_loss

```

Attention-Based model :

```
python evaluate_encoder_decoder.py --encoder_ckpt_path ./models/jointnlu-encoder.pkl --decoder_ckpt_path ./models/jointnlu-decoder.pkl

```

For the latter there is no exp_name so plots will be saved directly at the root of the folder ```plots```


## Do a demo with a sentence

This file take as input a sentence and prints its intent/slot classification

This time only one file for all models, with a flag stating which model to use. 

```
python demo.py --sentence add Diamonds by Rihanna to the party playlist --model_type joint --ckpt_path {path_to_a_join_checkpoint}

// separate
python demo.py --sentence add Diamonds by Rihanna to the party playlist --model_type separate --ckpt_path {path_to_intent_checkpoint} {path_to_slots_checkpoint}

// separate example 

python demo.py --sentence add Diamonds by Rihanna to the party playlist --model_type separate --ckpt_path ./checkpoints/separate_training_ce_loss/best_intent.pt ./checkpoints/separate_training_ce_loss/best_slots.pt

//attention
python demo.py --sentence add Diamonds by Rihanna to the party playlist --model_type attention --ckpt_path {encoder_path} {decoder_path}

//attention example
python demo.py --sentence add Diamonds by Rihanna to the party playlist --model_type attention --ckpt_path ./models/jointnlu-encoder.pkl ./models/jointnlu-decoder.pkl

``` 



