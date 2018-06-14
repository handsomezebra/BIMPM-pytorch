# BIMPM-pytorch
Re-implementation of [BIMPM](https://arxiv.org/abs/1702.03814)(Bilateral Multi-Perspective Matching for Natural Language Sentences, Zhiguo Wang et al., IJCAI 2017) on Pytorch.
The code is forked from [this version](https://github.com/galsang/BIMPM-pytorch) with several improvements. 

## Results

Dataset: [SNLI](https://nlp.stanford.edu/projects/snli/)

| Model        |  ACC(%)   | 
|--------------|:----------:|
| BiMPM paper (Single BiMPM)	|  86.9    |    
| galsang's version |			 86.5 |  
| Re-implementation |			 86.2 |  

Dataset: [Quora](https://drive.google.com/file/d/0B0PlTAo--BnaQWlsZl9FZ3l1c28/view) (provided by the authors)

| Model        |  ACC(%)   | 
|--------------|:----------:|
| BiMPM paper (Single BiMPM)     	|  88.17   |
| galsang's version 			| 87.3 |  
| Re-implementation 			| 88.0 |  

## Development Environment
- OS: Ubuntu 16.04 LTS (64bit)
- Language: Python 3.6.5.

## Requirements

Please install the following library requirements specified in the **requirements.txt** first.

    torch==0.4.0
    torchtext==0.3.0

## Training

> python train.py --help

	usage: train.py [-h] [--data-type DATA_TYPE] [--batch-size BATCH_SIZE] [--epoch EPOCH] [--gpu GPU]
                    [--print-freq PRINT_FREQ] [--dropout DROPOUT] [--learning-rate LEARNING_RATE]
                    [--max-sent-len MAX_SENT_LEN] [--max-word-len MAX_WORD_LEN] [--num-perspective NUM_PERSPECTIVE]
                    [--word-dim WORD_DIM] [--char-dim CHAR_DIM] [--char-lstm-dim CHAR_LSTM_DIM]
                    [--context-lstm-dim CONTEXT_LSTM_DIM] [--context-layer-num CONTEXT_LAYER_NUM]
                    [--aggregation-lstm-dim AGGREGATION_LSTM_DIM] [--aggregation-layer-num AGGREGATION_LAYER_NUM]
                    [--wo-char] [--wo-full-match] [--wo-maxpool-match] [--wo-attentive-match] [--wo-max-attentive-match]
    
    optional arguments:
      -h, --help                            show this help message and exit
      --data-type DATA_TYPE                 data type, available: SNLI or Quora (default: Quora)
      --batch-size BATCH_SIZE               batch size (default: 48)
      --epoch EPOCH                         number of epoch (default: 20)
      --gpu GPU                             gpu id, -1 to use cpu (default: 0)
      --print-freq PRINT_FREQ               number of batches to evaluate and checkpoint (default: 500)
      --dropout DROPOUT                     dropout rate (default: 0.1)
      --learning-rate LEARNING_RATE         learning rate (default: 0.001)
      --max-sent-len MAX_SENT_LEN           max number of words per sentence, if -1, it accepts any length (default: -1)
      --max-word-len MAX_WORD_LEN           max number of chars per word, if -1, it accepts any length (default: -1)
      --num-perspective NUM_PERSPECTIVE     number of perspective (default: 20)
      --word-dim WORD_DIM                   word embedding dimension (default: 300)
      --char-dim CHAR_DIM                   character embedding dimension (default: 20)
      --char-lstm-dim CHAR_LSTM_DIM         character LSTM's hidden size (default: 100)
      --context-lstm-dim CONTEXT_LSTM_DIM   context LSTM's hidden size (default: 100)
      --context-layer-num CONTEXT_LAYER_NUM
                                            context LSTM's layer number (default: 2)
      --aggregation-lstm-dim AGGREGATION_LSTM_DIM
                                            aggregation LSTM's hidden size (default: 100)
      --aggregation-layer-num AGGREGATION_LAYER_NUM
                                            aggregation LSTM's layer number (default: 2)
      --wo-char                             whether to learn without character features (default: False)
      --wo-full-match                       whether to learn without full match (default: False)
      --wo-maxpool-match                    whether to learn without max pool match (default: False)
      --wo-attentive-match                  whether to learn without attentive match (default: False)
      --wo-max-attentive-match              whether to learn without max attentive match (default: False)



## Test

> python test.py --help

	usage: test.py [-h] --model-path MODEL_PATH [--batch-size BATCH_SIZE] [--data-type DATA_TYPE] [--gpu GPU]
                   [--max-sent-len MAX_SENT_LEN] [--max-word-len MAX_WORD_LEN]
    
    optional arguments:
      -h, --help                   show this help message and exit
      --model-path MODEL_PATH      path of trained model (default: None)
      --batch-size BATCH_SIZE      batch size (default: 16)
      --data-type DATA_TYPE        data type, available: SNLI or Quora (default: Quora)
      --gpu GPU                    gpu id, -1 to use cpu (default: 0)
      --max-sent-len MAX_SENT_LEN  max number of words per sentence, if -1, it accepts any length (default: -1)
      --max-word-len MAX_WORD_LEN  max number of chars per word, if -1, it accepts any length (default: -1)


