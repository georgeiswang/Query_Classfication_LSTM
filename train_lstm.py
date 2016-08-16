import sys
from config_reader import ConfigReader
from train_dnn_lstm import TrainDNN
from train_autoencoder import TrainAutoencoder

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print 'Usage: python train_lstm.py config_file'
        exit(1)
    config_file = sys.argv[1]
    conf = ConfigReader(config_file)
    conf.ReadConfig()
    #TrainAutoencoder(conf)
    TrainDNN(conf)
