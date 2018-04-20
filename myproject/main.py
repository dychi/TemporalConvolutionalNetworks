from data_loader.tcn_data_loader import DataGenerator
from models.keras_model import ED_TCN
from trainers.trainer import EDTCNTrainer
from utils.config import process_config
from utils.dirs import create_dirs
from utils.utils import get_args

def main():
    try:
        args = get_args()
        config = process_config(args.config)
    except:
        print("missing or invalid arguments")
        exit(0)
    
    print("Create the data generator.")
    data_generator = DataGenerator(config)
    
    print("Create the model.")
    model = ED_TCN(config)

    print("Create the trainer.")
    trainer = EDTCNTrainer(model.model, data_loader.get_train_data(), config)
    
    print('Start training the model.')
    trainer.train()
    
    # print('Visualize the losses.')
    # trainer.visualize()
   
if __name__ == '__main__()':
    main()
