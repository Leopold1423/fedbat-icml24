import os
import sys
import copy
import time
import argparse
import warnings
sys.path.append('.')
sys.path.append('..')
warnings.filterwarnings("ignore")
import flwr as fl
from http import client
from collections import OrderedDict
import torch
from train.trainer import train
from model.models import get_model
from utils.logger import get_log
from utils.tool import get_device
from dataloader.dataloader import get_client_train_dataloader
from model.bat import bat_set_parameters, bat_get_parameters


class Client(fl.client.NumPyClient):
    def __init__(self, args):
        self.device = get_device(args.gpu)
        self.pt_path = os.path.join(args.log_dir, "results")
        self.logger = get_log(args.log_dir, args.log_name+"-"+str(args.id))
        self.logger.info(args)
        self.id, self.com_type, self.past_parameters = args.id, args.com_type, None
        self.model_name, self.dataset_name = args.model, args.dataset
        self.part_strategy, self.num_client, self.val_ratio = args.part_strategy, args.num_client, args.val_ratio
        self.model = get_model(self.model_name, self.dataset_name)

        self.keys_client = list(self.model.state_dict().keys())
        if self.com_type == "fedbat":
            self.keys_server = [item for item in self.keys_client if 'update' not in item and 'alpha' not in item and 'rho' not in item]
            self.keys_weight = [item[:-6] for item in self.keys_client if 'update' in item]

    def set_parameters(self, parameters, config):
        if self.com_type == 'fedavg':
            params_zip = zip(self.model.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_zip})
            self.past_parameters = copy.deepcopy(state_dict)
            self.model.load_state_dict(state_dict, strict=True)
        
        if self.com_type == 'fedbat':
            state_dict = copy.deepcopy(self.model.state_dict())
            params_zip = zip(self.keys_server, parameters)
            params_dict = OrderedDict({k: torch.tensor(v) for k, v in params_zip})
            for key in self.keys_server:
                state_dict[key] = params_dict[key]
            self.model.load_state_dict(state_dict, strict=True)
            bat_set_parameters(self.model, config["rho"])

    def get_parameters(self, config):
        if self.com_type == 'fedavg':
            state_dict = copy.deepcopy(self.model.state_dict())
            return [val.cpu().numpy() for _, val in state_dict.items()]
        if self.com_type == 'fedbat':
            bat_get_parameters(self.model)
            state_dict = copy.deepcopy(self.model.state_dict())
            for key in self.keys_weight:
                del state_dict[key+'update']
                del state_dict[key+'alpha']
                del state_dict[key+'alpha0']
                del state_dict[key+'rho']
            return [val.cpu().numpy() for _, val in state_dict.items()]

    def fit(self, parameters, config):
        config["ids"] = [int(num) for num in config["ids"].split('.')]
        self.set_parameters(parameters, config)
        trainloader, valloader = get_client_train_dataloader(self.dataset_name, self.part_strategy, self.num_client, config["ids"][self.id], config["batch_size"], self.val_ratio)
        results = train(self.model, trainloader, valloader, config, self.device)
        self.logger.info("round %d client #%d, val loss: %.4f, val acc: %.4f" %(config["round"], config["ids"][self.id], results["val_loss"], results["val_accuracy"]))
        parameters_prime = self.get_parameters(config)
        return parameters_prime, len(trainloader), results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="client")
    parser.add_argument("--com_type", type=str, default="fedbat", help="type")
    parser.add_argument("--model", type=str, default="bat_cnn4", help="model")
    parser.add_argument("--dataset", type=str, default="fmnist", help="dataset")    
    parser.add_argument("--part_strategy", type=str, default="iid", help="iid")
    parser.add_argument("--num_client", type=int, default=30, choices=range(2, 200), help="num_client")
    parser.add_argument("--id", type=int, default=9, choices=range(0, 99), help="client id")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="dataset")
    parser.add_argument("--gpu", type=int, default=4, help="-1 0 1")
    parser.add_argument("--ip", type=str, default="0.0.0.0:12345", help="server address")
    parser.add_argument("--log_dir", type=str, default="./log/debug/", help="dir")
    parser.add_argument("--log_name", type=str, default="debug", help="log")
    args = parser.parse_args()
    client = Client(args)
    while True:
        flags_path = os.path.join(client.pt_path, "flags_"+str(args.id)+".npy")
        if os.path.exists(flags_path):
            os.remove(flags_path)
            time.sleep(1)
            break
        else:
            time.sleep(1)
    print("start client {}".format(args.id))
    fl.client.start_numpy_client(server_address=args.ip, client=client)
