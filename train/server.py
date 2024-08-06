import os
import sys
import warnings
sys.path.append('.')
sys.path.append('..')
warnings.filterwarnings("ignore")
import torch
import random
import argparse
import numpy as np
from typing import Dict
from collections import OrderedDict
import flwr as fl
from flwr.server.strategy import FedAvg
from train.trainer import test
from model.models import get_model
from utils.logger import get_log
from utils.tool import get_device, save_npy_record
from dataloader.dataloader import get_server_test_dataloader

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="server")
    parser.add_argument("--com_type", type=str, default="fedavg", help="communication type")
    parser.add_argument("--rho", type=float, default=6, help="alpha coefficient")
    parser.add_argument("--phi", type=float, default=0.5, help="warmup ratio")
    
    parser.add_argument("--model", type=str, default="resnet10", help="model")
    parser.add_argument("--dataset", type=str, default="cifar10", help="dataset")
    parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
    parser.add_argument("--momentum", type=float, default=0, help="momentum")
    parser.add_argument("--l2", type=float, default=0.0, help="l2")
    parser.add_argument("--rounds", type=int, default=100, help="rounds")
    parser.add_argument("--epochs", type=int, default=3, help="epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="batch_size")
    parser.add_argument("--save_round", type=int, default=0, help="save_round")
    parser.add_argument("--num_per_round", type=int, default=10, choices=range(2, 200), help="num_per_round") 
    parser.add_argument("--num_client", type=int, default=30, choices=range(2, 200), help="num_client")  
    parser.add_argument("--gpu", type=int, default=4, help="-1 0 1")
    parser.add_argument("--ip", type=str, default="0.0.0.0:12345", help="server address")
    parser.add_argument("--log_dir", type=str, default="./log/debug/", help="dir")
    parser.add_argument("--log_name", type=str, default="debug", help="log")
    args = parser.parse_args()

    device = get_device(args.gpu)
    logger = get_log(args.log_dir, args.log_name)
    logger.info(args)
    pt_path = os.path.join(args.log_dir, "results")
    os.makedirs(pt_path, exist_ok=True)

    model = get_model(args.model, args.dataset)
    model_parameters = [val.cpu().numpy() for _, val in model.state_dict().items()]
    print(model)
    record = {"accuracy":[], "loss":[]}
    com_dict={"ids": "0.1.2.3.4.5.6.7.8.9"}

    def fit_config(server_round: int):
        config = {
            "com_type": args.com_type,
            "round": server_round,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "momentum":args.momentum,
            "l2": args.l2,
            "ids": com_dict["ids"], 
            
            "rho": args.rho,
            "phi": args.phi,
        }
        return config

    def evaluate_config(server_round: int):
        config = {
            "round": server_round,
            "batch_size": args.batch_size,
        }
        return config
    
    def get_evaluate_fn(model, dataset: str):
        test_loader = get_server_test_dataloader(dataset, batch_size=args.batch_size)
        def evaluate(server_round: int, parameters: fl.common.NDArrays, config: Dict[str, fl.common.Scalar]):
            if server_round==0:
                print("starting server evalutation...")
                loss, accuracy = 0.0, 0.0
                record["accuracy"].append(accuracy)
                record["loss"].append(loss)
                logger.info("round %d - server test loss:%.4f; acc:%.4f" %(server_round, loss, accuracy))
                ids = random.sample(list(range(args.num_client)), args.num_per_round)
                com_dict["ids"] = '.'.join(map(str, ids))
                return loss, {"accuracy": accuracy}

            params_dict = zip(model.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            model.load_state_dict(state_dict, strict=True)
            print("starting server evalutation...")
            loss, accuracy = test(model, test_loader, None, device)
            record["accuracy"].append(accuracy)
            record["loss"].append(loss)
            logger.info("round %d - server test loss:%.4f; acc:%.4f" %(server_round, loss, accuracy))
            if args.save_round:
                torch.save(model.state_dict(), os.path.join(pt_path, str(server_round)+".pt"))
            if accuracy >= np.max(np.array(record["accuracy"])):
                torch.save(model.state_dict(), os.path.join(pt_path, "best.pt"))
            
            ids = random.sample(list(range(args.num_client)), args.num_per_round)
            com_dict["ids"] = '.'.join(map(str, ids))
            return loss, {"accuracy": accuracy}
        return evaluate

    strategy = FedAvg(
        fraction_fit=0.1,
        fraction_evaluate=0.0,
        min_fit_clients=args.num_per_round,
        min_evaluate_clients=0,
        min_available_clients=args.num_per_round,
        evaluate_fn=get_evaluate_fn(model, args.dataset),
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        initial_parameters=fl.common.ndarrays_to_parameters(model_parameters),
    )
    
    for i in range(args.num_per_round):
        np.save(os.path.join(pt_path, "flags_"+str(i)+".npy"), [])

    fl.server.start_server(
        server_address=args.ip,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )

    best_round = np.argmax(np.array(record["accuracy"]))
    best_acc = record["accuracy"][best_round]
    best_loss = record["loss"][best_round]
    save_npy_record(pt_path, record)
    logger.info("* best round: %d; best acc: %.4f; best loss: %.4f" %(best_round, best_acc, best_loss))

    