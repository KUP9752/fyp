from canvas import play_game, MOVES
from torch_bc import AgentNetwork_Classification, AgentNetwork_Regression, PositionPredictor
import pickle
import random
from argparse import ArgumentParser


def move(args):
  ## extract args
  moveFunc = args.movement
  modelPath = args.model_path
  
  print(f"{moveFunc = }")
  print(f"{modelPath = }")
  play_game(moveAgent=moveFunc, 
            loadModelFromFile=modelPath)

def train(args):
  dataPath = args.dataset
  coords = f"{dataPath}/ss-coords.pkl"
  
  size = args.size # default to 1000
  # modelName = f"regression-{"1k" if size == 1000 else size}.pth"
  # modelPath = f"{args.model_path}/{modelName}"
  modelPath = args.model_path
  # with open(args.dataset, "rb") as f:
  #   data = pickle.load(f)
  
  # points = int(len(data) * (size / 1000))  
  # samples = random.sample(data, points)
  
  model = PositionPredictor().train_on_images(doPrints=False, 
                                              imagesDir= dataPath, 
                                              imageCoordsPath=coords, 
                                              modelPath=modelPath)


if __name__ == "__main__":
  # trainingData = learn_game(agentMovement = human_interaction, n = 10)
  parser = ArgumentParser()
  subparsers = parser.add_subparsers(required=True, help="sub-command help")
  
  ## Training
  trainParser = subparsers.add_parser("train",
                                      help="Whether the script should run in training mode or not [need to also provide size]")
  trainParser.add_argument("-s", "--size", 
                           default = 1000, 
                           type=int, 
                           help="Number of data the dataset should be trained on")
  trainParser.add_argument("-d","--dataset", 
                           type=str, 
                           required=True, 
                           help="path of the dataset to train on")  
  trainParser.add_argument("-f", "--model-path", 
                           required=True, 
                           type=str, 
                           help="filepath of where the model should be saved")
  trainParser.set_defaults(func=train) ## call train()
  ## Eval/Movement
  evalParser: ArgumentParser = subparsers.add_parser("move", aliases= ["eval"], 
                                                     help="Whether the script should run in evaluation mode or not")
  evalParser.add_argument("-m", "--movement", required=True, choices=MOVES)
  evalParser.add_argument("-f", "--model-path", 
                           required=True, 
                           type=str, 
                           help="File path of the model to load")
  evalParser.set_defaults(func=move) ## call move()
  
  
  
  args = parser.parse_args()
  
  args.func(args)
  
    
    