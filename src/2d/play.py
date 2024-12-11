from canvas import learn_game, play_game, human_interaction, move_arrowkeys, move_regression, move_classification
from torch_bc import AgentNetwork_Classification, AgentNetwork_Regression, PositionPredictor
import pickle
import random
from argparse import ArgumentParser


if __name__ == "__main__":
  # trainingData = learn_game(agentMovement = human_interaction, n = 10)
  parser = ArgumentParser()
  parser.add_argument_group
  parser.add_argument("-t", "--train", default = False, action="store_true", help="Whether the script should run in training mode or not [need to also provide size]")
  parser.add_argument("-m", "--model-path", required=True, type=str, help="File path of the model to load")
  parser.add_argument("-s", "--size", default = 1000, type=int, help="Number of data the dataset should be trained on")
  parser.add_argument("--dataset", type=str, default="./datasets/1k-targets.pkl", help="path of the dataset to train on")  
  args = parser.parse_args()
  
  
  isTraining = args.train
  size = args.size # default to 1000
  # modelName = f"regression-{"1k" if size == 1000 else size}.pth"
  # modelPath = f"{args.model_path}/{modelName}"
  modelPath = args.model_path
  
  
  if isTraining:
    ss = "./datasets/screenshots"
    coords = f"{ss}/ss-coords.pkl"
    # with open(args.dataset, "rb") as f:
    #   data = pickle.load(f)
    
    # points = int(len(data) * (size / 1000))  
    # samples = random.sample(data, points)
    
    model = PositionPredictor().train_on_images(doPrints=True, imagesDir= ss, imageCoordsPath=coords, modelPath=modelPath)
  else:
    play_game(move_agent=move_regression, modelType=AgentNetwork_Regression, loadModelFromFile=modelPath)
    
    
    