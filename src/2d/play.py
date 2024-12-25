from canvas import play_game, MOVES
from torch_bc import AgentNetwork, AgentNetwork_Classification, AgentNetwork_Regression, PositionPredictor, CNN_Regression
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


modelToClass = {
  "coord-class": AgentNetwork_Classification,
  "coord-regr": AgentNetwork_Regression,
  "pos-pred": PositionPredictor,
  "cnn-regr": CNN_Regression
}
def train(args):
  dataPath = args.dataset
  modelPath = args.model_path
  frac = args.size_fraction
  overrideDevice = args.override_device
  modelType: AgentNetwork | PositionPredictor | CNN_Regression = modelToClass[args.model_type]
  doPrints = args.do_prints
  
  # samples = random.sample(data, points)
  
  print(f"Training a {modelType} model")
  
  
  match args.model_type:
    case "coord-class" | "coord-regr":
      model = modelType().train_on_behaviour(
        doPrints=doPrints, 
        dataFilepath= dataPath, 
        modelPath=modelPath,
        overwriteDevice=overrideDevice
        )
    case "pos-pred" | "cnn-regr": 
      coords = f"{dataPath}/ss-info.pkl"
      model = modelType(
        ).train_on_behaviour(
        doPrints=doPrints, 
        sizeFrac=frac,
        imagesDir= dataPath, 
        imageLabelsPath=coords, 
        modelPath=modelPath,
        overwriteDevice=overrideDevice
        )




MODELS = [ "coord-class", "coord-regr", "pos-pred", "cnn-regr"]
if __name__ == "__main__":
  # trainingData = learn_game(agentMovement = human_interaction, n = 10)
  parser = ArgumentParser()
  subparsers = parser.add_subparsers(required=True, help="sub-command help")
  
  ## Training
  trainParser = subparsers.add_parser("train",
                                      help="Whether the script should run in training mode or not [need to also provide size]")
  trainParser.add_argument("-s", "--size-fraction", 
                           default = 1.0, 
                           type=float, 
                           help="Fraction of the dataset the model should be trained on")
  trainParser.add_argument("-m","--model-type", 
                           required=True, 
                           choices = MODELS,
                           help="path of the dataset to train on")  
  trainParser.add_argument("-d","--dataset", 
                           type=str, 
                           required=True, 
                           help="path of the dataset to train on")  
  trainParser.add_argument("-f", "--model-path", 
                           required=True, 
                           type=str, 
                           help="filepath of where the model should be saved")
  trainParser.add_argument("-od", "--override-device", 
                           choices = ["cpu", "cuda", None],
                           default = None,  
                           type=str, 
                           help="Force a device to be used")
  trainParser.add_argument("-p", "--do-prints",
                          action="store_true",
                          help="Whether the model should print out training information"
  )
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
  
    
    