from canvas import learn_game, play_game, human_interaction, move_arrowkeys, move_regression, move_classification
from torch_bc import AgentNetwork_Classification, AgentNetwork_Regression
import pickle

if __name__ == "__main__":
  # # trainingData = learn_game(agentMovement = human_interaction, n = 10)
  # with open("./src/2d/datasets/1k-targets.pkl", "rb") as f:
  #   data = pickle.load(f)
    
  # # point = int(len(data) * 0.1)  # 10% of the data = 100 points
  # model = AgentNetwork_Regression().train_on_behaviour(data=data, modelName="regression-1k.pth", overwriteDevice="cpu")
  
  play_game(move_agent=move_regression, modelType=AgentNetwork_Regression, loadModelFromFile="./src/2d/models/regression-1k.pth")