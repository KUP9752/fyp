from canvas import learn_game, play_game, human_interaction, move_arrowkeys, move_regression, move_classification
from torch_bc import AgentNetwork_Classification, AgentNetwork_Regression
import pickle
import random

if __name__ == "__main__":
  # trainingData = learn_game(agentMovement = human_interaction, n = 10)
  
  isTraining = False
  size: int = 250
  modelName = f"regression-{size}.pth"
  
  if isTraining:
    with open("./src/2d/datasets/1k-targets.pkl", "rb") as f:
      data = pickle.load(f)
    
    points = int(len(data) * (250 / 1000))  
    samples = random.sample(data, points)
    
    model = AgentNetwork_Regression().train_on_behaviour(data=samples, modelName=f"{modelName}", overwriteDevice="cpu")
  else:
    play_game(move_agent=move_regression, modelType=AgentNetwork_Regression, loadModelFromFile=f"./src/2d/models/{modelName}")