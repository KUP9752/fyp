from canvas import learn_game, play_game, human_interaction
from torch_bc import train_on_behaviour
import pickle

if __name__ == "__main__":
  # trainingData = learn_game(agentMovement = human_interaction, n = 10)
  with open("./src/2d/datasets/10k-targets.pkl", "rb") as f:
    data = pickle.load(f)
    
  model = train_on_behaviour(modelName = "agent-network-10k.pth", data = data, overwriteDevice="cpu")
  play_game(model)
    