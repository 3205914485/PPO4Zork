import json
from jericho import *

env = FrotzEnv("../z-machine-games-master/jericho-game-suite/zork1.z5")
initial_observation, info = env.reset()
print(initial_observation)

with open('sft/data/raw/zork1_human_reward.jsonl', 'w', encoding='utf-8') as f:
    done = False
    step = 0
    while not done:
        location = env.get_player_location().name
        inventory = [item.name for item in env.get_inventory()]
        action = input(">>")
        print(action)

        observation, reward, done, info = env.step(action)
        print(observation)
        data = {
            "location": location,
            "inventory": inventory,
            "gold_action": action,
            "reward": reward
        }
        f.write(json.dumps(data) + '\n')
        step += 1