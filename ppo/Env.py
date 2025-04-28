from jericho import FrotzEnv

class ZorkEnv:
    def __init__(self, env_name, max_history=3, max_repeat=20):
        path = f"../z-machine-games-master/jericho-game-suite/{env_name}"
        self.env = FrotzEnv(path)
        self.max_history = max_history
        self.max_repeat = max_repeat
        self.history = []
        self.cache = None
        self.terminal = False
        self.score = 0
        self.action_history = []
        self.location_history = []
        self.visited_locations = set()
        self.collected_items = set()
        self.unique_actions = set() 

    def reset(self):
        obs, info = self.env.reset()
        self.action_history = []
        self.location_history = []
        self.terminal = False
        self.history = []
        self.score = 0
        self.visited_locations = set()
        self.collected_items = set()
        self.unique_actions = set()         

        inventory = []
        valued_actions = self.env.get_valid_actions()
        location = self.env.get_player_location().name
        self.visited_locations.add(location)

        step_info = {
            "observation": obs,
            "inventory": inventory,
            "candidates": valued_actions,
            "location": location,
            "reward": 0,
            "score": 0,
            "action": None
        }
        self.cache = step_info
        return step_info

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.terminal = done
        self.score = info['score']

        inventory = [item.name for item in self.env.get_inventory()]
        valued_actions = self.env.get_valid_actions()
        location = self.env.get_player_location().name

        
        prev_items = self.collected_items.copy()
        prev_locations = self.visited_locations.copy()

        self.visited_locations.add(location)
        self.collected_items.update(inventory)

        # ===== reward shaping =====
        shaped_reward = reward

        if reward == 0:
            new_place = location not in prev_locations
            new_item = len(self.collected_items - prev_items) > 0

            if new_place:
                shaped_reward += 2  # new place
            if new_item:
                shaped_reward += 2  # new item
            if shaped_reward == 0:
                shaped_reward = 0 # penalty

        # ===== exploration bonus / penalty =====
        if action not in self.unique_actions:
            shaped_reward += 0.5  # new action
            self.unique_actions.add(action)

        step_info = {
            "observation": obs,
            "inventory": inventory,
            "candidates": valued_actions,
            "location": location,
            "reward": shaped_reward,
            "score": info['score'],
            "action": action
        }

        # 保存历史
        self.cache['action'] = action
        if len(self.history) >= self.max_history:
            self.history.pop(0)
        self.history.append(self.cache)
        self.cache = step_info

        return step_info


    def check_stuck(self, action, location):
        self.action_history.append(action)
        self.location_history.append(location)
        if len(self.action_history) > self.max_repeat:
            recent_actions = self.action_history[-self.max_repeat:]
            if len(set(recent_actions)) <= 3:
                return True
            recent_locations = self.location_history[-self.max_repeat:]
            if len(set(recent_locations)) <= 3:
                return True
        return False
