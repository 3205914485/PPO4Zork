import torch
import re

class MemoryAgent:
    def __init__(self, current_location):
        self.current_location = current_location
        self.prev_location = None
        self.map = {}  # {location: {direction: neighbor}}
        self.visited = set()
        self.inventory = set()

    def update(self, action, new_location, inventory):
        # 更新地图
        if self.current_location and self.current_location != new_location:
            direction = self.extract_direction(action)
            if direction:
                self.map.setdefault(self.current_location, {})[direction] = new_location
                # 可选：双向连接（如果确定无单向门）
                self.map.setdefault(new_location, {})[self.reverse_direction(direction)] = self.current_location

        self.prev_location = self.current_location
        self.current_location = new_location
        self.visited.add(new_location)
        self.inventory = inventory

    def extract_direction(self, action: str) -> str:
        action = action.strip().lower()

        direction_aliases = {
            "n": "north", "s": "south", "e": "east", "w": "west",
            "u": "up", "d": "down",
            "ne": "northeast", "nw": "northwest",
            "se": "southeast", "sw": "southwest",
            "north": "north", "south": "south", "east": "east", "west": "west",
            "North": "north", "South": "south", "West": "west", "East": "east",
             "NORTH": "north", "SOUTH": "south", "EAST": "east", "WEST": "west",
            "up": "up", "down": "down",
            "northeast": "northeast", "northwest": "northwest",
            "southeast": "southeast", "southwest": "southwest"
        }

        # tokenize input, assume direction is the first or only word
        tokens = action.split()
        for token in tokens:
            if token in direction_aliases:
                return direction_aliases[token]

        return None


    def reverse_direction(self, direction: str) -> str:
        reverse = {
            "north": "south",
            "south": "north",
            "east": "west",
            "west": "east",
            "up": "down",
            "down": "up",
        }
        return reverse.get(direction, None)

    def summarize(self):
        visited_rooms = ", ".join(sorted(self.visited))
        items = ", ".join(self.inventory)

        exits = self.map.get(self.current_location, {})
        if exits:
            exits_str = "; ".join([f"{dir} → {room}" for dir, room in exits.items()])
        else:
            exits_str = "Unknown"

        return (
            f"You are in '{self.current_location}'.\n"
            f"Inventory: {items if items else 'Nothing'}.\n"
            f"Visited Rooms: {visited_rooms}.\n"
            f"Known Exits from here: {exits_str}."
        )
        
    def get_map(self):
        return self.map
