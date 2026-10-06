# 04_Recursive_Best_First_Search.py
import math

heuristics = {
    'Arad': 366, 'Zerind': 374, 'Oradea': 380, 'Sibiu': 253,
    'Timisoara': 329, 'Lugoj': 244, 'Mehadia': 241, 'Drobeta': 242,
    'Craiova': 160, 'Rimnicu Vilcea': 193, 'Fagaras': 178,
    'Pitesti': 98, 'Bucharest': 0, 'Giurgiu': 77, 'Urziceni': 80,
    'Hirsova': 151, 'Eforie': 161, 'Vaslui': 199, 'Iasi': 226, 'Neamt': 234
}
romania_map = {
    'Arad': {'Zerind': 75, 'Timisoara': 118, 'Sibiu': 140},
    'Zerind': {'Arad': 75, 'Oradea': 71},
    'Oradea': {'Zerind': 71, 'Sibiu': 151},
    'Sibiu': {'Arad': 140, 'Oradea': 151, 'Fagaras': 99, 'Rimnicu Vilcea': 80},
    'Timisoara': {'Arad': 118, 'Lugoj': 111},
    'Lugoj': {'Timisoara': 111, 'Mehadia': 70},
    'Mehadia': {'Lugoj': 70, 'Drobeta': 75},
    'Drobeta': {'Mehadia': 75, 'Craiova': 120},
    'Craiova': {'Drobeta': 120, 'Rimnicu Vilcea': 146, 'Pitesti': 138},
    'Rimnicu Vilcea': {'Sibiu': 80, 'Craiova': 146, 'Pitesti': 97},
    'Fagaras': {'Sibiu': 99, 'Bucharest': 211},
    'Pitesti': {'Rimnicu Vilcea': 97, 'Craiova': 138, 'Bucharest': 101},
    'Bucharest': {'Fagaras': 211, 'Pitesti': 101, 'Giurgiu': 90, 'Urziceni': 85},
    'Giurgiu': {'Bucharest': 90},
    'Urziceni': {'Bucharest': 85, 'Hirsova': 98, 'Vaslui': 142},
    'Hirsova': {'Urziceni': 98, 'Eforie': 86},
    'Eforie': {'Hirsova': 86},
    'Vaslui': {'Urziceni': 142, 'Iasi': 92},
    'Iasi': {'Vaslui': 92, 'Neamt': 87},
    'Neamt': {'Iasi': 87}
}

class Node:
    def __init__(self, city, parent=None, g=0):
        self.city = city
        self.parent = parent
        self.g = g
        self.h = heuristics[city]
        self.f = max(self.g + self.h, getattr(parent, 'f', 0))

def rbfs(node, goal, f_limit):
    print(f'Visiting {node.city}, f={node.f}, limit={f_limit}')
    if node.city == goal:
        return node, 0
    successors = []
    for neighbor, cost in romania_map[node.city].items():
        if node.parent and neighbor == node.parent.city:
            continue
        child = Node(neighbor, parent=node, g=node.g + cost)
        successors.append(child)
    if not successors:
        return None, math.inf
    while True:
        successors.sort(key=lambda x: x.f)
        best = successors[0]
        if best.f > f_limit:
            return None, best.f
        alternative = successors[1].f if len(successors) > 1 else math.inf
        result, best.f = rbfs(best, goal, min(f_limit, alternative))
        if result is not None:
            return result, best.f

def extract_path(node):
    path = []
    while node:
        path.append(node.city)
        node = node.parent
    return path[::-1]

if __name__ == '__main__':
    start_node = Node('Arad')
    result, _ = rbfs(start_node, 'Bucharest', math.inf)
    if result:
        print('\nPath found:')
        print(' -> '.join(extract_path(result)))
        print('Total cost:', result.g)
    else:
        print('No path found.')
