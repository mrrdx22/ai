# 02_Depth_First_Search.py
# Note: The PDF labels this iterative DFS, but supplies recursive DFS.
# This file preserves the PDF algorithm.
graph = {
    'A': ['B', 'C'], 'B': ['D', 'E'], 'C': ['F'],
    'D': [], 'E': ['G', 'H'], 'F': ['I'],
    'G': ['T'], 'H': [], 'I': [], 'T': []
}
visited = []
goal = 'F'
depth = 0

def ids(graph, node, visited, goal, depth):
    if node not in visited:
        visited.append(node)
        for n in graph[node]:
            depth += 1
            print(depth)
            if goal in visited:
                print('goal reached')
                break
            else:
                print('n is ', n)
                print('visited is ', visited)
                ids(graph, n, visited, goal, depth)
    return visited

visited = ids(graph, 'A', visited, goal, depth)
print(visited)
