### What is Dijkstra's Algorithm?

At its core, Dijkstra's algorithm is a **greedy algorithm** used to find the shortest paths from a single source node to all other nodes in a weighted graph.

**Key characteristics and constraints:**

  * **Single Source Shortest Path:** It finds the shortest path from one starting node (the "source") to every other node.
  * **Weighted Graph:** The connections (edges) between nodes (vertices) must have a numerical weight or cost.
  * **Non-Negative Edge Weights:** This is the most important constraint. Dijkstra's algorithm does not work correctly if the graph has negative edge weights. For such cases, you would use an algorithm like Bellman-Ford.

### The Core Idea (Analogy)

Think of it like finding the fastest way to get from your home to all other locations in a city.

1.  **Start at Home:** You are at the source node. The distance to your home is 0. The distance to everywhere else is "unknown" (we'll represent this with infinity).
2.  **Explore the Closest Place:** From all the places you can get to directly from your current location, you pick the one that is closest (has the minimum travel time/distance).
3.  **Update Your Map:** You've now found the "shortest path" to this new place. You mark it as "visited." Now, from this newly visited place, you look at its neighbors. For each neighbor, you ask: "Is the path through my newly visited location shorter than any path I've seen before?" If it is, you update your map with this new, shorter route.
4.  **Repeat:** You repeat step 2: From all the unvisited places you now know a path to, pick the one with the absolute shortest path from the original source. Then repeat step 3.

You continue this process—always exploring the unvisited node with the smallest known distance from the source—until you have visited all the nodes you can reach.

### The Algorithm Step-by-Step

To implement this efficiently, we use a few key data structures:

  * A **distances dictionary (or array)** to store the shortest distance found so far from the source to every other node. We initialize all distances to infinity, except for the source node, which is 0.
  * A **priority queue (min-heap)** to efficiently determine the next unvisited node with the smallest distance. The priority queue will store tuples of `(distance, node)`.
  * A **graph representation**, typically an adjacency list (e.g., a dictionary where keys are nodes and values are lists of `(neighbor, weight)` tuples).

Here are the formal steps:

1.  **Initialization:**

      * Create a `distances` dictionary and initialize the distance to the `start_node` as 0 and all other nodes to infinity.
      * Create a priority queue and add the starting node with a priority of 0: `(0, start_node)`.

2.  **Main Loop:**

      * While the priority queue is not empty:
        
        a. Pop the element with the smallest distance (priority) from the queue. Let's call it `(current_distance, current_node)`.
        
        b. If `current_distance` is greater than the distance we already have recorded for `current_node` in our `distances` dictionary, it means we've found a shorter path to this node already. So, we skip it and

      * continue to the next iteration.

        c. For each `neighbor` of the `current_node`:

          i. Calculate the distance to this neighbor through the `current_node`: `distance = current_distance + weight_of_edge`.

          ii. **Relaxation Step:** If this new `distance` is smaller than the known distance to the `neighbor` (stored in the `distances` dictionary), it means we have found a new shorter path.

          iii. Update the `distances` dictionary with this new shorter distance and push `(new_distance, neighbor)` to the priority queue.

4.  **Termination:**

      * When the loop finishes, the `distances` dictionary will contain the shortest path distances from the source node to all other reachable nodes.

-----

### Python Code

This implementation uses Python's `heapq` module, which provides an efficient min-heap, perfect for our priority queue.

```python
import heapq
import collections

def dijkstra(graph, start_node):
    """
    Implements Dijkstra's algorithm to find the shortest path from a start node to all other nodes.

    Args:
        graph (dict): A dictionary representing the graph as an adjacency list.
                      Example: {'A': [('B', 10), ('C', 3)], 'B': [('C', 1), ('D', 2)], ...}
        start_node: The starting node.

    Returns:
        dict: A dictionary mapping each node to the shortest distance from the start_node.
    """
    # The graph should be represented as an adjacency list:
    # graph = {
    #     'A': [('B', 10), ('C', 3)],
    #     'B': [('D', 2)],
    #     'C': [('B', 4), ('D', 8), ('E', 2)],
    #     'D': [('E', 7)],
    #     'E': [('D', 9)]
    # }

    # 1. Initialize distances with infinity, except for the start node which is 0.
    # We can use a defaultdict for convenience.
    distances = collections.defaultdict(lambda: float('inf'))
    distances[start_node] = 0

    # 2. Priority queue to store (distance, node)
    # Python's heapq is a min-heap, so it's perfect for this.
    priority_queue = [(0, start_node)]

    while priority_queue:
        # 3. Get the node with the smallest distance
        current_distance, current_node = heapq.heappop(priority_queue)

        # If we have found a shorter path already, skip
        if current_distance > distances[current_node]:
            continue

        # 4. Iterate over neighbors and relax edges
        if current_node in graph:
            for neighbor, weight in graph[current_node]:
                distance = current_distance + weight

                # If we found a shorter path to the neighbor
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    heapq.heappush(priority_queue, (distance, neighbor))

    return distances

# --- Example Usage ---
if __name__ == '__main__':
    # Graph represented as an adjacency list
    # The weight is the second element in the tuple
    graph = {
        'A': [('B', 10), ('C', 3)],
        'B': [('C', 1), ('D', 2)],
        'C': [('B', 4), ('D', 8), ('E', 2)],
        'D': [('E', 7)],
        'E': [('D', 9)]
    }

    start_node = 'A'
    shortest_paths = dijkstra(graph, start_node)

    print(f"Shortest paths from node '{start_node}':")
    # Convert defaultdict to a regular dict for cleaner printing
    print(dict(shortest_paths)) 
    # Expected output: {'A': 0, 'C': 3, 'B': 7, 'E': 5, 'D': 9}
```

### Complexity Analysis

Let $V$ be the number of vertices (nodes) and $E$ be the number of edges in the graph.

#### **Time Complexity: $O(E \\log V)$**

  * **Initialization:** Initializing the `distances` dictionary takes $O(V)$ time.
  * **Priority Queue Operations:**
      * Every time we find a shorter path, we add an edge to the priority queue (`heapq.heappush`). In the worst case, we might do this for every edge in the graph. A push operation on a heap of size $V$ takes $O(\\log V)$ time. Therefore, all push operations combined take $O(E \\log V)$.
      * Each vertex is extracted from the priority queue exactly once (`heapq.heappop`). A pop operation takes $O(\\log V)$ time. Total time for all pops is $O(V \\log V)$.
  * **Total:** The overall time complexity is the sum of these operations: $O(V \\log V + E \\log V)$. In a connected graph, $E \\ge V-1$. Thus, the $E \\log V$ term dominates, and the complexity simplifies to **$O(E \\log V)$**.

#### **Space Complexity: $O(V + E)$**

  * **Graph Storage:** The adjacency list representation of the `graph` itself requires $O(V + E)$ space.

  * **Distances Dictionary:** The `distances` dictionary stores a value for each vertex, requiring $O(V)$ space.

  * **Priority Queue:** In the worst-case scenario (a "star" graph where the source is connected to all other nodes), the priority queue could hold an entry for each vertex. Therefore, it requires $O(V)$ space.

  * **Total:** Combining these, the total space complexity is **$O(V + E)$**.

-----

### Sample LeetCode Problems

Here are some classic LeetCode problems where Dijkstra's algorithm is the intended solution.

1.  **[LeetCode 743. Network Delay Time](https://leetcode.com/problems/network-delay-time/)**

      * **Problem:** You are given a network of `n` nodes, a list of `times` represented as `(u, v, w)` where `w` is the time it takes for a signal to travel from node `u` to `v`. Starting from a node `k`, find the minimum time it takes for the signal to reach *all* nodes.
      * **How Dijkstra fits:** This is a direct application. The nodes are the graph vertices, the `times` are the directed, weighted edges, and `k` is the source node. You run Dijkstra from `k` to find the shortest path to all other nodes. The answer is the maximum of these shortest paths.

2.  **[LeetCode 1514. Path with Maximum Probability](https://leetcode.com/problems/path-with-maximum-probability/)**

      * **Problem:** You are given an undirected graph and the probability of success for traversing each edge. Find the path with the maximum probability from a `start` node to an `end` node.
      * **How Dijkstra fits:** This is a clever variation. Dijkstra minimizes a sum (distance), but here we need to maximize a product (probability). We can adapt it in two ways:
        1.  **Transform weights:** Since $\\log(a \\times b) = \\log(a) + \\log(b)$, we can maximize the product of probabilities by maximizing the sum of their logarithms. Run Dijkstra on edge weights of $-\\log(\\text{probability})$.
        2.  **Use a max-heap:** Instead of a min-heap, use a max-heap to always explore the path with the current highest probability.

3.  **[LeetCode 505. The Maze II](https://leetcode.com/problems/the-maze-ii/)**

      * **Problem:** A ball in a maze can roll up, down, left, or right, but it won't stop until it hits a wall. Find the shortest distance for the ball to travel from a `start` position to a `destination`.
      * **How Dijkstra fits:** This is a shortest path problem on an implicit graph. The "nodes" are the `(row, col)` cells of the maze. The "edges" are the full rolls from one cell to another until a wall is hit. The "weight" of an edge is the number of steps taken in that roll. Dijkstra's algorithm is perfect for finding the shortest total distance.

4.  **[LeetCode 787. Cheapest Flights Within K Stops](https://leetcode.com/problems/cheapest-flights-within-k-stops/)**

      * **Problem:** Find the cheapest flight from a source `src` to a destination `dst` with at most `k` stops.
      * **How Dijkstra fits (with modification):** Standard Dijkstra doesn't work here because it's greedy on distance only. A path that is short in distance might use too many stops. You need to modify Dijkstra to keep track of an additional state: the number of stops. The state in the priority queue becomes `(cost, current_node, stops_taken)`. This variant explores paths that might be more expensive but use fewer stops, which could lead to a valid solution whereas the absolute cheapest path might be invalid due to the stop limit.
