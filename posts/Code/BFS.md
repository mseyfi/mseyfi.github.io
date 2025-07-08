
### What is Breadth-First Search (BFS)?

Breadth-First Search (BFS) is a graph traversal algorithm that explores a graph "layer by layer." It starts at a source node, explores all of its direct neighbors, then explores the neighbors of those neighbors, and so on. Think of it as the ripple effect when you drop a stone in water—it explores outward one level at a time.

Its primary use is to find the **shortest path** from a starting node to a target node in an **unweighted graph**.

-----

### When to Use BFS

You should immediately think of using BFS when a problem asks for:

  * The **shortest path** between two nodes in an unweighted graph or a grid.
  * The **minimum number of steps**, **fewest moves**, or **shortest time** to get from a start to an end state where each step has a uniform cost (i.e., each move costs 1).
  * Traversing a tree or graph in **level order**.

-----

### Algorithm and Core Components

BFS relies on two key data structures:

1.  **Queue**: A First-In-First-Out (FIFO) data structure to keep track of the nodes to visit next. This ensures the level-by-level traversal.
2.  **Seen Set**: A hash set to store the states we have already visited. This is crucial to avoid redundant work and getting stuck in infinite loops in graphs with cycles.

#### General Code Template

This code provides a template for BFS on a graph represented by an adjacency list.

```python
from collections import deque

def bfs_template(graph, start_node):
    """
    A general template for Breadth-First Search.
    
    Args:
        graph: A dictionary representing the adjacency list (e.g., {'A': ['B', 'C']}).
        start_node: The node to begin the search from.
    """
    # 1. Initialize the queue with the starting state.
    # The state can be a simple node or a tuple with more info (node, distance).
    queue = deque([start_node])
    
    # 2. Initialize the seen set to track visited states.
    seen = {start_node}
    
    while queue:
        # 3. Get the current state from the front of the queue.
        current_node = queue.popleft()
        
        # --- Process the current node here ---
        # (e.g., check if it's the target, add to a result list, etc.)
        
        # 4. Explore neighbors.
        for neighbor in graph.get(current_node, []):
            # 5. VERY IMPORTANT: Check if the new state has been seen before.
            if neighbor not in seen:
                # 6. If not seen, add it to the seen set and the queue.
                seen.add(neighbor)
                queue.append(neighbor)
```

-----

### Managing State and the Seen Set

The most critical part of solving BFS problems is correctly identifying the **state**. The state is the minimum set of information you need to uniquely define a position in your search. **What you put in the seen set must be the same as what you put in the queue.**

  * **Simple State**: In basic problems, the state is just the node itself. For a grid, this would be its coordinates.

      * **What to add to queue/seen**: A tuple like `(row, col)`.
      * **Check**: `if (next_row, next_col) not in seen:`

  * **Complex State**: In harder problems, you need more than just coordinates. For example, if you are carrying keys or have a limited ability to break walls, that information is part of the state.

      * **Example**: You're in a grid and can break up to `k` walls. Arriving at `(r, c)` with `k=3` remaining breaks is a different, better state than arriving with `k=1`.
      * **What to add to queue/seen**: A tuple like `(row, col, k_remaining)`.
      * **Check**: `if (next_row, next_col, new_k) not in seen:`

Rule of thumb: If a condition changes how you can explore from a node, it must be part of your state.

-----

### Complexity Analysis

Let $V$ be the number of vertices (nodes) and $E$ be the number of edges in the graph.

  * **Time Complexity**: $O(V + E)$

      * Every vertex is enqueued and dequeued exactly once, costing $O(V)$.
      * Every edge is checked exactly once when exploring the neighbors of a vertex, costing $O(E)$.

  * **Space Complexity**: $O(V)$

      * In the worst case, the queue can hold a large number of nodes at a single level. For a bushy graph, this can be proportional to $V$. The seen set can also store up to $V$ states.

-----

### Practice Problems on LeetCode

Here are some excellent problems to practice BFS, ranging from classic to more advanced.

1.  **Binary Tree Level Order Traversal (LeetCode 102)**

      * **Hint**: This is the most direct application of BFS. The state is simply the **tree node** itself.

2.  **Number of Islands (LeetCode 200)**

      * **Hint**: Iterate through the grid. When you find land ('1'), start a BFS to find all connected land parts and mark them as seen. The state is `(row, col)`.

3.  **01 Matrix (LeetCode 542)**

      * **Hint**: This is a classic for "multi-source" BFS. Start the BFS with all cells containing `0` in the queue *at the same time*. The state is `(row, col)`.

4.  **Shortest Path in a Grid with Obstacles Elimination (LeetCode 1293)**

      * **Hint**: This is a great example of a complex state. You can't just track your location. You must also track how many obstacles you can still eliminate. The state is `(row, col, k_remaining)`.

5.  **Word Ladder (LeetCode 127)**

      * **Hint**: Think of words as nodes in a graph. An edge exists between two words if they are one letter apart. The state is the **word** and its **distance** from the start: `(word, distance)`.

6.  **Jump Game III (LeetCode 1306)**

      * **Hint**: The array indices are the nodes. From an index `i`, you can jump to `i + arr[i]` or `i - arr[i]`. The state is just the **index**.
