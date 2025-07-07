
### What is Topological Sorting?

Imagine you have a list of tasks, but some tasks must be completed before others. For example, you have to put on your socks before you put on your shoes. You can't put on your shoes first.

Topological sorting provides a linear ordering of these tasks (or "nodes" in a graph) such that for every directed edge from node `u` to node `v`, node `u` comes before node `v` in the ordering.

**Key Concepts:**

1.  **Directed Acyclic Graph (DAG):** Topological sorting is only possible on a **Directed Acyclic Graph (DAG)**. A DAG is a directed graph that contains no cycles. If a graph has a cycle (e.g., Task A depends on B, B depends on C, and C depends on A), it's impossible to create a valid linear ordering.
2.  **Not Unique:** For a given DAG, there can be multiple valid topological sorts.

**Analogy: Course Prerequisites**
A classic example is scheduling university courses. The courses are the nodes, and the prerequisites are the directed edges. If "Calculus I" is a prerequisite for "Calculus II", there's a directed edge from "Calculus I" to "Calculus II". A topological sort would give you a valid sequence in which you can take the courses.

-----

### 1\. DFS-based Topological Sort

The intuition behind the DFS-based approach is: "If we start a path at node `A`, which leads to node `B`, we will finish exploring all of `B`'s descendants before we finish exploring `A`'s." This means that the node that *finishes* last should come *first* in the topological sort.

We can achieve this by performing a post-order traversal. When a DFS call for a node finishes (i.e., it has explored all its neighbors), we add that node to the *front* of our sorted list.

#### Algorithm (DFS)

1.  Initialize an empty list, `topological_order`, which will store the result.
2.  Initialize a `visited` set to keep track of visited nodes to avoid redundant computations.
3.  Iterate through each node in the graph. If a node has not been visited, perform a DFS from that node.
4.  **DFS function (`dfs(node)`):**
    a. Mark the current `node` as visited by adding it to the `visited` set.
    b. For each `neighbor` of the current `node`:
    i. If the `neighbor` has not been visited, recursively call `dfs(neighbor)`.
    c. After the loop (meaning all descendants have been visited), add the current `node` to the **beginning** of the `topological_order` list.
5.  After the main loop finishes, `topological_order` contains the sorted nodes.

#### Python Code (DFS)

```python
from collections import defaultdict

def topological_sort_dfs(graph, n):
    """
    Performs topological sort on a directed acyclic graph (DAG) using DFS.
    
    Args:
        graph: A dictionary representing the graph as an adjacency list.
               Example: {'A': ['B', 'C'], 'B': ['D'], 'C': ['D'], 'D': []}
               
    Returns:
        A list representing the topologically sorted order of nodes.
        Returns an empty list if the graph is empty.
    """
    
    visit = [0] * n  # 0 = unvisited, 1 = visiting, 2 = visited
    topo = []

    def dfs(node):
        if visit[node] == 1:
            return False  # cycle detected
        if visit[node] == 2:
            return True   # already visited

        visit[node] = 1  # mark as visiting
        for neighbor in graph[node]:
            if not dfs(neighbor):
                return False

        visit[node] = 2  # mark as visited
        topo.append(node)
        return True

    for node in range(n):
        if visit[node] == 0:
            if not dfs(node):
                return []  # return early on cycle

    topo.reverse()
    return topo

# Example Usage:
# A -> B, C
# B -> D
# C -> D
# D -> E
graph_dict = {
    'A': ['B', 'C'],
    'B': ['D'],
    'C': ['D'],
    'D': ['E'],
    'E': []
}
print(f"DFS Topological Sort: {topological_sort_dfs(graph_dict)}")
# Possible output: ['A', 'C', 'B', 'D', 'E'] or ['A', 'B', 'C', 'D', 'E']
```

#### Complexity Analysis (DFS)

  * **Time Complexity:** $O(V + E)$
      * $V$ is the number of vertices (nodes).
      * $E$ is the number of edges.
      * The algorithm visits every vertex once and traverses every edge once. The `insert(0, ...)` operation on a list can be $O(V)$ in the worst case, but since each vertex is inserted only once, the total time for all insertions is still bounded by the overall DFS traversal. A more efficient implementation would use a `collections.deque` and `appendleft()`, which is an $O(1)$ operation.
  * **Space Complexity:** $O(V)$
      * The `visited` set can store up to $V$ vertices.
      * The recursion stack can go as deep as $V$ in the worst-case scenario (e.g., a graph like A -\> B -\> C -\> ...).
      * The `topological_order` list stores all $V$ vertices.

-----

### 2\. BFS-based Topological Sort (Kahn's Algorithm)

This approach is more intuitive. We start with all the nodes that have no incoming edges (no prerequisites). These nodes can be the first in our sorted list. We add them to a queue.

Then, we process the nodes in the queue. When we process a node, we "remove" it and its outgoing edges from the graph. This might cause some of its neighbors to now have no other incoming edges. If a neighbor's "in-degree" (count of incoming edges) drops to zero, we add it to the queue. We repeat this until the queue is empty.

#### Algorithm (BFS / Kahn's Algorithm)

1.  **Compute In-degrees:** Calculate the in-degree for every node in the graph. An in-degree is the count of incoming edges.
2.  **Initialize Queue:** Create a queue and add all nodes with an in-degree of 0 to it.
3.  **Process Queue:**
    a. Initialize an empty list, `topological_order`, for the result.
    b. While the queue is not empty:
    i. Dequeue a node, let's call it `u`.
    ii. Add `u` to the `topological_order` list.
    iii. For each `neighbor` `v` of `u`:
    \- Decrement the in-degree of `v` by 1.
    \- If the in-degree of `v` becomes 0, enqueue `v`.
4.  **Check for Cycles:** After the loop, if the number of nodes in `topological_order` is less than the total number of nodes in the graph, it means there was a cycle. Otherwise, `topological_order` holds a valid sort.

#### Python Code (BFS / Kahn's Algorithm)

```python
from collections import defaultdict, deque

def topological_sort_bfs(graph):
    """
    Performs topological sort on a DAG using Kahn's algorithm (BFS).
    
    Args:
        graph: A dictionary representing the graph as an adjacency list.
               
    Returns:
        A list representing the topologically sorted order of nodes.
        If the graph has a cycle, it returns a list that is shorter than the number of nodes.
    """
    # A dictionary to store in-degrees of all nodes
    in_degree = {node: 0 for node in graph}

    # Traverse the graph to fill in-degrees
    for node in graph:
        for neighbor in graph[node]:
            if neighbor in in_degree:
                in_degree[neighbor] += 1
    
    # Create a queue and enqueue all vertices with in-degree 0
    queue = deque([node for node in in_degree if in_degree[node] == 0])
    
    topological_order = []
    
    while queue:
        # Dequeue a vertex and add it to the topological order
        u = queue.popleft()
        topological_order.append(u)
        
        # Iterate through all its neighboring nodes
        # and decrease their in-degree by 1
        for v in graph[u]:
            if v in in_degree:
                in_degree[v] -= 1
                # If in-degree becomes 0, add it to the queue
                if in_degree[v] == 0:
                    queue.append(v)
    
    # Check if there was a cycle
    if len(topological_order) == len(graph):
        return topological_order
    else:
        # This indicates a cycle was present
        print("Graph has a cycle! Topological sort not possible.")
        return []

# Example Usage (same graph as before)
graph_dict = {
    'A': ['B', 'C'],
    'B': ['D'],
    'C': ['D'],
    'D': ['E'],
    'E': []
}
print(f"BFS Topological Sort: {topological_sort_bfs(graph_dict)}")
# Possible output: ['A', 'B', 'C', 'D', 'E'] or ['A', 'C', 'B', 'D', 'E']
```

#### Complexity Analysis (BFS)

  * **Time Complexity:** $O(V + E)$
      * Calculating in-degrees takes $O(V + E)$ because we iterate through all vertices and their edges.
      * Initializing the queue takes $O(V)$.
      * The `while` loop will execute $V$ times in a DAG. Inside the loop, the inner `for` loop will, over the course of the entire algorithm, traverse each edge exactly once. So, the loop part is $O(V + E)$.
  * **Space Complexity:** $O(V)$
      * The `in_degree` map stores $V$ entries.
      * The `queue` can store at most $V$ vertices in the worst case.
      * The `topological_order` list stores all $V$ vertices.

-----

### Cycle Detection in a Directed Graph

Detecting a cycle is a natural part of topological sorting.

#### Method 1: Using Kahn's Algorithm (BFS)

This is the simplest method. As mentioned above:

1.  Run Kahn's algorithm.
2.  At the end, compare the count of nodes in your `topological_order` list with the total number of nodes in the graph.
3.  If `len(topological_order) == total_nodes`, there is **no cycle**.
4.  If `len(topological_order) < total_nodes`, there **is a cycle**. The nodes that are not in the list are part of or dependent on a cycle, so their in-degrees never became zero.

#### Method 2: Using DFS

For DFS, we need to track the state of each node more carefully. A node can be in one of three states:

1.  **UNVISITED:** We haven't visited this node yet.
2.  **VISITING:** We are currently in the recursion stack for this node (i.e., `dfs(node)` has been called, but has not yet returned).
3.  **VISITED:** We have finished visiting this node and all its descendants.

A cycle is detected when our DFS traversal encounters a node that is currently in the **VISITING** state. This means we found a "back edge" – an edge going from a node to one of its ancestors in the DFS tree.

**Algorithm (DFS Cycle Detection):**

1.  Initialize two sets: `visiting` (for the current recursion path) and `visited` (for completed nodes).
2.  Iterate through all nodes in the graph.
3.  For each node, if it's not in `visited`, call a helper `is_cyclic_util(node)`.
4.  **`is_cyclic_util(node)` function:**
    a. Add `node` to `visiting` and `visited`.
    b. For each `neighbor` of `node`:
    i. If `neighbor` is in `visiting`: **CYCLE DETECTED\!** Return `True`.
    ii. If `neighbor` is not in `visited`: If `is_cyclic_util(neighbor)` returns `True`, then propagate the cycle detection by returning `True`.
    c. Remove `node` from `visiting` (backtrack).
    d. Return `False` (no cycle found from this path).

```python
def has_cycle_directed(graph, n):
    visited = [False] * n
    recStack = [False] * n

    def dfs(u):
        visited[u] = True
        recStack[u] = True

        for v in graph[u]:
            if not visited[v]:
                if dfs(v):
                    return True
            elif recStack[v]:  # back edge detected
                return True

        recStack[u] = False
        return False

    for node in range(n):
        if not visited[node]:
            if dfs(node):
                return True
    return False
```

-----

### Sample LeetCode Problems

Here are some popular LeetCode problems that directly use or are variations of topological sorting.

1.  [**207. Course Schedule**](https://leetcode.com/problems/course-schedule/)

      * **Problem:** Given a set of courses and their prerequisites, determine if you can finish all courses.
      * **Solution:** This is a classic cycle detection problem in a directed graph. You can use either the DFS or BFS (Kahn's) approach to detect a cycle. If there's no cycle, you can finish all courses.

2.  [**210. Course Schedule II**](https://leetcode.com/problems/course-schedule-ii/)

      * **Problem:** Same as above, but if you can finish all courses, you must return one valid order in which to take them.
      * **Solution:** This is a direct application of topological sort. You build the graph and then run either the DFS or BFS algorithm to produce the sorted order. If you detect a cycle, you return an empty list.

3.  [**269. Alien Dictionary**](https://leetcode.com/problems/alien-dictionary/) (Premium)

      * **Problem:** You are given a list of words sorted lexicographically by the rules of a new alien language. Derive the order of letters in this language.
      * **Solution:** This is a more advanced problem where you first have to **build the graph**. You compare adjacent words in the list to find ordering rules (e.g., if `["wrt", "wrf"]` is given, you know `t` must come before `f`). Once you've built the graph of character dependencies, you perform a topological sort to find the alphabet order.

4.  [**444. Sequence Reconstruction**](https://leetcode.com/problems/sequence-reconstruction/) (Premium)

      * **Problem:** Given an original sequence and a list of subsequences, determine if the original sequence is the only possible shortest supersequence that can be reconstructed.
      * **Solution:** You build a dependency graph from the subsequences. Then, you run Kahn's algorithm. For the original sequence to be the *unique* solution, the queue in Kahn's algorithm must never contain more than one element at any time.

5.  [**2392. Build a Matrix With Conditions**](https://leetcode.com/problems/build-a-matrix-with-conditions/)

      * **Problem:** You are given row and column conditions for placing numbers 1 to k in a k x k matrix. You need to return a valid matrix.
      * **Solution:** This is a creative application. You run topological sort twice: once on the row conditions to get the order of numbers in the rows, and once on the column conditions to get the order in the columns. Then you combine the results to place the numbers in the matrix.
