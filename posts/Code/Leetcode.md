# 1. Array and String

## Two pointers

> Start the pointers at the edges of the input. Move them towards each other until they meet.

```python
function fn(arr):
    left = 0
    right = arr.length - 1

    while left < right:
        Do some logic here depending on the problem
        Do some more logic here to decide on one of the following:
            1. left++
            2. right--
            3. Both left++ and right--
```
The strength of this technique is that we will never have more than  $O(n)$ iterations for the while loop because the pointers start $n$ away from each other and move at least one step closer in every iteration. Therefore, if we can keep the work inside each iteration at $O(1)$, this technique will result in a linear runtime, which is usually the best possible runtime. 

#### sample Questions
- Given a string `s`, return `true` if it is a `palindrome`, `false` otherwise
- Given a sorted array of unique integers and a target integer, return `true` if there exists a pair of numbers that sum to target, `false` otherwise.
- [Squares of a Sorted Array](https://leetcode.com/problems/squares-of-a-sorted-array/description/)
- [Reverse-words-in-a-string-iii](https://leetcode.com/problems/reverse-words-in-a-string-iii/description/)
- [minimum-consecutive-cards-to-pick-up](https://leetcode.com/problems/minimum-consecutive-cards-to-pick-up/description/)


> Move along both inputs simultaneously until all elements have been checked.

```python
function fn(arr1, arr2):
    i = j = 0
    while i < arr1.length AND j < arr2.length:
        Do some logic here depending on the problem
        Do some more logic here to decide on one of the following:
            1. i++
            2. j++
            3. Both i++ and j++

    // Step 4: make sure both iterables are exhausted
    // Note that only one of these loops would run
    while i < arr1.length:
        Do some logic here depending on the problem
        i++

    while j < arr2.length:
        Do some logic here depending on the problem
        j++
```
#### Sample Questions:
- Given two sorted integer arrays arr1 and arr2, return a new array that combines both of them and is also sorted.

Similar to the first method we looked at, this method will have a linear time complexity of $O(n+m)$ if the work inside the while loop is 
$O(1)$, where $n = arr1.length$ and $m = arr2.length$. This is because at every iteration, we move at least one pointer forward, and the pointers cannot be moved forward more than $n + m$ times without the arrays being exhausted. Let's look at some examples.

## Sliding Window
#### When should we use sliding window?
There is a very common group of problems involving subarrays that can be solved efficiently with sliding window. Let's talk about how to identify these problems.

**First**, the problem will either explicitly or implicitly define criteria that make a subarray "valid". There are 2 components regarding what makes a subarray valid:

A constraint metric. This is some attribute of a subarray. It could be the sum, the number of unique elements, the frequency of a specific element, or any other attribute.
A numeric restriction on the constraint metric. This is what the constraint metric should be for a subarray to be considered valid.
For example, let's say a problem declares a subarray is valid if it has a sum less than or equal to 10. The constraint metric here is the sum of the subarray, and the numeric restriction is <= 10. A subarray is considered valid if its constraint metric conforms to the numeric restriction, i.e. the sum is less than or equal to 10.

**Second**, the problem will ask you to find valid subarrays in some way.

The most common task you will see is finding the best valid subarray. The problem will define what makes a subarray better than another. For example, a problem might ask you to find the longest valid subarray.

Another common task is finding the number of valid subarrays. We will take a look at this later in the article.

> Whenever a problem description talks about subarrays, you should figure out if sliding window is a good option by analyzing the problem description. If you can find the things mentioned above, then it's a good bet.

#### Sample Questions:
- Given an array of **positive** integers nums and an integer $k$, find the length of the longest subarray whose sum is less than or equal to $k$. ***solve for when negatives are there.***
- Find the longest substring that has at most one "0"
- Find the number of subarrays that have a product less than $k$
- [Max Consecutive Ones III](https://leetcode.com/problems/max-consecutive-ones-iii/description/)


```python
function fn(arr):
    left = 0
    for (int right = 0; right < arr.length; right++):
        Do some logic to "add" element at arr[right] to window

        while WINDOW_IS_INVALID:
            Do some logic to "remove" element at arr[left] from window
            left++

        Do some logic to update the answer
```

sample solution to sum of array = k:

```python
function fn(nums, k):
    left = 0
    curr = 0
    answer = 0
    for (int right = 0; right < nums.length; right++):
        curr += nums[right]
        while (curr > k):
            curr -= nums[left]
            left++

        answer = max(answer, right - left + 1)

    return answer
```

#### Complexity
is $O(n)$ with **amortized** $O(1)$ in the inner while loop.

## Number of subarrays
If a problem asks for the number of subarrays that fit some constraint, we can still use sliding window, but we need to use a neat math trick to calculate the number of subarrays.

Let's say that we are using the sliding window algorithm we have learned and currently have a window `(left, right)`. How many valid windows end at index `right`?

There's the current window `(left, right)`, then `(left + 1, right)`, `(left + 2, right)`, and so on until `(right, right)` (only the element at `right`).

You can fix the right bound and then choose any value between `left` and `right` inclusive for the `left` bound. Therefore, the number of valid windows ending at index `right` is equal to the size of the window, which we know is `right - left + 1`.

## Prefix Sum
Prefix sum is a technique that can be used on arrays (of numbers). The idea is to create an array prefix where `prefix[i]` is the sum of all elements up to the index `i` (inclusive). For example, given `nums = [5, 2, 1, 6, 3, 8]`, we would have `prefix = [5, 7, 8, 14, 17, 25]`.

When a subarray starts at index `0`, it is considered a "prefix" of the array. A prefix sum represents the sum of all prefixes.

Prefix sums allow us to find the sum of any subarray in `O(1)`. If we want the sum of the subarray from `i` to `j` (inclusive), then the answer is `prefix[j] - prefix[i - 1]`, or `prefix[j] - prefix[i] + nums[i]` if you don't want to deal with the out of bounds case when `i = 0`.

This works because `prefix[i - 1]` is the sum of all elements before index `i`. When you subtract this from the sum of all elements up to index `j`, you are left with the sum of all elements starting at index `i` and ending at index `j`, which is exactly what we are looking for.

# 2. Hashing:
Hashing can be used for existance or for counting. Since existance is easy we jump into counting:
### Counting with Hash
Counting is a very common pattern with hash maps. By "counting", we are referring to tracking the frequency of things. This means our hash map will be mapping keys to integers. Anytime you need to count anything, think about using a hash map to do it.

Recall that when we were looking at **sliding windows**, some problems had their constraint as limiting the amount of a certain element in the window. For example, longest substring with at most `k` `0`s. In those problems, we could simply use an integer variable curr because we are only focused on **one element** (we only cared about `0`). A hash map opens the door to solving problems where the constraint involves **multiple elements**. Let's start by looking at a sliding window example that leverages a hash map.


## Count the number of subarrays with an "exact" constraint
In the sliding window article from previous section, we talked about a pattern "find the number of subarrays/substrings that fit a constraint". In those problems, if you had a window between left and right that fit the constraint, then all windows from `x` to `right` also fit the constraint, where `left < x <= right`.

For this pattern, we will be looking at problems with stricter constraints, so that the property just mentioned is not necessarily true.

For example, **"Find the number of subarrays that have a sum less than k"** with an input that only has positive numbers would be solved with sliding window. In this section, we would be talking about questions like **"Find the number of subarrays that have a sum exactly equal to k"**.

At first, some of these problems seem very difficult. However, the pattern is very simple once you learn it, and you'll see how similar the code is for each problem that falls in this pattern. To understand the algorithm, we need to recall the concept of **prefix sums**.

With a prefix sum, you can find the sum of subarrays by taking the difference between two prefix sums. Let's say that you wanted to find subarrays that had a sum exactly equal to k, and you also had a prefix sum of the input. You know that any difference in the prefix sum equal to k represents a subarray with a sum equal to k. So how do we find these differences?

Let's first declare a hash map counts that maps prefix sums to how often they occur (a number could appear multiple times in a prefix sum if the input has negative numbers, for example, given `nums = [1, -1, 1]`, the prefix sum is `[1, 0, 1]` and `1` appears twice). We need to initialize `counts[0] = 1`. This is because the empty `prefix []` has a sum of `0`. You'll see why this is necessary in a second.

Next, let's declare our `answer` variable and `curr`. As we iterate over the input, `curr` will represent the sum of all elements we have iterated over so far (the sum of the prefix up to the current element).

Now, we iterate over the input. At each element, we update `curr` and also maintain counts by incrementing the frequency of curr by `1` 
- remember, `counts` is counting how many times we have seen each prefix sum. Before we update counts however, we first need to update the `answer`.

How do we update the answer? Recall that in the sliding window article, when we were looking for the "number of subarrays", we focused on each index and figured out how many valid subarrays ended at the current index. We will do the same thing here. Let's say that we're at an index `i`. We know a few things:

- Up until this point, `curr` stores the prefix of all elements up to `i`.
- We have stored all other prefixes before `i` inside of `counts`.
The difference between any two prefix sums represents a subarray. For example, if you wanted the subarray starting at index `3` and ending at index `8`, you would take the prefix up to `8` and subtract the prefix up to `2` from it.
Now, imagine there exists a subarray ending at `i` with a sum of `k`. We don't know where this subarray starts, we just know it exists - let's say it starts at index `j`. What would the prefix sum be ending at `j - 1`? Well, according to our assumptions, the sum of the subarray from `j` to `i` is `k` and the sum of the prefix up to `i` is `curr`. Thus, you can find the prefix sum ending at `j - 1` by subtracting these two: it's `curr - k`.

This is the key idea: if we saw the prefix sum `curr - k` before, it necessarily implies that there is a subarray ending at `i` with a sum of `k`. Again, we don't know where the beginning of this subarray is; we just know it exists, but that's enough to solve the problem.

Therefore, we can increment our answer by `counts[curr - k]`. If the prefix `curr - k` occurred multiple times before (due to negative numbers), then each of those prefixes could be used as a starting point to form a subarray ending at the current index with a sum of `k`. That's why we need to track the frequency.
> Let's use a concrete example to better illustrate this idea. Imagine we had `nums = [0, 1, 2, 3, 4]` and `k = 5`. Let's jump to `i = 3`.
> 
> Currently, `curr = 6` (remember, `curr` is tracking the prefix sum up to `i`). We also have `0`, `1`, and `3` in `counts` (all the prefix sums we have encountered so far).
> 
> At this point, we can see that there is a subarray ending at `i` with a sum of `k`
> - it's `[2, 3]`. How does our algorithm see it though?
>
> The current prefix sum is `6`. We want a subarray with a sum of `5`. Thus, if there was a prefix sum of `1` earlier, you could just subtract that prefix from the current one, and you'll get a subarray sum of `5`. In > > this case, we had a `prefix [0, 1]` which has a prefix sum of `1`. We can subtract that from the current prefix `[0, 1, 2, 3]` and we're left with `[2, 3]`, which has our target sum.
>
> Here's another mathematical way to look at it: we have `curr` and we need to subtract `x` from it to find `k`. The equation is `curr - x = k`. We can rearrange for `x` to get `x = curr - k`.
>

> Given an integer array nums and an integer `k`, find the number of subarrays whose sum is equal to `k`.

> [!Note]
```python
from collections import defaultdict
class Solution:
    def subarraySum(self, nums: List[int], k: int) -> int:
        counter = defaultdict(int)
        prefix = ans = 0
        counter[0] = 1
        for item in nums:
            prefix+=item
            ans +=counter[prefix - k]
            counter[prefix]+=1
        
        return ans
```
### Sample Questions:
- [Contiguous-Array](https://leetcode.com/problems/contiguous-array/)
- [Group-Anagrams](https://leetcode.com/problems/group-anagrams/)
- [max-sum-of-a-pair-with-equal-sum-of-digits](https://leetcode.com/problems/max-sum-of-a-pair-with-equal-sum-of-digits/)

## Tricks:
- To check if string `s1` is a permutation of string `s2` you have to check if `Counter(s1) == Counter(s2)`
- Given two strings s and t, determine if they are isomorphic.

Two strings `s` and `t` are isomorphic if the characters in s can be replaced to get `t`.
All occurrences of a character must be replaced with another character while preserving the order of characters. No two characters may map to the same character, but a character may map to itself.

 
Example :
Input: s = "egg", t = "add"
Output: true
Explanation:
The strings s and t can be made identical by:

Mapping 'e' to 'a'.
Mapping 'g' to 'd'.

```python
  class Solution:
    def isIsomorphic(self, s: str, t: str) -> bool:
        if len(set(s))==len(set(t))==len(set(zip(s,t))):
            return True
        else:
            return False    
 ```

# 3. Trees

```python
class TreeNode:
    def __init__(self, val, left, right):
        self.val = val
        self.left = left
        self.right = right
```
## Depth-first search (DFS)
In a DFS, we prioritize depth by traversing as far down the tree as possible in one direction (until reaching a leaf node) before considering the other direction. For example, let's say we choose left as our priority direction. We move exclusively with `node.left` until the left subtree has been fully explored. Then, we move up one step and explore the right subtree ( the right leaf will be the parent we we do DFS on that again) when done again we move up one level and do everything again until we finish traversing all the nodes.

Trees are named as such because they resemble real-life trees. You can think of the paths of a binary tree as branches growing from the root. DFS chooses a branch and **goes as far down as possible. Once it fully explores the branch, it backtracks until it finds another unexplored branch.**

Because we need to backtrack up the tree after reaching the end of a branch, DFS is typically implemented using recursion, although it is also sometimes done iteratively using a stack. Here is a simple example of recursive DFS to visit every node:
```python
def dfs(node):
    if node == None:
        return

    dfs(node.left)
    dfs(node.right)
    return
```
<p align="center">
<img src="../../images/DFS.gif"  width="500" height="500">
p>
  
> [!IMPORTANT]
> **You should also be comfortable with the idea that during the DFS, many calls to dfs exist simultaneously with their own versions of node.**
The good news is that the structure for performing a DFS is very similar across all problems. It goes as follows:

1. Handle the base case(s). Usually, an empty tree `(node = null)` is a base case.
2. Do some logic for the current node
3. Recursively call on the current node's children
4. Return the answer

As we will see in a moment, steps 2 and 3 may happen in different orders.

> [!IMPORTANT]
The most important thing to understand when it comes to solving binary tree problems is that **each function call solves and returns the answer to the original problem as if the subtree rooted at the current node was the input.** The logic that will be done at each call (step 2) will depend on the problem.


## Preorder traversal

In preorder traversal, logic is done on the current node before moving to the children. Let's say that we wanted to just print the value of each node in the tree to the console. In that case, at any given node, we would print the current node's value, then recursively call the left child, then recursively call the right child.
```python
def preorder_dfs(node):
    if not node:
        return

    print(node.val)
    preorder_dfs(node.left)
    preorder_dfs(node.right)
    return
```

Running the above code on the example tree, we would see the nodes printed in this order: `0, 1, 3, 4, 6, 2, 5`.

Because the logic (printing) is done immediately at the start of each function call, preorder handles nodes in the same order that the function calls happen.
<p align="center">
<img src="../../images/DFS.png"  width="500" height="300">
p>
## Inorder traversal

For inorder traversal, we first recursively call the left child, then perform logic (print in this case) on the current node, and then recursively call the right child. This means no logic will be done until we reach a node without a left child since calling on the left child takes priority over performing logic.
```python
def inorder_dfs(node):
    if not node:
        return

    inorder_dfs(node.left)
    print(node.val)
    inorder_dfs(node.right)
    return
```

Running the above code on the example tree, we would see the nodes printed in this order: `3, 1, 4, 6, 0, 2, 5`.

Notice that for any given node, its value is not printed until all values in the left subtree are printed, and values in its right subtree are not printed until after that.

## Postorder traversal

In postorder traversal, we recursively call on the children first and then perform logic on the current node. This means no logic will be done until we reach a leaf node since calling on the children takes priority over performing logic. In a postorder traversal, the root is the last node where logic is done.
```python
def postorder_dfs(node):
    if not node:
        return

    postorder_dfs(node.left)
    postorder_dfs(node.right)
    print(node.val)
    return
```

Running the above code on the example tree, we would see the nodes printed in this order: `3, 6, 4, 1, 5, 2, 0`.

Notice that for any given node, no values in its right subtree are printed until all values in its left subtree are printed, and its own value is not printed until after that.

> The name of each traversal is describing when the current node's logic is performed.
>
> Pre -> before children
>
> In -> in the middle of children
>
> Post -> after children


## Solving Tree Problems with Recursion
> - for post-order: Assume the `func` returns the correct values for the left node and for the right node then decide how left return value and right return value contribute to the final result.
> - for in-order: ???

***Assume the left tree and the right tree work fine and return the correct value, then write the logic for the current node given the values of L and R***

## Sample Important and Classic Question:
[Lowest Common Ancestor of a Binary Tree](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree/description/)

## BFS code implementations

Just like DFS, the code/implementations for BFS is very similar across different problems. Here is a general format (we're printing the values of the nodes as an example):
```python
from collections import deque

def print_all_nodes(root):
    queue = deque([root])
    while queue:
        nodes_in_current_level = len(queue)
        # do some logic here for the current level

        for _ in range(nodes_in_current_level):
            node = queue.popleft()
            
            # do some logic here on the current node
            print(node.val)

            # put the next level onto the queue
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
```
#  Binary search trees
## A trick:
- ***if a problem is BTS then it has to have unique values***
  
A binary search tree (BST) is a type of binary tree. A BST has the following property:

For each node, all values in its left subtree are less than the value in the node, and all values in its right subtree are greater than the value in the node.
With a binary search tree, operations like searching, adding, and removing can be done in 
`O(logn)` time on average, where `n` is the number of nodes in the tree, using something called binary search, which is the focus of an upcoming chapter.
>[!IMPORTANT]
**Trivia to know: an inorder DFS traversal prioritizing left before right on a BST will handle the nodes in sorted order.**


# 4. Graphs
- ## Sample Questions

> [!IMPORTANT]
**This problem is very important to learn how to solve**
- [reorder-routes-to-make-all-paths-lead-to-the-city-zero](https://leetcode.com/problems/reorder-routes-to-make-all-paths-lead-to-the-city-zero/description/)
```python
from collections import defaultdict

class Solution:
    def minReorder(self, n: int, connections: List[List[int]]) -> int:

        graph = defaultdict(list)
        
        for i, j in connections:
            graph[i].append(j)
            graph[j].append(i)

        direct_path = set([(c1, c2) for c1, c2 in connections])
        seen = [0] * n
        def dfs(node):
            ans = 0
            for neighbor in graph[node]:
                if seen[neighbor] == 0:
                    seen[neighbor] = 1
                    # if the path from node->neighbir is in direct path that means we have to swap it
                    if (node, neighbor) in direct_path: 
                        ans +=1
                    ans +=dfs(neighbor)

            return ans
        seen[0] = 1
        return dfs(0)
```
- [course-schedule](https://leetcode.com/problems/course-schedule/)
>[!IMPORTANT]
**Find if there is a cycle in the graph (directed or undirected)**
```python
from collections import defaultdict

class Graph:
    def __init__(self, num_nodes):
        self.graph = defaultdict(list)
        self.num_nodes = num_nodes

    def add_edge(self, u, v):
        self.graph[u].append(v)

    def is_cyclic_util(self, node, visited, rec_stack):
        visited[node] = True
        rec_stack[node] = True

        for neighbor in self.graph[node]:
            if not visited[neighbor]:
                if self.is_cyclic_util(neighbor, visited, rec_stack):
                    return True
            elif rec_stack[neighbor]:
                return True  # Cycle detected

        rec_stack[node] = False
        return False

    def is_cyclic(self):
        visited = [False] * self.num_nodes
        rec_stack = [False] * self.num_nodes

        for node in range(self.num_nodes):
            if not visited[node]:
                if self.is_cyclic_util(node, visited, rec_stack):
                    return True
        return False

# Example usage:
g = Graph(4)
g.add_edge(0, 1)
g.add_edge(1, 2)
g.add_edge(2, 0)  # Cycle here
g.add_edge(3, 3)  # Self-loop (also a cycle)

print("Graph has cycle?" , g.is_cyclic())

```

## Graph BFS
*In graphs, it is mostly the case when you are asked to find the **shortest path.***

Recall that in binary trees, BFS would visit all nodes at a depth `d` before visiting any node at a depth `d + 1`. BFS visited the nodes according to their distance from the root.

99% of the time, a graph will not have a tree structure. But even then, the same logic still applies. Imagine whatever node you start from as a "root". Then, the neighbors of the root represent the next level, and the neighbors of those nodes represent the level after that.

BFS on a graph always visits nodes according to their distance from the starting point. This is the key idea behind BFS on graphs - every time you visit a node, you must have reached it in the minimum steps possible from wherever you started your BFS.

The above statement was always the case on binary trees, even if you did a DFS, because there is only one possible path to any node from the root. In a graph, there could be many paths from a given starting point to any other node. Using BFS will ensure that out of all possible paths, you take the shortest one.

>[!IMPORTANT]
**Do we need to track the visited nodes?:**
- In classic BFS or DFS We track visited to avoid revisiting the same node — because we're only looking for the shortest path in terms of number of edges. Once we visit a node, there's no point in visiting it again.
- But in problems that we do not look for the shortest path we dont  need it. e.e.g **[cheapest-flights-within-k-stops](https://leetcode.com/problems/cheapest-flights-within-k-stops/)**
  - We're looking for minimum cost, not minimum number of steps, and we allow up to k stops. So: A city might be reachable using 1 stop at a higher cost, and again with 2 stops at a lower cost. Therefore, you must allow re-visiting the same node if you're reaching it
  - With fewer stops but higher cost, or
  - With more stops but cheaper cost.




 ## Implicit graphs
 Sometimes, a graph is more subtle. The input may look nothing like one of the formats we have talked about. Remember that a graph is any abstract collection of elements (nodes) connected by some abstract relationship (edges). **If a problem involves transitioning between states, then try to think about if the states can be nodes and the transition criteria can be edges. Additionally, if the problem wants the shortest path or fewest operations etc., it is a great candidate for BFS.**
>[!IMPORTANT]
**I recommended to solve the BST problems using iterative way by Stack first**
 
- ## Sample Questions
> [!IMPORTANT]
**These problems are very important to learn how to solve**
- [cracking-the-safe](https://leetcode.com/problems/cracking-the-safe/)

- [evaluate-division](https://leetcode.com/problems/evaluate-division/description/)  #very important
- [insert-into-a-binary-search-tree](https://leetcode.com/problems/insert-into-a-binary-search-tree/)

# 5. Binrary Search

**General template:**
```python
def binary_search(arr, target):
    left = 0
    right = len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            # do something
            return
        if arr[mid] > target:
            right = mid - 1
        else:
            left = mid + 1
    
    # target is not in arr, but left is at the insertion point
    return left
```

**Duplicate elements**
If your input has duplicates, you can modify the binary search template to find either the first or the last position of a given element. If target appears multiple times, then the following template will find the left-most index:
learn these templates:
```python
def binary_search(arr, target):
    left = 0
    right = len(arr)
    while left < right:
        mid = (left + right) // 2
        if arr[mid] >= target:
            right = mid
        else:
            left = mid + 1

    return left
```

The following template will find the right-most insertion point (the index of the right-most element plus one):

```python
def binary_search(arr, target):
    left = 0
    right = len(arr)
    while left < right:
        mid = (left + right) // 2
        if arr[mid] > target:
            right = mid
        else:
            left = mid + 1

    return left
```
## something to remember:
if you walk on a sorted array from left to right the first algo gives you the index of the target in the array immediately after seeing the right spot. So if e.g. `target = 5` and `arr = [0, 1, 2, 3, 3, 3, 5, 5, 5, 6]` then the first and second algorithm return `left = 6`. If `target = 4` the first and second algorithm return `left = 6` again. The last algorithm, however; is more patient. It walks from left to right on the array and chacks target with the elements from left to right. If and `target== element` it goes to the next one untill target < element. So it chooses the index *after* the last possible place. Here e.g. if `target=5` the last algorithm returns `9` it means put the new element in this position. If `target = 4` it retuns `6`.

## Binary Search On solution spaces (sort is not needed because we do not have an array to find the index for)

There is a more creative way to use binary search - on a solution space/answer. A very common type of problem is "what is the max/min that something can be done". Binary search can be used if the following criteria are met:

You can quickly (in `O(n)` or better) verify if the task is possible for a given number x.

1. If the task is possible for a number `x`, and you are looking for:
  - A maximum, then it is also possible for all numbers less than `x`.
  - A minimum, then it is also possible for all numbers greater than `x`.
2. If the task is not possible for a number `x`, and you are looking for:
  - A maximum, then it is also impossible for all numbers greater than `x`.
  - A minimum, then it is also impossible for all numbers less than `x`.

The 2nd and 3rd requirements imply that there are two "zones". One where it is possible and one where it is impossible. The zones have no breaks, no overlap, and are separated by a threshold.


## Super Important 
[path-with-minimum-effort](https://leetcode.com/problems/path-with-minimum-effort/description/)




## A note on implementation return left for minimization and right for maximization problems
If we ask for a minimum, in all solutions, we return `left`.

If a problem is instead asking for a maximum, then `left` will not actually be the correct answer at the end. Instead, we should return `right`.

Why does left point to the answer when looking for a minimum, but right points to the answer when looking for a maximum?

Let's say we're looking for a minimum and the answer is `x`. After doing `check(x)`, we set `right = x - 1` because `check(x)` will return `true`, and we move the right bound to look for a better answer. As you can see, the correct answer is actually outside of our search space now. That means every future iteration of check is going to fail, which means we will continuously increase `left` until eventually we try `check(x - 1)`. This will fail and set `left = (x - 1) + 1 = x`. Our while loop terminates because `left > right`, and `left` is at the answer.

If we are instead looking for a maximum, after performing check(x), we set `left = x + 1`. Again, the correct answer is outside of the search space and all future checks will fail. Eventually, we try `check(x + 1)`, fail, and set `right = (x + 1) - 1 = x`. The loop terminates because `right < left`, and right is pointing at the answer.


# 6. Backtracking

In backtracking we have a couple of imporatant templates

Try building a tree for yourself before solving backtracking problems, it helps understanding how to manage the index

## temlate 1: Permutation of non repetitious numbers: if you saw something before you dont want to see it again.

```python
nums = [1, 2, 3, 4]
def backtrack(curr):
    if condition:
        ans.append(curr[:])

    for item in nums:
        if not item in curr:
            curr.append(item)
            backtrack(curr)
            curr.pop()
```
## temlate 2: combination of non repetitious numbers: if you incorporated a node, then put that node aside and don't work with it anymore.
```python
#combanation of c(n,k)
class Solution:
    def combine(self, n: int, k: int) -> List[List[int]]:
        def backtrack(curr, i):
            if len(curr) == k:
                ans.append(curr[:])
                return
            
            for num in range(i, n + 1):
                curr.append(num)
                backtrack(curr, num + 1)
                curr.pop()
        
        ans = []
        backtrack([], 1)
        return ans
```
## temlate 3: combination of repetitious numbers: if you incorporated a node, then put that node aside and don't work with it anymore.
Given an array of distinct positive integers candidates and a target integer target, return a list of all unique combinations of candidates where the chosen numbers sum to target. The same number may be chosen from candidates an unlimited number of times. Two combinations are unique if the frequency of at least one of the chosen numbers is different.

***We use this template if we don't want to use the pre-used members e.g. [(1,1,6), (1, 6, 1), (6, 1, 1)] are all the same, therefore we put `i` in the arguments so that when transitioning to a new parent children we do not use the previous members***

<p align="center">
<img src="../../images/recursion.jpg"  width="500" height="500">
p>

```python
class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        def backtrack(path, start, curr):
            if curr == target:
                ans.append(path[:])
                return

            for i in range(start, len(candidates)):
                num = candidates[i]
                if curr + num <= target:
                    path.append(num)
                    backtrack(path, i, curr + num)
                    path.pop()     
        
        ans = []
        backtrack([], 0, 0)
        return and
```

## Note to remember: 
Try to create a tree. Look at curr after each backtrack return. When a backtrack hits return that means it has hit a **node leaf**. Look at curr at that moment. Sometimes our tree do not append on leaf nodes, but they append on nodes  themeselves. At this time you should not return anything in your backtracking (look at subset example). Right after the return we have this tem `curr.pop()` try to imaging if curr pops the next item is  the item in  `for ... in ...` if we are done with all the children of the first child, that means we are at the end of the for loop and a new number will be chosen from scratch. That means `for i in range(start, end)` now what should this `start` be?  Also a very important thing you should do is to create a tree and beside each node write the status of the variables so you can track them.

# Template 4 Partitioning.
In many questions you are asked to partition the input based on a criteria the best way is to create segments like th code below

```python
def backtrack(input, curr, start=0):
  if condition:
    ans.append(curr[:])
  return

  for end in range(start + 1, len(input)):
    segment = input[start: end]
    if condition:
      curr.append(seqment)
      backtrack(input, curr, end)
      curr.pop()
```


# Sample Questions


[Word Search](https://leetcode.com/problems/word-search/?envType=problem-list-v2&envId=backtracking)

**partitioning**
[Valid IP ADDRESS](https://leetcode.com/problems/restore-ip-addresses/description/?envType=problem-list-v2&envId=backtracking)
[Punishment Number](https://leetcode.com/problems/find-the-punishment-number-of-an-integer/description/?envType=problem-list-v2&envId=backtracking)


# 7. Heap
in heap the first element to pop in the minimum element in the heap. If we push tuples in the heap the fitst element to pop is the tuple with minimum first element. If two tuples have same first element then the tuple with the second element being minimum is poped.


One common type of interview problem is one that asks you to find the k best elements, with "best" being defined by the problem. The easiest way to solve these problems is to just sort the input according to the criteria defined in the problem, and then return the top `k` elements. This has a time complexity of  `O(n⋅logn)` if `n` is the length of the input.

Using a heap, we can instead find the top k elements in  `O(n⋅logk)`. Logically, `k < n`, so this is an improvement. Practically, because 
`log` is so fast anyway, it probably isn't a big deal in terms of a speed increase. But when interviewers give you these kinds of problems, it is these small improvements that they are looking for.

What is the improvement? Create a max heap at the start, iterate over the input while pushing every element on the heap (according to the problem's criteria), and pop from the heap once the size exceeds `k`. Because the heap's size is bounded by `k`, then all heap operations are at worst 
`O(logk)`. Multiply this by n iterations to get `O(n⋅logk)`. Because we are using a max heap and we are popping from the heap according to the problem criteria, pops remove the "worst" elements, so at the end, the `k` "best" elements will remain in the heap.

Q: Given an integer array nums and an integer k, return the k most frequent elements. It is guaranteed that the answer is unique.

```python
from collections import Counter
import heapq

class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        counts = Counter(nums)
        heap = []
        
        for key, val in counts.items():
            heapq.heappush(heap, (val, key))
            if len(heap) > k:
                heapq.heappop(heap)
        
        return [pair[1] for pair in heap]
```
