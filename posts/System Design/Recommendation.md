# Video Recommendation System

## Requirement Clarification:

1- Do we recommend similar videos to the one we are watching now or do we personalize the recommendation?(personalized)

2- Users are allocated world wide with different languages?(yes)

3- Can I construct a dataset based on the user interaction? (yes)

4- Howmany videos do we have? billions

5- How fast the system should be performing 200ms? (yes)

## some objectives:  

1- Maximize the user click (not good, you may click but not watch)

2- Maximize completed videos(not good, you may watch only short videos)

3- Maxiize total watch time(could be)

**4- Maximize relevance: choose the videos that the user presses like or finishes at least half of it.**

## system input output

Input: user, user features, 

output: ranked videos

## ML category:
1- Content based filtering: Use video features to recommend. (Good for cols-start, but doesnt explore new interests)

2- Collaborative filtering: look at what you liked and others liked and recommend things others like you liked. (Efficient, explores new interests but not niche interests, bad for cold-start)

3- Hybrid   collaboratve first and content base second. 

## Data Engineering
1- User data

2- Video Data

3-User interaction Data

## Videos

| ID  | Tags  | Length | Language | Number of Likes | Comments | Views | Rating (PG, PG-13) |
------|-------|--------|----------|------------------|----------|--------|---------------------|




