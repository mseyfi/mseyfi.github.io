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

| ID  | Tags  | Title  | Length | Language | Number of Likes | Comments | Views | Rating (PG, PG-13) |
------|-------|--------|--------|----------|-----------------|----------|-------|--------------------|
Embedding| CBOW|BERT | log+normalize|Embedding|log+normalize|BERT|log+normalzie|Embedding



## User-Video Interactions

| User ID | Video ID | Interaction Type | Interaction Value      | Location     | Time Stamp  |
|---------|----------|------------------|-------------------------|--------------|-------------|
| 4       | 18       | like             |                         | 38.89-077.1  | 128648624   |
| 15      | 6        | watch            | 88mins                  | 38.9-23.2    | 12872496    |
| 9       | -        | search           | basics of clustering    | 22.3-54.2    | 3864123     |
| 8       | 12       | comment          | amazing job             | 37.23-12293  | 125143653   |
|Embedding|Embedding | Embedding |  BERT or log + normalization| cluster + embedding| Bucket + embedding|

For video Interaction history we create a feature for each user. For each video id we create interaction feature for the specific user and then aggregate all of them.
We can alternatively create features for searched videos and videos with comments, and liked videos, each separately and then concatenate the result.


## User 

### Demographics

| Age | Gender | Location | Username | ID  | Language | Time Zone |
|-----|--------|----------|----------|-----|----------|-----------|
| bucke+Embedding|Embedding|Clustered+embedding|BERT|Embedding|Embedding|Embedding|


### Contextual Information
|Time of Day | Device | Day of Week|
-------------|--------|------------|
bucket+embedding|embedding|embedding|


User:  | Demographics | Contextual | Historical Interactions |
       |--------------|------------|-------------------------| 


## Model development
For each user, we craft user features and concatenate them with their video-interaction history. For each user, we have a specific feature. For each video, we also have a feature

## Matrix Factorization: Collaborative Filtering
If observation is the feedback a user give to a video, we create a feedback matrix and try to optimize the loss below using gradient descent

$$
\sum_{(i,j) \in obs} (1 - \lambda) (A_{ij} - U_i^T V_j)^2 + \lambda \sum_{(i,j) \notin obs} (A_{ij} - U_i^T V_j)^2
$$

user i1 likes video j

user i2 also likes video j

Look for other videos that user i2 likes and multiply those video features by user i2 features and get the top k.

## Two Towers

User features and their interactions with tower 1 and video features to tower 2, both create embedding. 

we can train them using contrastive learning or BCE

positive videos: those who the user liked, or watched at least half
Negative Videos: those who the user disliked and randomly select from the one the user did not watch.

Inference:
if its not cold start we start with collaborative filtering to narrow down some videos. Then select from those

ANN and FAISS is used to search for video embeddings in the system,

## Evaluation:

**offline**
precision@k: Proportion of number of relevant videos among the top k videos for multiple k values
mAP: 
$mAp = \frac{\sum_i^k \text{Precision@i if i's video is relevant to the user}}{total relevant items}$

Diversity: How much the videos are diverse. for this we can average the 1-1 similarity scores betweenthe videos and they should be close to zero

**online**
Click Through Rate = number of clicked videos / number of recommended videos

Number of completed videos

Total watch time

Explicit User feedback

## Serving


