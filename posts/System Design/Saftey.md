# Multimodal Harmful posts

## **Objective**
We want to design a harmful contentdetectionsystem which identifies harmful posts, then deletes or demotes them and informs user why the post was identified as harmful. A post content might be text, image, video or any combination of these, and content can be in different languages. Users can report harmful posts.


![img1](/images/harmful_sys_dsgn.png)


### Late Fusion: 
is not good because each modality might not be harmful by its own but the combination may be harmful. So using different models for each modality and then combine the results would not perform as well.

### Early fusion: 
1. Extract features for each modality
2. fuse them
3. pass them trough a shared backbone
4. get a single fused transformed feature
5. pass them through multi task classification heads. (binary classification for each label)
  5.1- Single Binay head: Doenst work because we a would not know what was harmful (nudity, gore, violence)
  5.2- One binary classifiedr per harmful class: Its not ideal because different labels might need different proccessign and we need expert heads for each label.
  5.3- Multi-task classifier: one head per llabel.

## **Data Engineering**:
1- Post(text, video, image)
2- Reactions(likes, reposts, reposrts, comments)
3- User(username, id, location, age, gender, followers, violence history, report history)


## **models**
Text: BERT or DistilmBERT, get the text and take a CLS tike in the output as it semantics.
Image: ViT or ClIP based transformers, same we can the CLS token as the embedding.
User reactions= [no. likes, no. reposts, no. reposrts] normalize and get the embedding
Comments: use DistilBERT to get the embedding for each and then concatenate all
get the feature from each model and fuse them.

## **Dataset Preparation**:
We can use the human annotators for posts to create validation set, thi sis time consuming. So we can have smaller dataset.
We can use the user's reports for labeling

## **loss function**
We use Binary cross entropy for each task. We can handle class imbalance using focal loss, resampling or weighted loss funcction.

## **Evaluation**
**offline:** PR-AUC or ROC-AUc are best, because precision or recall by itself cant be reliable because of the long tail characteristic of profanity and violence.
**online:** 

$$
\begin{align}
\text{Apeals} &=& \frac{\text{Number of reversed apeals}}{\text{Number of harmful posts detected by the system}}\\
\text{Proactive Rate}&=&\frac{\text{Number of harmful posts detected by the system}}{\text{Number of harmful posts detected by the system + reposted by users}}
\end{align}
$$


## **Serving:** 
1- **Harmful detection service** is called and it has two parts (hash tables for images/text-data if we find specific words in the text, or image DNA(finger print) we can immediately report it without using the model.
2- If a user with harmfull content history posts something, we reroute it to a heavier AI model.
3- We use a load balancer for the traffic, the model is loaded to Kubernetes, for processing the data. 
4- if a post is flagged as harmful, a notification is sent to the user accordingly by the **Violation enforcement service** and the post will be eliminated. The user will also be added to the hash table of harmful users.
5- if a post has a low confidence for harmful then we send it to manual review.
6- We evaluate the system performance using the KPIs. If a post is reported as harmful but we missed detecting it or if a post was flagged harmful but was apealed and passed we add the to our hard dataset for future retraining.

load balancer can route the job to the clusters that are getting their batch full. So we can do batch processing for budget saving. If a user is with violence history we can immediately process the post.
