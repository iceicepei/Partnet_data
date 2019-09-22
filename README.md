# PartNet_Dataset
Dataset for PartNet: A Recursive Part Decomposition Network for Fine-grained and Hierarchical Shape Segmentation

### Introduction

We present PartNet: a recursive part decomposition network for fine-grained and hierarchical shape segmentation. We expand another dataset for it，which consists of 24 object categories.This dataset is derived from the dataset of the paper published by Stanford University called PartNet(A Large-scale Benchmark for Fine-grained and Hierarchical Part-level 3D Object Understanding). We organize their dataset into the form of our PartNet dataset for the purpose of expanding our dataset.This dataset could enable and serve as a catalyst for many tasks such as shape analysis, dynamic 3D scene modeling and simulation, affordance analysis, and others. 

### Dataset information

The dataset contains 22699 3D shapes covering 24 shape categories:

|  category_name  |  lamp   | table | cutting_instrument   |  bag   | bed  | bottle   | bowl   | clock   | display   | dishwasher   | door   | earphone   | faucet   | hat   | storage   | keyboard   | laptop   | microwave   | mug   | refrigerator   | scissors   | trashcan   | vase   | chair   |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| number of shapes | 2603 | 5701 | 486 | 158 | 115 | 511 | 100 | 426 | 329 | 198 | 198 | 269 | 826 | 251 | 2546 | 109 | 92 | 81 | 232 | 209 | 112 | 296 | 411 | 6440 |
| number of parts | 12200 | 28958 | 1571 | 358 | 2420 | 1432 | 207 | 1151 | 1174 | 838 | 585 | 1193 | 4025 | 588 | 34564 | 5587 | 270 | 346 | 291 | 947 | 394 | 2565 | 1013 | 40879 |
| maximum parts per shape | 122 | 47 | 17 | 27 | 21 | 9 | 25 | 14 | 5 | 8 | 9 | 8 | 18 | 3 | 100 | 63 | 3 | 8 | 4 | 11 | 5 | 43 | 8 | 30 |
| minimum parts per shape | 2 | 2 | 2 | 2 | 4 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 13 | 2 | 3 | 2 | 2 | 2 | 2 | 2 | 2 |

### Demo

We use a simple table as an example to illustrate how our data is organized. This chapter is specifically divided into three parts to elaborate the demo. In the first part, we will illustrate how a model is represented by an obj file and an obb file. In the second part, we will illustrate the structure of a partnet tree. In the third part, we will explain the details of each folder.

#### 1. Representing the model with obj file
![image](https://github.com/PeppaZhu/Partnet_data/blob/master/pictures/picture1.png)  
As shown in the figure above, this is an obj model displayed in Deep Exploration, which represents a simple table.The table has five parts, each of which is represented by a box with labels ranging from box 1 to box 5. Five parts are identified as two categories: desktop and legs; desktop is marked with number 0 and legs are marked with number 1. Among them, box 1 and box 4 are symmetrical, while box 2 and box 3 are symmetrical.

#### 2. Structure of a partnet tree

![image](https://github.com/PeppaZhu/Partnet_data/blob/master/pictures/picture2.png)
 
We organize the model into a partnet tree, as shown in the figure above. Each leaf node represents a box, or part. So why does the example model mentioned above have five boxes and this tree only has three leaf nodes? We find that because box 1 and box 4 are symmetrical and box 2 and box 3 are symmetrical , only one representative box needs to be stored in a symmetrical relationship, and then only one symmetrical parameter needs to be stored to obtain the other boxes of the symmetrical relationship according to the symmetrical parameters.

We classify nodes of a partnet tree into three categories: 0 for leaf node, 1 for adjacent node (its left child is adjacent to the right child), and 2 for symmetric node (it has only left child, and the left child is a representative box in the symmetric relationship. Then store the symmetric parameters.)

#### 3. Folder instructions

There are seven folders in each of our data models. 

##### A. the ops folder
Each mat file in the ops folder stores a corresponding type of the node of a partnet tree, as shown in the figure below for the corresponding type of the node of a simple table.(0 for leaf nodes, 1 for adjacent nodes and 2 for symmetric nodes).
![image](https://github.com/PeppaZhu/Partnet_data/blob/master/pictures/picture3.png)  

##### B. the part_fix folder
The mat file under the part_fix folder stores the box index corresponding to the leaf node of a model. As shown in the following figure, leaf nodes with serial numbers 3, 1 and 2 correspond to box 5, box 4 and box 3, respectively.  
![image](https://github.com/PeppaZhu/Partnet_data/blob/master/pictures/picture5.png)

##### C. the boxes folder
The mat file under the boxes folder stores the box corresponding to each leaf node of a model.

##### D. the labels folder
The mat file under the labels folder corresponds to the type identification of each leaf node. As shown in the following figure, leaf node 3 represents the desktop (marked by number 0), and leaf nodes 1 and 2 represent legs (marked by number 1).  
![image](https://github.com/PeppaZhu/Partnet_data/blob/master/pictures/picture6.png)

##### E. the syms folder
The mat file under the syms folder stores symmetrical parameters of each symmetrical node. In the example of a simple table, there are two sets of symmetrical relationships, so two symmetrical parameters are stored. In partnet tree, corresponding nodes 4 and 5 respectively.

##### F. the models folder & the obbs folder
The models folder stores models in .obj form. The obbs folder stores the corresponding obb files for each model.

### Accessing to the dataset
You could get the dataset [Here](https://www.dropbox.com/sh/o04yue60joxwkml/AACS0HmBybSgEruM3C5bmAvJa?dl=0).
