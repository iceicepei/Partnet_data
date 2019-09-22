# PartNet_Dataset
Dataset for PartNet: A Recursive Part Decomposition Network for Fine-grained and Hierarchical Shape Segmentation

### Introduction

We present PartNet: a recursive part decomposition network for fine-grained and hierarchical shape segmentation. We add another dataset for it，which consists of 24 object categories.This dataset is derived from the dataset of the paper published by Stanford University called PartNet(A Large-scale Benchmark for Fine-grained and Hierarchical Part-level 3D Object Understanding). We organize their dataset into the form of our PartNet dataset for the purpose of expanding our dataset.This dataset could enable and serve as a catalyst for many tasks such as shape analysis, dynamic 3D scene modeling and simulation, affordance analysis, and others. 

### About the Dataset

The dataset contains 22699 3D shapes covering 24 shape categories: lamp (2603),table (5701),cutting_instrument (486),bag (158), table (115),bottle (511),bowl (100),clock (426),display (329),dishwasher (198),door (198),earphone (269),faucet (826),hat (251),storage (2546),keyboard (109),laptop (92),microwave (81),mug (232),refrigerator (209),scissors (112),trashcan (296),vase (411) and chair (6440).

### Demo

We use a simple table as an example to illustrate how our data is organized.

#### 1. Representing the model with obj file
![image](https://github.com/PeppaZhu/Partnet_data/blob/master/pictures/picture1.png)  
As shown in the figure above, this is an obj model displayed in Deep Exploration, which represents a simple table.The table has five parts, each of which is represented by a box with labels ranging from box 1 to box 5. Five parts are identified as two categories: desktop and legs; desktop is marked with number 0 and legs are marked with number 1. Among them, box 1 and box 4 are symmetrical, while box 2 and box 3 are symmetrical.

#### 2. Structure of a partnet tree

![image](https://github.com/PeppaZhu/Partnet_data/blob/master/pictures/picture2.png)
 
We organize the model into a partnet tree, as shown in the figure above. Each leaf node represents a box, or part. So why does the example model mentioned above have five boxes and this tree only has three leaf nodes? We find that because box 1 and box 4 are symmetrical and box 2 and box 3 are symmetrical , only one representative box needs to be stored in a symmetrical relationship, and then only one symmetrical parameter needs to be stored to obtain the other boxes of the symmetrical relationship according to the symmetrical parameters.

We classify nodes of a partnet tree into three categories: 0 for leaf node, 1 for adjacent node (its left child is adjacent to the right child), and 2 for symmetric node (it has only left child, and the left child is a representative box in the symmetric relationship. Then store the symmetric parameters.)

There are seven folders in each of our data models. Each mat file in the ops folder stores a corresponding type of the node of a partnet tree, as shown in the figure below for the corresponding type of the node of a simple table.(0 for leaf nodes, 1 for adjacent nodes and 2 for symmetric nodes).
![image](https://github.com/PeppaZhu/Partnet_data/blob/master/pictures/picture3.png)  

The mat file under the part_fix folder stores the box index corresponding to the leaf node of a model. As shown in the following figure, leaf nodes with serial numbers 3, 1 and 2 correspond to box 5, box 4 and box 3, respectively.  
![image](https://github.com/PeppaZhu/Partnet_data/blob/master/pictures/picture5.png)

The mat file under the boxes folder stores the box corresponding to each leaf node of a model.

The mat file under the labels folder corresponds to the type identification of each leaf node. As shown in the following figure, leaf node 3 represents the desktop (marked by number 0), and leaf nodes 1 and 2 represent legs (marked by number 1).  
![image](https://github.com/PeppaZhu/Partnet_data/blob/master/pictures/picture6.png)

The mat file under Syms folder stores symmetrical parameters of each symmetrical node. In the example of a simple table, there are two sets of symmetrical relationships, so two symmetrical parameters are stored. In partnet tree, corresponding nodes 4 and 5 respectively.

The models folder stores models in .obj form. The obbs folder stores the corresponding obb files for each model.
