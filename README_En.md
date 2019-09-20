# Partnet_data
We use a simple table as an example to illustrate how our data is organized.
![image](https://github.com/PeppaZhu/Partnet_data/blob/master/pictures/picture1.png)  
As shown in the figure above, the table has five parts, each of which is represented by a box with labels ranging from box 1 to box 5. Five parts are identified as two categories: desktop and legs; desktop is marked with number 0 and legs are marked with number 1. Among them, box 1 and box 4 are symmetrical, while box 2 and box 3 are symmetrical.

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

