# Partnet_data
We use a simple table as an example to illustrate how our data is organized.
![image](https://github.com/PeppaZhu/Partnet_data/blob/master/pictures/picture1.png)  
As shown in the figure above, the table has five parts, each of which is represented by a box with labels ranging from box 1 to box 5. Five parts are identified as two categories: desktop and legs; desktop is marked with number 0 and legs are marked with number 1. Among them, box 1 and box 4 are symmetrical, while box 2 and box 3 are symmetrical.

![image](https://github.com/PeppaZhu/Partnet_data/blob/master/pictures/picture2.png)
 
We organize the model into a partnet tree, as shown in the figure above. Each leaf node represents a box, or part. So why does the example model mentioned above have five boxes and this tree only has three leaf nodes? We find that because box 1 and box 4 are symmetrical and box 2 and box 3 are symmetrical , only one representative box needs to be stored in a symmetrical relationship, and then only one symmetrical parameter needs to be stored to obtain the other boxes of the symmetrical relationship according to the symmetrical parameters.

我们把一颗partnet tree的节点分为三类，0表示叶子节点，1表示邻接节点（它的左孩子与右孩子为邻接关系），2表示对称节点（它只有左孩子，左孩子是一个对称关系中的代表box。然后再存储对称参数即可）。

我们的每一类数据模型一共有7个文件夹。  ops文件夹下每一个mat文件存储一颗partnet tree的节点对应类型，如下图所示为简单桌子的partnet tree的节点对应类型。
![image](https://github.com/PeppaZhu/Partnet_data/blob/master/pictures/picture3.png)  

part_fix文件夹下的mat文件存储一个模型的叶子节点对应的box索引。如下图所示，序号为3、1、2的叶子节点分别对应box5、box4、box3。  
![image](https://github.com/PeppaZhu/Partnet_data/blob/master/pictures/picture5.png)

boxes文件夹下的mat文件存储一个模型的每个叶子节点对应的box。

labels文件夹下的mat文件对应每个叶子节点的类型标识。如下图所示，叶子节点3表示桌面（标为0），叶子节点1和2表示腿（标为1）。  
![image](https://github.com/PeppaZhu/Partnet_data/blob/master/pictures/picture6.png)

syms文件夹下的mat文件对应存储每个对称节点的对称参数。简单桌子的例子中有两组对称关系，因此存储了两个对称参数。在partnet tree中分别对应节点4和节点5。

models文件夹下存储.obj形式的模型。obbs文件夹下存储每个模型对应的obb文件。

