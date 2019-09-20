# Partnet_data
我们用简单桌子作为一个例子来说明我们的数据是如何组织的。
![image](https://github.com/PeppaZhu/Partnet_data/blob/master/pictures/picture1.png)  
如上图所示，这个桌子有五个part，每个part用一个box来表示，标号从box1到box5。五个part被标识为两类：桌面和腿；用数字0标识桌面，用数字1标识腿。其中box1和box4具有对称关系，box2和box3具有对称关系。

![image](https://github.com/PeppaZhu/Partnet_data/blob/master/pictures/picture2.png)
 
我们将该模型组织成一颗partnet tree，如上图所示。每个叶子节点都代表一个box，也即一个part。那为什么上面提到的例子模型有五个box，而这棵树只有3个叶子节点呢？我们发现因为其中box1和box4为一个对称关系，box2和box3为另一个对称关系，因此在一个对称关系中只需要存储一个代表box，然后只需要再存储对称参数，就可以根据对称参数得到该对称关系的其他box。

我们把一颗partnet tree的节点分为三类，0表示叶子节点，1表示邻接节点（它的左孩子与右孩子为邻接关系），2表示对称节点（它只有左孩子，左孩子是一个对称关系中的代表box。然后再存储对称参数即可）。

