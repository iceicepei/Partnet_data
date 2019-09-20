# Partnet_data
我们用简单桌子作为一个例子来说明我们的数据是如何组织的。
![image](https://github.com/PeppaZhu/Partnet_data/blob/master/pictures/picture1.png)  
如上图所示，这个桌子有五个part，每个part用一个box来表示，标号从box1到box5。五个part被标识为两类：桌面和腿；用数字0标识桌面，用数字1标识腿。其中box1和box4具有对称关系，box2和box3具有对称关系。

![image](https://github.com/PeppaZhu/Partnet_data/blob/master/pictures/picture2.png)
 
我们将该模型组织成一颗partnet tree，如上图所示。每个叶子节点都代表一个box，也即一个part。那为什么上面提到的例子模型有五个box，而这棵树只有3个叶子节点呢？我们发现因为其中box1和box4为一个对称关系，box2和box3为另一个对称关系，因此在一个对称关系中只需要存储一个代表box，然后只需要再存储对称参数，就可以根据对称参数得到该对称关系的其他box。

我们把一颗partnet tree的节点分为三类，0表示叶子节点，1表示邻接节点（它的左孩子与右孩子为邻接关系），2表示对称节点（它只有左孩子，左孩子是一个对称关系中的代表box。然后再存储对称参数即可）。

我们的每一类数据模型一共有7个文件夹。  ops文件夹下每一个mat文件存储一颗partnet tree的节点对应类型，如下图所示为简单桌子的partnet tree的节点对应类型。
![image](https://github.com/PeppaZhu/Partnet_data/blob/master/pictures/picture3.png)  

part_fix文件夹下的mat文件存储一个模型的叶子节点对应的box索引。如下图所示，序号为3、1、2的叶子节点分别对应box5、box4、box3。  
![image](https://github.com/PeppaZhu/Partnet_data/blob/master/pictures/picture5.png)

boxes文件夹下的mat文件存储一个模型的每个叶子节点对应的box。

labels文件夹下的mat文件对应每个叶子节点的类型标识。如下图所示，叶子节点3表示桌面（标为0），叶子节点1和2表示腿（标为1）。  
![image](https://github.com/PeppaZhu/Partnet_data/blob/master/pictures/picture6.png)

syms文件夹下的mat文件对应存储每个对称节点的对称参数。简单桌子的例子中有两组对称关系，因此存储了两个对称参数。在partnet tree中分别对应节点4和节点5。

