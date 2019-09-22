# PartNet_Dataset
Dataset for PartNet: A Recursive Part Decomposition Network for Fine-grained and Hierarchical Shape Segmentation

### Introduction

We present PartNet: a recursive part decomposition network for fine-grained and hierarchical shape segmentation. We add another dataset for it，which consists of 24 object categories.This dataset is derived from the dataset of the paper published by Stanford University called PartNet(A Large-scale Benchmark for Fine-grained and Hierarchical Part-level 3D Object Understanding). We organize their dataset into the form of our PartNet dataset for the purpose of expanding our dataset.This dataset could enable and serve as a catalyst for many tasks such as shape analysis, dynamic 3D scene modeling and simulation, affordance analysis, and others. 

### About the Dataset

The dataset contains 22699 3D shapes covering 24 shape categories: lamp (2603),table (5701),cutting_instrument (486),bag (158), table (115),bottle (511),bowl (100),clock (426),display (329),dishwasher (198),door (198),earphone (269),faucet (826),hat (251),storage (2546),keyboard (109),laptop (92),microwave (81),mug (232),refrigerator (209),scissors (112),trashcan (296),vase (411) and chair (6440).

### Demo

我们用简单桌子作为一个例子来说明我们的数据是如何组织的。具体分为三个部分进行说明。在第一部分，我们通过一个图来说明一个模型是如何通过obj文件和obb文件来表示的。在第二部分，我们也通过图示说明一颗partnet tree的结构。在第三部分，我们将说明每个文件夹的具体含义。

#### 1. Representing the model with obj file
![image](https://github.com/PeppaZhu/Partnet_data/blob/master/pictures/picture1.png)  
如上图所示，这是一个显示在Deep Exploration中的obj模型，它表示一个简单桌子。这个桌子有五个part，每个part用一个box来表示，标号从box1到box5。五个part被标识为两类：桌面和腿；用数字0标识桌面，用数字1标识腿。其中box1和box4具有对称关系，box2和box3具有对称关系。

#### 2. Structure of a partnet tree

![image](https://github.com/PeppaZhu/Partnet_data/blob/master/pictures/picture2.png)
 
我们将该模型组织成一颗partnet tree，如上图所示。每个叶子节点都代表一个box，也即一个part。那为什么上面提到的例子模型有五个box，而这棵树只有3个叶子节点呢？我们发现因为其中box1和box4为一个对称关系，box2和box3为另一个对称关系，因此在一个对称关系中只需要存储一个代表box，然后只需要再存储对称参数，就可以根据对称参数得到该对称关系的其他box。

我们把一颗partnet tree的节点分为三类，0表示叶子节点，1表示邻接节点（它的左孩子与右孩子为邻接关系），2表示对称节点（它只有左孩子，左孩子是一个对称关系中的代表box。然后再存储对称参数即可）。

#### 3. Folder instructions

我们的每一类数据模型一共有7个文件夹。  

##### A. ops文件夹
ops文件夹下每一个mat文件存储一颗partnet tree的节点对应类型，如下图所示为简单桌子的partnet tree的节点对应类型。（0表示叶子节点，1表示邻接节点，2表示对称节点）。
![image](https://github.com/PeppaZhu/Partnet_data/blob/master/pictures/picture3.png)  

##### B. part_fix文件夹
part_fix文件夹下的mat文件存储一个模型的叶子节点对应的box索引。如下图所示，序号为3、1、2的叶子节点分别对应box5、box4、box3。  
![image](https://github.com/PeppaZhu/Partnet_data/blob/master/pictures/picture5.png)

##### C. boxes文件夹
boxes文件夹下的mat文件存储一个模型的每个叶子节点对应的box。

##### D. labels文件夹
labels文件夹下的mat文件对应每个叶子节点的类型标识。如下图所示，叶子节点3表示桌面（标为0），叶子节点1和2表示腿（标为1）。  
![image](https://github.com/PeppaZhu/Partnet_data/blob/master/pictures/picture6.png)

##### E. syms文件夹
syms文件夹下的mat文件对应存储每个对称节点的对称参数。简单桌子的例子中有两组对称关系，因此存储了两个对称参数。在partnet tree中分别对应节点4和节点5。

##### F. models文件夹和obbs文件夹
models文件夹下存储.obj形式的模型。obbs文件夹下存储每个模型对应的obb文件。

### 数据集获取
你可以在[这里](https://www.dropbox.com/sh/o04yue60joxwkml/AACS0HmBybSgEruM3C5bmAvJa?dl=0)下载数据集。
