# Partnet_data
我们用简单桌子作为一个例子来说明我们的数据是如何组织的。
![image](https://github.com/PeppaZhu/Partnet_data/blob/master/pictures/picture1.png)  
如上图所示，这个桌子有五个part，每个part用一个box来表示，标号从box1到box5。五个part被标识为两类：桌面和腿；用数字0标识桌面，用数字1标识腿。其中box1和box4具有对称关系，box2和box3具有对称关系。

我们将模型组织成一颗partnet tree，如下图所示。
