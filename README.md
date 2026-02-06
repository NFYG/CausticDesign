参考：High-contrast Computational Caustic Design
必须先安装 CGAL 环境，绝大多数内容依赖该计算几何库。
项目毫无架构设计，绝大多数内容都直接塞进 main.cpp 中，lbfgs.hpp 优化算法库提供 L-BFGS 优化算法，stb_image.h 负责读取图像，tinyxml2.cpp/h 负责将中间结果绘制为 svg 矢量图，便于可视化地检查中间过程的输出。
两个python文件负责用B样条曲面拟合cpp文件算出来的法向量，并用cadquery库生成stl文件
