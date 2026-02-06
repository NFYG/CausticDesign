# import cadquery as cq
import torch
import torch.nn as nn
import numpy as np
import math
import pickle


class calculate_biquadratic_bspline():

    def __init__(self, x0: float, y0: float, grid_size: float, m: int, n: int, ctrl_points_z: np.ndarray):

        assert(ctrl_points_z.ndim == 2 and ctrl_points_z.shape[0] == n and ctrl_points_z.shape[1] == m)

        self.x0 = x0
        self.y0 = y0
        self.grid_size = grid_size
        self.m = m # X
        self.n = n # Y
        self.ctrl_points_z = ctrl_points_z
    
    def calculate(self, x: float, y: float):
        temp_x = (x - self.x0)/self.grid_size + 0.5
        temp_y = (y - self.y0)/self.grid_size + 0.5
        i = math.floor( temp_x )
        j = math.floor( temp_y )
        # assert(i > 0 and j > 0),f"xy:{x,y}, ij:{i,j}"
        u = temp_x - i
        w = temp_y - j

        cu = u*u
        bu = -2*cu + 2*u + 1
        au = 2 - bu - cu

        cw = w*w
        bw = -2*cw + 2*w + 1
        aw = 2 - bw - cw


        # z0 = self.ctrl_points_z[j-1, i-1]
        # z1 = self.ctrl_points_z[j-1, i  ]
        # z2 = self.ctrl_points_z[j-1, i+1]
        # z3 = self.ctrl_points_z[j  , i-1]
        # z4 = self.ctrl_points_z[j  , i  ]
        # z5 = self.ctrl_points_z[j  , i+1]
        # z6 = self.ctrl_points_z[j+1, i-1]
        # z7 = self.ctrl_points_z[j+1, i  ]
        # z8 = self.ctrl_points_z[j+1, i+1]


        # ==========================================
        uarray = np.array([au,bu,cu]).reshape((1,3))
        warray = np.array([aw,bw,cw]).reshape((3,1))
        uw_matrix = warray@uarray

        z_matrix = self.ctrl_points_z[j-1:j+2,i-1:i+2]

        return (uw_matrix*z_matrix).sum()/4


class biquadratic_bspline_surface_fit(nn.Module):
    def __init__(self, 
                 xmin: float, 
                 ymin: float, 
                 xmax: float, 
                 ymax: float, 
                 data_num: int, 
                 data_xy: torch.Tensor,
                 k: float = 1.0):
        '''
        数据点xy可能出现在 [xmin, xmax] × [ymin, ymax] 的矩形区域内。我们只关心拟合出的Bspline曲面在 [xmin, xmax] × [ymin, ymax] 范围内的面型。
        data_num: 数据点的数量
        '''
        super(biquadratic_bspline_surface_fit, self).__init__()
        
        # 确定控制点在XY平面内的等间隔分布，Pij = (x0+i*grid_size, y0+j*grid_size) i=0,1,2,..,m-1;  j=0,1,2,...,n-1;
        W = xmax - xmin
        H = ymax - ymin
        grid_size = k*2*np.sqrt(H*W/data_num)      # (H/grid_size)*(W/grid_size) = data_num/4 控制点在XY平面内等间隔排列
        x0, y0 = xmin - grid_size, ymin - grid_size
        m = int( (np.floor(W/grid_size+0.5)+1)+2 ) # X方向控制点总数
        n = int( (np.floor(H/grid_size+0.5)+1)+2 ) # Y方向控制点总数

        # 控制点范围检查，确保控制点决定的Bspline可以把我们关心的矩形区域包含进来
        assert(x0+0.5*grid_size < xmin and y0+0.5*grid_size < ymin)
        assert(x0+(m-1-0.5)*grid_size > xmax and y0+(n-1-0.5)*grid_size > ymax)
        self.grid_size: float = grid_size
        self.x0: float = x0
        self.y0: float = y0
        self.m: int = m
        self.n: int = n
        self.data_num = data_num
        
        # 将输入数据点的xy坐标转换为ijuw坐标，i,j表示所处的面片patch的编号，u,w表示在patch内部的参数坐标
        assert(data_xy.shape[0] == data_num and data_xy.shape[1] == 2)
        self.ijuw_lst = self.xy2ijuw(data_xy)

        # 设置初始参数，即所有控制点的初始z坐标
        ctrl_points_z = -50 + 10*torch.ones((self.n, self.m), dtype=torch.float32) # 注意xyz坐标单位不一定是mm也，可能是像素，实际生成STL模型时需要变为毫米mm
        self.ctrl_points_z = nn.Parameter(ctrl_points_z, requires_grad=True)

    def xy2ijuw(self, data_xy):
        temp_x = (data_xy[:,0] - self.x0)/self.grid_size + 0.5
        temp_y = (data_xy[:,1] - self.y0)/self.grid_size + 0.5
        i_lst = torch.floor( temp_x )
        j_lst = torch.floor( temp_y )
        u_lst = temp_x - i_lst
        w_lst = temp_y - j_lst

        ijuw_lst = torch.stack((i_lst, j_lst, u_lst, w_lst), dim=1)
        assert(ijuw_lst.shape[0] == data_xy.shape[0] and ijuw_lst.shape[1] == 4)
        return ijuw_lst

    
    
    def forward(self, data_normal: torch.Tensor): # 
        assert(data_normal.shape[0] == self.data_num and data_normal.shape[1] == 3)

        j_lst = self.ijuw_lst[:,1].int() # y方向
        i_lst = self.ijuw_lst[:,0].int() # x方向
        z0 = self.ctrl_points_z[j_lst-1, i_lst-1]#.unsqueeze(1)
        z1 = self.ctrl_points_z[j_lst-1, i_lst  ]#.unsqueeze(1)
        z2 = self.ctrl_points_z[j_lst-1, i_lst+1]#.unsqueeze(1)
        z3 = self.ctrl_points_z[j_lst  , i_lst-1]#.unsqueeze(1)
        z4 = self.ctrl_points_z[j_lst  , i_lst  ]#.unsqueeze(1)
        z5 = self.ctrl_points_z[j_lst  , i_lst+1]#.unsqueeze(1)
        z6 = self.ctrl_points_z[j_lst+1, i_lst-1]#.unsqueeze(1)
        z7 = self.ctrl_points_z[j_lst+1, i_lst  ]#.unsqueeze(1)
        z8 = self.ctrl_points_z[j_lst+1, i_lst+1]#.unsqueeze(1)

        # z_lst = torch.stack((z0,z1,z2,z3,z4,z5,z6,z7,z8), dim=1)

        u_lst = self.ijuw_lst[:,2]
        w_lst = self.ijuw_lst[:,3]
        
        cu, dcu = u_lst*u_lst, 2*u_lst
        bu, dbu = -2*cu + dcu + 1, 2 - 4*u_lst
        au, dau = cu - dcu + 1, 2*u_lst - 2 

        cw, dcw = w_lst*w_lst, 2*w_lst
        bw, dbw = -2*cw + dcw + 1, 2 - 4*w_lst
        aw, daw = cw - dcw + 1, 2*w_lst - 2
        # print(u_lst[:10],w_lst[:10],i_lst[:10],j_lst[:10])
        # 计算法向量
        pzpu = (dau*(aw*z0 + bw*z3 + cw*z6) + dbu*(aw*z1 + bw*z4 + cw*z7) + dcu*(aw*z2 + bw*z5 + cw*z8))/(4*self.grid_size)
        pzpw = (au*(daw*z0 + dbw*z3 + dcw*z6) + bu*(daw*z1 + dbw*z4 + dcw*z7) + cu*(daw*z2 + dbw*z5 + dcw*z8))/(4*self.grid_size)
        assert(len(pzpu) == len(pzpw) and len(pzpu) == self.data_num)
        negative1_lst = -torch.ones((self.data_num,1), dtype=torch.float32)
        pzpu = pzpu.unsqueeze(1)
        pzpw = pzpw.unsqueeze(1)
        normal_lst = torch.hstack([pzpu, pzpw, negative1_lst])
        normal_lst = normal_lst/torch.norm(normal_lst, p=2, dim=1, keepdim=True)

        return normal_lst
        
        

if __name__ == "__main__":

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    '''
    xmin,ymin,xmax,ymax = 0,0,800,800
    # 设置数据点
    N = 1200
    x_lst = torch.from_numpy( np.linspace(xmin, xmax, N+1)[1:N] )
    y_lst = torch.from_numpy( np.linspace(ymin, ymax, N+1)[1:N] )
    data_num = len(x_lst)*len(y_lst)
    H, W = len(x_lst), len(y_lst)
    
    data_xy = torch.zeros((data_num, 2), dtype=torch.float32)
    for i in range(data_num):
        r = i//W
        c = i - r*W
        data_xy[i,0], data_xy[i,1] = x_lst[c], y_lst[r]
    '''
    # 如何根据表面法向量的数据，衡量表面的粗糙度？
    # 在根据粗糙度决定控制点的多少，以及学习率的大小，这里的关系是什么样的？
    # 如何衡量两个单位方向向量的差？MSELoss合理吗？合理MSELoss本质上就是两个向量的余弦夹角
    '''
    data_normal = torch.hstack( [2*(data_xy[:,0:1]-400)/1e3, 2*(data_xy[:,1:2]-400)/1e3, -torch.ones((data_num,1))] )
    # data_normal = 2*torch.rand( (data_num,3) ) - 1 # 对于不光滑的表面法向量，必须加密控制点，即grid_size要缩小，学习率也要缩小
    # data_normal[:,2] = -1

    # 归一化
    data_normal = data_normal/data_normal.norm(dim=1, keepdim=True)
    '''
    # ==================================================================================================
    # target_path = "D:/myproject/JSTJ/Target_GhostInTheShell_128000/"
    target_path = "D:/myproject/JSTJ/Target_USTC_128000/"
    filename = target_path+"stage3_Grid_Normal_mapping.txt"
    xmin,ymin,xmax,ymax = 0,0,800,800 # NOTE:手动填
    
    data_all = np.loadtxt(filename, delimiter=',', dtype=np.float64)
    data_num = len(data_all)
    data_xy = torch.tensor( data_all[:,:2] )
    data_normal = torch.tensor( data_all[:,2:] )

    # ==================================================================================================
    '''
    grid_size 正比于k, k=1时(默认情况)控制点的数量为数据点的1/4, 即调节k可以控制每个面片(patch)中有多少数据, k越大则 grid_size 越大, 则每个patch中数据点越多
    网格不能太细小，否则生成曲面有很多毛刺。
    问题：如何通过设置loss让生成的曲面尽量平滑
    '''
    bspline_fit = biquadratic_bspline_surface_fit(xmin, ymin, xmax, ymax, data_num, data_xy,k=1.0) # np.sqrt(2)/2
    
    loss_fn = nn.MSELoss(reduction="sum")
    optimizer = torch.optim.Adam(bspline_fit.parameters(), lr=2.5e-1)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)

    cnt = 0
    while(1):
        cnt += 1
        out_normal = bspline_fit(data_normal)

        loss = loss_fn(out_normal, data_normal)

        if(cnt%50 == 0):
            print(f"{cnt}--学习率:{optimizer.param_groups[0]['lr']:.6f}--loss:{loss.item():.12f}")
            if(loss.item() < 2.2):
                break
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # if(optimizer.param_groups[0]['lr'] > 0.1):
        scheduler.step()

    # ==========================================================================================
    bspline_surface = calculate_biquadratic_bspline(bspline_fit.x0,
                                                    bspline_fit.y0,
                                                    bspline_fit.grid_size,
                                                    bspline_fit.m,
                                                    bspline_fit.n,
                                                    bspline_fit.ctrl_points_z.detach().numpy())
    print(bspline_surface.x0, 
          bspline_surface.y0, 
          bspline_surface.grid_size, 
          bspline_surface.m, 
          bspline_surface.n)

    # 以二进制写入模式打开文件（已存在会覆盖，不存在会新建）
    
    with open(target_path+"bspline_surface.pickle", "wb") as f:
        pickle.dump(bspline_surface, f)
    

    


        