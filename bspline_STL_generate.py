import OCP.Geom
import cadquery as cq
import OCP
import numpy as np
import pickle
import time
from bspline_fit_normal import calculate_biquadratic_bspline

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 空间三维画图




if __name__ == "__main__":
    # target_path = "D:/myproject/JSTJ/Target_GhostInTheShell_128000/"
    target_path = "D:/myproject/JSTJ/Target_USTC_128000/"
    edge_len = 75
    mm2px = 1.0*800/edge_len # 800像素对应75mm
    offset = 12 # 10mm offset

    with open(target_path+"bspline_surface.pickle", "rb") as f:
        bspline_surface_loaded = pickle.load(f)
    
    # print(bspline_surface_loaded.x0, 
    #       bspline_surface_loaded.y0,
    #       bspline_surface_loaded.grid_size,
    #       bspline_surface_loaded.m, 
    #       bspline_surface_loaded.n)
    
    print(bspline_surface_loaded.calculate(0.1,0.1))
    print(bspline_surface_loaded.calculate(0,800))
    print(bspline_surface_loaded.calculate(800,0))
    print(bspline_surface_loaded.calculate(800,800))
    print(bspline_surface_loaded.calculate(400,400))
    

    def Bsurface(u, v):
        x, y = u, v           # x,y单位默认是mm
        global mm2px, offset
        x_px, y_px = mm2px*x, mm2px*y
        z_px = bspline_surface_loaded.calculate(x_px, y_px)
        z = z_px/mm2px + offset
        return (x,y,z)
    
    if(1):
        x0, y0 = 4, 4
        grid = 4
        xy_lst = np.array( [[x0+j*grid, y0+i*grid] for i in range(200) for j in range(200)] )
        z_lst = [bspline_surface_loaded.calculate(xy[0], xy[1]) for xy in xy_lst]
        for i,each in enumerate(z_lst):
            if(each > -20):
                print(xy_lst[i],each)
        fig = plt.figure()
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.scatter(xy_lst[:,0], xy_lst[:,1], z_lst, c='black', marker="^",s=0.1)
        ax1.set_zlim(-120,-10)

        plt.show()
        input("plot finished")


    '''
    如何计算曲面在XY范围内的Z的范围？
    '''
    param_surface = cq.Workplane().parametricSurface(Bsurface, N=1200, start=-0.001, stop=edge_len+0.001, tol=1e-6,\
                                                     minDeg=2, maxDeg=2, smoothing=None)
    print("parametricSurface generated")
    
    # shape = cq.Workplane().center(edge_len/2.0, edge_len/2.0).box(edge_len,edge_len,30,centered=(25,25,0.001)).split(param_surface).solids('>Z')
    shape = cq.Workplane().center(edge_len/2.0, edge_len/2.0).cylinder(30, edge_len/2.0, centered=(edge_len/2.0, edge_len/2.0, 0.001)).split(param_surface).solids('>Z')
    print("shape generated")
    
    
    stl_name =  target_path + "caustic_model2"+".stl"
    shape.val().exportStl(stl_name,tolerance=1e-4,angularTolerance=0.01,parallel=True) # exportStep API
    print("STL file generated")

    end_time = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
    print(f"EOF,time:{end_time}")

