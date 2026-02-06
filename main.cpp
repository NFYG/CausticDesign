#include <iostream>
#include <limits>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <boost/algorithm/string.hpp>
#include <boost/math/complex/asin.hpp>
#include <boost/math/constants/constants.hpp>
#include "stdio.h"
#include "assert.h"
#include <random>
#include <thread>
#include <mutex>
#include <chrono>
#include <fstream>
#include <map>
#include <functional>
#include <ranges>
#include <mutex>
#include "tinyxml2.h"
#include "lbfgs.hpp"



#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Delaunay_triangulation_2.h>
#include <CGAL/Voronoi_diagram_2.h>
#include <CGAL/Delaunay_triangulation_adaptation_traits_2.h>
#include <CGAL/Delaunay_triangulation_adaptation_policies_2.h>
#include <CGAL/draw_triangulation_2.h>
#include <CGAL/natural_neighbor_coordinates_2.h>
#include <CGAL/Simple_cartesian.h>
#include <CGAL/convex_hull_2.h>
#include <CGAL/Polygon_2.h>
#include <CGAL/Polygon_with_holes_2.h>
#include <CGAL/intersections.h>
#include <CGAL/Boolean_set_operations_2.h>
#include <CGAL/centroid.h>
#include <CGAL/point_generators_2.h>
// #include <CGAL/power_diagram_2.h>
#include <CGAL/Object.h>
#include <CGAL/Regular_triangulation.h>
#include <CGAL/Regular_triangulation_2.h>
#include <CGAL/Triangulation_2.h>
#include <CGAL/Regular_triangulation_face_base_2.h>
#include <CGAL/Regular_triangulation_vertex_base_2.h>
#include <CGAL/Regular_triangulation_adaptation_traits_2.h>
#include <CGAL/Polygon_set_2.h>
#include <CGAL/natural_neighbor_coordinates_2.h>


#include <Windows.h>

typedef CGAL::Exact_predicates_exact_constructions_kernel                               CGALKernel;
// typedef CGAL::Exact_predicates_inexact_constructions_kernel                             CGALKernel;
typedef CGALKernel::FT                                                                  Coord_type;
typedef CGALKernel::Point_2                                                             Point;
typedef CGALKernel::Weighted_point_2                                                    Weighted_point_2;
typedef CGALKernel::Segment_2                                                           Segment;
typedef CGALKernel::Ray_2                                                               Ray;
typedef CGALKernel::Line_2                                                              Line;
typedef CGAL::Vector_2<CGALKernel>                                                      Vector_2;
typedef CGAL::Polygon_2<CGALKernel>                                                     Polygon_2;
typedef CGAL::Polygon_set_2<CGALKernel>                                                 Polygon_set_2;
typedef CGAL::Polygon_with_holes_2<CGALKernel>                                          Polygon_with_holes_2;
typedef CGAL::Delaunay_triangulation_2<CGALKernel>                                      DelaunayTri;
typedef CGAL::Delaunay_triangulation_adaptation_traits_2<DelaunayTri>                   AT;
typedef CGAL::Delaunay_triangulation_caching_degeneracy_removal_policy_2<DelaunayTri>   AP;
typedef CGAL::Voronoi_diagram_2<DelaunayTri,AT,AP>                                      Voronoi_diagram;
typedef Voronoi_diagram::Halfedge_handle                                                Halfedge_handle;

typedef CGAL::Regular_triangulation_2<CGALKernel>                                       Regular_triangulation;
typedef Regular_triangulation::Vertex_handle                                            Vertex_handle;
typedef Regular_triangulation::Edge                                                     Edge;
typedef Regular_triangulation::Face_handle                                              Face_handle;

#define mprint(x)                                                                       std::cout<< (x) <<std::endl
#define LLOYD_ALGORITHM_THREADS_NUM                                                     (6)
#define INTEGRAL_IN_POWERCELL_THREADS_NUM                                               (12)
#define GRID_NN_INTERPOLATE_THREADS_NUM                                                 (12)


double area(double r)
{
   using namespace boost::math::double_constants;
   return pi * r * r;
}



class voronoi_cell{

public:
    Point site;
    int cell_idx;
    bool is_cell_bounded;
    bool is_all_vd_vertexs_in_rectangle; // 是否所有voronoi图生成的vertex都在窗口矩形内部
    std::vector<Point> cell_vertexs_lst; // 顺时针排列的cell的边界点

    bool is_have_corner;
    bool is_intersect_boundary;
    // double cell_mass;    // cell的总质量
    // Point cell_centroid;

    // 构造函数
    voronoi_cell(){
        is_have_corner=false;
        is_intersect_boundary = false;
    }
    
    // 构造函数重载
    voronoi_cell(Point site_pt, int cell_num): site(site_pt), cell_idx(cell_num) {
        is_have_corner=false;
        is_intersect_boundary = false;
    }

    // 析构函数
    ~voronoi_cell(){
        this->cell_vertexs_lst.clear();
        this->cell_vertexs_lst.shrink_to_fit();
    }
    
};



/// @brief 将图片作为灰度图读取并将灰度值放入矩阵中
/// @param PicFilePath_str 图片路径
/// @return pic_mat
auto load_pic_as_gray(std::string PicFilePath_str)
{
    // std::string 转为 const char* 
    const char* PicFilePath = PicFilePath_str.c_str(); 

    int width, height, channels;

    // 读取图像，使用stbi_load函数
    unsigned char *image = stbi_load(PicFilePath, &width, &height, &channels, 1); // 最后的参数1表示把图片加载为灰度图
    if (image == nullptr) {
        std::cerr << "Failed to load image" << std::endl;
    }

    //Eigen::MatrixXd pic_mat(height, width);
    Eigen::Matrix<uint8_t,-1,-1> pic_mat(height, width); // Eigen::Matrix<double,-1,-1>与Eigen::MatrixXd是等价的
    for(size_t r = 0; r < height; r++){
        for(size_t c = 0; c < width; c++){
            pic_mat(r, c) = image[r * width + c];
        }
    }
    // eigen DOC https://eigen.tuxfamily.org/dox/group__TutorialMatrixArithmetic.html
    return pic_mat;
}



/// @brief 对图片矩阵每行按行累加求和，再对每行的总和进行按列累加求和
/// @param pic_mat 图片矩阵
/// @return (pic_mat_row_cumsum, pic_mat_col_cumsum)
auto pic_mat_cumsum(Eigen::Matrix<uint8_t,-1,-1> pic_mat)
{
    int H = pic_mat.rows();
    int W = pic_mat.cols();

    Eigen::Matrix<int,-1,-1> pic_mat_row_cumsum(H,W);  //等价于Eigen::MatrixXi pic_mat_row_cumsum = Eigen::MatrixXi::Zero(H,W);
    
    pic_mat_row_cumsum.block(0,0,H,1) = pic_mat.block(0,0,H,1).cast<int>();
    for(size_t c = 1; c < W; c++){
        pic_mat_row_cumsum.block(0,c,H,1) = pic_mat_row_cumsum.block(0,c-1,H,1) + pic_mat.block(0,c,H,1).cast<int>();
    }

    // for(size_t r = 0; r < H; r++){
    //     for(size_t c = 0; c < W; c++){
    //         assert((pic_mat.cast<int>())(r,c) >= 0);
    //     }
    // }

    Eigen::Vector<LONG64,-1> pic_mat_col_cumsum(H);
    pic_mat_col_cumsum(0) = pic_mat_row_cumsum(0,W-1);
    for(size_t r = 1; r < H; r++){
        pic_mat_col_cumsum(r) = pic_mat_col_cumsum(r-1) + pic_mat_row_cumsum(r,W-1);
    }

    LONG64 pic_mat_sum = pic_mat_row_cumsum.block(0,W-1,H,1).sum();

    assert(pic_mat_sum >= 0);
    assert(pic_mat_col_cumsum(H-1) == pic_mat_sum);
    
    // 归一化
    Eigen::MatrixXd pic_mat_row_cumsum_normalized = pic_mat_row_cumsum.cast<double>();
    for(size_t r = 0; r < H; r++){
        pic_mat_row_cumsum_normalized.block(r,0,1,W) = pic_mat_row_cumsum_normalized.block(r,0,1,W)/pic_mat_row_cumsum_normalized(r,W-1);
    }
    Eigen::VectorXd pic_mat_col_cumsum_normalized = pic_mat_col_cumsum.cast<double>()/pic_mat_col_cumsum(H-1);

    return std::make_tuple(pic_mat_row_cumsum_normalized, pic_mat_col_cumsum_normalized);

}



/// @brief 以灰度值作为二维概率密度分布，进行随机采样生成N个随机点，作为Voronoi图的初始sites
/// @param N 随机点的数目
/// @param pic_row_cumsum_n 按行累加求和
/// @param pic_col_cumsum_n 每行总和按列累加求和
/// @return sites_lst
auto generate_sites_lst(int N, Eigen::MatrixXd pic_row_cumsum_n, Eigen::VectorXd pic_col_cumsum_n)
{
    int H = pic_row_cumsum_n.rows();
    int W = pic_row_cumsum_n.cols();
    assert(H == pic_col_cumsum_n.size());

    // 创建N行2列的0~1的double类型随机数矩阵
    Eigen::MatrixXd random_number = Eigen::MatrixXd::Random(N,2);
    random_number = 0.5*(random_number.array() + 1.0);

    // 创建N行2列的0~1的double类型随机数矩阵作为像素内部的随机偏移，防止分配到同一个像素的多个sites完全重合
    Eigen::MatrixXd in_pix_bias = Eigen::MatrixXd::Random(N,2);
    in_pix_bias = 0.5*(in_pix_bias.array() + 1.0);

    // 随机矩阵random_number每行确定一个sites坐标，每行第一个元素确定sites的y，第二个元素确定sites的x
    std::vector<Point> sites_lst;

    for(size_t i = 0; i < N; i++){
        
        double r1 = random_number(i,0);
        double r2 = random_number(i,1);

        if(r1 > 1.0 - 1e-8){
            r1 = 1.0 - 1e-8;
        }
        if(r2 > 1.0 - 1e-8){
            r2 = 1.0 - 1e-8;
        }

        Eigen::Index index1, index2;
        bool found;
        found = (pic_col_cumsum_n.array() > r1).maxCoeff(&index1);

        if(!found){
            index1 = H - 1;
            // std::cout<<"fail r1 r2:"<<r1<<", "<<r2<<std::endl;
            // assert(false);
        }
        found = (pic_row_cumsum_n(index1 , Eigen::all).array() > r2).maxCoeff(&index2); // 或者用pic_row_cumsum_n.row(index1)获取索引index1行
        if(!found){
            index2 = W - 1;
            // std::cout<<"fail r1 r2:"<<r1<<", "<<r2<<std::endl;
            // std::cout<<"fail index1 index2:"<<index1<<", "<<index2<<std::endl;
            // assert(false);
        }
        //std::cout<<"--------------------------------------"<<std::endl;
        //std::cout<<i<<":"<< (int)index1<< ", "<<(int)index2<<std::endl;
        Point sites_point(((int)index2) + in_pix_bias(i,0), ((int)index1) + in_pix_bias(i,1));
        //std::cout<< sites_point[0]<< ", "<<sites_point[1]<<std::endl;
        
        // assert(!std::isnan(static_cast<double>(sites_point[0]))); 
        // assert(!std::isnan(static_cast<double>(sites_point[1])));

        sites_lst.push_back(sites_point);
        
    }

    return sites_lst;
}



/// @brief 计算射线ray和线段seg的交点
/// @param ray 射线
/// @param seg 线段
/// @return 返回元组，第一个元素true/false表示是否存在交点，第二个元素给出交点
auto ray_segment_intersect(Ray ray, Segment seg)
{
    Point intersect_pt(-1,-1);
    CGAL::Object result = CGAL::intersection(ray, seg);

    if(const Point  *tmp_pt = CGAL::object_cast<Point>(&result)){
        // 有1个交点
        intersect_pt = *(tmp_pt);
        return std::make_tuple(true, intersect_pt);
    }else if(const Segment *iseg = CGAL::object_cast<Segment>(&result)){
        std::cerr<< "Runtime Assertion failed: The ray coincides with the segment."<<std::endl;
        assert(false);
        return std::make_tuple(false, intersect_pt);
    }else{
        // 无交点
        return std::make_tuple(false, intersect_pt);  
    }
}



/// @brief 计算两个线段的交点
/// @param seg1 线段1
/// @param seg2 线段2
/// @return 
auto segment_segment_intersect(Segment seg1, Segment seg2)
{
    Point intersect_pt(-1,-1);
    CGAL::Object result = CGAL::intersection(seg1, seg2);

    if(const Point  *tmp_pt = CGAL::object_cast<Point>(&result)){
        // 有1个交点，返回flag 1以及1个交点和1个无效点（确保返回值类型不变）
        // intersect_pt = *(tmp_pt);
        return std::make_tuple(1, *(tmp_pt), intersect_pt);
    }else if(const Segment *iseg = CGAL::object_cast<Segment>(&result)){
        std::cerr<< "Runtime Assertion failed: The seg1 coincides with the seg2."<<std::endl;
        assert(false);
        // 若是2线段重合，则返回flag 2以及重合线段的2个端点
        return std::make_tuple(2, iseg->source(), iseg->target());
    }else{
        // 无交点，则返回flag 0以及两个无效点
        return std::make_tuple(0, intersect_pt, intersect_pt);  
    }
}



/// @brief 计算以坐标(sx,sy)为左上角点，高H，宽W的矩形与射线的交点(假设射线与窗框只有1个交点)
/// @param ray_pt 射线端点
/// @param ray_dir 射线方向
/// @param sx 左上角点x坐标
/// @param sy 左上角点y坐标
/// @param H 矩形的高
/// @param W 矩形的宽
/// @return 
auto ray_rectangle_intersect(Point ray_pt, Vector_2 ray_dir, int sx, int sy, int H, int W){
    // 归一化
    // auto length = CGAL::sqrt(ray_dir*ray_dir);
    ray_dir = ray_dir.direction().vector();
    // 计算以坐标(sx,sy)为左上角点，高H，宽W的矩形与射线的交点
    Point UL(static_cast<double>(sx)  ,static_cast<double>(sy)  );
    Point UR(sx+W,sy  );
    Point DL(sx  ,sy+H);
    Point DR(sx+W,sy+H);
    Segment U_boundary(UL,UR);
    Segment D_boundary(DL,DR);
    Segment L_boundary(UL,DL);
    Segment R_boundary(UR,DR);
    // 创建射线ray并计算与矩形的四个边（线段）的交点
    Ray ray(ray_pt, ray_dir);
    auto [is_intersect_U, pt_U] = ray_segment_intersect(ray, U_boundary);
    auto [is_intersect_D, pt_D] = ray_segment_intersect(ray, D_boundary);
    auto [is_intersect_L, pt_L] = ray_segment_intersect(ray, L_boundary);
    auto [is_intersect_R, pt_R] = ray_segment_intersect(ray, R_boundary);

    if(is_intersect_U && (!is_intersect_D) && (!is_intersect_L) && (!is_intersect_R)){
        return std::make_tuple("U_boundary", pt_U);
    }else if((!is_intersect_U) && (is_intersect_D) && (!is_intersect_L) && (!is_intersect_R)){
        return std::make_tuple("D_boundary", pt_D);
    }else if((!is_intersect_U) && (!is_intersect_D) && (is_intersect_L) && (!is_intersect_R)){
        return std::make_tuple("L_boundary", pt_L);
    }else if((!is_intersect_U) && (!is_intersect_D) && (!is_intersect_L) && (is_intersect_R)){
        return std::make_tuple("R_boundary", pt_R);
    }else if((is_intersect_U) && (!is_intersect_D) && (is_intersect_L) && (!is_intersect_R)){
        return std::make_tuple("UL_corner", UL);
    }else if((is_intersect_U) && (!is_intersect_D) && (!is_intersect_L) && (is_intersect_R)){
        return std::make_tuple("UR_corner", UR);
    }else if((!is_intersect_U) && (is_intersect_D) && (is_intersect_L) && (!is_intersect_R)){
        return std::make_tuple("DL_corner", DL);
    }else if((!is_intersect_U) && (is_intersect_D) && (!is_intersect_L) && (is_intersect_R)){
        return std::make_tuple("DR_corner", DR);
    }else{

        std::cerr<< "ERROR:ray "<<ray_pt<<","<<ray_dir<<" do not intersect with rectangle"<<std::endl;
        assert(false);
        Point pt_error(-1,-1);
        return std::make_tuple("ERROR", pt_error);
    }
}



/// @brief 计算以坐标(sx,sy)为左上角点，高H，宽W的矩形与线段的交点
/// @param seg 
/// @param sx 
/// @param sy 
/// @param H 
/// @param W 
/// @return 
auto seg_rectangle_intersect(Segment seg, int sx, int sy, int H, int W){
    // 计算以坐标(sx,sy)为左上角点，高H，宽W的矩形与线段seg的交点
    Point UL(static_cast<double>(sx)  ,static_cast<double>(sy)  );
    Point UR(sx+W,sy  );
    Point DL(sx  ,sy+H);
    Point DR(sx+W,sy+H);
    Segment U_boundary(UL,UR);
    Segment D_boundary(DL,DR);
    Segment L_boundary(UL,DL);
    Segment R_boundary(UR,DR);

    // 计算seg和四个边框的交点
    auto [is_intersect_U, pt_U, _1] = segment_segment_intersect(seg, U_boundary);
    auto [is_intersect_D, pt_D, _2] = segment_segment_intersect(seg, D_boundary);
    auto [is_intersect_L, pt_L, _3] = segment_segment_intersect(seg, L_boundary);
    auto [is_intersect_R, pt_R, _4] = segment_segment_intersect(seg, R_boundary);
    
    // 忽略线段与某1边框重合的情况
    std::vector<Point> intersect_pt_lst;
    bool is_intersect = false;
    if(1 == is_intersect_U){
        intersect_pt_lst.push_back(pt_U);
        is_intersect = true;
    }
    if(1 == is_intersect_D){
        intersect_pt_lst.push_back(pt_D);
        is_intersect = true;
    }
    if(1 == is_intersect_L){
        intersect_pt_lst.push_back(pt_L);
        is_intersect = true;
    }
    if(1 == is_intersect_R){
        intersect_pt_lst.push_back(pt_R);
        is_intersect = true;
    }
    return std::make_tuple(is_intersect, intersect_pt_lst);
}


/// @brief 判断一个点在不在矩形内部（包括边上）
/// @param pt 点
/// @param sx 矩形左上角点x坐标
/// @param sy 矩形左上角点y坐标
/// @param H 矩形的高
/// @param W 矩形的宽
/// @return 在内部（包括边上）返回true，在外部返回false.
auto is_point_in_rectangle(Point pt, int sx, int sy, int H, int W){
    auto x = pt.x();
    auto y = pt.y();
    if(x>=sx && x<=(sx+W) && y>=sy && y<=(sy+H) ){
        return true;
    }else{
        return false;
    }
}



/// @brief 
/// @param sites_lst 
/// @param N 
/// @param sx 
/// @param sy 
/// @param H 
/// @param W 
/// @return 
// 优化速度(大头在画Voronoi_diagram) 验证点的顺序是否正确(的确都是顺时针) 画图
auto GetVoronoiDiagram(std::vector<Point> sites_lst, int N, double sx, double sy, int H, int W)
{
    assert(sites_lst.size() == N);
    
    // 创建最终返回值
    std::vector<voronoi_cell> voronoi_cell_lst(N); 

    //  Delaunay 三角剖分
    DelaunayTri delaunay_triang;
    delaunay_triang.insert(sites_lst.begin(), sites_lst.end());
    
    // 创建voronoi图，并检验有效性
    // auto start = std::chrono::high_resolution_clock::now();
    Voronoi_diagram vd(delaunay_triang);
    assert(vd.is_valid());
    // auto end = std::chrono::high_resolution_clock::now();
    // auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    // std::cout << "Time taken by function: " << duration.count() << " milliseconds" << std::endl;

    // step 1 遍历所有voronoi face并将相关信息加入voronoi_cell_lst列表
    int face_idx = 0;
    for(auto face = vd.faces_begin(); face != vd.faces_end(); face++, face_idx++){

        auto vertex = face->dual();
        Point site_pt = vertex->point();
        // assert(is_point_in_rectangle(site_pt, static_cast<int>(sx), static_cast<int>(sy), H, W));

        voronoi_cell_lst[face_idx].site     = site_pt;                      // 获取该面的sites坐标
        voronoi_cell_lst[face_idx].cell_idx = face_idx;                     // 获取编号
        voronoi_cell_lst[face_idx].is_cell_bounded = !face->is_unbounded(); // 获取是否有界信息
        
        // 遍历节点，判断是否有节点在窗框外部
        auto ccb_e = face->ccb();
        auto ccb_start = ccb_e;
        voronoi_cell_lst[face_idx].is_all_vd_vertexs_in_rectangle = true;
        do{
            Point pt(-1,-1);
            if(ccb_e->is_segment()){
                pt = ccb_e->target()->point();
            }else if(ccb_e->is_ray()){
                if( ccb_e->has_source() ){
                    pt = ccb_e->source()->point();
                }else{
                    pt = ccb_e->target()->point();
                }
            }else{
                assert(false);
            }
            auto x = pt.x();
            auto y = pt.y();
            if(x < sx || x > (sx+W) || y < sy || y > (sy+H)){
                voronoi_cell_lst[face_idx].is_all_vd_vertexs_in_rectangle = false;
                break;
            }
            ccb_e++;
        }while(ccb_e != ccb_start);
    }
    // mprint("step 1 finished");
    // step 2 再次遍历所有voronoi face并将相关信息加入voronoi_cell_lst列表
    face_idx = 0;
    for(auto face = vd.faces_begin(); face != vd.faces_end(); face++, face_idx++){
        // case 1：无界cell，所有点都在内部
        if(face->is_unbounded() && voronoi_cell_lst[face_idx].is_all_vd_vertexs_in_rectangle){
            // 该face是无界的情况，意味着face边界有射线(2条ray)
            // 遍历边界的射线&线段，只有射线会与边界相交产生新的vertex
            auto ccb_e = face->ccb();
            auto ccb_start = ccb_e;
            do{
                if(ccb_e->is_ray()){
                    // 首先获取ray的唯一的端点
                    Point ray_pt;
                    if( ccb_e->has_source() ){
                        ray_pt = ccb_e->source()->point();
                    }else if( ccb_e->has_target() ){
                        ray_pt = ccb_e->target()->point();
                    }else{
                        std::cerr<< "Runtime Assertion failed: Edge Has NO Source and Target"<<std::endl;
                        assert(false);
                    }
                    
                    // 每条边voronoi dege都是其对应德劳内三角edge的中垂线，以此计算ray的方向
                    int _idx = ccb_e->dual().second;
                    auto _pt1 = ccb_e->dual().first->vertex((1+_idx)%3)->point();
                    auto _pt2 = ccb_e->dual().first->vertex((2+_idx)%3)->point();
                    Segment _seg(_pt1, _pt2);
                    Vector_2 dir0 = _seg.to_vector();
                    Vector_2 ray_dir(-dir0.y(), dir0.x());

                    // 修正ray方向的正负，上述得出的ray方向总是从source指向target，若target存在则ray要反向才能指向无穷远
                    if( ccb_e->has_target() ){
                        ray_dir = -ray_dir;
                    }
                    // mprint(ray_dir);

                    // 计算射线与窗口边框的交点（有且仅有一个交点）并加入vertex列表
                    auto [intersect_info, intersect] = ray_rectangle_intersect(ray_pt, ray_dir, static_cast<int>(sx), static_cast<int>(sy), H, W);
                    voronoi_cell_lst[face_idx].cell_vertexs_lst.push_back(intersect);
                    voronoi_cell_lst[face_idx].is_intersect_boundary = true; // 该cell与边界相交
                    // mprint(ray_pt);
                    //mprint(intersect);
                    if( ccb_e->has_target() ){
                        // 我们统一约定，只把target加入vertex列表，不把source放进去
                        voronoi_cell_lst[face_idx].cell_vertexs_lst.push_back(ccb_e->target()->point());
                    }


                }else if(ccb_e->is_segment()){
                    // 对于线段，我们把它的target加入vertex列表
                    voronoi_cell_lst[face_idx].cell_vertexs_lst.push_back(ccb_e->target()->point());
                }else{
                    std::cerr<< "Runtime Assertion failed: Edge must be ray or segment"<<std::endl;
                    assert(false);
                }
                ccb_e++;
            }while(ccb_e != ccb_start);

        }
        // case 2：有界cell，所有点都在内部
        else if(!face->is_unbounded() &&  voronoi_cell_lst[face_idx].is_all_vd_vertexs_in_rectangle){
            // 该face是有界的情况，意味着face边界全为线段
             // 遍历边界的线段
            auto ccb_e = face->ccb();
            auto ccb_start = ccb_e;
            do{
                if(ccb_e->is_segment()){
                    // 对于线段，我们把它的target加入vertex列表
                    voronoi_cell_lst[face_idx].cell_vertexs_lst.push_back(ccb_e->target()->point());
                }else{
                    std::cerr<< "Runtime Assertion failed: Edge in bounded face must be segment"<<std::endl;
                    assert(false);
                }
                ccb_e++;
            }while(ccb_e != ccb_start);
        }
        // case 3：无界cell，有点在外部，有点在外部的线段也要计算与窗框的交点，但是端点在外部的ray无需考虑
        else if( face->is_unbounded() && !(voronoi_cell_lst[face_idx].is_all_vd_vertexs_in_rectangle)){
            auto ccb_e = face->ccb();
            auto ccb_start = ccb_e;
            do{
                if(ccb_e->is_ray()){
                    // 首先获取ray的唯一的端点
                    Point ray_pt;
                    if( ccb_e->has_source() ){
                        ray_pt = ccb_e->source()->point();
                    }else if( ccb_e->has_target() ){
                        ray_pt = ccb_e->target()->point();
                    }else{
                        std::cerr<< "Runtime Assertion failed: Edge Has NO Source and Target"<<std::endl;
                        assert(false);
                    }

                    // 判断ray端点是否在窗框内部，不在内部必定无法与窗框相交，直接跳至下一边
                    auto is_ray_pt_in = is_point_in_rectangle(ray_pt, static_cast<int>(sx), static_cast<int>(sy), H, W);
                    if(!is_ray_pt_in){ 
                        ccb_e++;
                        continue; 
                    }

                    // 每条边voronoi dege都是其对应德劳内三角edge的中垂线，以此计算ray的方向
                    int _idx = ccb_e->dual().second;
                    auto _pt1 = ccb_e->dual().first->vertex((1+_idx)%3)->point();
                    auto _pt2 = ccb_e->dual().first->vertex((2+_idx)%3)->point();
                    Segment _seg(_pt1, _pt2);
                    Vector_2 dir0 = _seg.to_vector();
                    Vector_2 ray_dir(-dir0.y(), dir0.x());

                    // 修正ray方向的正负，上述得出的ray方向总是从source指向target，若target存在则ray要反向才能指向无穷远
                    if( ccb_e->has_target() ){
                        ray_dir = -ray_dir;
                    }
                    // mprint(ray_dir);

                    // 计算射线与窗口边框的交点（有且仅有一个交点）并加入vertex列表
                    auto [intersect_info, intersect] = ray_rectangle_intersect(ray_pt, ray_dir, static_cast<int>(sx), static_cast<int>(sy), H, W);
                    voronoi_cell_lst[face_idx].cell_vertexs_lst.push_back(intersect);
                    voronoi_cell_lst[face_idx].is_intersect_boundary = true; // 该cell与边界相交

                    if( ccb_e->has_target() ){
                        // 我们统一约定，只把target加入vertex列表，不把source放进去
                        voronoi_cell_lst[face_idx].cell_vertexs_lst.push_back(ccb_e->target()->point());
                    }
                    


                }else if(ccb_e->is_segment()){
                    auto source_pt = ccb_e->source()->point();
                    auto target_pt = ccb_e->target()->point();
                    auto is_source_pt_in = is_point_in_rectangle(source_pt, static_cast<int>(sx), static_cast<int>(sy), H, W);
                    auto is_target_pt_in = is_point_in_rectangle(target_pt, static_cast<int>(sx), static_cast<int>(sy), H, W);
                    // 对于线段，我们把它的target加入vertex列表
                    if(is_source_pt_in && is_target_pt_in){
                        voronoi_cell_lst[face_idx].cell_vertexs_lst.push_back(target_pt);
                    }else if(is_source_pt_in && !is_target_pt_in){
                        Segment _seg(source_pt, target_pt);
                        auto [intersect_info, intersect] = ray_rectangle_intersect(source_pt, _seg.to_vector(), static_cast<int>(sx), static_cast<int>(sy), H, W);
                        voronoi_cell_lst[face_idx].cell_vertexs_lst.push_back(intersect);
                        voronoi_cell_lst[face_idx].is_intersect_boundary = true; // 该cell与边界相交
                    }else if(!is_source_pt_in && is_target_pt_in){
                        Segment _seg(target_pt, source_pt);
                        auto [intersect_info, intersect] = ray_rectangle_intersect(target_pt, _seg.to_vector(), static_cast<int>(sx), static_cast<int>(sy), H, W);
                        voronoi_cell_lst[face_idx].cell_vertexs_lst.push_back(intersect);
                        voronoi_cell_lst[face_idx].cell_vertexs_lst.push_back(target_pt);
                        voronoi_cell_lst[face_idx].is_intersect_boundary = true; // 该cell与边界相交
                    }else{
                        // 即便线段2个端点都在边框外部，线段也可能与矩形边框相交
                        Segment _seg(source_pt, target_pt);
                        auto [is_intersect, intersect_pt_lst] = seg_rectangle_intersect(_seg, static_cast<int>(sx), static_cast<int>(sy), H, W);
                        if(is_intersect && intersect_pt_lst.size() > 0){
                            if(1 == intersect_pt_lst.size()){
                                voronoi_cell_lst[face_idx].cell_vertexs_lst.push_back(intersect_pt_lst[0]);
                            }else if(2 == intersect_pt_lst.size()){
                                // 若线段与边框有2个交点，要按照从source到target的顺序将他们添加到列表中
                                auto d1 = CGAL::squared_distance(intersect_pt_lst[0], source_pt);
                                auto d2 = CGAL::squared_distance(intersect_pt_lst[1], source_pt);
                                if(d1 < d2){
                                    voronoi_cell_lst[face_idx].cell_vertexs_lst.push_back(intersect_pt_lst[0]);
                                    voronoi_cell_lst[face_idx].cell_vertexs_lst.push_back(intersect_pt_lst[1]);
                                }else{
                                    voronoi_cell_lst[face_idx].cell_vertexs_lst.push_back(intersect_pt_lst[1]);
                                    voronoi_cell_lst[face_idx].cell_vertexs_lst.push_back(intersect_pt_lst[0]);
                                }
                            }else{
                                assert(false);
                            }
                            // for(auto pt : intersect_pt_lst){
                            //     // voronoi_cell_lst[face_idx].cell_vertexs_lst.push_back(pt);
                            //     std::cout<< "3:" << pt << "face_idx:" << face_idx << ","<< face->dual()->point() << std::endl;
                            // }
                            voronoi_cell_lst[face_idx].is_intersect_boundary = true; // 该cell与边界相交
                        }
                    }
                }else{
                    std::cerr<< "Runtime Assertion failed: Edge must be ray or segment"<<std::endl;
                    assert(false);
                }
                ccb_e++;
            }while(ccb_e != ccb_start);
        }
        // case 4：有界cell，有点在外部
        else if(!face->is_unbounded() && !(voronoi_cell_lst[face_idx].is_all_vd_vertexs_in_rectangle)){
            auto ccb_e = face->ccb();
            auto ccb_start = ccb_e;
            do{
                if(ccb_e->is_segment()){
                    auto source_pt = ccb_e->source()->point();
                    auto target_pt = ccb_e->target()->point();
                    auto is_source_pt_in = is_point_in_rectangle(source_pt, static_cast<int>(sx), static_cast<int>(sy), H, W);
                    auto is_target_pt_in = is_point_in_rectangle(target_pt, static_cast<int>(sx), static_cast<int>(sy), H, W);
                    // 对于线段，我们把它的target加入vertex列表
                    if(is_source_pt_in && is_target_pt_in){
                        voronoi_cell_lst[face_idx].cell_vertexs_lst.push_back(target_pt);
                    }else if(is_source_pt_in && !is_target_pt_in){
                        Segment _seg(source_pt, target_pt);
                        auto [intersect_info, intersect] = ray_rectangle_intersect(source_pt, _seg.to_vector(), static_cast<int>(sx), static_cast<int>(sy), H, W);
                        voronoi_cell_lst[face_idx].cell_vertexs_lst.push_back(intersect);
                        voronoi_cell_lst[face_idx].is_intersect_boundary = true; // 该cell与边界相交
                    }else if(!is_source_pt_in && is_target_pt_in){
                        Segment _seg(target_pt, source_pt);
                        auto [intersect_info, intersect] = ray_rectangle_intersect(target_pt, _seg.to_vector(), static_cast<int>(sx), static_cast<int>(sy), H, W);
                        voronoi_cell_lst[face_idx].cell_vertexs_lst.push_back(intersect);
                        voronoi_cell_lst[face_idx].cell_vertexs_lst.push_back(target_pt);
                        voronoi_cell_lst[face_idx].is_intersect_boundary = true; // 该cell与边界相交
                    }else{
                        // 即便线段2个端点都在边框外部，线段也可能与矩形边框相交
                        Segment _seg(source_pt, target_pt);
                        auto [is_intersect, intersect_pt_lst] = seg_rectangle_intersect(_seg, static_cast<int>(sx), static_cast<int>(sy), H, W);
                        if(is_intersect && intersect_pt_lst.size() > 0){
                            if(1 == intersect_pt_lst.size()){
                                voronoi_cell_lst[face_idx].cell_vertexs_lst.push_back(intersect_pt_lst[0]);
                            }else if(2 == intersect_pt_lst.size()){
                                auto d1 = CGAL::squared_distance(intersect_pt_lst[0], source_pt);
                                auto d2 = CGAL::squared_distance(intersect_pt_lst[1], source_pt);
                                if(d1 < d2){
                                    voronoi_cell_lst[face_idx].cell_vertexs_lst.push_back(intersect_pt_lst[0]);
                                    voronoi_cell_lst[face_idx].cell_vertexs_lst.push_back(intersect_pt_lst[1]);
                                }else{
                                    voronoi_cell_lst[face_idx].cell_vertexs_lst.push_back(intersect_pt_lst[1]);
                                    voronoi_cell_lst[face_idx].cell_vertexs_lst.push_back(intersect_pt_lst[0]);
                                }
                            }else{
                                assert(false);
                            }
                            // for(auto pt : intersect_pt_lst){
                            //     // voronoi_cell_lst[face_idx].cell_vertexs_lst.push_back(pt);
                            //     std::cout<< "4:" << pt << "face_idx:" << face_idx << ","<< face->dual()->point() <<std::endl;
                            // }
                            voronoi_cell_lst[face_idx].is_intersect_boundary = true; // 该cell与边界相交
                        }
                    }
                }else{
                    std::cerr<< "Runtime Assertion failed: Edge must be ray or segment"<<std::endl;
                    assert(false);
                }
                ccb_e++;
            }while(ccb_e != ccb_start);
        }
        // 验证vertex获取没问题
        assert(voronoi_cell_lst[face_idx].cell_vertexs_lst.size() > 1);
    }
    
    // mprint("step 2 finished");
    // 计算四角点属于哪一个voronoi cell
    Point UL(sx  ,sy  );
    Point UR(sx+W,sy  );
    Point DL(sx  ,sy+H);
    Point DR(sx+W,sy+H);

    double UL_distance_min = (H*H + W*W)*1e8;
    double UR_distance_min = UL_distance_min;
    double DL_distance_min = UL_distance_min;
    double DR_distance_min = UL_distance_min;

    int UL_nearest_cell_idx = -1;
    int UR_nearest_cell_idx = -1;
    int DL_nearest_cell_idx = -1;
    int DR_nearest_cell_idx = -1;

    for(face_idx = 0; face_idx < N; face_idx++){
        if(voronoi_cell_lst[face_idx].is_intersect_boundary || !(voronoi_cell_lst[face_idx].is_cell_bounded)){
            
            auto site_pt = voronoi_cell_lst[face_idx].site;
            auto idx = voronoi_cell_lst[face_idx].cell_idx;
            assert(idx == face_idx);
            
            auto UL_squared_distance = CGAL::squared_distance(UL, site_pt);
            auto UR_squared_distance = CGAL::squared_distance(UR, site_pt);
            auto DL_squared_distance = CGAL::squared_distance(DL, site_pt);
            auto DR_squared_distance = CGAL::squared_distance(DR, site_pt);
            
            if(UL_squared_distance < UL_distance_min){
                UL_distance_min = UL_squared_distance.exact().convert_to<double>() ; 
                UL_nearest_cell_idx = idx;
            }
            if(UR_squared_distance < UR_distance_min){
                UR_distance_min = UR_squared_distance.exact().convert_to<double>();
                UR_nearest_cell_idx = idx;
            }
            if(DL_squared_distance < DL_distance_min){
                DL_distance_min = DL_squared_distance.exact().convert_to<double>();
                DL_nearest_cell_idx = idx;
            }
            if(DR_squared_distance < DR_distance_min){
                DR_distance_min = DR_squared_distance.exact().convert_to<double>();
                DR_nearest_cell_idx = idx;
            }

        }
    }

    voronoi_cell_lst[UL_nearest_cell_idx].cell_vertexs_lst.push_back(UL);
    voronoi_cell_lst[UR_nearest_cell_idx].cell_vertexs_lst.push_back(UR);
    voronoi_cell_lst[DL_nearest_cell_idx].cell_vertexs_lst.push_back(DL);
    voronoi_cell_lst[DR_nearest_cell_idx].cell_vertexs_lst.push_back(DR);

    voronoi_cell_lst[UL_nearest_cell_idx].is_have_corner=true;
    voronoi_cell_lst[UR_nearest_cell_idx].is_have_corner=true;
    voronoi_cell_lst[DL_nearest_cell_idx].is_have_corner=true;
    voronoi_cell_lst[DR_nearest_cell_idx].is_have_corner=true;

    // 用凸包算法更新vertex列表中点的顺序
    std::vector<int> idx_lst = {UL_nearest_cell_idx, UR_nearest_cell_idx, DL_nearest_cell_idx, DR_nearest_cell_idx};
    for(auto idx:idx_lst){
        std::vector<Point> convex_hull_v_lst;
        auto v_lst = voronoi_cell_lst[idx].cell_vertexs_lst;
        CGAL::convex_hull_2(v_lst.begin(), v_lst.end(), std::back_inserter(convex_hull_v_lst));
        voronoi_cell_lst[idx].cell_vertexs_lst = convex_hull_v_lst;
    }

    return voronoi_cell_lst;

}



/// @brief 将voronoi图画成svg并保存
/// @param voronoi_cell_lst voronoi图所有单元的相关信息的列表 
/// @param H 图像的高
/// @param W 图像的宽
/// @param svg_save_path SVG图像保存路径
/// @return 
int draw_voronoi_polygon(std::vector<voronoi_cell> voronoi_cell_lst, int H, int W, std::string svg_save_path){
    
    int N = voronoi_cell_lst.size();

    // 创建一个XML文档对象
    tinyxml2::XMLDocument doc;
    // 创建XML声明并添加到文档中
    tinyxml2::XMLDeclaration* declaration = doc.NewDeclaration();
    doc.InsertFirstChild(declaration);
    // 创建SVG根元素并设置必要属性
    tinyxml2::XMLElement* svgElement = doc.NewElement("svg");
    svgElement->SetAttribute("xmlns", "http://www.w3.org/2000/svg");
    svgElement->SetAttribute("width", std::to_string(W).c_str());
    svgElement->SetAttribute("height", std::to_string(H).c_str());
    doc.InsertEndChild(svgElement);

    


    for(size_t i = 0; i < voronoi_cell_lst.size(); i++){
        auto cell = voronoi_cell_lst[i];
        // if(!cell.is_cell_bounded){
        //     continue;
        // }
        std::string points_str = "";
        
        for(auto pt: cell.cell_vertexs_lst){
            points_str += std::to_string(pt.exact().x().convert_to<double>()) + "," +  std::to_string(pt.exact().y().convert_to<double>()) + " ";
        }
        
        if(!points_str.empty()){
            points_str.pop_back();
        }
        // 创建圆形元素表示点
        tinyxml2::XMLElement* pointElement = doc.NewElement("circle");
        pointElement->SetAttribute("cx", cell.site.exact().x().convert_to<double>());
        pointElement->SetAttribute("cy", cell.site.exact().y().convert_to<double>());
        pointElement->SetAttribute("r", 1.0);
        pointElement->SetAttribute("fill", "red");

        svgElement->InsertEndChild(pointElement);
        
        // 创建多边形元素
        tinyxml2::XMLElement* polygonElement = doc.NewElement("polygon");

        polygonElement->SetAttribute("points", points_str.c_str()); 
        polygonElement->SetAttribute("fill", "none"); // 设置不填充颜色
        polygonElement->SetAttribute("stroke", "black"); // 设置边框颜色为蓝色
        polygonElement->SetAttribute("stroke-width", "0.8"); // 设置边框宽度为2

        svgElement->InsertEndChild(polygonElement);
    }

    // 保存XML文档为SVG文件
    // auto filename = (std::string("voronoi_polygon") + std::to_string(N) + std::string(".svg")).c_str();
    tinyxml2::XMLError eResult = doc.SaveFile(svg_save_path.c_str());
    if (eResult != tinyxml2::XML_SUCCESS) {
        std::cerr << "Failed to save the SVG file." << std::endl;
        return 1;
    }

    std::cout << "SVG file has been created successfully." << std::endl;
    return 0;

}


// ====================================================================================



/// @brief 计算直线line和线段seg的交点
/// @param line 直线
/// @param seg 线段
/// @return 返回元组，第一个元素true/false表示是否存在交点，第二个元素给出交点
auto line_segment_intersect(Line line, Segment seg)
{
    Point intersect_pt(-1,-1);
    CGAL::Object result = CGAL::intersection(line, seg);

    if(const Point  *tmp_pt = CGAL::object_cast<Point>(&result)){
        // 有1个交点
        intersect_pt = *(tmp_pt);
        return std::make_tuple(0, intersect_pt);
    }else if(const Segment *iseg = CGAL::object_cast<Segment>(&result)){
        // 直线与线段重合
        return std::make_tuple(1, intersect_pt);
    }else{
        // 无交点
        return std::make_tuple(2, intersect_pt);  
    }
}



// 计算单个cell在图像中的总质量和质心
auto calculate_cell_mass_and_centroid(std::vector<Point>& cell_vertexs_lst, Eigen::Matrix<uint8_t,-1,-1>& pic_mat, Point& cell_site){
    
    assert(cell_vertexs_lst.size() > 2);

    auto H = static_cast<int>(pic_mat.rows());
    auto W = static_cast<int>(pic_mat.cols());

    auto cell_x_min = std::numeric_limits<double>::max();
    auto cell_y_min = std::numeric_limits<double>::max();
    auto cell_x_max = std::numeric_limits<double>::min();
    auto cell_y_max = std::numeric_limits<double>::min();

    for(auto vertex_pt : cell_vertexs_lst){
        // auto x = vertex_pt.x().exact().convert_to<double>();
        // auto y = vertex_pt.y().exact().convert_to<double>();
        auto x = vertex_pt.exact().x().convert_to<double>();
        auto y = vertex_pt.exact().y().convert_to<double>();

        if(x < cell_x_min){
            cell_x_min = x;
        }
        if(x > cell_x_max){
            cell_x_max = x;
        }
        if(y < cell_y_min){
            cell_y_min = y;
        }
        if(y > cell_y_max){
            cell_y_max = y;
        }
        
    }

    int idx_x_min = std::max(static_cast<int>(cell_x_min),0);
    int idx_x_max = std::min(static_cast<int>(cell_x_max),W-1);
    int idx_y_min = std::max(static_cast<int>(cell_y_min),0);
    int idx_y_max = std::min(static_cast<int>(cell_y_max),H-1);
    
    // 构造多边形
    Polygon_2 cell_poly(cell_vertexs_lst.begin(), cell_vertexs_lst.end());

    double cell_mass = 0;
    double centroid_x = 0;
    double centroid_y = 0;

    for(int r = idx_y_min; r < idx_y_max+1; r++){
        // 构造扫描线
        Vector_2 vector(1, 0); // 水平方向向量
        Point p1 = Point(0,r);
        Point p2 = Point(0,r+1);
        Line scan_line1(p1, vector);
        Line scan_line2(p2, vector);

        // 遍历多边形的每条边，计算扫描线与多边形的交点，确定扫描线横坐标的范围
        double current_row_x_min = static_cast<double>(idx_x_max);
        double current_row_x_max = static_cast<double>(idx_x_min);
        
        std::vector<Segment> seg_lst;
        for (auto edge_ = cell_poly.edges_begin(); edge_ != cell_poly.edges_end(); edge_++) {

            auto seg = *edge_;
            bool is_seg_intersect = false;
            auto [intersect_type1, intersection1] = line_segment_intersect(scan_line1, seg);
            if(0 == intersect_type1){
                is_seg_intersect = true;
                auto x = intersection1.exact().x().convert_to<double>();
                if(x < current_row_x_min){
                    current_row_x_min = x;
                }
                if(x > current_row_x_max){
                    current_row_x_max = x;
                }
            }else if(1 == intersect_type1){
                is_seg_intersect = true;
                auto x = seg.source().exact().x().convert_to<double>();
                if(x < current_row_x_min){
                    current_row_x_min = x;
                }
                if(x > current_row_x_max){
                    current_row_x_max = x;
                }

                x = seg.target().exact().x().convert_to<double>();
                if(x < current_row_x_min){
                    current_row_x_min = x;
                }
                if(x > current_row_x_max){
                    current_row_x_max = x;
                }
            }

            auto [intersect_type2, intersection2] = line_segment_intersect(scan_line2, seg);
            if(0 == intersect_type2){
                is_seg_intersect = true;
                auto x = intersection2.exact().x().convert_to<double>();
                if(x < current_row_x_min){
                    current_row_x_min = x;
                }
                if(x > current_row_x_max){
                    current_row_x_max = x;
                }
            }else if(1 == intersect_type2){
                is_seg_intersect = true;
                auto x = seg.source().exact().x().convert_to<double>();
                if(x < current_row_x_min){
                    current_row_x_min = x;
                }
                if(x > current_row_x_max){
                    current_row_x_max = x;
                }

                x = seg.target().exact().x().convert_to<double>();
                if(x < current_row_x_min){
                    current_row_x_min = x;
                }
                if(x > current_row_x_max){
                    current_row_x_max = x;
                }
            }

            if(is_point_in_rectangle(seg.source() ,idx_x_min, r, 1, idx_x_max - idx_x_min + 1)){
                is_seg_intersect = true;
                auto x = seg.source().exact().x().convert_to<double>();
                if(x < current_row_x_min){
                    current_row_x_min = x;
                }
                if(x > current_row_x_max){
                    current_row_x_max = x;
                }
            }
            if(is_point_in_rectangle(seg.target() ,idx_x_min, r, 1, idx_x_max - idx_x_min + 1)){
                is_seg_intersect = true;
                auto x = seg.target().exact().x().convert_to<double>();
                if(x < current_row_x_min){
                    current_row_x_min = x;
                }
                if(x > current_row_x_max){
                    current_row_x_max = x;
                }
            }

            // 记录所有与扫描线相交的线段
            if(is_seg_intersect = true){
                seg_lst.push_back(seg);
            }
        }

        int row_x_min = std::max(static_cast<int>(current_row_x_min), 0);
        int row_x_max = std::min(static_cast<int>(current_row_x_max), W-1);
        
        // 计算扫描行的每个像素矩形与多边形重合的面积
        for(int c = row_x_min; c < row_x_max+1; c++){

            auto pixel_mass = static_cast<int>(pic_mat(r,c));
            if(0 == pixel_mass){ continue; }

            // 像素中心点
            Point px_middle_pt(c+0.5, r+0.5);
            bool is_inner_px = true;
            // 计算像素中心点到线段的距离以判断其是否为内部像素
            for(auto seg: seg_lst){
                auto squared_dist = CGAL::squared_distance(px_middle_pt, seg);
                if(CGAL::to_double(squared_dist) <= std::sqrt(2.0)/2){
                    is_inner_px = false;   // TODO 20250519 真的是以0.5为判断标准吗？
                    break;
                }
            }

            if(is_inner_px){
                cell_mass += pixel_mass;
                centroid_x += (c+0.5)*pixel_mass;
                centroid_y += (r+0.5)*pixel_mass;

            }else{
                // 构造像素矩形
                std::vector<Point> pixel_pt_lst = { Point(c, r), Point(c+1, r), Point(c+1, r+1), Point(c, r+1) };
                Polygon_2 pixel_poly(pixel_pt_lst.begin(), pixel_pt_lst.end());

                // 计算两个凸多边形的交集
                std::vector<Polygon_with_holes_2> result;
                // CGAL::intersection(pixel_poly, cell_poly, std::back_inserter(result)); // TODO 20250518 有可能要换一种求两个多边形交集的方法
                
                Polygon_set_2 ps1, ps2, intersection_result;
                ps1.insert(pixel_poly);
                ps2.insert(cell_poly);
                intersection_result = ps1;
                intersection_result.intersection(ps2);
                intersection_result.polygons_with_holes(std::back_inserter(result));
                

                // 遍历交集结果，计算每个部分的面积并累加
                double overlapping_area = 0;
                if(1 == result.size()){

                    auto poly_with_holes = result[0];
                    // assert(!poly_with_holes.has_holes());
                    auto poly = poly_with_holes.outer_boundary();
                    overlapping_area += poly.area().exact().convert_to<double>();
                    auto poly_centroid = CGAL::centroid(poly.vertices_begin(), poly.vertices_end(), CGALKernel());
                        
                    assert(overlapping_area <= 1.0 && overlapping_area > 0);
                    
                    cell_mass += overlapping_area*pixel_mass;
                    centroid_x += poly_centroid.exact().x().convert_to<double>()*overlapping_area*pixel_mass;
                    centroid_y += poly_centroid.exact().y().convert_to<double>()*overlapping_area*pixel_mass;

                }
                else if(0 == result.size()){

                }
                else{
                    assert(false);
                }
            }

        }
    }

    centroid_x /= cell_mass;
    centroid_y /= cell_mass;

    if(std::abs(cell_mass) < 1e-18){
        std::cout<<"cell_mass:"<<cell_mass<<std::endl;
        std::cout<< cell_site <<std::endl;
        for(int i = 0; i < cell_vertexs_lst.size(); i++){
            auto vertex = cell_vertexs_lst[i];
            auto vertex_x = vertex.exact().x().convert_to<double>();
            auto vertex_y = vertex.exact().y().convert_to<double>();
            std::cout<< vertex_x << ",  "<< vertex_y <<std::endl;
        }
    }

    assert(cell_mass>1e-18);
    assert(!std::isnan(centroid_x) && !std::isnan(centroid_y));
    assert(centroid_x > 0 && centroid_x < W);
    assert(centroid_y > 0 && centroid_y < H);

    return std::make_tuple(cell_mass, centroid_x, centroid_y);

}



void lloyd_algorithm_subprocess(std::vector<voronoi_cell>& vc_lst, Eigen::Matrix<uint8_t,-1,-1>& pic_mat, \
                                std::vector<std::vector<double>>& cmass_cx_cy_result, size_t start_idx, size_t end_idx){

    // int i = 0;
    // for(auto acell: v_sub_lst){
    for(int i = start_idx; i < end_idx; i++){
        auto acell = vc_lst[i];
        auto [cmass,cx,cy] = calculate_cell_mass_and_centroid(acell.cell_vertexs_lst, pic_mat, acell.site);
        // mprint("--" + std::to_string(i) + "--");
        cmass_cx_cy_result[i][0] = cmass;
        cmass_cx_cy_result[i][1] = cx;
        cmass_cx_cy_result[i][2] = cy;
    }
}



/// @brief (多线程)计算每个voronoi cell的质心坐标
/// @param vc_lst 
/// @param pic_mat 
/// @return 
auto lloyd_algorithm(std::vector<voronoi_cell> vc_lst, Eigen::Matrix<uint8_t,-1,-1> pic_mat){

    size_t N = vc_lst.size();

    // N很大时若线程数过多程序可能会莫名其妙跳出
    size_t threads_num = LLOYD_ALGORITHM_THREADS_NUM;
    if(N >= 32000){
        threads_num = LLOYD_ALGORITHM_THREADS_NUM;
    }

    // 计算每份子列表的大小
    size_t sub_lst_size = N / threads_num;

    // 创建线程池
    std::vector<std::thread> threads;
    // std::mutex lock1;

    // 存放结果的二维列表
    std::vector<std::vector<double>> cmass_cx_cy_result(N, std::vector<double>(3, 0.0));
    // mprint("lloyd_algorithm 1");
    for(int i = 0; i < threads_num; i ++){

        // 计算当前子列表的起始和结束位置
        size_t start = i * sub_lst_size;
        size_t end = ((threads_num - 1) == i) ? N : (i + 1) * sub_lst_size;

        // 创建子列表
        // std::vector<voronoi_cell> sub_lst(vc_lst.begin() + start, vc_lst.begin() + end);
        // assert(sub_lst.size() == 3200);

        // 创建子线程
        threads.emplace_back(lloyd_algorithm_subprocess, std::ref(vc_lst), std::ref(pic_mat), std::ref(cmass_cx_cy_result), start, end);
    }
    // mprint("lloyd_algorithm 2");

    // 等待所有线程完成
    for (auto& thread : threads) {
        thread.join();
    }
    // mprint("lloyd_algorithm 3");

    // 整理结果得到返回值
    double total_mass = 0;
    std::vector<Point> new_sites_lst;
    std::vector<double> cmass_lst;

    for(int i = 0; i < N; i++){
        // 统计总质量
        total_mass += cmass_cx_cy_result[i][0];
        // 得到新的site坐标
        if((!std::isnan(cmass_cx_cy_result[i][1])) && (!std::isnan(cmass_cx_cy_result[i][2]))){
            Point new_site(cmass_cx_cy_result[i][1], cmass_cx_cy_result[i][2]);
            new_sites_lst.push_back(new_site);
        }else{
            assert(false);
        }
    }

    // 计算每个cell的归一化质量
    for(int i = 0; i < N; i++){
        cmass_lst.push_back(cmass_cx_cy_result[i][0] / total_mass);
    }
    
    // std::cout<<"total_mass:"<<static_cast<LONG64>(total_mass)<<std::endl;
    assert(new_sites_lst.size() == cmass_lst.size());
    return std::make_tuple(total_mass, new_sites_lst, cmass_lst);

}



/// @brief 将sites_lst和每个cell对应的面积(cmass_lst)写入一个文本文件
/// @param sites_lst 
/// @param cmass_lst 
/// @param filename 
void save_sites_lst(const std::vector<Point>& sites_lst, const std::vector<double>& cmass_lst, const std::string& filename){

    int N = sites_lst.size();
    assert(cmass_lst.size() == N);

    // 以追加模式打开文件
    std::ofstream outFile(filename, std::ios::app);

    // 检查文件是否成功打开
    if (outFile.is_open()) {
        // 设置写入精度
        outFile << std::fixed << std::setprecision(12);

        outFile << "#" << N << std::endl;

        // 遍历sites_lst
        size_t idx = 0;
        for (auto site_pt : sites_lst) {
            auto pt_x = site_pt.exact().x().convert_to<double>();
            auto pt_y = site_pt.exact().y().convert_to<double>();
            // 将double数据写入文件
            outFile << pt_x << "," << pt_y << "," << cmass_lst[idx] << std::endl;
            idx += 1;
        }

        // 关闭文件流
        outFile.close();
    } else {
        // 若文件打开失败，输出错误信息
        std::cerr << "Unable to open file: " << filename << std::endl;
        assert(false);
    }
}



/// @brief 以x_quantizer, y_quantizer作为累计概率在2D平面采样N个点，生成VoronoiDiagram的vc_lst，再用lloyd_algorithm算每个cell的质心，
///        作为新sites_lst生成新vc_lst，迭代60次，将最终的sites_lst存如"source_sites.txt"
/// @param pic_mat 
/// @param N 
/// @param x_quantizer 
/// @param y_quantizer 
/// @return 
auto lloyd_iteration(Eigen::Matrix<uint8_t,-1,-1> pic_mat, 
                     int N, 
                     Eigen::MatrixXd x_quantizer, 
                     Eigen::VectorXd y_quantizer, 
                     std::string txt_save_path, 
                     std::string svg_save_path){
    // 计算总质量
    LONG64 pic_sum = pic_mat.cast<LONG64>().sum();
    std::cout << "pic_sum:"<< pic_sum << std::endl;
    
    int H = x_quantizer.rows();
    int W = x_quantizer.cols();

    // 初始节点的voronoi diagram
    auto sites_lst = generate_sites_lst(N, x_quantizer, y_quantizer);
    auto vc_lst = GetVoronoiDiagram(sites_lst, N, 0, 0, H, W);

    std::cout << "-------- Start Lloyd Iteration --------" << std::endl;

    int i = 60;
    while(i--){
        
        // 计算质心作为新节点
        auto [total_mass, new_sites_lst, cmass_lst] = lloyd_algorithm(vc_lst, pic_mat);
        
        // 打印总质量验证质量&质心计算是否出错
        auto total_mass_L64 = static_cast<LONG64>(total_mass+0.5);
        std::cout<<i<<"--: "<<total_mass_L64<<", sites_num: "<<new_sites_lst.size()<<std::endl;
        // assert(total_mass_L64 == pic_sum);
        
        vc_lst = GetVoronoiDiagram(new_sites_lst, new_sites_lst.size(), 0, 0, H, W);
        // std::cout<< "----" << std::endl;
        
    }

    // 迭代结束后计算最终每个cell的质量
    auto [_1, _2, cmass_lst] = lloyd_algorithm(vc_lst, pic_mat);
    
    // 提取sites形成列表并存入TXT文件
    std::vector<Point> result_sites_lst;
    for(auto cell : vc_lst){
        result_sites_lst.push_back(cell.site);
    }
    
    save_sites_lst(result_sites_lst, cmass_lst, txt_save_path); // std::string("source_sites.txt")
    
    // 将voronoi图画为svg
    draw_voronoi_polygon(vc_lst, H, W, svg_save_path);

    return result_sites_lst;
}



/// @brief 找到sites_lst1中距离sites_lst2每个点最近的点的索引号，并以追加模式存入文本文件
/// @param sites_lst1 
/// @param sites_lst2 
/// @return 
auto find_nearest_site_pt(std::vector<Point> sites_lst1, std::vector<Point> sites_lst2, const std::string& filename){
    
    int N = sites_lst2.size();
    std::vector<int> index_lst(N);
    
    for(int i = 0; i < N; i++){

        auto pt = sites_lst2[i];
        
        int nearest_idx = -1;
        double nearest_distance = std::numeric_limits<double>::max();

        for(int j = 0; j < sites_lst1.size(); j++){
            auto pt_near = sites_lst1[j];
            auto dis2 = CGAL::squared_distance(pt, pt_near).exact().convert_to<double>();
            if(dis2 < nearest_distance){
                nearest_distance = dis2;
                nearest_idx = j;
            }
        }

        index_lst[i] = nearest_idx;

    }

    // 保存到文件，std::ios::app是追加模式标志，写入的数据会被添加到文件的末尾，不会覆盖文件原有内容。若文件不存在，则创建一个新文件。
    std::ofstream outFile(filename, std::ios::app);
    // 检查文件是否成功打开
    if (outFile.is_open()) {
        outFile << "#" << index_lst.size() << "-->" << sites_lst1.size() << std::endl;
        for (auto index : index_lst) {
            outFile << index << std::endl;
        }
        outFile.close(); // 关闭文件流
    } else {
        // 若文件打开失败，输出错误信息
        std::cerr << "Unable to open file: " << filename << std::endl;
        assert(false);
    }

    return index_lst;
}



// stage 2 ===================================================================================




/// @brief 读取文件获取其中的sites_lst（voronoi图的节点坐标）和lambda_lst（对应voronoi cell的面积）
/// @param file_path 
/// @param N_lst 
/// @return 
auto read_sites_and_lambda(std::string& file_path, std::vector<int>& N_lst){
    std::vector<Point> all_sites_lst;
    std::vector<double> all_lambda_lst;

    std::ifstream file(file_path);
    // 检查文件是否成功打开
    if (!file.is_open()) {
        std::cerr << "read_sites_and_lambda ERROR: cannot open file" << std::endl;
        assert(false);
    }
    
    // 逐行读取文件内容
    std::string line;
    int N_idx = 0, cnt = 0;
    auto N = N_lst[0];
    while (std::getline(file, line)) {
        if('#' == line[0]){
            //std::cout << line << std::endl;
            auto num_str = line.substr(1);    // 截取从索引1开始的子字符串
            boost::algorithm::trim(num_str);  // 去除首尾空字符
            auto num = std::stoi(num_str);    // 验证N是否正确
            if(N_idx > 0){
                assert(cnt == N_lst[N_idx-1]);
                cnt = 0;
            }
            assert(num == N_lst[N_idx++]);
            
        }else{
            std::vector<std::string> substr_lst;
            boost::algorithm::split(substr_lst, line, boost::is_any_of(","));
            assert(3 == substr_lst.size());
            auto site_x = std::stod(substr_lst[0]);
            auto site_y = std::stod(substr_lst[1]);
            auto lambda = std::stod(substr_lst[2]);
            all_sites_lst.push_back(Point(site_x, site_y));
            all_lambda_lst.push_back(lambda);
            cnt++;
            
        } 
        
    }
    int N_lst_sum = std::accumulate(N_lst.begin(), N_lst.end(), 0);
    assert(all_lambda_lst.size() == N_lst_sum && all_sites_lst.size() == N_lst_sum);
    
    return std::make_tuple(all_sites_lst, all_lambda_lst);

}



/// @brief 
/// @param file_path 
/// @param N_lst 
/// @return 
auto read_nearest_info(std::string& file_path, std::vector<int>& N_lst){
    // nearest_index.txt
    std::vector<int> all_info_lst;

    std::ifstream file(file_path);
    // 检查文件是否成功打开
    if (!file.is_open()) {
        std::cerr << "read_nearest_info ERROR: cannot open file" << std::endl;
        assert(false);
    }
    
    // 逐行读取文件内容
    std::string line;
    int N_idx = 1, cnt = 0;
    while (std::getline(file, line)) {
        if('#' == line[0]){
            if(N_idx > 1){
                assert(cnt == N_lst[N_idx-1]);
                cnt = 0;
            }
            N_idx++;
        }else{
            std::string line_info = line;
            boost::algorithm::trim(line_info);
            auto index = std::stoi(line_info);
            all_info_lst.push_back(index);
            cnt++;
            
        } 
        
    }
    int N_lst_sum = std::accumulate(N_lst.begin()+1, N_lst.end(), 0);
    assert(all_info_lst.size() == N_lst_sum);
    
    return all_info_lst;
}




class power_cell{
public:
    Weighted_point_2 wsite;
    int cell_idx;
    bool is_cell_bounded;
    bool is_all_pd_vertexs_in_rectangle; // 是否所有voronoi图生成的vertex都在窗口矩形内部
    std::vector<Point> cell_vertexs_lst; // 顺时针排列的cell的边界点

    bool is_have_corner;
    bool is_intersect_boundary;
    bool is_hidden;
    // double cell_mass;    // cell的总质量
    // Point cell_centroid;

    // 构造函数
    power_cell(){
        is_have_corner=false;
        is_intersect_boundary = false;
        cell_idx = -114514;
    }
    
    // 构造函数重载
    power_cell(Weighted_point_2 wsite_pt, int cell_num): wsite(wsite_pt), cell_idx(cell_num) {
        is_have_corner=false;
        is_intersect_boundary = false;
    }

    // 析构函数
    ~power_cell(){
        this->cell_vertexs_lst.clear();
        this->cell_vertexs_lst.shrink_to_fit();
    }

    
};



/// @brief 
/// @param sites_lst 
/// @param weight_lst 
/// @param N 
/// @param sx 
/// @param sy 
/// @param H 
/// @param W 
/// @return 
auto GetPowerDiagram(std::vector<Point> sites_lst, Eigen::VectorXd weight_lst, int N, double sx, double sy, int H, int W)
{
    assert(sites_lst.size() == N && weight_lst.size() == N);
    
    // 创建最终返回值
    std::vector<power_cell> all_power_cell_lst(N);

    // 创建权重点并形成列表&字典
    std::vector<Weighted_point_2> weighted_pt_lst;
    std::map<Weighted_point_2, int> sites_dict;   // 字典用于后续恢复power cell列表顺序，确保power cell 列表顺序与sites_lst相同
    for(int i = 0; i < N; i++){
        auto wpt = Weighted_point_2(sites_lst[i], weight_lst(i));
        weighted_pt_lst.push_back(wpt);
        sites_dict[wpt] = i;
    }

    // Regular 三角剖分
    Regular_triangulation rt;
    rt.insert(weighted_pt_lst.begin(), weighted_pt_lst.end());
    assert(rt.is_valid() && 2 == rt.dimension());

    // hidden cell和正常 cell分开来储存
    int N_pc = rt.number_of_vertices();
    int N_hidden = rt.number_of_hidden_vertices();
    assert(N == (N_pc + N_hidden));
    std::vector<power_cell> power_cell_lst(N_pc); 
    std::vector<power_cell> power_cell_hidden_lst(N_hidden);

    // 先处理hidden cell
    int idx_hidden = 0;
    for(auto vit = rt.hidden_vertices_begin(); vit != rt.hidden_vertices_end(); vit++){
        power_cell_hidden_lst[idx_hidden].wsite = vit->point();
        power_cell_hidden_lst[idx_hidden].is_hidden = true;
        idx_hidden++;
    }
    assert(idx_hidden == N_hidden);

    // 在处理正常cell
    // 规则三角剖分的顶点对偶于 Power Diagram 的一个面
    // 规则三角剖分的边对偶于 Power Diagram 的一条边
    // 规则三角剖分的面（二维情况下是三角形）对偶于 Power Diagram 的一个顶点
    int idx = 0;
    for(auto vit = rt.finite_vertices_begin(); vit != rt.finite_vertices_end(); vit++){
        if(vit->is_hidden()){
            // 这里是不会出现hidden cell的
            assert(0);
        }
        assert(!vit->is_hidden());
        assert(vit->is_valid());

        

        auto wpt = vit->point();  // 权重点
        power_cell_lst[idx].cell_idx = idx;
        power_cell_lst[idx].wsite = wpt;
        power_cell_lst[idx].is_cell_bounded = true;
        power_cell_lst[idx].is_hidden = false;
        power_cell_lst[idx].is_all_pd_vertexs_in_rectangle = true;

        
        // 遍历所有cell边界点，判断该cell是否有部分在矩形框架外
        auto start_eit = rt.incident_edges(vit);
        auto eit = start_eit;
        do{
            Edge edge = *eit;
            assert(edge.first->is_valid());

            Point pt(-1,-1);
            if(!rt.is_infinite(edge)){
                auto dual_e = rt.dual(edge);    // 找到对偶边，即power cell的边
                const Segment* dual_edge = CGAL::object_cast<Segment>(&dual_e);
                if(dual_edge){
                    // 对偶边是线段Segment
                    pt = dual_edge->target();
                }else{
                    // 对偶边为射线Ray
                    const Ray* dual_ray = CGAL::object_cast<Ray>(&dual_e);
                    if(dual_ray){  
                        pt = dual_ray->source();  // 转换得来的Ray只有source没有target
                        power_cell_lst[idx].is_cell_bounded = false; // 有射线必定是无解cell
                    }else{ assert(false); }
                }
                auto x = pt.x(), y = pt.y();
                if((x < sx) || (x > (sx+W)) || (y < sy) || (y > (sy+H))){
                    power_cell_lst[idx].is_all_pd_vertexs_in_rectangle = false;
                }
            }else{
                // auto endpt1 = edge.first->vertex(Regular_triangulation::ccw(edge.second))->point().point();
                // auto endpt2 = edge.first->vertex(Regular_triangulation::cw(edge.second))->point().point();
            }
            eit++;
        }while(eit != start_eit );
        idx++;
    }
    assert(idx == N_pc);

    idx = 0;
    for(auto vit = rt.finite_vertices_begin(); vit != rt.finite_vertices_end(); vit++){
        if(vit->is_hidden()){ assert(0); }


        // case 1：无界cell，所有点都在内部
        if( !power_cell_lst[idx].is_cell_bounded && power_cell_lst[idx].is_all_pd_vertexs_in_rectangle){
            // 遍历所有cell边界点
            auto start_eit = rt.incident_edges(vit);
            auto eit = start_eit;
            do{
                Edge edge = *eit;
                assert(edge.first->is_valid());

                if(!rt.is_infinite(edge)){
                    auto dual_e = rt.dual(edge);    // 找到对偶边，即power cell的边
                    const Segment* dual_edge = CGAL::object_cast<Segment>(&dual_e);
                    if(dual_edge){
                        // 边为线段Segment 对于线段，我们把它的target加入vertex列表
                        auto pt = dual_edge->target();
                        power_cell_lst[idx].cell_vertexs_lst.push_back(pt);
                    }else{
                        const Ray* dual_ray = CGAL::object_cast<Ray>(&dual_e);
                        if(dual_ray){  
                            // 边为射线Ray
                            auto ray_pt = dual_ray->source();  // 转换得来的Ray只有source没有target
                            auto ray_dir = dual_ray->direction().to_vector();
                            // 计算射线与窗口边框的交点（有且仅有一个交点）并加入vertex列表
                            auto [intersect_info, intersect] = ray_rectangle_intersect(ray_pt, ray_dir, static_cast<int>(sx), static_cast<int>(sy), H, W);
                            power_cell_lst[idx].cell_vertexs_lst.push_back(intersect);
                            power_cell_lst[idx].is_intersect_boundary = true; // 该cell与边界相交
                            power_cell_lst[idx].cell_vertexs_lst.push_back(ray_pt); // TODO
                            //
                            // auto next_eit = eit++;
                            // auto next_edge = *next_eit;
                            // auto next_dual_e = rt.dual(next_edge);
                            // const Segment* next_dual_seg = CGAL::object_cast<Segment>(&next_dual_e);
                            // if(next_dual_seg){
                            //     if(next_dual_seg->source() == ray_pt){
                            //         power_cell_lst[idx].cell_vertexs_lst.push_back(ray_pt); // TODO
                            //     }
                            // }else{
                            //     const Ray* next_dual_ray = CGAL::object_cast<Ray>(&next_dual_e);
                            //     if(next_dual_ray){

                            //     }else{
                            //         assert(false);
                            //     }
                            // }
                            
                        }else{ 
                            assert(false);
                        }
                    }
                }
                eit++;
            }while(eit != start_eit );
        }
        // case 2：有界cell，所有点都在内部
        else if( power_cell_lst[idx].is_cell_bounded && power_cell_lst[idx].is_all_pd_vertexs_in_rectangle){
            // 遍历所有cell边界点
            auto start_eit = rt.incident_edges(vit);
            auto eit = start_eit;
            do{
                Edge edge = *eit;
                assert(edge.first->is_valid());

                if(!rt.is_infinite(edge)){
                    auto dual_e = rt.dual(edge);    // 找到对偶边，即power cell的边
                    const Segment* dual_edge = CGAL::object_cast<Segment>(&dual_e);
                    if(dual_edge){
                        // 边为线段Segment 我们把它的target加入vertex列表
                        auto pt = dual_edge->target();
                        power_cell_lst[idx].cell_vertexs_lst.push_back(pt);
                    }else{
                        assert(false); // 有界cell的边只能是线段
                    }
                }
                eit++;
            }while(eit != start_eit );
        }
        // case 3：无界cell，有些点在外部
        else if(!power_cell_lst[idx].is_cell_bounded && !power_cell_lst[idx].is_all_pd_vertexs_in_rectangle){
            // 遍历所有cell边界点
            auto start_eit = rt.incident_edges(vit);
            auto eit = start_eit;
            do{
                Edge edge = *eit;
                assert(edge.first->is_valid());

                if(!rt.is_infinite(edge)){
                    auto dual_e = rt.dual(edge);    // 找到对偶边，即power cell的边
                    const Segment* dual_edge = CGAL::object_cast<Segment>(&dual_e);
                    if(dual_edge){
                        // 边为线段Segment
                        auto source_pt = dual_edge->source();
                        auto target_pt = dual_edge->target();
                        auto is_source_pt_in = is_point_in_rectangle(source_pt, static_cast<int>(sx), static_cast<int>(sy), H, W);
                        auto is_target_pt_in = is_point_in_rectangle(target_pt, static_cast<int>(sx), static_cast<int>(sy), H, W);
                        // 对于线段，我们把它的target加入vertex列表
                        if(is_source_pt_in && is_target_pt_in){
                            power_cell_lst[idx].cell_vertexs_lst.push_back(target_pt);
                        }else if(is_source_pt_in && !is_target_pt_in){
                            Segment _seg(source_pt, target_pt);
                            auto [intersect_info, intersect] = ray_rectangle_intersect(source_pt, _seg.to_vector(), static_cast<int>(sx), static_cast<int>(sy), H, W);
                            power_cell_lst[idx].cell_vertexs_lst.push_back(intersect);
                            power_cell_lst[idx].is_intersect_boundary = true; // 该cell与边界相交
                        }else if(!is_source_pt_in && is_target_pt_in){
                            Segment _seg(target_pt, source_pt);
                            auto [intersect_info, intersect] = ray_rectangle_intersect(target_pt, _seg.to_vector(), static_cast<int>(sx), static_cast<int>(sy), H, W);
                            power_cell_lst[idx].cell_vertexs_lst.push_back(intersect);
                            power_cell_lst[idx].cell_vertexs_lst.push_back(target_pt);
                            power_cell_lst[idx].is_intersect_boundary = true; // 该cell与边界相交
                        }else{
                            // 即便线段2个端点都在边框外部，线段也可能与矩形边框相交
                            Segment _seg(source_pt, target_pt);
                            auto [is_intersect, intersect_pt_lst] = seg_rectangle_intersect(_seg, static_cast<int>(sx), static_cast<int>(sy), H, W);
                            if(is_intersect && intersect_pt_lst.size() > 0){
                                if(1 == intersect_pt_lst.size()){
                                    power_cell_lst[idx].cell_vertexs_lst.push_back(intersect_pt_lst[0]);
                                }else if(2 == intersect_pt_lst.size()){
                                    // 若线段与边框有2个交点，要按照从source到target的顺序将他们添加到列表中
                                    auto d1 = CGAL::squared_distance(intersect_pt_lst[0], source_pt);
                                    auto d2 = CGAL::squared_distance(intersect_pt_lst[1], source_pt);
                                    if(d1 < d2){
                                        power_cell_lst[idx].cell_vertexs_lst.push_back(intersect_pt_lst[0]);
                                        power_cell_lst[idx].cell_vertexs_lst.push_back(intersect_pt_lst[1]);
                                    }else{
                                        power_cell_lst[idx].cell_vertexs_lst.push_back(intersect_pt_lst[1]);
                                        power_cell_lst[idx].cell_vertexs_lst.push_back(intersect_pt_lst[0]);
                                    }
                                }else{
                                    assert(false); // 线段与矩形不可能有超过2个交点
                                }

                                power_cell_lst[idx].is_intersect_boundary = true; // 该cell与边界相交
                            }
                        }
                    }else{
                        const Ray* dual_ray = CGAL::object_cast<Ray>(&dual_e);
                        if(dual_ray){  
                            // 边为射线Ray
                            auto ray_pt = dual_ray->source();  // 转换得来的Ray只有source没有target
                            auto ray_dir = dual_ray->direction().to_vector();

                            // 判断ray端点是否在窗框内部，不在内部必定无法与窗框相交，直接跳至下一边
                            auto is_ray_pt_in = is_point_in_rectangle(ray_pt, static_cast<int>(sx), static_cast<int>(sy), H, W);
                            if(!is_ray_pt_in){ eit++;  continue; }

                            // 计算射线与窗口边框的交点（有且仅有一个交点）并加入vertex列表
                            auto [intersect_info, intersect] = ray_rectangle_intersect(ray_pt, ray_dir, static_cast<int>(sx), static_cast<int>(sy), H, W);
                            power_cell_lst[idx].cell_vertexs_lst.push_back(intersect);
                            power_cell_lst[idx].is_intersect_boundary = true; // 该cell与边界相交
                            power_cell_lst[idx].cell_vertexs_lst.push_back(ray_pt); // TODO  
                        }else{ 
                            assert(false);
                        }
                    }
                }
                eit++;
            }while(eit != start_eit );
        }
        // case 4：有界cell，有些点在外部
        else if( power_cell_lst[idx].is_cell_bounded && !power_cell_lst[idx].is_all_pd_vertexs_in_rectangle){
            // 遍历所有cell边界点
            auto start_eit = rt.incident_edges(vit);
            auto eit = start_eit;
            do{
                Edge edge = *eit;
                assert(edge.first->is_valid());

                if(!rt.is_infinite(edge)){
                    auto dual_e = rt.dual(edge);    // 找到对偶边，即power cell的边
                    const Segment* dual_edge = CGAL::object_cast<Segment>(&dual_e);
                    if(dual_edge){
                        // 边为线段Segment
                        auto source_pt = dual_edge->source();
                        auto target_pt = dual_edge->target();
                        auto is_source_pt_in = is_point_in_rectangle(source_pt, static_cast<int>(sx), static_cast<int>(sy), H, W);
                        auto is_target_pt_in = is_point_in_rectangle(target_pt, static_cast<int>(sx), static_cast<int>(sy), H, W);
                        // 对于线段，我们把它的target加入vertex列表
                        if(is_source_pt_in && is_target_pt_in){
                            power_cell_lst[idx].cell_vertexs_lst.push_back(target_pt);
                        }else if(is_source_pt_in && !is_target_pt_in){
                            Segment _seg(source_pt, target_pt);
                            auto [intersect_info, intersect] = ray_rectangle_intersect(source_pt, _seg.to_vector(), static_cast<int>(sx), static_cast<int>(sy), H, W);
                            power_cell_lst[idx].cell_vertexs_lst.push_back(intersect);
                            power_cell_lst[idx].is_intersect_boundary = true; // 该cell与边界相交
                        }else if(!is_source_pt_in && is_target_pt_in){
                            Segment _seg(target_pt, source_pt);
                            auto [intersect_info, intersect] = ray_rectangle_intersect(target_pt, _seg.to_vector(), static_cast<int>(sx), static_cast<int>(sy), H, W);
                            power_cell_lst[idx].cell_vertexs_lst.push_back(intersect);
                            power_cell_lst[idx].cell_vertexs_lst.push_back(target_pt);
                            power_cell_lst[idx].is_intersect_boundary = true; // 该cell与边界相交
                        }else{
                            
                            // 即便线段2个端点都在边框外部，线段也可能与矩形边框相交
                            Segment _seg(source_pt, target_pt);
                            auto [is_intersect, intersect_pt_lst] = seg_rectangle_intersect(_seg, static_cast<int>(sx), static_cast<int>(sy), H, W);
                            if(is_intersect && intersect_pt_lst.size() > 0){
                                if(1 == intersect_pt_lst.size()){
                                    power_cell_lst[idx].cell_vertexs_lst.push_back(intersect_pt_lst[0]);
                                }else if(2 == intersect_pt_lst.size()){
                                    // 若线段与边框有2个交点，要按照从source到target的顺序将他们添加到列表中
                                    auto d1 = CGAL::squared_distance(intersect_pt_lst[0], source_pt);
                                    auto d2 = CGAL::squared_distance(intersect_pt_lst[1], source_pt);
                                    if(d1 < d2){
                                        power_cell_lst[idx].cell_vertexs_lst.push_back(intersect_pt_lst[0]);
                                        power_cell_lst[idx].cell_vertexs_lst.push_back(intersect_pt_lst[1]);
                                    }else{
                                        power_cell_lst[idx].cell_vertexs_lst.push_back(intersect_pt_lst[1]);
                                        power_cell_lst[idx].cell_vertexs_lst.push_back(intersect_pt_lst[0]);
                                    }
                                }else{
                                    assert(false); // 线段与矩形不可能有超过2个交点
                                }

                                power_cell_lst[idx].is_intersect_boundary = true; // 该cell与边界相交
                            }else{

                            }
                        }
                    }else{
                        assert(false);   
                    }
                }
                eit++;
            }while(eit != start_eit );
        }

        idx++;
    }
    assert(idx == N_pc);

    // 计算四角点属于哪一个power cell
    Point UL(sx  ,sy  );
    Point UR(sx+W,sy  );
    Point DL(sx  ,sy+H);
    Point DR(sx+W,sy+H);

    double UL_distance_min = (H*H + W*W)*1e8;
    double UR_distance_min = UL_distance_min;
    double DL_distance_min = UL_distance_min;
    double DR_distance_min = UL_distance_min;

    int UL_nearest_cell_idx = -1;
    int UR_nearest_cell_idx = -1;
    int DL_nearest_cell_idx = -1;
    int DR_nearest_cell_idx = -1;

    for(idx = 0; idx < N_pc; idx++){
        if(power_cell_lst[idx].is_intersect_boundary || !(power_cell_lst[idx].is_cell_bounded)){
            
            auto site_pt = power_cell_lst[idx].wsite.point();
            auto cell_weight = power_cell_lst[idx].wsite.weight();
            auto cell_idx = power_cell_lst[idx].cell_idx;
            if(idx != cell_idx){
                std::cerr<<idx<<"--"<<cell_idx<<std::endl;
            }
            assert(idx == cell_idx);
            
            auto UL_squared_distance = CGAL::squared_distance(UL, site_pt) - cell_weight;
            auto UR_squared_distance = CGAL::squared_distance(UR, site_pt) - cell_weight;
            auto DL_squared_distance = CGAL::squared_distance(DL, site_pt) - cell_weight;
            auto DR_squared_distance = CGAL::squared_distance(DR, site_pt) - cell_weight;
            
            if(UL_squared_distance < UL_distance_min){
                UL_distance_min = UL_squared_distance.exact().convert_to<double>() ; 
                UL_nearest_cell_idx = idx;
            }
            if(UR_squared_distance < UR_distance_min){
                UR_distance_min = UR_squared_distance.exact().convert_to<double>();
                UR_nearest_cell_idx = idx;
            }
            if(DL_squared_distance < DL_distance_min){
                DL_distance_min = DL_squared_distance.exact().convert_to<double>();
                DL_nearest_cell_idx = idx;
            }
            if(DR_squared_distance < DR_distance_min){
                DR_distance_min = DR_squared_distance.exact().convert_to<double>();
                DR_nearest_cell_idx = idx;
            }
        }
    }

    power_cell_lst[UL_nearest_cell_idx].cell_vertexs_lst.push_back(UL);
    power_cell_lst[UR_nearest_cell_idx].cell_vertexs_lst.push_back(UR);
    power_cell_lst[DL_nearest_cell_idx].cell_vertexs_lst.push_back(DL);
    power_cell_lst[DR_nearest_cell_idx].cell_vertexs_lst.push_back(DR);

    power_cell_lst[UL_nearest_cell_idx].is_have_corner=true;
    power_cell_lst[UR_nearest_cell_idx].is_have_corner=true;
    power_cell_lst[DL_nearest_cell_idx].is_have_corner=true;
    power_cell_lst[DR_nearest_cell_idx].is_have_corner=true;

    // 用凸包算法更新vertex列表中点的顺序
    std::vector<int> idx_lst = {UL_nearest_cell_idx, UR_nearest_cell_idx, DL_nearest_cell_idx, DR_nearest_cell_idx};
    for(auto idx:idx_lst){
        std::vector<Point> convex_hull_v_lst;
        auto v_lst = power_cell_lst[idx].cell_vertexs_lst;
        CGAL::convex_hull_2(v_lst.begin(), v_lst.end(), std::back_inserter(convex_hull_v_lst));
        power_cell_lst[idx].cell_vertexs_lst = convex_hull_v_lst;
    }

    // 对于有可能存在重复点、点顺序不正确的cell，用凸包算法重新算出多边形点列表
    for(idx = 0; idx < N_pc; idx++){
        if(!power_cell_lst[idx].is_cell_bounded || power_cell_lst[idx].is_intersect_boundary){
            std::vector<Point> convex_hull_v_lst;
            auto v_lst = power_cell_lst[idx].cell_vertexs_lst;
            CGAL::convex_hull_2(v_lst.begin(), v_lst.end(), std::back_inserter(convex_hull_v_lst));
            power_cell_lst[idx].cell_vertexs_lst = convex_hull_v_lst;
        }
    }

    // 最后将正常cell列表和hidden cell列表根据输入sites列表的顺序重新融为一体
    double eps = 1e-8;
    for(int i = 0; i < power_cell_lst.size(); i++){
        auto wpt = power_cell_lst[i].wsite;
        auto it = sites_dict.find(wpt);
        if(it != sites_dict.end()){
            auto j = sites_dict[wpt];
            all_power_cell_lst[j] = power_cell_lst[i];
        }else{
            assert(0);
        }
        
        // auto pt = power_cell_lst[i].wsite.point();
        // auto ptx = pt.x(), pty = pt.y();
        // for(int j = 0; j < N; j++){
        //     auto pt0 = sites_lst[j];
        //     if(CGAL::to_double( CGAL::abs(ptx - pt0.x()) ) > 0.1 || CGAL::to_double( CGAL::abs(pty - pt0.y()) ) > 0.1){ 
        //         continue; 
        //     }
        //     else if(CGAL::to_double( CGAL::squared_distance(pt, pt0) ) < eps){
        //         all_power_cell_lst[j] = power_cell_lst[i];
        //         break;
        //     }else{
        //         continue; 
        //     }
        // }
    }
    for(int i = 0; i < power_cell_hidden_lst.size(); i++){
        auto wpt = power_cell_hidden_lst[i].wsite;
        auto it = sites_dict.find(wpt);
        if(it != sites_dict.end()){
            auto j = sites_dict[wpt];
            all_power_cell_lst[j] = power_cell_hidden_lst[i];
        }else{
            assert(0);
        }

        // auto pt = power_cell_hidden_lst[i].wsite.point();
        // auto ptx = pt.x(), pty = pt.y();
        // for(int j = 0; j < N; j++){
        //     auto pt0 = sites_lst[j];
        //     if(CGAL::abs(ptx - pt0.x()) > 0.1 || CGAL::abs(pty - pt0.y()) > 0.1){ 
        //         continue; 
        //     }
        //     else if(CGAL::squared_distance(pt, pt0) < eps){
        //         all_power_cell_lst[j] = power_cell_hidden_lst[i];
        //         break;
        //     }else{
        //         continue; 
        //     }
        // }
    }

    //mprint("power diagram generated");
    return all_power_cell_lst;

}



/// @brief 将power图画成svg并保存
/// @param power_cell_lst power图所有单元的相关信息的列表 
/// @param H 图像的高
/// @param W 图像的宽
/// @return 
int draw_powerdiagram_polygon(std::vector<power_cell> power_cell_lst, int H, int W, std::string filename){
    
    int N = power_cell_lst.size();

    // 创建一个XML文档对象
    tinyxml2::XMLDocument doc;
    // 创建XML声明并添加到文档中
    tinyxml2::XMLDeclaration* declaration = doc.NewDeclaration();
    doc.InsertFirstChild(declaration);
    // 创建SVG根元素并设置必要属性
    tinyxml2::XMLElement* svgElement = doc.NewElement("svg");
    svgElement->SetAttribute("xmlns", "http://www.w3.org/2000/svg");
    svgElement->SetAttribute("width", std::to_string(W).c_str());
    svgElement->SetAttribute("height", std::to_string(H).c_str());
    doc.InsertEndChild(svgElement);

    


    for(size_t i = 0; i < power_cell_lst.size(); i++){
        auto cell = power_cell_lst[i];
        if(cell.is_hidden || cell.cell_vertexs_lst.size() < 3){
            continue;
        }
        std::string points_str = "";
        
        for(auto pt: cell.cell_vertexs_lst){
            points_str += std::to_string(pt.exact().x().convert_to<double>()) + "," +  std::to_string(pt.exact().y().convert_to<double>()) + " ";
        }
        
        if(!points_str.empty()){
            points_str.pop_back();
        }
        // 创建圆形元素表示点
        // tinyxml2::XMLElement* pointElement = doc.NewElement("circle");
        // pointElement->SetAttribute("cx", cell.wsite.point().exact().x().convert_to<double>());
        // pointElement->SetAttribute("cy", cell.wsite.point().exact().y().convert_to<double>());
        // pointElement->SetAttribute("r", 1.0);
        // pointElement->SetAttribute("fill", "red");

        //svgElement->InsertEndChild(pointElement);
        
        // 创建多边形元素
        tinyxml2::XMLElement* polygonElement = doc.NewElement("polygon");

        polygonElement->SetAttribute("points", points_str.c_str()); 
        polygonElement->SetAttribute("fill", "none"); // 设置不填充颜色
        polygonElement->SetAttribute("stroke", "black"); // 设置边框颜色为蓝色
        polygonElement->SetAttribute("stroke-width", "0.8"); // 设置边框宽度为0.8

        svgElement->InsertEndChild(polygonElement);
    }

    // 保存XML文档为SVG文件
    // auto filename = (std::string("Stage2_powerdiagram_polygon") + std::to_string(N) + std::string(".svg")).c_str();
    tinyxml2::XMLError eResult = doc.SaveFile(filename.c_str());
    if (eResult != tinyxml2::XML_SUCCESS) {
        std::cerr << "Failed to save the SVG file." << std::endl;
        return 1;
    }

    std::cout << "power diagram SVG file has been created successfully." << std::endl;
    return 0;

}


// 定义互斥锁
// std::mutex mtx;
/// @brief 计算一个power cell中的3项积分（优化要用） TODO 重点检查
/// @param power_cell 
/// @param pic_mat_normalized 
/// @return 
auto integral_in_powercell(power_cell power_cell, Eigen::Matrix<double,-1,-1>& pic_mat_normalized, std::mutex& mtx, int H, int W, int mode = 0){
    
    if(power_cell.is_hidden){
        return std::make_tuple(0.0, 0.0, 0.0);
    }

    if(power_cell.cell_vertexs_lst.size() < 3){
        //assert(false);
        return std::make_tuple(0.0, 0.0, 0.0); // 有些非hidden的power cell会被挤到画框外面
    }
    // mtx.lock();
    // int H = pic_mat_normalized.rows(), W = pic_mat_normalized.cols();
    // mtx.unlock();
    // 获取site点坐标以及权重
    auto si_pt = power_cell.wsite.point();
    auto weight = power_cell.wsite.weight().exact().convert_to<double>();

    auto cell_vertexs_lst = power_cell.cell_vertexs_lst;
    
    auto cell_x_min = std::numeric_limits<double>::max();
    auto cell_y_min = std::numeric_limits<double>::max();
    auto cell_x_max = std::numeric_limits<double>::min();
    auto cell_y_max = std::numeric_limits<double>::min();

    for(auto& vertex_pt : cell_vertexs_lst){
        auto x = vertex_pt.exact().x().convert_to<double>();
        auto y = vertex_pt.exact().y().convert_to<double>();

        if(x < cell_x_min){ cell_x_min = x; }
        if(x > cell_x_max){ cell_x_max = x; }
        if(y < cell_y_min){ cell_y_min = y; }
        if(y > cell_y_max){ cell_y_max = y; }
        
    }

    int idx_x_min = std::max(static_cast<int>(cell_x_min),0);
    int idx_x_max = std::min(static_cast<int>(cell_x_max),W-1);
    int idx_y_min = std::max(static_cast<int>(cell_y_min),0);
    int idx_y_max = std::min(static_cast<int>(cell_y_max),H-1);

    // 构造多边形
    Polygon_2 cell_poly(cell_vertexs_lst.begin(), cell_vertexs_lst.end());

    double cell_mass = 0;
    double x_si_2_mass = 0;
    double cx_mass = 0, cy_mass = 0;
    // auto x_si_2_mass = CGAL::squared_distance(si_pt, si_pt) * 0;

    for(int r = idx_y_min; r < idx_y_max+1; r++){
        // 构造扫描线
        Vector_2 vector(1, 0); // 水平方向向量
        Point p1 = Point(0,r), p2 = Point(0,r+1);
        Line scan_line1(p1, vector), scan_line2(p2, vector);

        // 遍历多边形的每条边，计算扫描线与多边形的交点，确定扫描线横坐标的范围
        double current_row_x_min = static_cast<double>(idx_x_max);
        double current_row_x_max = static_cast<double>(idx_x_min);
        
        std::vector<Segment> seg_lst;
        for (auto edge_ = cell_poly.edges_begin(); edge_ != cell_poly.edges_end(); edge_++) {

            auto seg = *edge_;
            bool is_seg_intersect = false;
            auto [intersect_type1, intersection1] = line_segment_intersect(scan_line1, seg); // intersect_type1为0表示有一个交点，为1表示线段与直线重合
            if(0 == intersect_type1){
                is_seg_intersect = true;
                auto x = intersection1.exact().x().convert_to<double>();
                if(x < current_row_x_min){ current_row_x_min = x; }
                if(x > current_row_x_max){ current_row_x_max = x; }
            }else if(1 == intersect_type1){
                is_seg_intersect = true;
                auto x = seg.source().exact().x().convert_to<double>();
                if(x < current_row_x_min){ current_row_x_min = x; }
                if(x > current_row_x_max){ current_row_x_max = x; }

                x = seg.target().exact().x().convert_to<double>();
                if(x < current_row_x_min){ current_row_x_min = x; }
                if(x > current_row_x_max){ current_row_x_max = x; }
            }

            auto [intersect_type2, intersection2] = line_segment_intersect(scan_line2, seg);
            if(0 == intersect_type2){
                is_seg_intersect = true;
                auto x = intersection2.exact().x().convert_to<double>();
                if(x < current_row_x_min){ current_row_x_min = x; }
                if(x > current_row_x_max){ current_row_x_max = x; }
            }else if(1 == intersect_type2){
                is_seg_intersect = true;
                auto x = seg.source().exact().x().convert_to<double>();
                if(x < current_row_x_min){ current_row_x_min = x; }
                if(x > current_row_x_max){ current_row_x_max = x; }

                x = seg.target().exact().x().convert_to<double>();
                if(x < current_row_x_min){ current_row_x_min = x; }
                if(x > current_row_x_max){ current_row_x_max = x; }
            }
            // TODO 20250519
            if(is_point_in_rectangle(seg.source() ,idx_x_min, r, 1, idx_x_max - idx_x_min + 1)){
                is_seg_intersect = true;
                auto x = seg.source().exact().x().convert_to<double>();
                if(x < current_row_x_min){ current_row_x_min = x; }
                if(x > current_row_x_max){ current_row_x_max = x; }
            }
            if(is_point_in_rectangle(seg.target() ,idx_x_min, r, 1, idx_x_max - idx_x_min + 1)){
                is_seg_intersect = true;
                auto x = seg.target().exact().x().convert_to<double>();
                if(x < current_row_x_min){ current_row_x_min = x; }
                if(x > current_row_x_max){ current_row_x_max = x; }
            }

            // 记录所有与扫描线相交的线段
            if(is_seg_intersect = true){
                seg_lst.push_back(seg);
            }
        }

        int row_x_min = std::max(static_cast<int>(std::floor(current_row_x_min)), 0);
        int row_x_max = std::min(static_cast<int>(std::floor(current_row_x_max)), W-1);
        
        // 计算扫描行的每个像素矩形与power cell多边形重合的面积
        for(int c = row_x_min; c < row_x_max+1; c++){

            // auto pixel_mass = static_cast<double>(pic_mat_normalized(r,c));
            double pixel_mass;
            { 
                //std::lock_guard<std::mutex> lock(mtx);
                pixel_mass = pic_mat_normalized(r,c);
            }

            if(0 == pixel_mass ){ continue; }

            // 像素中心点
            Point px_middle_pt(c+0.5, r+0.5);
            bool is_inner_px = true;
            // 计算像素中心点到线段的距离以判断其是否为内部像素
            for(auto& seg: seg_lst){
                auto squared_dist = CGAL::squared_distance(px_middle_pt, seg);
                if(CGAL::to_double(squared_dist) < std::sqrt(2)/2){ is_inner_px = false; break; }
            }
            
            
            if(is_inner_px){
                // 多边形内部像素
                // std::lock_guard<std::mutex> lock(mtx);
                cell_mass += pixel_mass;
                x_si_2_mass += CGAL::to_double( CGAL::squared_distance(px_middle_pt, si_pt) )*pixel_mass;
                cx_mass += px_middle_pt.exact().x().convert_to<double>()*pixel_mass;
                cy_mass += px_middle_pt.exact().y().convert_to<double>()*pixel_mass;

            }else{
                // 有可能存在相交（不是一定相交）
                // std::lock_guard<std::mutex> lock(mtx);
                // 构造像素矩形
                std::vector<Point> pixel_pt_lst = { Point(c, r), Point(c+1, r), Point(c+1, r+1), Point(c, r+1) };
                
                Polygon_2 pixel_poly(pixel_pt_lst.begin(), pixel_pt_lst.end());

                
                // 计算两个凸多边形的交集
                std::vector<Polygon_with_holes_2> result;
                // {
                    // std::lock_guard<std::mutex> lock(mtx);
                    // try {
                        // CGAL::intersection(pixel_poly, cell_poly, std::back_inserter(result));
                    // } catch (const char* e) {
                    //     mtx.lock();
                    //     CGAL::intersection(pixel_poly, cell_poly, std::back_inserter(result));
                    //     mtx.unlock();
                    // }
                Polygon_set_2 ps1, ps2, intersection_result;
                ps1.insert(pixel_poly);
                ps2.insert(cell_poly);
                intersection_result = ps1;
                // mtx.lock();
                intersection_result.intersection(ps2);
                // mtx.unlock();
                intersection_result.polygons_with_holes(std::back_inserter(result));
                    
                // }
                
                
                // 遍历交集结果，计算每个部分的面积并累加
                double overlapping_area = 0;
                if(1 == result.size()){
                    
                    auto poly_with_holes = result[0];
                    auto poly = poly_with_holes.outer_boundary();
                    overlapping_area += poly.area().exact().convert_to<double>();

                    Point poly_centroid = CGAL::centroid(poly.vertices_begin(), poly.vertices_end());
                    // {   
                        // std::lock_guard<std::mutex> lock(mtx);
                        // poly_centroid = CGAL::centroid(poly.vertices_begin(), poly.vertices_end(), CGALKernel());
                        // poly_centroid = CGAL::centroid(poly.vertices_begin(), poly.vertices_end());
                    // }

                    assert(overlapping_area <= 1.0 && overlapping_area > 0);
                    
                    auto dmass = overlapping_area*pixel_mass;
                    cell_mass += dmass;
                    x_si_2_mass += CGAL::to_double(CGAL::squared_distance(poly_centroid, si_pt)) * dmass;
                    cx_mass += poly_centroid.exact().x().convert_to<double>() * dmass;
                    cy_mass += poly_centroid.exact().y().convert_to<double>() * dmass;

                }
                else if(0 == result.size()){
                    // 像素中心点到线段的距离小于sqrt(2)/2，但其实是内部像素
                    // cell_mass += pixel_mass;
                    // x_si_2_mass += CGAL::to_double( CGAL::squared_distance(px_middle_pt, si_pt) )*pixel_mass;
                    // cx_mass += px_middle_pt.exact().x().convert_to<double>()*pixel_mass;
                    // cy_mass += px_middle_pt.exact().y().convert_to<double>()*pixel_mass;

                }
                else{
                    std::cerr<< "ERROR in function <integral_in_powercell>, result.size(): "<< std::to_string(result.size()) <<std::endl;
                    assert(false);
                }
            }

        }
    }

    //assert(cell_mass>1e-18);
    double x_si_2_mass_double = x_si_2_mass;//.exact().convert_to<double>();
    double cell_mass_weighted = cell_mass*weight;
    if(0 == mode){
        return std::make_tuple(cell_mass, x_si_2_mass_double, cell_mass_weighted);
    }else if(1 == mode){
        return std::make_tuple(cell_mass, cx_mass, cy_mass);
    }else{
        return std::make_tuple(cell_mass, x_si_2_mass_double, cell_mass_weighted);
    }
}   



/// @brief 计算一部分 power cell的积分
/// @param power_cell_lst 
/// @param pic_mat_normalized 
/// @param result 
/// @param start 
/// @param end 
void integral_in_powercell_sublst(std::vector<power_cell>& power_cell_lst, Eigen::Matrix<double,-1,-1>& pic_mat_normalized,\
    Eigen::MatrixXd& result, std::mutex& mtx, size_t start, size_t end, int H, int W)
{

        // for(int i = 0; i < end-start; i++){
    for(int i = start; i < end; i++){
        //std::cout<< start << ": " << i<< std::endl;
        auto pc = power_cell_lst[i];  // 加锁
        auto [cell_mass, x_si_2_mass, cell_mass_weighted] = integral_in_powercell(pc, std::ref(pic_mat_normalized), std::ref(mtx), H, W);
        // result[i][0] = cell_mass;
        // result[i][1] = x_si_2_mass;
        // result[i][2] = cell_mass_weighted;

        result(i,0) = cell_mass;   // 加锁
        result(i,1) = x_si_2_mass;
        result(i,2) = cell_mass_weighted;
        
    }

}



/// @brief 以多线程的方式计算power cell的积分，结果放入一个N*3的矩阵中，N代表power cell的数量，3代表在一个power cell内部要计算的3项积分项
/// @param power_cell_lst 
/// @param pic_mat_normalized 
auto integral_in_powercell_multiprocessing(std::vector<power_cell> power_cell_lst, Eigen::Matrix<double,-1,-1> pic_mat_normalized){
    size_t N = power_cell_lst.size();

    // N很大时若线程数过多程序可能会莫名其妙跳出 不知道为什么？ TODO
    size_t threads_num = INTEGRAL_IN_POWERCELL_THREADS_NUM;//std::thread::hardware_concurrency();
    if(N >= 8000){
        threads_num = INTEGRAL_IN_POWERCELL_THREADS_NUM;
    }


    // 计算每份子列表的大小
    size_t sub_lst_size = N / threads_num;

    std::vector<size_t> start_idx_lst;
    std::vector<size_t> end_idx_lst;
    for(int i = 0; i < threads_num; i++){
        // 计算子列表的起始和结束位置
        size_t start = i * sub_lst_size;
        size_t end = ((threads_num - 1) == i) ? N : (i + 1) * sub_lst_size;
        start_idx_lst.push_back(start);
        end_idx_lst.push_back(end);
    }
    
    // 创建线程池
    std::vector<std::thread> threads;

    // 存放结果的二维数组
    Eigen::MatrixXd result(N, 3);
    result.setZero();

    // 定义互斥锁
    std::mutex mtx;
    int H = pic_mat_normalized.rows(), W = pic_mat_normalized.cols();
    for(int i = 0; i < threads_num; i++){
        // 创建子线程
        threads.emplace_back(integral_in_powercell_sublst, std::ref(power_cell_lst), std::ref(pic_mat_normalized), std::ref(result), std::ref(mtx), \
                             start_idx_lst[i], end_idx_lst[i], H, W);
    }

    // 等待所有线程完成
    for (auto& thread: threads) {
        thread.join();
    }


    return result;  // result是N行3列的数组，第1列代表每个power cell的质量，第2列代表|x-si|^2dμ，第3列是power cell质量乘权重，用于convex function优化计算的是2,3列数据
}



/// @brief 计算当前level的优化用积分，进而得到当前的cost function和梯度信息
auto level_x(Eigen::VectorXd weight_lst, std::vector<Point> sites_lst, Eigen::VectorXd lambda_lst, int N, \
            Eigen::Matrix<double,-1,-1> pic_mat_normalized, double sx, double sy, int H, int W)
{
    // 根据当前sites_lst和weight_lst得到power diagram
    auto power_cell_lst = GetPowerDiagram(sites_lst, weight_lst, N, sx, sy, H, W);
    assert(sites_lst.size() == power_cell_lst.size());
    mprint("power diagram generated.");

    //draw_powerdiagram_polygon(power_cell_lst, H, W);

    Eigen::MatrixXd result(power_cell_lst.size(), 3);
    try{
        result = integral_in_powercell_multiprocessing(power_cell_lst, pic_mat_normalized);
    }
    catch(int e){
        std::cerr << "Exception caught: " << e << std::endl;
    }   
    mprint("integral in powercell multiprocessing finished.");

    auto mass_lst = result.col(0);
    double total_mass = mass_lst.sum();
    assert(std::abs(total_mass - 1.0) < 1e-8);
    //std::cout<< "total_mass:" << total_mass << std::endl;
    auto x_si_2_term = result.col(1).sum();
    auto weighted_mass_term = result.col(2).sum();
    
    auto term1 = weight_lst.dot(lambda_lst);
    double convex_function_val = term1 - (x_si_2_term - weighted_mass_term);
    Eigen::VectorXd convex_function_diff = mass_lst -  lambda_lst; // N维梯度向量
    // grad = convex_function_diff;
    return std::make_tuple(convex_function_val, convex_function_diff);
}



void save_2d_list(const std::vector<std::vector<double>>& data_lst, const std::string& filename){


    std::ofstream outFile(filename, std::ios::out);

    // 检查文件是否成功打开
    if (outFile.is_open()) {
        // 设置写入精度
        outFile << std::fixed << std::setprecision(12);

        // 遍历data_lst
        for (auto data : data_lst) {
            // 将double数据写入文件
            outFile << data[0] << "," << data[1] << "," << data[2] << "," << data[3] << std::endl;
        }

        // 关闭文件流
        outFile.close();
    } else {
        // 若文件打开失败，输出错误信息
        std::cerr << "Unable to open file: " << filename << std::endl;
        assert(false);
    }
}



void save_2d_list(const Eigen::MatrixXd& data_lst, const std::string& filename){



    std::ofstream outFile(filename, std::ios::out);

    // 检查文件是否成功打开
    if (outFile.is_open()) {
        // 设置写入精度
        outFile << std::fixed << std::setprecision(12);

        // 遍历data_lst
        // for (int i = 0; i < data_lst.rows(); i++) {
            
        //     // 将double数据写入文件
        //     outFile << data_lst(i,0) << "," << data_lst(i,1) << "," << data_lst(i,2) << "," << data_lst(i,3) << std::endl;
        // }
        outFile << std::fixed << std::setprecision(12)
        << data_lst.format(Eigen::IOFormat(
            Eigen::StreamPrecision,   // 继承流的精度设置
            Eigen::DontAlignCols,     // 不进行列对齐
            ", ",                     // 数值间用逗号+空格分隔
            "\n"                      // 行间用换行符分隔
        ));

        // 关闭文件流
        outFile.close();
    } else {
        // 若文件打开失败，输出错误信息
        std::cerr << "Unable to open file: " << filename << std::endl;
        assert(false);
    }
}



/// @brief 
class LBFGSmin
{
public:
    Eigen::VectorXd weight_lst;      // 权重列表
    std::vector<Point> sites_lst;
    Eigen::VectorXd lambda_lst;
    int N;                           // sites_lst长度，例如500 2000 8000 32000
    Eigen::Matrix<double,-1,-1> pic_mat_normalized;
    double sx;
    double sy;
    int H;
    int W;

    LBFGSmin(Eigen::VectorXd weight_lst, std::vector<Point> sites_lst, Eigen::VectorXd lambda_lst, \
            int N, Eigen::Matrix<double,-1,-1> pic_mat_normalized, double sx, double sy, int H, int W){
        this->weight_lst = weight_lst;
        this->sites_lst = sites_lst;
        this->lambda_lst = lambda_lst;
        this->N = N;
        this->pic_mat_normalized = pic_mat_normalized;
        this->sx = sx;
        this->sy = sy;
        this->H = H;
        this->W = W;
    }

    auto run(void)
    {
        double finalCost;
        int N = weight_lst.size();
        Eigen::VectorXd x(N);

        /* Set the initial guess */
        x = this->weight_lst;

        /* Set the minimization parameters */
        lbfgs::lbfgs_parameter_t params;
        params.g_epsilon = 0*1.0e-10;
        params.past = 10;
        params.delta = 1.0e-10;

        /* Start minimization */
        int ret = lbfgs::lbfgs_optimize(x, finalCost, costFunction, \
                                        nullptr, monitorProgress, this, params);

        /* Report the result. */
        std::cout << std::setprecision(8)
                  << "=============LBFGS-END===============" << std::endl
                  << "L-BFGS Optimization Returned: " << ret  << ", Minimized Cost: " << finalCost << std::endl;
                //   << "Optimal Variables: " << std::endl
                //   << x.transpose() << std::endl;
        return std::make_tuple(ret, x);
    }

private:
    static double costFunction(void *instance,
                               const Eigen::VectorXd &x,
                               Eigen::VectorXd &g)
    {
        auto self = ((LBFGSmin*)instance);
        // self->weight_lst
        auto [cost, _grad] = level_x(x, self->sites_lst, self->lambda_lst, self->N, \
                                    self->pic_mat_normalized, self->sx, self->sy, self->H, self->W);
        g = _grad;
        return cost;
    }

    static int monitorProgress(void *instance,
                               const Eigen::VectorXd &x,
                               const Eigen::VectorXd &g,
                               const double fx,
                               const double step,
                               const int k,
                               const int ls)
    {
        auto self = ((LBFGSmin*)instance);
        std::cout << std::setprecision(8)
                  //<< "================================" << std::endl
                  << "N:" << self->weight_lst.size() <<", Iteration: " << k << ", Function Value: " << fx << std::endl;
                  //<< "Gradient Inf Norm: " << g.cwiseAbs().maxCoeff() << std::endl;
                  //<< "Variables: " << std::endl
                  //<< x.transpose() << std::endl;
        return 0;
    }
};



/// @brief LBFGS最优输运优化，得到每个power cell的权重，使得光源面的一个voronoi cell的质量尽可能等于对应的power cell的质量。\
最后输出voronoi cell的site和power cell的质心的映射关系（注意剔除hidden power cell和质量为0的power cell）
/// @param all_sites_lst 
/// @param all_lambda_lst 
/// @param N_lst 
/// @param nearest_info_file_path 
/// @param target_file_path 
/// @param name_tag 
/// @return 
auto OTM_LBFGS_optimization(std::vector<Point> all_sites_lst, std::vector<double> all_lambda_lst, std::vector<int> N_lst, \
                            std::string& nearest_info_file_path, std::string& target_file_path, std::string& name_tag, std::string save_dir)
{
    auto target_pic_mat = load_pic_as_gray(target_file_path);
    int H = target_pic_mat.rows();
    int W = target_pic_mat.cols();

    // 获取光源面的sites_lst和对应的power cell的质心的，我们希望光源面site的光，经过折射后，会打在对应的power cell的质心处。
    std::vector<std::vector<double>> SrcSites2Centriod;

    // 目标图矩阵归一化 TODO
    Eigen::Matrix<double,-1,-1> target_pic_mat_double = target_pic_mat.cast<double>();
    auto img_sum = target_pic_mat_double.sum();
    target_pic_mat_double = target_pic_mat_double/img_sum;

    // N_lst累加
    std::vector<int> cumN_lst(N_lst.size());
    for(int i = 0; i < N_lst.size(); i++){
        if(0 == i){
            cumN_lst[i] = N_lst[i];
        } else{
            cumN_lst[i] = N_lst[i] + cumN_lst[i-1];
        }
    }

    // 读取nearest信息
    auto all_nearest_info = read_nearest_info(nearest_info_file_path, N_lst);


    Eigen::VectorXd weight_last_level;
    for(int i = 0; i < N_lst.size(); i++){

        auto start = std::chrono::high_resolution_clock::now();

        auto N = N_lst[i];
        std::vector<Point> this_level_sites_lst;
        std::vector<double> this_level_lambda_lst;
        Eigen::VectorXd this_level_lambda_eigenvec;
        Eigen::VectorXd weight_init(N);
        // 准备初始权重
        if(0 == i){
            this_level_sites_lst = std::vector<Point>(all_sites_lst.begin(), all_sites_lst.begin() + cumN_lst[i]);
            this_level_lambda_lst = std::vector<double>(all_lambda_lst.begin(), all_lambda_lst.begin() + cumN_lst[i]);
            assert(N == this_level_sites_lst.size() && N == this_level_lambda_lst.size());
            this_level_lambda_eigenvec = Eigen::Map<Eigen::VectorXd>(this_level_lambda_lst.data(), this_level_lambda_lst.size());

            // 初始权重全为0
            weight_init.setZero();
        }else{
            this_level_sites_lst = std::vector<Point>(all_sites_lst.begin() + cumN_lst[i-1], all_sites_lst.begin() + cumN_lst[i]);
            this_level_lambda_lst = std::vector<double>(all_lambda_lst.begin() + cumN_lst[i-1], all_lambda_lst.begin() + cumN_lst[i]);
            assert(N == this_level_sites_lst.size() && N == this_level_lambda_lst.size());
            this_level_lambda_eigenvec = Eigen::Map<Eigen::VectorXd>(this_level_lambda_lst.data(), this_level_lambda_lst.size());

            std::vector<int> nearest_info(all_nearest_info.begin() + cumN_lst[i-1] - N_lst[0], all_nearest_info.begin() + cumN_lst[i] - N_lst[0]); 
            for(int j = 0; j < N; j++){
                weight_init(j) = weight_last_level(nearest_info[j]);
            }
        }
        
        // ===============================================================================================================================
        
        auto opt = LBFGSmin(weight_init, this_level_sites_lst, this_level_lambda_eigenvec, N, target_pic_mat_double, 0.0, 0.0, H, W);
        auto [state, weight_optimized] = opt.run();

        // 优化结束，画图
        auto power_cell_lst = GetPowerDiagram(this_level_sites_lst, weight_optimized, N, 0.0, 0.0, H, W);

        std::string file_name = std::string("Stage2_powerdiagram_polygon") + std::to_string(N) + std::string(".svg");
        draw_powerdiagram_polygon(power_cell_lst, H, W, save_dir+file_name);

        weight_last_level.resize(weight_optimized.size());
        weight_last_level = weight_optimized;

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "Time consumed(seconds):" << duration.count()/1000.0 << std::endl;


        // ================================================================================================================================
        // 最后一轮循环，记录光源面sites和对应power cell的质心并存入文件
        if(i == N_lst.size()-1){
            
            assert(this_level_sites_lst.size() == power_cell_lst.size());

            // 统计有多少个有效的pwoer cell
            int notHidden_pc_cnt = 0;
            for(int k = 0; k < power_cell_lst.size(); k++){
                if(!power_cell_lst[k].is_hidden){ notHidden_pc_cnt += 1; }
            }

            
            for(int k = 0; k < power_cell_lst.size(); k++){
                auto pc = power_cell_lst[k];
                auto src_pt = this_level_sites_lst[k];

                // 剔除隐藏power cell
                if(pc.is_hidden){continue;}

                // 检查对应关系
                auto distance2 = CGAL::to_double( CGAL::squared_distance(pc.wsite.point(), src_pt) );
                assert(distance2 < 1e-8);

                std::mutex mtx;
                auto [pc_mass, pc_cx_mass, pc_cy_mass] = integral_in_powercell(pc, target_pic_mat_double, mtx, H, W, 1);
                if(pc_mass <= 1e-9){
                    std::cout<<"too small pc_mass:"<<pc_mass<<std::endl;
                }
                // assert(pc_mass > 1e-9);

                // 剔除无质量pc（无质量表示该power cell内部对应的target图像素和为0）理论上经过OTM优化的不该有质量为0的power cell
                // 不知道为什么还真有质量为0的power cell
                if(pc_mass <= 1e-9){
                    continue;
                }else{
                    std::vector<double> temp(4);
                    temp[0] = src_pt.exact().x().convert_to<double>();
                    temp[1] = src_pt.exact().y().convert_to<double>();
                    temp[2] = pc_cx_mass/pc_mass;
                    temp[3] = pc_cy_mass/pc_mass;
                    SrcSites2Centriod.push_back(temp);
                }
                // SrcSites2Centriod 每行4个数字, 前2个是光源面site的xy坐标, 后2个是对应的power cell的质心xy坐标
                // 并且存入stage2_SrcSites2Centriod.txt文件中
            }

            // assert(notHidden_pc_cnt == SrcSites2Centriod.size());
            save_2d_list(SrcSites2Centriod, save_dir+"stage2_SrcSites2Centriod.txt");
            // return std::make_tuple( SrcSites2Centriod, power_cell_lst);
        }

    }
    return SrcSites2Centriod;
    // assert(weight_last_level.size() == N_lst[N_lst.size()-1]);
    // return weight_last_level;


}





// stage 3 ===================================================================================
// Stage2结束保存2d列表，保存power_cell_lst对象。边界sites挪到边界上。网格化。自然领域插值。

// 1-加载txt文件读取SrcSites2Centriod数据，并添加边界点，边界点不发生任何折射

/// @brief 读取txt文件中SrcSites2Centriod数据，得到一个二维列表，每行4个元素，分别代表src site的xy坐标和target power cell的质心xy坐标
/// @param file_path 
/// @return 
auto read_src2centriod_file(std::string& file_path){

    std::vector<std::vector<double>> SrcSites2Centriod_lst;

    std::ifstream file(file_path);

    // 检查文件是否成功打开
    if (!file.is_open()) {
        std::cerr << "read_src2centriod_file ERROR: cannot open file" << std::endl;
        assert(false);
    }
    
    // 逐行读取文件内容
    std::string line;
    while (std::getline(file, line)) {
        std::vector<double> temp;

        std::stringstream ss(line);
        std::string item;
        while (std::getline(ss, item, ',')) {
            temp.push_back(std::stod(item));
        }
        
        assert(4 == temp.size());
        SrcSites2Centriod_lst.push_back(temp);
    }

    
    return SrcSites2Centriod_lst;
}


// 2-对图片范围做网格化，然后用做自然邻域插值，得到网格格点上的目标坐标

/// @brief 对输入的单个点执行自然邻域插值
/// @param DT_nn 
/// @param query 
/// @param src_points_value_map 
/// @return 
auto natural_neighbor_interpolate(const DelaunayTri& DT_nn,
                                  const Point query, 
                                  std::map<Point, std::pair<double, double>>& src_points_value_map, 
                                  double shapen_factor=-1){
    std::vector<std::pair<Point, Coord_type>> coords;
    // Coord_type norm;
    
    // 参考：https://github.com/CGAL/cgal/blob/master/Interpolation/examples/Interpolation/nn_coordinates_2.cpp
    auto nn_result = CGAL::natural_neighbor_coordinates_2(DT_nn, query, std::back_inserter(coords));
    auto norm = nn_result.second;
    auto success = nn_result.third;

    double interpolate_x = 0, interpolate_y = 0;
    if(success){
        
        double norm_val = norm.exact().convert_to<double>();
        double area = 0;
        double sum_exp = 0;
        std::vector<double> coef_lst;
        for(size_t i = 0; i < coords.size(); i++){
            // auto neighbor_pt = coords[i].first;
            auto part_area = coords[i].second.exact().convert_to<double>();
            area += part_area;
            auto coef = (part_area / norm_val);  // 归一化的系数
            sum_exp += std::exp(shapen_factor * coef);
            coef_lst.push_back(coef);
            
            // interpolate_x += src_points_value_map[neighbor_pt].first  * coef;
            // interpolate_y += src_points_value_map[neighbor_pt].second * coef;
        }
        
        // 检查是否归一
        auto e = std::abs(area - norm_val);
        if(e >= 1e-7){
            std::cout<< std::setprecision(12) << "error:"<< e << std::endl;
        }
        assert((e < 1e-7));

        // 插值
        if(shapen_factor > 0){
            for(size_t i = 0; i < coords.size(); i++){
                auto neighbor_pt = coords[i].first;
                auto exp_coef = std::exp(shapen_factor * coef_lst[i])/sum_exp;
                interpolate_x += src_points_value_map[neighbor_pt].first  * exp_coef;
                interpolate_y += src_points_value_map[neighbor_pt].second * exp_coef;
            }
        }else{
            for(size_t i = 0; i < coords.size(); i++){
                auto neighbor_pt = coords[i].first;
                interpolate_x += src_points_value_map[neighbor_pt].first  * coef_lst[i];
                interpolate_y += src_points_value_map[neighbor_pt].second * coef_lst[i];
            }
        }
    }else{
        assert(false && "ERROR: point is not in convex hull");
        // 插值点不在凸包内部，TODO 如何处理？
        // interpolate_x = query.exact().x().convert_to<double>();
        // interpolate_y = query.exact().y().convert_to<double>();
    }
    return std::make_pair(interpolate_x, interpolate_y);
}



/// @brief for循环做自然邻域插值(deprecated)
/// @param DT_nn 
/// @param grid_pt 
/// @param src_points_value_map 
/// @return 
auto grid_nn_interpolate(const DelaunayTri& DT_nn, const std::vector<Point>& grid_pt, std::map<Point, std::pair<double, double>>& src_points_value_map){
    int N = grid_pt.size();
    Eigen::MatrixXd nn_result(N, 4);
    for(size_t i = 0; i < grid_pt.size(); i++){
        auto query_pt = grid_pt[i];
        auto [interpolate_x, interpolate_y] = natural_neighbor_interpolate(std::ref(DT_nn), query_pt, std::ref(src_points_value_map));
        nn_result(i,0) = query_pt.x().exact().convert_to<double>();
        nn_result(i,1) = query_pt.y().exact().convert_to<double>();
        nn_result(i,2) = interpolate_x;
        nn_result(i,3) = interpolate_y;
    }
    return nn_result;
}



/// @brief 对待插值点的子列表用for循环做自然邻域插值(deprecated)
/// @param DT_nn 
/// @param grid_pt 
/// @param src_points_value_map 
/// @return 
auto grid_nn_interpolate_sublst(const DelaunayTri& DT_nn, 
                                const std::vector<Point>& grid_pt, 
                                std::map<Point, std::pair<double, double>>& src_points_value_map,
                                Eigen::MatrixXd& nn_result,
                                int start_idx,
                                int end_idx)
{
    auto time1 = std::chrono::high_resolution_clock::now();
    auto time2 = std::chrono::high_resolution_clock::now();
    for(size_t i = start_idx; i < end_idx; i++){
        auto query_pt = grid_pt[i];
        auto [interpolate_x, interpolate_y] = natural_neighbor_interpolate(std::ref(DT_nn), query_pt, std::ref(src_points_value_map));

        nn_result(i,0) = query_pt.x().exact().convert_to<double>();
        nn_result(i,1) = query_pt.y().exact().convert_to<double>();
        nn_result(i,2) = interpolate_x;
        nn_result(i,3) = interpolate_y;
        if((i - start_idx)%10000 == 0 && (i - start_idx > 1000)){
            time1 = time2;
            time2 = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(time2 - time1);
            std::cout<<""<< i <<","<< duration.count()/1000 <<" sec"<<std::endl;

        }
    }

}



/// @brief 多线程自然邻域插值
/// @param DT_nn 
/// @param grid_pt 
/// @param src_points_value_map 
/// @return 
auto grid_nn_interpolate_multiprocessing(const DelaunayTri& DT_nn, 
                                         const std::vector<Point>& grid_pt, 
                                         std::map<Point, std::pair<double, double>>& src_points_value_map){
    auto N = grid_pt.size();
    size_t threads_num = GRID_NN_INTERPOLATE_THREADS_NUM;//std::thread::hardware_concurrency();

    // 计算每份子列表的大小
    size_t sub_lst_size = N / threads_num;

    std::vector<size_t> start_idx_lst;
    std::vector<size_t> end_idx_lst;
    for(int i = 0; i < threads_num; i++){
        // 计算子列表的起始和结束位置
        size_t start = i * sub_lst_size;
        size_t end = ((threads_num - 1) == i) ? N : (i + 1) * sub_lst_size;
        start_idx_lst.push_back(start);
        end_idx_lst.push_back(end);
    }
    
    // 创建线程池
    std::vector<std::thread> threads;

    // 存放结果的二维数组
    Eigen::MatrixXd result(N, 4);
    result.setZero();

    for(int i = 0; i < threads_num; i++){
        // 创建子线程
        threads.emplace_back(grid_nn_interpolate_sublst, 
                             std::ref(DT_nn), 
                             std::ref(grid_pt), 
                             std::ref(src_points_value_map), 
                             std::ref(result),
                             start_idx_lst[i], 
                             end_idx_lst[i]);
    }

    // 等待所有线程完成
    for (auto& thread: threads) {
        thread.join();
    }


    return result;
}


/// @brief 生成网格点，一般是每个像素分配一个格点
/// @param H_num 纵向格点数目
/// @param W_num 横向格点数目
/// @param rangeH 高，单位是像素
/// @param rangeW 宽，单位是像素
/// @return 
auto generate_grid_points(int H_num, int W_num, int rangeH, int rangeW, std::string shape=std::string("square")){
    double grid_size_x = (1.0*rangeW)/W_num;
    double grid_size_y = (1.0*rangeH)/H_num;
    std::vector<Point> grid_points;
    if(std::string("square") == shape){
        assert(rangeH == rangeW && rangeH > 0);
        for(int i = 0; i < H_num; i++){
            for(int j = 0; j < W_num; j++){
                Point grid_pt((j+0.5)*grid_size_x, (i+0.5)*grid_size_y);
                grid_points.push_back(grid_pt);
            }
        }
        std::cout<<"Generate square grid points number: " + std::to_string(grid_points.size())<<std::endl;
    }else if(std::string("circle") == shape){
        assert(rangeH == rangeW && rangeH > 0);
        for(int i = 0; i < H_num; i++){
            for(int j = 0; j < W_num; j++){
                double x = (j+0.5)*grid_size_x;
                double y = (i+0.5)*grid_size_y;
                double R2 = (x-rangeW/2.0)*(x-rangeW/2.0) + (y-rangeH/2.0)*(y-rangeH/2.0);
                // if(R2 < rangeW*rangeW/4.0 - 0.01){
                //     Point grid_pt(x, y);
                //     grid_points.push_back(grid_pt);
                // }
                // else if(R2 <= (1.02*rangeW)*(1.02*rangeW)/4.0){
                //     // 边界往外扩一圈
                //     Point grid_pt(x, y);
                //     grid_points.push_back(grid_pt);
                // }else{

                // }
                Point grid_pt(x, y);
                grid_points.push_back(grid_pt);
                
            }
        }
        std::cout<<"Generate circle grid points number: " + std::to_string(grid_points.size())<<std::endl;
    }else{
        assert(false && "undefined shape");
    }
    
    return grid_points;
}


/// @brief 
/// @param points 
/// @param filename 
/// @param H 
/// @param W 
/// @return 
int draw_scatter_points_svg(Eigen::MatrixXd points, const std::string& filename, int H, int W){


    // 创建一个XML文档对象
    tinyxml2::XMLDocument doc;
    // 创建XML声明并添加到文档中
    tinyxml2::XMLDeclaration* declaration = doc.NewDeclaration();
    doc.InsertFirstChild(declaration);
    // 创建SVG根元素并设置必要属性
    tinyxml2::XMLElement* svgElement = doc.NewElement("svg");
    svgElement->SetAttribute("xmlns", "http://www.w3.org/2000/svg");
    svgElement->SetAttribute("width", std::to_string(W).c_str());
    svgElement->SetAttribute("height", std::to_string(H).c_str());
    doc.InsertEndChild(svgElement);

    


    for(size_t i = 0; i < points.rows(); i++){
        auto x = points(i,0);
        auto y = points(i,1);

        // 创建圆形元素表示点
        tinyxml2::XMLElement* pointElement = doc.NewElement("circle");
        pointElement->SetAttribute("cx", x);
        pointElement->SetAttribute("cy", y);
        pointElement->SetAttribute("r", 0.2);
        pointElement->SetAttribute("fill", "red");
        svgElement->InsertEndChild(pointElement);
    }

    // 保存XML文档为SVG文件
    tinyxml2::XMLError eResult = doc.SaveFile(filename.c_str());
    if (eResult != tinyxml2::XML_SUCCESS) {
        std::cerr << "Failed to save the SVG file." << std::endl;
        return 1;
    }

    std::cout << "SVG file has been created successfully." << std::endl;
    return 0;

}



/// @brief 返回每个网格点处的法向量
/// @param grid_mapping 
/// @param distance_px 目标屏幕到源图的距离（单位px）
/// @param n 材料折射率
/// @param scale_center 缩放中心，该点在缩放操作中保持不变
/// @param scale 缩放因子，scale<=0表示不做任何缩放
/// @return grid_normal_mapping
auto calculate_normal(Eigen::MatrixXd grid_mapping, double distance_px, double n, Eigen::Vector2d scale_center, double scale=-1){

    assert(distance_px > 0 && "distance_px以像素为单位, 表示源面到像面的距离, 必须>0");
    int N = grid_mapping.rows();
    Eigen::MatrixXd grid_normal_mapping(N, 5);
    
    Eigen::Vector3d incident_dir(0, 0, -1.0);
    for(size_t i = 0; i < N; i++){
        auto xs = grid_mapping(i,0);
        auto ys = grid_mapping(i,1);
        auto xt = grid_mapping(i,2);
        auto yt = grid_mapping(i,3);
        if(scale > 0){
            // 执行缩放
            xt = scale*(xt - scale_center(0)) + scale_center(0);
            yt = scale*(yt - scale_center(1)) + scale_center(1);
        }

        Eigen::Vector3d refract_dir(xt-xs, yt-ys, -distance_px);
        refract_dir = refract_dir/refract_dir.norm();

        
        Eigen::Vector3d normal = n*incident_dir - refract_dir;
        normal = normal/normal.norm();
        assert(normal(2) < 0.0);

        grid_normal_mapping(i,0) = xs;
        grid_normal_mapping(i,1) = ys;
        grid_normal_mapping(i,2) = normal(0);
        grid_normal_mapping(i,3) = normal(1);
        grid_normal_mapping(i,4) = normal(2);
    }
    return grid_normal_mapping;
}
// 3-利用折射定律，算出网格格点的法向量，并人为添加法向量垂直向上的边界格点(边界格点上梯度为0)，由法向量得到格点梯度
// 4-积分得到格点处高度？
// 4-B样条曲面拟合，使得B样条曲面在格点处的法向量尽可能贴近自然邻域插值给出的数据

auto add_boundary_mapping(std::vector<std::vector<double>> src2centriod_lst_Nx4, int H, int W, std::string shape){

    if(std::string("square") == shape){
        assert(H > 0 && H == W);
        // 对于方形，边界就是4条边
        for(int i = 0; i < H+1; i++){
            std::vector<double> temp_(4);
            temp_[0] = 0;
            temp_[1] = i;
            temp_[2] = temp_[0];
            temp_[3] = temp_[1];
            src2centriod_lst_Nx4.push_back(temp_);
            
        }
        for(int i = 0; i < H+1; i++){
            std::vector<double> temp_(4);
            temp_[0] = W;
            temp_[1] = i;
            temp_[2] = temp_[0];
            temp_[3] = temp_[1];
            src2centriod_lst_Nx4.push_back(temp_);
        }
        for(int j = 0; j < W+1; j++){
            std::vector<double> temp_(4);
            temp_[0] = j;
            temp_[1] = 0;
            temp_[2] = temp_[0];
            temp_[3] = temp_[1];
            src2centriod_lst_Nx4.push_back(temp_);
        }
        for(int j = 0; j < W+1; j++){
            std::vector<double> temp_(4);
            temp_[0] = H;
            temp_[1] = 0;
            temp_[2] = temp_[0];
            temp_[3] = temp_[1];
            src2centriod_lst_Nx4.push_back(temp_);
        }
        return src2centriod_lst_Nx4;

    }else if(std::string("circle") == shape){

        assert(H > 0 && H == W);

        // 提取点
        std::vector<Point> src_sites_lst_pre;
        for(int i = 0; i < src2centriod_lst_Nx4.size(); i++){
            auto x = src2centriod_lst_Nx4[i][0], y = src2centriod_lst_Nx4[i][1];
            Point pt(x, y);
            src_sites_lst_pre.push_back(pt);
        }

        // 运用凸包算法
        std::vector<Point> convex_hull_pt_lst;
        CGAL::convex_hull_2(src_sites_lst_pre.begin(), src_sites_lst_pre.end(), std::back_inserter(convex_hull_pt_lst));

        // 
        double cx = W/2.0, cy = H/2.0, R = W/2.0;
        double my_pi = 4*std::atan(1.0);
        assert(std::abs(my_pi - 3.14159265358) < 1e-9);
        
        for(int i = 0; i < convex_hull_pt_lst.size(); i++){
            
            auto pt = convex_hull_pt_lst[i];
            auto it = std::find(src_sites_lst_pre.begin(), src_sites_lst_pre.end(), pt);
            int index = 0;
            if (it != src_sites_lst_pre.end()) {
                // 找到边界点原来的索引
                index = std::distance(src_sites_lst_pre.begin(), it);
            }else{
                assert(false && "cannot find pt in src_sites_lst_pre");
            }

            auto x = pt.exact().x().convert_to<double>();
            auto y = pt.exact().y().convert_to<double>();
            auto dx = x-cx, dy = y-cy;
            auto theta = std::atan2(dy,dx); // -pi~pi
            // std::cout<< convex_hull_pt_lst[i] << ","<<theta <<std::endl;
            double new_x,new_y;
            if(theta >= -my_pi/4 && theta < my_pi/4){
                new_x = W;
                new_y = cy + R*std::tan(theta);
            }else if(theta >= my_pi/4 && theta < 3*my_pi/4){
                new_x = cx - R*std::tan(theta - my_pi/2);
                new_y = H;
            }else if(theta >= -3*my_pi/4 && theta < -my_pi/4){
                new_x = cx + R*std::tan(theta + my_pi/2);
                new_y = 0;
            }else{
                new_x = 0;
                new_y = cy - R*std::tan(theta-my_pi);
            }
            std::vector<double> temp_(4);
            temp_[0] = new_x;
            temp_[1] = new_y;
            temp_[2] = src2centriod_lst_Nx4[index][2];
            temp_[3] = src2centriod_lst_Nx4[index][3];
            src2centriod_lst_Nx4.push_back(temp_);
        }
        std::vector<double> temp_LU(4);
        temp_LU[0] = 0;
        temp_LU[1] = 0;
        temp_LU[2] = temp_LU[0];
        temp_LU[3] = temp_LU[1];
        std::vector<double> temp_LD(4);
        temp_LD[0] = 0;
        temp_LD[1] = H;
        temp_LD[2] = temp_LD[0];
        temp_LD[3] = temp_LD[1];
        std::vector<double> temp_RU(4);
        temp_RU[0] = W;
        temp_RU[1] = 0;
        temp_RU[2] = temp_RU[0];
        temp_RU[3] = temp_RU[1];
        std::vector<double> temp_RD(4);
        temp_RD[0] = W;
        temp_RD[1] = H;
        temp_RD[2] = temp_RD[0];
        temp_RD[3] = temp_RD[1];
        src2centriod_lst_Nx4.push_back(temp_LU);
        src2centriod_lst_Nx4.push_back(temp_LD);
        src2centriod_lst_Nx4.push_back(temp_RU);
        src2centriod_lst_Nx4.push_back(temp_RD);


        
        
        // 对于圆形，边界就是圆的边
        
        // int extra_num = 4*H;

        
        // double d_theta = 2*my_pi/extra_num;
        
        // for(int i = 0; i < extra_num; i++){
        //     double x = cx + R*std::cos(i*d_theta);
        //     double y = cy + R*std::sin(i*d_theta);

        //     std::vector<double> temp_(4);
        //     temp_[0] = x;
        //     temp_[1] = y;
        //     temp_[2] = temp_[0];
        //     temp_[3] = temp_[1];
        //     src2centriod_lst_Nx4.push_back(temp_);
        // }

        return src2centriod_lst_Nx4;
    }else{
        assert(false && "undefined shape");
    }

    return src2centriod_lst_Nx4;

}



/// @brief 
/// @param grid_mapping 
/// @param target_file_path 
/// @param pic_shape 
/// @return 
Eigen::MatrixXd grid_mapping_filter(Eigen::MatrixXd grid_mapping, std::string target_file_path, std::string pic_shape){

    auto target_pic_mat = load_pic_as_gray(target_file_path);
    int H = target_pic_mat.rows();
    int W = target_pic_mat.cols();

    int N = grid_mapping.rows();
    Eigen::MatrixXd filtered_grid_mapping(N, 4);
    int cnt = 0;
    for(int i = 0; i < N; i++){
        auto xs = grid_mapping(i,0);
        auto ys = grid_mapping(i,1);
        auto xt = grid_mapping(i,2);
        auto yt = grid_mapping(i,3);
        int c = static_cast<int>(std::floor(xt));
        int r = static_cast<int>(std::floor(yt));
        if(pic_shape == std::string("circle")){
            if( (xs-W/2.0)*(xs-W/2.0) + (ys-H/2.0)*(ys-H/2.0) <= H*H/4.0 && target_pic_mat(r,c) <= 120 ){
                continue;
            }else{
                filtered_grid_mapping(cnt, 0) = xs;
                filtered_grid_mapping(cnt, 1) = ys;
                filtered_grid_mapping(cnt, 2) = xt;
                filtered_grid_mapping(cnt, 3) = yt;
                cnt += 1;
            }
        }else if(pic_shape == std::string("square")){
            if( target_pic_mat(r,c) <= 120 ){
                continue;
            }else{
                filtered_grid_mapping(cnt, 0) = xs;
                filtered_grid_mapping(cnt, 1) = ys;
                filtered_grid_mapping(cnt, 2) = xt;
                filtered_grid_mapping(cnt, 3) = yt;
                cnt += 1;
            }
        }else{
            assert(false && "undefined shape");
        }
    }
    assert(cnt >= 1 && cnt <= N);
    Eigen::MatrixXd result = filtered_grid_mapping.block(0, 0, cnt, 4);

    return result;
}








int main(){

    if(1){
        // Point p1(0,1);
        // Point p2(1,0);
        // Segment s1(p1,p2);

        // std::cout << "Hello, World!" << std::endl;
        // std::string s = "Boris Schaling"; 
        // std::cout << boost::algorithm::to_upper_copy(s) << std::endl; 
        // std::cout << area(2.0) << std::endl; 
        // std::cout << p1 << std::endl;

        // auto pic_mat = load_pic_as_gray("fu.jpg");
        // auto pic_mat = load_pic_as_gray("WhiteSource.jpg");
        // std::cout << "size:"<<pic_mat.size() << std::endl;
        // std::cout << "rows:"<<pic_mat.rows() << std::endl;
        // std::cout << "cols:"<<pic_mat.rows() << std::endl;
        
        // auto [x_quantizer,y_quantizer] = pic_mat_cumsum(pic_mat);
        
        // auto sites_lst = generate_sites_lst(32000, x_quantizer, y_quantizer);
        
        // auto start = std::chrono::high_resolution_clock::now();
        
        // auto vc_lst = GetVoronoiDiagram(sites_lst, 32000, 0, 0, x_quantizer.rows(), x_quantizer.cols());

        // auto end = std::chrono::high_resolution_clock::now();
        // auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        // std::cout << "Time taken by function: " << duration.count() << " milliseconds" << std::endl;


        std::string src_pic_name = "WhiteSource_Circle.jpg";
        std::string src_pic_path = "D:/myproject/JSTJ/WhiteSource_Circle_128000/";

        // std::string tgt_pic_path = "D:/myproject/JSTJ/Target_GhostInTheShell_128000/";
        // std::string tgt_pic_name = "GhostInTheShell_inv.png";
        std::string tgt_pic_path = "D:/myproject/JSTJ/Target_USTC_128000/";
        std::string tgt_pic_name = "USTC_inv.jpg";
        
        auto pic_mat = load_pic_as_gray(src_pic_path+src_pic_name);
        std::string pic_shape = "circle"; // square circle
        bool use_128000 = true;

        if(0){
            // Stage-1 在相同光源下，只需要执行一次。根据光源分布进行lloyd采样获得各个level的sites_lst(和各cell的归一化面积)以及level之间的邻接关系。
            // 计算结果存储在source_sites.txt和nearest_index.txt中
            auto start = std::chrono::high_resolution_clock::now();

            
            std::string src_txt_name = "source_sites.txt";
            std::string src_svg_name = "voronoi_polygon";

            std::string txt_path = src_pic_path + src_txt_name;       // 源图voronoi节点TXT保存路径
            std::string svg_path = src_pic_path + src_svg_name;       // 源图各level的svg矢量图保存路径

            
            auto [x_quantizer,y_quantizer] = pic_mat_cumsum(pic_mat);

            auto result_sites_lst1 = lloyd_iteration(pic_mat, 500,    x_quantizer, y_quantizer, txt_path, svg_path+std::to_string(500)   +".svg");
            auto result_sites_lst2 = lloyd_iteration(pic_mat, 2000,   x_quantizer, y_quantizer, txt_path, svg_path+std::to_string(2000)  +".svg");
            auto result_sites_lst3 = lloyd_iteration(pic_mat, 8000,   x_quantizer, y_quantizer, txt_path, svg_path+std::to_string(8000)  +".svg");
            auto result_sites_lst4 = lloyd_iteration(pic_mat, 32000,  x_quantizer, y_quantizer, txt_path, svg_path+std::to_string(32000) +".svg");
            auto result_sites_lst5 = lloyd_iteration(pic_mat, 128000, x_quantizer, y_quantizer, txt_path, svg_path+std::to_string(128000)+".svg");
            find_nearest_site_pt(result_sites_lst1, result_sites_lst2, src_pic_path+std::string("nearest_index.txt"));
            find_nearest_site_pt(result_sites_lst2, result_sites_lst3, src_pic_path+std::string("nearest_index.txt"));
            find_nearest_site_pt(result_sites_lst3, result_sites_lst4, src_pic_path+std::string("nearest_index.txt"));
            find_nearest_site_pt(result_sites_lst4, result_sites_lst5, src_pic_path+std::string("nearest_index.txt"));

            

            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            std::cout << "Time taken by step-1: " << duration.count() << " milliseconds" << std::endl;
        }
        // return 0;
        if(0){
            // Stage-2
            std::string sites_file = src_pic_path+"source_sites.txt";
            std::vector<int> N_lst;
            if(use_128000){
                N_lst = {500,2000,8000,32000,128000}; // 复杂图案
            }else{
                N_lst = {500,2000,8000,32000}; // 普通的字符32000就够了
            }
            auto [all_sites_lst, all_lambda_lst] = read_sites_and_lambda(sites_file, N_lst);

            std::string near_info_file = src_pic_path+"nearest_index.txt";
            std::string target_filepath = tgt_pic_path+tgt_pic_name;//"IQ_inv800800.png";
            std::string name_tag = "test";
            mprint("++++++++++++++++++++");
            OTM_LBFGS_optimization(all_sites_lst, all_lambda_lst, N_lst, near_info_file, target_filepath, name_tag, tgt_pic_path);
            mprint("++++++++++++++++++++");
        }
        // return 0;
        if(1){
            // stage3-step1
            
            std::string src2centriod_file = tgt_pic_path+"stage2_SrcSites2Centriod.txt";
            auto src2centriod_lst_Nx4 = read_src2centriod_file(src2centriod_file);
            // 添加边界点，确保所有网格在数据点的凸包内
            int H = pic_mat.rows(), W = pic_mat.cols();
            mprint("Before add boundary mapping: "+std::to_string(src2centriod_lst_Nx4.size()));
            src2centriod_lst_Nx4 = add_boundary_mapping(src2centriod_lst_Nx4, H, W, pic_shape);
            mprint("After add boundary mapping: "+std::to_string(src2centriod_lst_Nx4.size()));

            // stage3-step2 得到德劳内三角剖分
            std::vector<Point> src_points_lst;
            std::map<Point, std::pair<double, double>> src_points_value_map;
            for(size_t i = 0; i < src2centriod_lst_Nx4.size(); i++){

                auto x = src2centriod_lst_Nx4[i][0];
                auto y = src2centriod_lst_Nx4[i][1];
                Point src_point(x,y);

                src_points_lst.push_back(src_point);

                auto target_x = src2centriod_lst_Nx4[i][2];
                auto target_y = src2centriod_lst_Nx4[i][3];
                auto target_val = std::make_pair(target_x, target_y);
                
                src_points_value_map[src_point] = target_val;
            }

            DelaunayTri DT_nn;    // 利用已有数据点创建 Delaunay 三角剖分，为后续自然邻域插值做准备
            DT_nn.insert(src_points_lst.begin(), src_points_lst.end());
            mprint("Delaunay triangulation success.");
            
            // stage3-step3 训练集: NN插值并计算法向量(作为后续优化B样条曲面的训练集)
            auto grid_pt = generate_grid_points(1200, 1200, H, W, pic_shape);
            auto grid_mapping = grid_nn_interpolate_multiprocessing(std::ref(DT_nn), std::ref(grid_pt), std::ref(src_points_value_map));
            
            // 过滤
            // std::string target_filepath = tgt_pic_path+tgt_pic_name;
            // auto grid_mapping_f = grid_mapping_filter(grid_mapping, target_filepath, pic_shape);
            auto grid_mapping_f = grid_mapping;
            
            // 画SVG图
            draw_scatter_points_svg(grid_mapping_f.block(0, 2, grid_mapping_f.rows(), 2), tgt_pic_path+"stage3_Grid_SrcSites2Centriod.svg", H, W);
            
            // 计算格点处法向量(折射率，成像距离，缩放比例，斜入射 TODO)
            Eigen::Vector2d scale_center;
            scale_center(0) = W/2;
            scale_center(1) = H/2;
            auto grid_normal_mapping = calculate_normal(grid_mapping_f, 200*(800/75), 1.496, scale_center, -1);
            
            // 映射数据和法向量数据都保存为TXT
            save_2d_list(std::ref(grid_mapping_f), tgt_pic_path+"stage3_Grid_SrcSites2Centriod.txt");
            save_2d_list(std::ref(grid_normal_mapping), tgt_pic_path+"stage3_Grid_Normal_mapping.txt");
            
        }
        return 0;
        // ============================================================================================



        

        std::default_random_engine engine(std::random_device{}());
        std::uniform_real_distribution<double> distribution(0.0, 1.0);
        // 生成 0 到 1 之间的随机数
        int cnt = 3;
        while((cnt--) > 0){
            double random_num = distribution(engine);
            std::cout << random_num << std::endl;
        }
        Eigen::Matrix<int,4,4> warped_eigen;
        //赋值
        warped_eigen<<1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16;

        Eigen::VectorXi histogram1 = warped_eigen.colwise().sum();
        Eigen::VectorXi histogram2 = warped_eigen.rowwise().sum();
        std::cout<<"histogram1"<<std::endl<<histogram1<<std::endl;   
        std::cout<<"histogram2"<<std::endl<<histogram2<<std::endl; 
    }
    


    
    return 0;
}

