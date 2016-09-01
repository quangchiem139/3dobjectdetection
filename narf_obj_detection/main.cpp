#include <cstdlib>
#include <iostream>
#include <string>
#include <sstream> 
#include <boost/thread/thread.hpp>
#include <boost/make_shared.hpp>
#include <pcl/ros/conversions.h>

#include <pcl/common/angles.h>
#include <pcl/common/transforms.h>
#include <pcl/common/common.h>
#include <pcl/correspondence.h>
#include <pcl/range_image/range_image.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/range_image_visualizer.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/keypoints/narf_keypoint.h>
#include <pcl/console/parse.h>
#include <pcl/console/print.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/kdtree/impl/kdtree_flann.hpp>
#include <pcl/point_representation.h>
#include "pcl/impl/point_types.hpp"

#include <pcl/features/range_image_border_extractor.h>
#include <pcl/features/narf_descriptor.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/board.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/correspondence_rejection_sample_consensus.h>
#include <pcl/registration/correspondence_rejection_median_distance.h>
#include <pcl/registration/correspondence_rejection_surface_normal.h>
#include <pcl/registration/transformation_estimation_svd.h>
#include <pcl/registration/transformation_validation_euclidean.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/registration/correspondence_estimation.h>
//#include <pcl/recognition/hv/hv_go.h>

typedef pcl::PointXYZ PointType;
typedef pcl::Normal NormalType;
typedef pcl::ReferenceFrame RFType;

struct CloudStyle
{
	double r;
	double g;
	double b;
	double size;

	CloudStyle(double r,
		double g,
		double b,
		double size) :
		r(r),
		g(g),
		b(b),
		size(size)
	{
	}
};
CloudStyle style_white(255.0, 255.0, 255.0, 4.0);
CloudStyle style_red(255.0, 0.0, 0.0, 3.0);
CloudStyle style_lime(0.0, 255.0, 0.0, 5.0);
CloudStyle style_cyan(93.0, 200.0, 217.0, 4.0);
CloudStyle style_violet(255.0, 0.0, 255.0, 8.0);
CloudStyle style_blue(0.0, 0.0, 255.0, 8.0);
CloudStyle style_yellow(255.0, 255.0, 0.0, 8.0);
CloudStyle style_magenta(255.0, 0.0, 255.0, 8.0);
CloudStyle style_green(0.0, 128.0, 0.0, 5.0);
CloudStyle style_purple(128.0, 0.0, 128.0,5.0);

// Define new PointRepresentation for NARF36
class NARFPointRepresenation : public pcl::PointRepresentation<pcl::Narf36>
{
public:
	NARFPointRepresenation()
	{
		this->nr_dimensions_ = 36;
	}

	void copyToFloatArray(const pcl::Narf36 &p, float *out) const
	{
		for (int i = 0; i < 36; ++i)
			out[i] = p.descriptor[i];
	}
};

//pcl::RangeImage::CoordinateFrame coordinate_frame = pcl::RangeImage::LASER_FRAME;
pcl::RangeImage::CoordinateFrame coordinate_frame = pcl::RangeImage::CAMERA_FRAME;
float model_angularResolution_ = 0.1;
float scene_angularResolution_ = 0.2;
float maxAngleWidth = pcl::deg2rad(360.0f);
float maxAngleHeight = pcl::deg2rad(180.0f);
bool live_update = true;
bool show_keypoints_ = false;
bool show_cluster_ = false;
bool debug_ = false;
float support_size_ = 0.05f; // keypoint for narf, decrease this value to increase number of keypoints.
bool rotation_invariant_ = true;
float cg_size_(0.25f);
float cg_thresh_(3.0f); // at least 3 samples are needed to compute the 6 DoF pose
float cluster_tolerance_(0.20f); // threshold to get cluster point cloud
bool use_hough_ = false;
float sac_inliers_(0.015f);
float max_corr_dist_(0.05f);
float min_corr_dist_(0.04f);
float icp_corr_distance_(0.015f);
float median_factor_(8);
float rejection_angle_ (60);
float validation_score_max_ (0.4);
float validation_score_min_ (0.1);
float feature_distance_model_scene_ (0.02);

float eu_epsilon_(1);
int tmp = 78;

void
readConfigureFile(const char* filename)
{
	std::ifstream fin(filename);
	std::string line;
	std::istringstream sin;
	std::string::size_type sz;     // alias of size_t
	while (std::getline(fin, line)) 
	{
		sin.str(line.substr(line.find("=")+1));
	//	float sin_number = std::stof(sin.str());
 		if (line.find("support_size") != std::string::npos) 
		{
  		std::cout<<"support_size = "<<sin.str()<<std::endl;
  		sin >> support_size_;
 		}
		else if (line.find("max_corr_dist") != std::string::npos) {
			std::cout<<"max_corr_dist_ = "<<sin.str()<<std::endl;
  		sin >> max_corr_dist_;
		}
		else if (line.find("min_corr_dist") != std::string::npos) {
			std::cout<<"min_corr_dist_ = "<<sin.str()<<std::endl;
  		sin >> min_corr_dist_;
		}
		else if (line.find("icp_corr_distance") != std::string::npos) {
			std::cout<<"icp_corr_distance_ = "<<sin.str()<<std::endl;
  		sin >> icp_corr_distance_;
		}	
		else if (line.find("model_angularResolution") != std::string::npos) {
			std::cout<<"model_angularResolution = "<<sin.str()<<std::endl;
  		sin >> model_angularResolution_;
		}	
		else if (line.find("scene_angularResolution") != std::string::npos) {
			std::cout<<"scene_angularResolution = "<<sin.str()<<std::endl;
  		sin >> scene_angularResolution_;
		}	
		else if (line.find("sac_inliers") != std::string::npos) {
			std::cout<<"sac_inliers = "<<sin.str()<<std::endl;
  		sin >> sac_inliers_;
		}	
		else if (line.find("median_factor") != std::string::npos) {
			std::cout<<"median_factor = "<<sin.str()<<std::endl;
  		sin >> median_factor_;
		}	
		else if (line.find("rejection_angle") != std::string::npos) {
			std::cout<<"rejection_angle = "<<sin.str()<<std::endl;
  		sin >> rejection_angle_;
		}	
    else if (line.find("validation_score_max") != std::string::npos) {                                                                                                                                  
			std::cout<<"validation_score_max = "<<sin.str()<<std::endl;
      sin >> validation_score_max_;
    }   
    else if (line.find("validation_score_min") != std::string::npos) {                                                                                                                                   
			std::cout<<"validation_score_min = "<<sin.str()<<std::endl;
      sin >> validation_score_min_;
    }
    else if (line.find("feature_distance_model_scene") != std::string::npos) { 
      std::cout<<"feature_distance_model_scene = "<<sin.str()<<std::endl;
      sin >> feature_distance_model_scene_;                                                                          
    }
		sin.clear();
	}
}

void 
pp_callback(const pcl::visualization::PointPickingEvent& event, void* cookie)
{
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer = *static_cast<boost::shared_ptr<pcl::visualization::PCLVisualizer> *> (cookie);
	int idx = event.getPointIndex();
	if (idx == -1)
		return;

	pcl::PointXYZ picked_pt;
	event.getPoint(picked_pt.x, picked_pt.y, picked_pt.z);
	std::stringstream ss;
	ss << picked_pt.x << " " << picked_pt.y << " " << picked_pt.z << endl;
	std::cout << ss.str() << std::endl;
	viewer->addText(ss.str(), 1, 1, "Mouse Position", 0);
}

double diff(Eigen::Matrix4f &in1, Eigen::Matrix4f &in2)
{
	return sqrt((in1(0, 3) - in2(0, 3))*(in1(0, 3) - in2(0, 3)) + (in1(1, 3) - in2(1, 3))*(in1(1, 3) - in2(1, 3)) + (in1(2, 3) - in2(2, 3))*(in1(2, 3) - in2(2, 3)));
}

void 
setViewerPose(pcl::visualization::PCLVisualizer& viewer, const Eigen::Affine3f& viewer_pose)
{
  Eigen::Vector3f pos_vector = viewer_pose * Eigen::Vector3f (0, 0, 0);
  Eigen::Vector3f look_at_vector = viewer_pose.rotation () * Eigen::Vector3f (0, 0, 1) + pos_vector;
  Eigen::Vector3f up_vector = viewer_pose.rotation () * Eigen::Vector3f (0, -1, 0);
  viewer.setCameraPosition(pos_vector[0], pos_vector[1], pos_vector[2],
                            look_at_vector[0], look_at_vector[1], look_at_vector[2],
                            up_vector[0], up_vector[1], up_vector[2]);
}

inline void
SetViewPoint(pcl::PointCloud<PointType>::Ptr cloud)
{

	cloud->sensor_origin_.setZero();
	cloud->sensor_orientation_.w() = 0.0;
	cloud->sensor_orientation_.x() = 1.0;
	cloud->sensor_orientation_.y() = 0.0;
	cloud->sensor_orientation_.z() = 0.0;
}

int main(int argc, char** argv) 
{

	//
	// 0. Loading file and configuration
	//
	pcl::console::print_highlight("Loading point clouds and configurations...\n");
	readConfigureFile("../cfg/robot.txt");

	pcl::console::parse_argument(argc, argv, "--cg_size", cg_size_);
	pcl::console::parse_argument(argc, argv, "--cg_thresh", cg_thresh_);
	pcl::console::parse_argument(argc, argv, "--cl_thresh", cluster_tolerance_);
	pcl::console::parse_argument(argc, argv, "--eu_ep", eu_epsilon_);
	pcl::console::parse_argument(argc, argv, "--sac_inliers", sac_inliers_);
	pcl::console::parse_argument(argc, argv, "--tmp", tmp);

	if (pcl::console::find_switch(argc, argv, "-show_kp")) show_keypoints_ = true;
	if (pcl::console::find_switch(argc, argv, "-show_cl")) show_cluster_ = true;
	if (pcl::console::find_switch(argc, argv, "-debug")) debug_ = true;

	std::string used_algorithm;
	if (pcl::console::parse_argument(argc, argv, "--algorithm", used_algorithm) != -1)
	{
		if (used_algorithm.compare("Hough") == 0)
		{
			use_hough_ = true;
		}
		else if (used_algorithm.compare("GC") == 0)
		{
			use_hough_ = false;
		}

		else
		{
			std::cout << "Wrong algorithm name.\n";
			/*showHelp(argv[0]);*/
			exit(-1);
		}
	}

	pcl::PointCloud<PointType>::Ptr model_ptr(new pcl::PointCloud<PointType>);
	pcl::PointCloud<PointType>& model = *model_ptr;
	pcl::PCLPointCloud2 model_cloud2;

	pcl::PointCloud<PointType>::Ptr scene_ptr(new pcl::PointCloud<PointType>);
	pcl::PointCloud<PointType>& scene = *scene_ptr;
	pcl::PCLPointCloud2 scene_cloud2;

	pcl::toPCLPointCloud2(model,model_cloud2);
	pcl::toPCLPointCloud2(scene,scene_cloud2);
	Eigen::Affine3f model_sensor_pose(Eigen::Affine3f::Identity());
	Eigen::Affine3f scene_sensor_pose(Eigen::Affine3f::Identity());
	Eigen::Matrix4f matrix = Eigen::Matrix4f::Identity();

	std::vector<int> pcd_filename_indices = pcl::console::parse_file_extension_argument(argc, argv, "pcd");
	if (!pcd_filename_indices.empty()) {
		//std::string filename = argv[pcd_filename_indices[0]];
		pcl::io::loadPCDFile(argv[pcd_filename_indices[0]], model);
		pcl::io::loadPCDFile(argv[pcd_filename_indices[1]], scene);
		std::cerr << "MODEL (" <<argv[pcd_filename_indices[0]] << ") has " << model.size() << " points " << std::endl;
		std::cerr << "SCENE (" << argv[pcd_filename_indices[1]] << ") has " << scene.size() << " points " << std::endl;

		matrix(0, 0) = 0.874829; 
		matrix(0, 1) = 0;
		matrix(0, 2) = 0.484433;
		matrix(0, 3) = -0.0925789;
		matrix(1, 0) = 0;
		matrix(1, 1) = 1;
		matrix(1, 2) = 0;
		matrix(1, 3) = 1.22088;
		matrix(2, 0) = -0.484433;
		matrix(2, 1) = 0;
		matrix(2, 2) = 0.874829;
		matrix(2, 3) = 0.596536;
		matrix(3, 0) = 0;
		matrix(3, 1) = 0;
		matrix(3, 2) = 0;
		matrix(3, 3) = 1;

		if (pcl::console::find_switch(argc, argv, "-man_pose")) 
		{ 
			//// Set manual view for modell and scene sensor pose
			//scene_sensor_pose = matrix; 
			//model_sensor_pose = matrix; 
		}
		else
		{
			//scene_sensor_pose = Eigen::Affine3f(Eigen::Translation3f(scene.sensor_origin_[0],
			//	scene.sensor_origin_[1],
			//	scene.sensor_origin_[2])) *
			//	Eigen::Affine3f(scene.sensor_orientation_);
		  model_sensor_pose = Eigen::Affine3f(Eigen::Translation3f(model.sensor_origin_[0], model.sensor_origin_[1], model.sensor_origin_[2])) * Eigen::Affine3f(model.sensor_orientation_);
		  scene_sensor_pose = Eigen::Affine3f(Eigen::Translation3f(scene.sensor_origin_[0], scene.sensor_origin_[1], scene.sensor_origin_[2])) * Eigen::Affine3f(scene.sensor_orientation_);
    	//pcl::PointCloud<pcl::PointWithViewpoint> tmp_model;
    	//pcl::PointCloud<pcl::PointWithViewpoint> tmp_scene;
    	//pcl::fromPCLPointCloud2(model_cloud2, tmp_model);
    	//pcl::fromPCLPointCloud2(scene_cloud2, tmp_model);
			//pcl::fromROSMsg(model_cloud2, tmp_model);
			//pcl::fromROSMsg(scene_cloud2, tmp_scene);
			//Eigen::Vector3f model_average_viewpoint = pcl::RangeImage::getAverageViewPoint(tmp_model);
    	//model_sensor_pose = Eigen::Translation3f(model_average_viewpoint) * model_sensor_pose;
			//Eigen::Vector3f scene_average_viewpoint = pcl::RangeImage::getAverageViewPoint(tmp_scene);
      //scene_sensor_pose = Eigen::Translation3f(scene_average_viewpoint) * scene_sensor_pose; 
		}
	}

	//
	// 1. Creating Range image of given point cloud
	//

	float noiseLevel = 0.0f;
	float minimumRange = 0.0f;
	int borderSize = 1;
	int imageSizeX = 640, imageSizeY = 480;
	float centerX = (640.0f / 2.0f), centerY = (480.0f / 2.0f);
	float focalLengthX = 525.0f, focalLengthY = focalLengthX;

	//boost::shared_ptr<pcl::RangeImage> rangeImage_ptr(new pcl::RangeImage);
	//pcl::RangeImage& rangeImage = *rangeImage_ptr;
	boost::shared_ptr<pcl::RangeImage> model_rangeImage_ptr(new pcl::RangeImage);
	pcl::RangeImage& model_rangeImage = *model_rangeImage_ptr;
  pcl::PointCloud<pcl::PointWithViewpoint> model_far_ranges;
  pcl::RangeImage::extractFarRanges(model_cloud2,model_far_ranges); 

	boost::shared_ptr<pcl::RangeImage> scene_rangeImage_ptr(new pcl::RangeImage);
	pcl::RangeImage& scene_rangeImage = *scene_rangeImage_ptr;
  pcl::PointCloud<pcl::PointWithViewpoint> scene_far_ranges;
  pcl::RangeImage::extractFarRanges(scene_cloud2,scene_far_ranges); 

	model_angularResolution_ = pcl::deg2rad(model_angularResolution_);
	scene_angularResolution_ = pcl::deg2rad(scene_angularResolution_);

	model_rangeImage.createFromPointCloud(model, model_angularResolution_, maxAngleWidth, maxAngleHeight, model_sensor_pose, coordinate_frame, noiseLevel, minimumRange, borderSize);
	model_rangeImage.integrateFarRanges(model_far_ranges); 
	model_rangeImage.setUnseenToMaxRange();
	scene_rangeImage.createFromPointCloud(scene, scene_angularResolution_, maxAngleWidth, maxAngleHeight, scene_sensor_pose, coordinate_frame, noiseLevel, minimumRange, borderSize);
	scene_rangeImage.integrateFarRanges(scene_far_ranges); 
	scene_rangeImage.setUnseenToMaxRange();

	// Show range Image
	//pcl::visualization::RangeImageVisualizer rangeImage_widget ("Range image");
	//rangeImage_widget.showRangeImage(model_rangeImage,-100,1000);
	//rangeImage_widget.showRangeImage(scene_rangeImage, -100, 1000);

	// 
	// 2. Extract NARF keypoints 
	// 
	pcl::console::print_highlight("Extract NARF keypoints...\n");

	pcl::RangeImageBorderExtractor model_rangeImage_border_extractor;
	pcl::NarfKeypoint narf_keypoint_detector;
	narf_keypoint_detector.setRangeImageBorderExtractor(&model_rangeImage_border_extractor);
	narf_keypoint_detector.setRangeImage(&model_rangeImage);
	narf_keypoint_detector.getParameters().support_size = support_size_;

	pcl::PointCloud<int> model_keypoint_indices;
	narf_keypoint_detector.compute(model_keypoint_indices);
	std::cout << "Found " << model_keypoint_indices.points.size() << " key points in model.\n";

	pcl::RangeImageBorderExtractor scene_rangeImage_border_extractor;
	narf_keypoint_detector.setRangeImageBorderExtractor(&scene_rangeImage_border_extractor);
	narf_keypoint_detector.setRangeImage(&scene_rangeImage);
	narf_keypoint_detector.getParameters().support_size = support_size_;

	pcl::PointCloud<int> scene_keypoint_indices;
	narf_keypoint_detector.compute(scene_keypoint_indices);
	std::cout << "Found " << scene_keypoint_indices.points.size() << " key points in scene.\n";

	// 
	// 3. Extract NARF descriptors for interest points
	// 
	pcl::console::print_highlight("Computing NARF descriptors...\n");

	std::vector<int> model_keypoint_indices2;
	model_keypoint_indices2.resize(model_keypoint_indices.points.size());
	for (unsigned int i = 0; i < model_keypoint_indices.size(); ++i) // This step is necessary to get the right vector type
		model_keypoint_indices2[i] = model_keypoint_indices.points[i];
	pcl::NarfDescriptor narf_descriptor(&model_rangeImage, &model_keypoint_indices2);
	narf_descriptor.getParameters().support_size = support_size_;
	narf_descriptor.getParameters().rotation_invariant = rotation_invariant_;
	pcl::PointCloud<pcl::Narf36>::Ptr model_narf_descriptors_ptr(new pcl::PointCloud<pcl::Narf36>);
	pcl::PointCloud<pcl::Narf36>& model_narf_descriptors = *model_narf_descriptors_ptr;
	narf_descriptor.compute(model_narf_descriptors);

	//for(int i = 0; i < model_narf_descriptors.size() , i++) 
	//for(int i = 0; i < 1 ; i++) 
	//{
	//	for (int k = 0 ; k < 36 ; k++)
	//	{
	//			std::cout << "model descritor at point" << i << "_"<< k << ": "<< model_narf_descriptors_ptr->at(i).descriptor[k] <<std::endl;
	//	} 
	//}

	cout << "Extracted " << model_narf_descriptors.size() << " descriptors for "
		<< model_keypoint_indices.points.size() << " keypoints.\n";
	// convert to 3D keypoint
	pcl::PointCloud<pcl::PointXYZ>::Ptr model_keypoints_ptr(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>& model_keypoints = *model_keypoints_ptr;
	model_keypoints.points.resize(model_keypoint_indices.points.size());
	for (size_t i = 0; i < model_keypoint_indices.points.size(); ++i)
		model_keypoints.points[i].getVector3fMap() = model_rangeImage.points[model_keypoint_indices.points[i]].getVector3fMap();

	std::vector<int> scene_keypoint_indices2;
	scene_keypoint_indices2.resize(scene_keypoint_indices.points.size());
	for (unsigned int i = 0; i < scene_keypoint_indices.size(); ++i) // This step is necessary to get the right vector type
		scene_keypoint_indices2[i] = scene_keypoint_indices.points[i];
	narf_descriptor.setRangeImage(&scene_rangeImage, &scene_keypoint_indices2);
	narf_descriptor.getParameters().support_size = support_size_;
	narf_descriptor.getParameters().rotation_invariant = rotation_invariant_;
	pcl::PointCloud<pcl::Narf36>::Ptr scene_narf_descriptors_ptr(new pcl::PointCloud<pcl::Narf36>);
	pcl::PointCloud<pcl::Narf36>& scene_narf_descriptors = *scene_narf_descriptors_ptr;
	narf_descriptor.compute(scene_narf_descriptors);
	cout << "Extracted " << scene_narf_descriptors.size() << " descriptors for "
		<< scene_keypoint_indices.points.size() << " keypoints.\n";
	pcl::PointCloud<pcl::PointXYZ>::Ptr scene_keypoints_ptr(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>& scene_keypoints = *scene_keypoints_ptr;
	scene_keypoints.points.resize(scene_keypoint_indices.points.size());
	for (size_t i = 0; i < scene_keypoint_indices.points.size(); ++i)
		scene_keypoints.points[i].getVector3fMap() = scene_rangeImage.points[scene_keypoint_indices.points[i]].getVector3fMap();

	//
	//  4. Find Model-Scene Correspondences with KdTree 
	// 

	pcl::console::print_highlight("Finding Correspondeces...\n");
	pcl::KdTreeFLANN<pcl::Narf36> match_search;
	pcl::CorrespondencesPtr model_scene_corrs(new pcl::Correspondences());
	std::cout << "calculated " << model_keypoints_ptr->size() << " keypoints for the model and " << scene_keypoints_ptr->size() << " keypoints for the scene" << std::endl;
	//std::cout <<"calculated " << model_descriptors_->size() << " for the model and " << scene_descriptors_->size() << " for the scene" <<std::endl;

	//// Select keypoint descriptor manually for debug.
	if(debug_)
	{
		pcl::Correspondence corr1(1,45,0.108024);
		pcl::Correspondence corr2(3,89,0.106604);
		//pcl::Correspondence corr3(32,120,0.101825);
		//pcl::Correspondence corr4(33,119,0.122719);
		//pcl::Correspondence corr5(5,78,0.150574);
		model_scene_corrs->push_back(corr1);
		model_scene_corrs->push_back(corr2);
		//model_scene_corrs->push_back(corr3);
		//model_scene_corrs->push_back(corr4);
		//model_scene_corrs->push_back(corr5);
	}
		
	match_search.setInputCloud(model_narf_descriptors_ptr);
	match_search.setPointRepresentation(boost::make_shared <NARFPointRepresenation>());

	//  For each scene keypoint descriptor, find nearest neighbor into the model keypoints descriptor cloud and add it to the correspondences vector.
	int k = 1;
	//for (size_t i = 0; i < scene_narf_descriptors_ptr->size(); ++i)
	for (size_t i = 0; i < 1; ++i)
	{
		std::vector<int> neigh_indices(k);
		std::vector<float> neigh_sqr_dists(k);
		if (!pcl_isfinite(scene_narf_descriptors_ptr->at(i).descriptor[0])) //skipping NaNs
		{
			continue;
		}

		if (match_search.point_representation_->isValid(scene_narf_descriptors_ptr->at(i)))
		{
			for (size_t j = 0; j < model_narf_descriptors_ptr->size(); ++j) 
			{
					// Consider distance between 2 descriptor vector
					//float neigh_sqr_dists = pcl::L1_Norm(scene_narf_descriptors_ptr->at(i).descriptor,model_narf_descriptors_ptr->at(j).descriptor,36);
					float descriptor_norm_dists_ = pcl::L1_Norm(scene_narf_descriptors_ptr->at(tmp).descriptor,model_narf_descriptors_ptr->at(j).descriptor,36);
					descriptor_norm_dists_ = descriptor_norm_dists_ / 36;
					if( descriptor_norm_dists_ < max_corr_dist_ && descriptor_norm_dists_ > min_corr_dist_ )
					{
						// Consider Euclidean distance of 2 point (xyz value)
						float x_diff = scene_narf_descriptors_ptr->at(tmp).x - model_narf_descriptors_ptr->at(j).x;
						float y_diff = scene_narf_descriptors_ptr->at(tmp).y - model_narf_descriptors_ptr->at(j).y;
						float z_diff = scene_narf_descriptors_ptr->at(tmp).z - model_narf_descriptors_ptr->at(j).z;
						float sqrt_diff = sqrt(x_diff*x_diff + y_diff*y_diff + z_diff*z_diff);
						std::cout << "sqrt_diff: " << sqrt_diff << std::endl;

						//pcl::Correspondence corr(static_cast<int> (j), static_cast<int> (i), neigh_sqr_dists);
						pcl::Correspondence corr(static_cast<int> (j), tmp, descriptor_norm_dists_);
						model_scene_corrs->push_back(corr);
						//std::cout << "neigh_sqr_dists:: "<< neigh_sqr_dists << std::endl;
						//std::cout << "(k_indices,index,k_distances): " << "(" << j << "," << i << "," << neigh_sqr_dists << ")" << std::endl;
						std::cout << "(k_indices,index,k_distances): " << "(" << j << "," << tmp << "," << descriptor_norm_dists_ << ")" << std::endl;
					}
					
			}
		}
	}
	std::cout << "\tFound " << model_scene_corrs->size() << " correspondences " << std::endl;
	
	float max_dist = (*model_scene_corrs.get())[0].distance;
	float min_dist = (*model_scene_corrs.get())[0].distance;
  for (int i = 0; i < model_scene_corrs->size(); i++)                                                                                                                                                 
  {
    std::cout << "(k_indices,index,k_distances): " << (*model_scene_corrs.get())[i] << std::endl; // Print good correspondences to check
		if ((*model_scene_corrs.get())[i].distance > max_dist) max_dist = (*model_scene_corrs.get())[i].distance;
		if ((*model_scene_corrs.get())[i].distance < min_dist) min_dist = (*model_scene_corrs.get())[i].distance;
 
  } 
	std::cout << "Max dist: " << max_dist <<std::endl;
	std::cout << "Min dist: " << min_dist <<std::endl;

	//
	//  5. Filtering Correspondences 
	//

	//// 5.2 RANSAC Correspondences Rejection - distance rejection
	//std::vector<int> IdxInliers;
	//pcl::CorrespondencesPtr good_model_scene_corrs(new pcl::Correspondences());	
	//pcl::registration::CorrespondenceRejectorSampleConsensus<PointType> crsac;
	////crsac.setInputSource(model_keypoints_ptr);
	////crsac.setInputTarget(scene_keypoints_ptr);
  //crsac.setInputSource(model_keypoints_ptr);                                                                                                                                                               
  //crsac.setInputTarget(scene_keypoints_ptr);
	//crsac.setInlierThreshold(sac_inliers_); // why does ppl set this value to 2.5 ?
	//crsac.setMaximumIterations(100000);
	//crsac.getInliersIndices(IdxInliers);
	//crsac.setInputCorrespondences(model_scene_corrs);
	//crsac.getCorrespondences(*good_model_scene_corrs);
	//std::cout << "\tFound " << good_model_scene_corrs->size() << " GOOD correspondences " << std::endl;

	//// Selecting correspondences
  // std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > rototranslations;
  // std::vector<pcl::CorrespondencesPtr> clustered_corrs;

  //pcl::registration::CorrespondenceRejectorMedianDistance rej;
  ////pcl::CorrespondencesPtr good_model_scene_corrs(new pcl::Correspondences);
  //rej.setMedianFactor (median_factor_);
  //rej.setInputCorrespondences (good_model_scene_corrs);
  ////rej.getCorrespondences (*good_model_scene_corrs);
  //
  //pcl::CorrespondencesPtr good_model_scene_corrs_tmp (new pcl::Correspondences);
  //rej.getCorrespondences (*good_model_scene_corrs_tmp);
	//std::cout << "Median distance:" << rej.getMedianDistance() << std::endl;
	//std::cout << "[rejectBadCorrespondences] Number of correspondences remaining after rejection: " << good_model_scene_corrs_tmp->size() << std::endl;	
  //for (int i = 0; i < good_model_scene_corrs_tmp->size(); i++)                                                                                                                                                 {
  //  std::cout << "(k_indices,index,k_distances): " << (*good_model_scene_corrs_tmp.get())[i] << std::endl; // Print good correspondences to check
	//}
	//// Filter correspondences by normal	
  //pcl::PointCloud<pcl::Normal>::Ptr model_normals_ptr (new pcl::PointCloud<pcl::Normal>);
	//pcl::PointCloud<pcl::Normal>& model_normals = *model_normals_ptr;
  //pcl::PointCloud<pcl::Normal>::Ptr scene_normals_ptr (new pcl::PointCloud<pcl::Normal>);
	//pcl::PointCloud<pcl::Normal>& scene_normals = *scene_normals_ptr;

	//pcl::NormalEstimationOMP<PointType, NormalType> norm_est;
  //norm_est.setKSearch (20);
  //norm_est.setInputCloud (model_ptr);
  //norm_est.compute (model_normals);

  //pcl::NormalEstimationOMP<PointType, NormalType> scene_norm_est;
  //scene_norm_est.setKSearch (20);
	//scene_norm_est.setInputCloud (scene_ptr);
  //norm_est.compute (scene_normals);
	//std::cout << "Model normals: " << model_normals_ptr->points.size () << std::endl;
	//std::cout << "Scene normals: " << scene_normals_ptr->points.size () << std::endl;

	//// Reject if the angle between the normals is really off
  //pcl::registration::CorrespondenceRejectorSurfaceNormal rej_normals;
  //rej_normals.setThreshold (cos (pcl::deg2rad (rejection_angle_)));
  //rej_normals.initializeDataContainer<PointType, NormalType> ();
  //rej_normals.setInputSource<PointType> (model_ptr);
  //rej_normals.setInputNormals<PointType, NormalType> (model_normals_ptr);
  //rej_normals.setInputTarget<PointType> (scene_ptr);
  //rej_normals.setTargetNormals<PointType, NormalType> (scene_normals_ptr);
  //rej_normals.setInputCorrespondences (good_model_scene_corrs_tmp);
  //rej_normals.getCorrespondences (*good_model_scene_corrs);
	//std::cout << "\tFound " << good_model_scene_corrs->size() << " GOOD correspondences " << std::endl;
	//for (int i = 0; i < good_model_scene_corrs->size(); i++)
	//{
	//	std::cout << "(k_indices,index,k_distances): " << (*good_model_scene_corrs.get())[i] << std::endl; // Print good correspondences to check
	//} 

	// Select 3 Correspondences to estimate Transformation Matrix
	pcl::CorrespondencesPtr good_model_scene_corrs(new pcl::Correspondences()); 
	good_model_scene_corrs = model_scene_corrs;
	std::vector<pcl::CorrespondencesPtr> clustered_corrs;
	for (int k = 2; k < good_model_scene_corrs->size(); k++)
	{
		for (int j = 1; j < k; j++)
		{
			for (int i = 0; i < j; i++)
			{
				pcl::CorrespondencesPtr corrs(new pcl::Correspondences());
				// Condition 1: Check if the metric distances between the three feature points in the scene are similar to the distances of the feature points in the model
				
				float diff_L1_ij_scene = (pcl::L1_Norm(scene_narf_descriptors_ptr->at((*good_model_scene_corrs.get())[i].index_match).descriptor,scene_narf_descriptors_ptr->at((*good_model_scene_corrs.get())[j].index_match).descriptor,36))/36;		
				float diff_L1_jk_scene = (pcl::L1_Norm(scene_narf_descriptors_ptr->at((*good_model_scene_corrs.get())[j].index_match).descriptor,scene_narf_descriptors_ptr->at((*good_model_scene_corrs.get())[k].index_match).descriptor,36))/36;		
				float diff_L1_ki_scene = (pcl::L1_Norm(scene_narf_descriptors_ptr->at((*good_model_scene_corrs.get())[k].index_match).descriptor,scene_narf_descriptors_ptr->at((*good_model_scene_corrs.get())[i].index_match).descriptor,36))/36;		
				float diff_L1_ij_model = (pcl::L1_Norm(model_narf_descriptors_ptr->at((*good_model_scene_corrs.get())[i].index_query).descriptor,model_narf_descriptors_ptr->at((*good_model_scene_corrs.get())[j].index_query).descriptor,36))/36;		
				float diff_L1_jk_model = (pcl::L1_Norm(model_narf_descriptors_ptr->at((*good_model_scene_corrs.get())[j].index_query).descriptor,model_narf_descriptors_ptr->at((*good_model_scene_corrs.get())[k].index_query).descriptor,36))/36;		
				float diff_L1_ki_model = (pcl::L1_Norm(model_narf_descriptors_ptr->at((*good_model_scene_corrs.get())[k].index_query).descriptor,model_narf_descriptors_ptr->at((*good_model_scene_corrs.get())[i].index_query).descriptor,36))/36;		
				std::cout << "diff_L1_ij_model_scene: " << std::abs(diff_L1_ij_model - diff_L1_ij_scene ) << std::endl;
				std::cout << "diff_L1_jk_model_scene: " << std::abs(diff_L1_jk_model - diff_L1_jk_scene ) << std::endl;
				std::cout << "diff_L1_ki_model_scene: " << std::abs(diff_L1_ki_model - diff_L1_ki_scene ) << std::endl;
				float diff_L1_ij_model_scene = std::abs(diff_L1_ij_model - diff_L1_ij_scene ); 
				float diff_L1_jk_model_scene = std::abs(diff_L1_jk_model - diff_L1_jk_scene );
				float diff_L1_ki_model_scene = std::abs(diff_L1_ki_model - diff_L1_ki_scene );
 				if ((diff_L1_ij_model_scene < 0.04) && (diff_L1_jk_model_scene < 0.04) && (diff_L1_ki_model_scene < 0.04))
				{
					// Condition 2: Discard triples of features that lie on a line
					int diff_ij = std::abs((*good_model_scene_corrs.get())[i].index_query - (*good_model_scene_corrs.get())[j].index_query);
					int diff_ik = std::abs((*good_model_scene_corrs.get())[i].index_query - (*good_model_scene_corrs.get())[k].index_query);
					int diff_jk = std::abs((*good_model_scene_corrs.get())[j].index_query - (*good_model_scene_corrs.get())[k].index_query);
					if ((diff_ij != 0) && (diff_ik != 0) && (diff_jk != 0)) 
					{
						corrs->push_back((*good_model_scene_corrs.get())[i]);
						corrs->push_back((*good_model_scene_corrs.get())[j]);
						corrs->push_back((*good_model_scene_corrs.get())[k]);
						clustered_corrs.push_back(corrs);
					}
				}
				//std::cout << "(i,j,k): " << i << "," << j << "," << k << std::endl;
			}
		}
	}
	
	
	std::cout << "Begin transform SVD" << std::endl;
	std::vector<pcl::PointCloud<PointType>::ConstPtr> rotated_instances;	
	std::vector<Eigen::Matrix4f> matrix_transformSVD_vector;
	pcl::registration::TransformationEstimationSVD<PointType, PointType> transformSVD;
	Eigen::Matrix4f matrix_transformSVD;

	if (!show_keypoints_)
	{
		try
		{
			for (int i = 0; i < clustered_corrs.size(); i++)
			{
				std::cout << "clustered_corrs.size(): " << clustered_corrs.size() << std::endl;
				pcl::Correspondences corrs = *clustered_corrs[i].get();
				for (int k = 0; k < corrs.size(); k++)
				{
					std::cout << "(k_indices,index,k_distances): " << corrs[k].index_query << "," << corrs[k].index_match << "," << corrs[k].distance << std::endl; // Print good correspondences group to check
				}
				transformSVD.estimateRigidTransformation(*model_keypoints_ptr, *scene_keypoints_ptr, corrs, matrix_transformSVD);
				pcl::PointCloud<PointType>::Ptr rotated_model(new pcl::PointCloud<PointType>());
				pcl::transformPointCloud(*model_ptr, *rotated_model, matrix_transformSVD);
				pcl::registration::TransformationValidationEuclidean<pcl::PointXYZ, pcl::PointXYZ> tve;
				double score_ = tve.validateTransformation (rotated_model, scene_ptr, matrix_transformSVD);
				if ((score_ < validation_score_max_) && (score_ > validation_score_min_))
				{
					float pose_gap_ = diff (matrix_transformSVD, matrix); // distance between estimated transform matrix and real pose (suggested by author)
					std::cout << "Pose gap: " << pose_gap_ << std::endl;
					std::cout << "Euclidean Transformation score: " << score_ << std::endl;
					rotated_instances.push_back(rotated_model);
					matrix_transformSVD_vector.push_back(matrix_transformSVD);
				}
			}
		}
		catch (const std::out_of_range& e) {}
		std::cout << "Finish transform SVD" << std::endl;
	}
	

	//std::vector<pcl::PointCloud<PointType>::ConstPtr> registered_instances;
	//pcl::PointCloud<PointType>::Ptr registered_model(new pcl::PointCloud<PointType>());
	//pcl::console::print_highlight("Refine transformation using ICP...\n");
	//std::vector<Eigen::Matrix4f> rototranslations_afterICP;

	//std::vector<int> skip_indexes; // skip instance which does not aligned in ICP
	//float score, score_afterICP;
	////for (size_t i = 0; i < rototranslations.size(); ++i)
	////for (size_t i = 0; i < matrix_transformSVD_vector.size(); ++i)
	//for (size_t i = 0; i < rotated_instances.size(); ++i)
	//{
	//	//Eigen::Matrix4f transform = rototranslations[i].block<4, 4>(0, 0);
	//	//pcl::transformPointCloud(*model_ptr, *rotated_model, transform);
	//	pcl::IterativeClosestPoint<PointType, PointType> icp;
	//	icp.setMaxCorrespondenceDistance(icp_corr_distance_); //float icp_corr_distance_ (0.005f);
	//	// Set the maximum number of iterations (criterion 1)
	//	icp.setMaximumIterations(1000000); // int icp_max_iter_ (5)
	//	// Set the transformation epsilon (criterion 2)
	//	icp.setTransformationEpsilon(1e-12);
	//	// Set the euclidean distance difference epsilon (criterion 3)
	//	icp.setEuclideanFitnessEpsilon(eu_epsilon_);

	//	icp.setInputTarget(scene_ptr);
	//	icp.setInputSource(rotated_instances[i]);
	//	//icp.setInputSource(model_ptr);
	//	pcl::PointCloud<PointType>::Ptr registered(new pcl::PointCloud<PointType>);
	//	icp.align(*registered);
	//	registered_instances.push_back(registered);

	//	//score = validator.validateTransformation(rotated_model, scene_ptr, transform);
	//	std::cout << "Instance " << i << " before ICP transformation score: " << score << std::endl;
	//	if (icp.hasConverged())
	//	{
	//		cout << "ICP Aligned!" << endl;
	//		Eigen::Matrix4f transformation_after_ICP = icp.getFinalTransformation();
	//		rototranslations_afterICP.push_back(transformation_after_ICP);
	//		std::cout << "Instance " << i << " after ICP transformation score: " << score_afterICP << std::endl;
	//		std::cout << "Instance " << i << " get fitness score: " << icp.getFitnessScore() << std::endl;
	//		//std::cout << "Instance " << i << " get EuclideanFitnessEpsilon: " << icp.getEuclideanFitnessEpsilon() << std::endl;
	//	}
	//	else
	//	{
	//		cout << "Not Aligned!" << endl;
	//		skip_indexes.push_back(i);
	//	}
	//}
	
	// 
	// 7. Print results
	//

	/**
	* Stop if no instances
	*/
	if (!show_keypoints_)
	{
		std::cout << "Recognized Instances: " << matrix_transformSVD_vector.size() << std::endl;
		for (size_t i = 0; i < matrix_transformSVD_vector.size(); ++i)
		{
			std::cout << "\n    Instance " << i + 1 << ":" << std::endl;
			//std::cout << "        Correspondences belonging to this instance: " << clustered_corrs[i].size() << std::endl;

			// Print the rotation matrix and translation vector
			Eigen::Matrix3f rotation = matrix_transformSVD_vector[i].block<3, 3>(0, 0);
			Eigen::Vector3f translation = matrix_transformSVD_vector[i].block<3, 1>(0, 3);

			printf("\n");
			printf("            | %6.3f %6.3f %6.3f | \n", rotation(0, 0), rotation(0, 1), rotation(0, 2));
			printf("        R = | %6.3f %6.3f %6.3f | \n", rotation(1, 0), rotation(1, 1), rotation(1, 2));
			printf("            | %6.3f %6.3f %6.3f | \n", rotation(2, 0), rotation(2, 1), rotation(2, 2));
			printf("\n");
			printf("        t = < %0.3f, %0.3f, %0.3f >\n", translation(0), translation(1), translation(2));

			//std::cout << "rototranslations_afterICP" << std::endl;
			////std::cout << "Matrix transform using RANSAC Rejection" << std::endl;

			//Eigen::Matrix3f rotation2 = rototranslations_afterICP[i].block<3, 3>(0, 0);
			//Eigen::Vector3f translation2 = rototranslations_afterICP[i].block<3, 1>(0, 3);
			//printf("\n");
			//printf("            | %6.3f %6.3f %6.3f | \n", rotation2(0, 0), rotation2(0, 1), rotation2(0, 2));
			//printf("        R = | %6.3f %6.3f %6.3f | \n", rotation2(1, 0), rotation2(1, 1), rotation2(1, 2));
			//printf("            | %6.3f %6.3f %6.3f | \n", rotation2(2, 0), rotation2(2, 1), rotation2(2, 2));
			//printf("\n");
			//printf("        t = < %0.3f, %0.3f, %0.3f >\n", translation2(0), translation2(1), translation2(2));

			//std::cout << "Distnace to ICP: " << diff(rototranslations_afterICP[i], matrix_transformSVD_vector[i]) << std::endl;
		}
	}

	
	

	//
	// 8. Visualization
	//
	pcl::visualization::PCLVisualizer viewer("Corresponding 3D Viewer");
	viewer.initCameraParameters();
	viewer.addPointCloud(scene_ptr, "scene");

	if ((show_keypoints_) && (!show_cluster_))
	{
		//
		// Show Keypoints in 3D Viewer
		//

		pcl::PointCloud<PointType>::Ptr off_scene_model_ptr(new pcl::PointCloud<PointType>());
		pcl::PointCloud<PointType>::Ptr off_scene_model_keypoints_ptr(new pcl::PointCloud<PointType>());
		pcl::transformPointCloud(*model_ptr, *off_scene_model_ptr, Eigen::Vector3f(-2, 1, 0), Eigen::Quaternionf(0, 0, 0, 0));
		pcl::transformPointCloud(*model_keypoints_ptr, *off_scene_model_keypoints_ptr, Eigen::Vector3f(-2, 1, 0), Eigen::Quaternionf(0, 0, 0, 0));

		viewer.addPointCloud(off_scene_model_ptr, "off_scene_model_ptr");

		CloudStyle model_keypointStyle = style_cyan;
		pcl::visualization::PointCloudColorHandlerCustom<PointType> model_keypoints_color_handler(off_scene_model_keypoints_ptr, model_keypointStyle.r, model_keypointStyle.g, model_keypointStyle.b);
		viewer.addPointCloud<PointType>(off_scene_model_keypoints_ptr, model_keypoints_color_handler, "model_keypoints_ptr");
		viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 7, "model_keypoints_ptr");

		CloudStyle scene_keypointStyle = style_violet;
		pcl::visualization::PointCloudColorHandlerCustom<PointType> scene_keypoints_color_handler(scene_keypoints_ptr, scene_keypointStyle.r, scene_keypointStyle.g, scene_keypointStyle.b);
		viewer.addPointCloud<PointType>(scene_keypoints_ptr, scene_keypoints_color_handler, "scene_keypoints_ptr");
		viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 7, "scene_keypoints_ptr");

		int index = 0;
		for (size_t j = 0; j < good_model_scene_corrs->size(); j++)
		{
			try
			{
				std::stringstream ss_line;
				ss_line << "correspondence_line" << j;
				std::cout << "Addline with ID: " << ss_line.str() << std::endl;
				float model_x = off_scene_model_keypoints_ptr->at((*good_model_scene_corrs.get())[j].index_query).x;
				float model_y = off_scene_model_keypoints_ptr->at((*good_model_scene_corrs.get())[j].index_query).y;
				float model_z = off_scene_model_keypoints_ptr->at((*good_model_scene_corrs.get())[j].index_query).z;
				float scene_x = scene_keypoints_ptr->at((*good_model_scene_corrs.get())[j].index_match).x;
				float scene_y = scene_keypoints_ptr->at((*good_model_scene_corrs.get())[j].index_match).y;
				float scene_z = scene_keypoints_ptr->at((*good_model_scene_corrs.get())[j].index_match).z;

				//std::cout << "clustered_corrs[" << i << "][" << j << "].index_query: " << clustered_corrs[i][j].index_query << std::endl;
				//std::cout << "clustered_corrs[" << i << "][" << j << "].index_match: " << clustered_corrs[i][j].index_match << std::endl;

				/*Eigen::Quaternion<float> transformation(0, 1, 0, 0);*/
				Eigen::Vector3f tmp(model_x, model_y, model_z);
				/*tmp = transformation._transformVector(tmp);*/
				pcl::PointXYZ model_point(tmp.x(), tmp.y(), tmp.z());

				Eigen::Vector3f tmp2(scene_x, scene_y, scene_z);
				/*tmp2 = transformation._transformVector(tmp2);*/
				pcl::PointXYZ scene_point(tmp2.x(), tmp2.y(), tmp2.z());

				CloudStyle corrStyle = style_red;
				std::stringstream ss_model,ss_scene;
				ss_model << "model" << (*good_model_scene_corrs.get())[j].index_query;
				ss_scene << "scene" << (*good_model_scene_corrs.get())[j].index_match;
				viewer.addSphere<PointType>(model_point,0.025,corrStyle.r,corrStyle.g,corrStyle.b,ss_model.str());				
				viewer.addSphere<PointType>(scene_point,0.025,corrStyle.r,corrStyle.g,corrStyle.b,ss_scene.str());				
																																											
				CloudStyle clusterStyle = style_lime;
				/*if (i == 1) clusterStyle = style_white;*/
				viewer.addLine<PointType, PointType>(model_point, scene_point, clusterStyle.r, clusterStyle.g, clusterStyle.b, ss_line.str());
			}
			catch (const std::out_of_range& e) {}
		}

	}
	else if ((!show_keypoints_) && (!show_cluster_))
	{
		std::stringstream ss_instance;
		for (int i = 0; i < rotated_instances.size(); i++)
		{	
			CloudStyle registeredStyles = style_white;
			int tmp = i % 4;
			switch (tmp)
			{
			case 0: registeredStyles = style_lime; break;
			case 1: registeredStyles = style_yellow; break;
			case 2: registeredStyles = style_blue; break;
			case 3: registeredStyles = style_cyan; break;
			}
			//CloudStyle registeredStyles = style_lime;
			ss_instance << "instance_" << i << "_rotated" << endl;
			pcl::visualization::PointCloudColorHandlerCustom<PointType> rotated_instance_color_handler(rotated_instances[i], registeredStyles.r,registeredStyles.g, registeredStyles.b);
			viewer.addPointCloud(rotated_instances[i], rotated_instance_color_handler, ss_instance.str());

			//registeredStyles = style_cyan;
			//ss_instance << "instance_" << i << "_registered" << endl;
			//pcl::visualization::PointCloudColorHandlerCustom<PointType> registered_instance_color_handler(registered_instances[i], registeredStyles.r, registeredStyles.g, registeredStyles.b);
			//viewer.addPointCloud(registered_instances[i], registered_instance_color_handler, ss_instance.str());

		}
	}

	setViewerPose(viewer, scene_rangeImage.getTransformationToWorldSystem());

	while (!viewer.wasStopped())
	{
		viewer.spinOnce();
		pcl_sleep(0.1);
	}
}
