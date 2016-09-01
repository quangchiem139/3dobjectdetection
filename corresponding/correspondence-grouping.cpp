#include <iostream>
#include <stdio.h>      /* printf, fgets */
#include <stdlib.h>     /* atoi */
#include <string.h>

#include <vector>
#include <Eigen/Core>
#include "pcl/point_types.h"
#include "pcl/point_cloud.h"
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/vtk_lib_io.h>
#include "pcl/kdtree/kdtree_flann.h"
#include <pcl/kdtree/impl/kdtree_flann.hpp>
#include <pcl/filters/voxel_grid.h>
#include <pcl/console/parse.h>
#include <pcl/features/normal_3d.h>
// For SHOT keypoint
#include <pcl/keypoints/uniform_sampling.h>
#include <pcl/keypoints/impl/uniform_sampling.hpp>
#include <pcl/features/shot_omp.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/recognition/cg/hough_3d.h>
#include <pcl/recognition/cg/geometric_consistency.h>
#include <pcl/common/transforms.h>
#include <pcl/registration/transforms.h>
// For SIFT keypoint
#include "inc/visualize_feature.h"
#include "inc/corresponding_SIFT.h"

void
downsample (pcl::PointCloud<pcl::PointXYZRGB>::Ptr &points, float leaf_size,
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr &downsampled_out)
{
  pcl::VoxelGrid<pcl::PointXYZRGB> vox_grid;
  vox_grid.setLeafSize (leaf_size, leaf_size, leaf_size);
  vox_grid.setInputCloud (points);
  vox_grid.filter (*downsampled_out);
}

void
compute_surface_normals (pcl::PointCloud<pcl::PointXYZRGB>::Ptr &points, float normal_radius,
                         pcl::PointCloud<pcl::Normal>::Ptr &normals_out)
{
  pcl::NormalEstimation<pcl::PointXYZRGB, pcl::Normal> norm_est;

  // Use a FLANN-based KdTree to perform neighborhood searches
  pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGB>());
  norm_est.setSearchMethod (tree);

  // Specify the size of the local neighborhood to use when computing the surface normals
  norm_est.setRadiusSearch (normal_radius);

  // Set the input points
  norm_est.setInputCloud (points);

  // Estimate the surface normals and store the result in "normals_out"
  norm_est.compute (*normals_out);
}

void
find_feature_correspondences(pcl::PointCloud<pcl::PFHSignature125>::Ptr &source_descriptors,
pcl::PointCloud<pcl::PFHSignature125>::Ptr &target_descriptors,
std::vector<int> &correspondences_out, std::vector<float> &correspondence_scores_out)
{
	// Resize the output vector
	correspondences_out.resize(source_descriptors->size());
	correspondence_scores_out.resize(source_descriptors->size());

	// Use a KdTree to search for the nearest matches in feature space
	pcl::KdTreeFLANN<pcl::PFHSignature125> descriptor_kdtree;
	descriptor_kdtree.setInputCloud(target_descriptors);

	// Find the index of the best match for each keypoint, and store it in "correspondences_out"
	const int k = 1;
	std::vector<int> k_indices(k);
	std::vector<float> k_squared_distances(k);
	for (size_t i = 0; i < source_descriptors->size(); ++i)
	{
		descriptor_kdtree.nearestKSearch(*source_descriptors, i, k, k_indices, k_squared_distances);
		correspondences_out[i] = k_indices[0];
		correspondence_scores_out[i] = k_squared_distances[0];
	}
}

int
main(int argc, char** argv)
{
	//
	// Create some new point clouds to hold our data
	//

	// For SIFT
	/*pcl::PointCloud<pcl::PointXYZRGB>::Ptr model (new pcl::PointCloud<pcl::PointXYZRGB>);
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr model_downsampled(new pcl::PointCloud<pcl::PointXYZRGB>);
	pcl::PointCloud<pcl::Normal>::Ptr model_normals(new pcl::PointCloud<pcl::Normal>);
	pcl::PointCloud<pcl::PointWithScale>::Ptr model_keypoints(new pcl::PointCloud<pcl::PointWithScale>);
	pcl::PointCloud<pcl::PFHSignature125>::Ptr model_descr(new pcl::PointCloud<pcl::PFHSignature125>);

	pcl::PointCloud<pcl::PointXYZRGB>::Ptr scene(new pcl::PointCloud<pcl::PointXYZRGB>);
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr scene_downsampled(new pcl::PointCloud<pcl::PointXYZRGB>);
	pcl::PointCloud<pcl::Normal>::Ptr scene_normals(new pcl::PointCloud<pcl::Normal>);
	pcl::PointCloud<pcl::PointWithScale>::Ptr scene_keypoints(new pcl::PointCloud<pcl::PointWithScale>);
	pcl::PointCloud<pcl::PFHSignature125>::Ptr scene_descr(new pcl::PointCloud<pcl::PFHSignature125>);*/
	
	// For SHOT
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr model(new pcl::PointCloud<pcl::PointXYZRGB>());
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr model_downsampled(new pcl::PointCloud<pcl::PointXYZRGB>);
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr model_keypoints(new pcl::PointCloud<pcl::PointXYZRGB>());
	
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr scene(new pcl::PointCloud<pcl::PointXYZRGB>());
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr scene_downsampled(new pcl::PointCloud<pcl::PointXYZRGB>);
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr scene_keypoints(new pcl::PointCloud<pcl::PointXYZRGB>());
	
	pcl::PointCloud<pcl::Normal>::Ptr model_normals(new pcl::PointCloud<pcl::Normal>());
	pcl::PointCloud<pcl::Normal>::Ptr scene_normals(new pcl::PointCloud<pcl::Normal>());
	pcl::PointCloud<pcl::SHOT352>::Ptr model_descriptors(new pcl::PointCloud<pcl::SHOT352>());
	pcl::PointCloud<pcl::SHOT352>::Ptr scene_descriptors(new pcl::PointCloud<pcl::SHOT352>());

	//
	// Load a point cloud
	//

	//1. Load PCD file
	pcl::io::loadPCDFile(argv[1], *model);
	std::cout << "MODEL HAS: " << model->size() << " POINTS" << std::endl;
	pcl::io::loadPCDFile(argv[2], *scene);
	std::cout << "SCENE HAS: " << scene->size() << " POINTS" << std::endl;

	//
	//  Downsample Clouds to Extract keypoints
	//
	//const float voxel_grid_leaf_size_model = 0.01;
	//downsample(model, voxel_grid_leaf_size_model, model_downsampled);
	//std::cout << "Model total points: " << model->size() << "; Model Downsample: " << model_downsampled->size() << std::endl;
	//const float voxel_grid_leaf_size_scene = 0.01;
	//downsample(scene, voxel_grid_leaf_size_scene, scene_downsampled);
	//std::cout << "Scene total points: " << scene->size() << "; Scene Downsample: " << scene_downsampled->size() << std::endl;
	
	
	//
	//  Compute Surface Normals
	//
	/*const float normal_radius = 0.03;
	compute_surface_normals(model_downsampled, normal_radius, model_normals);
	compute_surface_normals(scene_downsampled, normal_radius, scene_normals);*/

	//// Corresponding detection based on SIFT keypoints
	//corresponding_SIFT_demo(model, scene, model_downsampled, scene_downsampled, model_normal, scene_normal, model_keypoints, scene_keypoints, model_descr, scene_descr);

	//// Find feature correspondences
	//std::vector<int> correspondences;
	//std::vector<float> correspondence_scores;
	//find_feature_correspondences(model_descr, scene_descr, correspondences, correspondence_scores);

	//// Print out ( number of keypoints / number of points )
	//std::cout << "First cloud: Found " << model_keypoints->size() << " keypoints "
	//	<< "out of " << model_downsampled->size() << " total points." << std::endl;
	//std::cout << "Second cloud: Found " << scene_keypoints->size() << " keypoints "
	//	<< "out of " << scene_downsampled->size() << " total points." << std::endl;

	//// Visualize Keypoints or Correspondences 
	//bool parseTm = pcl::console::find_switch(argc, argv, "-vf");
	//std::cout << "parseTm = " << parseTm << std::endl;
	//if (parseTm) {
	//	visualize_keypoints(model, model_keypoints);
	//}
	//else {
	//	visualize_correspondences(model, model_keypoints, scene, scene_keypoints, correspondences, correspondence_scores);
	//}

	//
	//  Compute Normals
	//
	/*pcl::NormalEstimationOMP<pcl::PointXYZRGB, pcl::Normal> norm_est;
	norm_est.setKSearch(10);
	norm_est.setInputCloud(model);
	norm_est.compute(*model_normals);

	norm_est.setInputCloud(scene);
	norm_est.compute(*scene_normals);*/


	/**
	*  Downsample Clouds to Extract keypoints
	*/
	//float model_ss_(0.01f);
	//float scene_ss_(0.01f);
	//pcl::UniformSampling<pcl::PointXYZRGB> uniform_sampling;
	//uniform_sampling.setInputCloud(model);
	//uniform_sampling.setRadiusSearch(model_ss_);
	//pcl::PointCloud<int> keypointIndices;
	//uniform_sampling.compute(keypointIndices);
	//pcl::copyPointCloud(*model, keypointIndices.points, *model_keypoints);
	//std::cout << "Model total points: " << model->size() << "; Selected Keypoints: " << model_keypoints->size() << std::endl;

	//uniform_sampling.setInputCloud(scene);
	//uniform_sampling.setRadiusSearch(scene_ss_);
	////uniform_sampling.detectKeypoints(*scene_keypoints);
	//pcl::copyPointCloud(*scene, keypointIndices.points, *scene_keypoints);
	//std::cout << "Scene total points: " << scene->size() << "; Selected Keypoints: " << scene_keypoints->size() << std::endl;
	//// Corresponding detection based on SHOT descriptor
	//float descr_rad_(0.01f);
	//pcl::SHOTEstimationOMP<pcl::PointXYZRGB, pcl::Normal, pcl::SHOT352> descr_est;
	//descr_est.setRadiusSearch(descr_rad_);

	//descr_est.setInputCloud(model_keypoints);
	//descr_est.setInputNormals(model_normals);
	//descr_est.setSearchSurface(model);
	//descr_est.compute(*model_descriptors);

	//descr_est.setInputCloud(scene_keypoints);
	//descr_est.setInputNormals(scene_normals);
	//descr_est.setSearchSurface(scene);
	//descr_est.compute(*scene_descriptors);


	/**
	*  Find Model-Scene Correspondences with KdTree
	*/
	//pcl::CorrespondencesPtr model_scene_corrs(new pcl::Correspondences());
	//pcl::KdTreeFLANN<pcl::SHOT352> match_search;
	//match_search.setInputCloud(model_descriptors);
	//std::vector<int> model_good_keypoints_indices;
	//std::vector<int> scene_good_keypoints_indices;

	//for (size_t i = 0; i < scene_descriptors->size(); ++i)
	//{
	//	std::vector<int> neigh_indices(1);
	//	std::vector<float> neigh_sqr_dists(1);
	//	if (!pcl_isfinite(scene_descriptors->at(i).descriptor[0]))  //skipping NaNs
	//	{
	//		continue;
	//	}
	//	int found_neighs = match_search.nearestKSearch(scene_descriptors->at(i), 1, neigh_indices, neigh_sqr_dists);
	//	if (found_neighs == 1 && neigh_sqr_dists[0] < 0.25f)
	//	{
	//		pcl::Correspondence corr(neigh_indices[0], static_cast<int> (i), neigh_sqr_dists[0]);
	//		model_scene_corrs->push_back(corr);
	//		model_good_keypoints_indices.push_back(corr.index_query);
	//		scene_good_keypoints_indices.push_back(corr.index_match);
	//	}
	//}
	//pcl::PointCloud<pcl::PointXYZRGB>::Ptr model_good_kp(new pcl::PointCloud<pcl::PointXYZRGB>());
	//pcl::PointCloud<pcl::PointXYZRGB>::Ptr scene_good_kp(new pcl::PointCloud<pcl::PointXYZRGB>());
	//pcl::copyPointCloud(*model_keypoints, model_good_keypoints_indices, *model_good_kp);
	//pcl::copyPointCloud(*scene_keypoints, scene_good_keypoints_indices, *scene_good_kp);

	//std::cout << "Correspondences found: " << model_scene_corrs->size() << std::endl;

	//
	//  Actual Clustering
	//
	//std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > rototranslations;
	//std::vector<pcl::Correspondences> clustered_corrs;

	// // Using GeometricConsistency
	//
	//	pcl::GeometricConsistencyGrouping<pcl::PointXYZRGB, pcl::PointXYZRGB> gc_clusterer;
	//	gc_clusterer.setGCSize(0.01f);
	//	gc_clusterer.setGCThreshold(5.0f);

	//	gc_clusterer.setInputCloud(model_keypoints);
	//	gc_clusterer.setSceneCloud(scene_keypoints);
	//	gc_clusterer.setModelSceneCorrespondences(model_scene_corrs);

	//	//gc_clusterer.cluster (clustered_corrs);
	//	gc_clusterer.recognize(rototranslations, clustered_corrs);


	//
	//  Output results
	//
	//std::cout << "Model instances found: " << rototranslations.size() << std::endl;
	//for (size_t i = 0; i < rototranslations.size(); ++i)
	//{
	//	std::cout << "\n    Instance " << i + 1 << ":" << std::endl;
	//	std::cout << "        Correspondences belonging to this instance: " << clustered_corrs[i].size() << std::endl;

	//	// Print the rotation matrix and translation vector
	//	Eigen::Matrix3f rotation = rototranslations[i].block<3, 3>(0, 0);
	//	Eigen::Vector3f translation = rototranslations[i].block<3, 1>(0, 3);

	//	printf("\n");
	//	printf("            | %6.3f %6.3f %6.3f | \n", rotation(0, 0), rotation(0, 1), rotation(0, 2));
	//	printf("        R = | %6.3f %6.3f %6.3f | \n", rotation(1, 0), rotation(1, 1), rotation(1, 2));
	//	printf("            | %6.3f %6.3f %6.3f | \n", rotation(2, 0), rotation(2, 1), rotation(2, 2));
	//	printf("\n");
	//	printf("        t = < %0.3f, %0.3f, %0.3f >\n", translation(0), translation(1), translation(2));
	//}

	//
	//  Visualization
	//
	pcl::visualization::PCLVisualizer viewer("Correspondence Grouping");
	viewer.addPointCloud(scene, "scene_cloud");

	//pcl::PointCloud<pcl::PointXYZRGB>::Ptr off_scene_model(new pcl::PointCloud<pcl::PointXYZRGB>());
	//pcl::PointCloud<pcl::PointXYZRGB>::Ptr off_scene_model_keypoints(new pcl::PointCloud<pcl::PointXYZRGB>());

	//pcl::transformPointCloud(*model, *off_scene_model, Eigen::Vector3f(-1, 0, 0), Eigen::Quaternionf(1, 0, 0, 0));
	//pcl::transformPointCloud(*model_keypoints, *off_scene_model_keypoints, Eigen::Vector3f(-1, 0, 0), Eigen::Quaternionf(1, 0, 0, 0));

	//pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGB> off_scene_model_color_handler(off_scene_model, 255, 255, 128);
	//viewer.addPointCloud(off_scene_model, off_scene_model_color_handler, "off_scene_model");

	//for (size_t i = 0; i < rototranslations.size(); ++i)
	//{
	//	pcl::PointCloud<pcl::PointXYZRGB>::Ptr rotated_model(new pcl::PointCloud<pcl::PointXYZRGB>());
	//	pcl::transformPointCloud(*model, *rotated_model, rototranslations[i]);

	//	std::stringstream ss_cloud;
	//	ss_cloud << "instance" << i;

	//	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGB> rotated_model_color_handler(rotated_model, 255, 0, 0);
	//	viewer.addPointCloud(rotated_model, rotated_model_color_handler, ss_cloud.str());

	//	for (size_t j = 0; j < clustered_corrs[i].size(); ++j)
	//	{
	//		std::stringstream ss_line;
	//		ss_line << "correspondence_line" << i << "_" << j;
	//		pcl::PointXYZRGB& model_point = off_scene_model_keypoints->at(clustered_corrs[i][j].index_query);
	//		pcl::PointXYZRGB& scene_point = scene_keypoints->at(clustered_corrs[i][j].index_match);

	//		//  We are drawing a line for each pair of clustered correspondences found between the model and the scene
	//		viewer.addLine<pcl::PointXYZRGB, pcl::PointXYZRGB>(model_point, scene_point, 0, 255, 0, ss_line.str());
	//	}
	//}

	while (!viewer.wasStopped())
	{
		viewer.spinOnce();
	}

	//return (0);
}
