#ifndef CORRESPONDING_SIFT_H
#define CORRESPONDING_SIFT_H

#include "pcl/point_types.h"
#include "pcl/point_cloud.h"
#include "pcl/kdtree/kdtree_flann.h"
#include <pcl/kdtree/impl/kdtree_flann.hpp>
#include <pcl/features/normal_3d.h>
#include "pcl/features/pfh.h"
#include "pcl/keypoints/sift_keypoint.h"

void
detect_keypoints(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &points,
				float min_scale, int nr_octaves, int nr_scales_per_octave, float min_contrast,
				pcl::PointCloud<pcl::PointWithScale>::Ptr &keypoints_out)
{
	pcl::SIFTKeypoint<pcl::PointXYZRGB, pcl::PointWithScale> sift_detect;

	// Use a FLANN-based KdTree to perform neighborhood searches
	pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGB>());
	sift_detect.setSearchMethod(tree);

	// Set the detection parameters
	sift_detect.setScales(min_scale, nr_octaves, nr_scales_per_octave);
	sift_detect.setMinimumContrast(min_contrast);

	// Set the input
	sift_detect.setInputCloud(points);

	// Detect the keypoints and store them in "keypoints_out"
	sift_detect.compute(*keypoints_out);
}

void
compute_PFH_features_at_keypoints(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &points,
									pcl::PointCloud<pcl::Normal>::Ptr &normals,
									pcl::PointCloud<pcl::PointWithScale>::Ptr &keypoints, float feature_radius,
									pcl::PointCloud<pcl::PFHSignature125>::Ptr &descriptors_out)
{
	// Create a PFHEstimation object
	pcl::PFHEstimation<pcl::PointXYZRGB, pcl::Normal, pcl::PFHSignature125> pfh_est;

	// Set it to use a FLANN-based KdTree to perform its neighborhood searches
	pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGB>());
	pfh_est.setSearchMethod(tree);

	// Specify the radius of the PFH feature
	pfh_est.setRadiusSearch(feature_radius);

	/* This is a little bit messy: since our keypoint detection returns PointWithScale points, but we want to
	* use them as an input to our PFH estimation, which expects clouds of PointXYZRGB points.  To get around this,
	* we'll use copyPointCloud to convert "keypoints" (a cloud of type PointCloud<PointWithScale>) to
	* "keypoints_xyzrgb" (a cloud of type PointCloud<PointXYZRGB>).  Note that the original cloud doesn't have any RGB
	* values, so when we copy from PointWithScale to PointXYZRGB, the new r,g,b fields will all be zero.
	*/
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr keypoints_xyzrgb(new pcl::PointCloud<pcl::PointXYZRGB>);
	pcl::copyPointCloud(*keypoints, *keypoints_xyzrgb);
	std::cout << "Keypoint RGB: " << keypoints_xyzrgb->size() << std::endl;

	// Use all of the points for analyzing the local structure of the cloud
	pfh_est.setSearchSurface(points);
	pfh_est.setInputNormals(normals);

	// But only compute features at the keypoints
	pfh_est.setInputCloud(keypoints_xyzrgb);

	// Compute the features
	pfh_est.compute(*descriptors_out);
}

void 
corresponding_SIFT_demo(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &model,
						pcl::PointCloud<pcl::PointXYZRGB>::Ptr &scene,
						pcl::PointCloud<pcl::PointXYZRGB>::Ptr &model_downsampled,
						pcl::PointCloud<pcl::PointXYZRGB>::Ptr &scene_downsampled,
						pcl::PointCloud<pcl::Normal>::Ptr model_normal,
						pcl::PointCloud<pcl::Normal>::Ptr scene_normal,
						pcl::PointCloud<pcl::PointWithScale>::Ptr &model_keypoints,
						pcl::PointCloud<pcl::PointWithScale>::Ptr &scene_keypoints,
						pcl::PointCloud<pcl::PFHSignature125>::Ptr &model_descr,
						pcl::PointCloud<pcl::PFHSignature125>::Ptr &scene_descr)
{					
	//
	//  Compute Keypoints
	//
	const float min_scale = 0.01;
	const int nr_octaves = 3;
	const int nr_octaves_per_scale = 3;
	const float min_contrast = 10.0;
	detect_keypoints(model, min_scale, nr_octaves, nr_octaves_per_scale, min_contrast, model_keypoints);
	detect_keypoints(scene, min_scale, nr_octaves, nr_octaves_per_scale, min_contrast, scene_keypoints);
	std::cout << "Model keypoints: " << model_keypoints->size() <<  std::endl;
	std::cout << "Scene keypoints: " << scene_keypoints->size() << std::endl;

	// Compute PFH features
	const float feature_radius = 0.08;
	compute_PFH_features_at_keypoints(model_downsampled, model_normal, model_keypoints, feature_radius, model_descr);
	compute_PFH_features_at_keypoints(scene_downsampled, scene_normal, scene_keypoints, feature_radius, scene_descr);

}
	
#endif