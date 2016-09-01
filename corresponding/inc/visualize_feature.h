#ifndef VISUALIZE_FEATURE_H
#define VISUALIZE_FEATURE_H

#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/registration/transforms.h>

using pcl::visualization::PointCloudColorHandlerGenericField;
using pcl::visualization::PointCloudColorHandlerCustom;

void visualize_keypoints(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr points,
	const pcl::PointCloud<pcl::PointWithScale>::Ptr keypoints)
{
	// Add the points to the vizualizer
	pcl::visualization::PCLVisualizer viz;
	viz.addPointCloud(points, "points");

	// Draw each keypoint as a sphere
	for (size_t i = 0; i < keypoints->size(); ++i)
	{
		// Get the point data
		const pcl::PointWithScale & p = keypoints->points[i];

		// Pick the radius of the sphere *
		float r = 2 * p.scale;
		// * Note: the scale is given as the standard deviation of a Gaussian blur, so a
		//   radius of 2*p.scale is a good illustration of the extent of the keypoint

		// Generate a unique string for each sphere
		std::stringstream ss("keypoint");
		ss << i;

		// Add a sphere at the keypoint
		viz.addSphere(p, 2 * p.scale, 1.0, 0.0, 0.0, ss.str());
	}

	// Give control over to the visualizer
	viz.spin();
}

void visualize_correspondences(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr points1,
								const pcl::PointCloud<pcl::PointWithScale>::Ptr keypoints1,
								const pcl::PointCloud<pcl::PointXYZRGB>::Ptr points2,
								const pcl::PointCloud<pcl::PointWithScale>::Ptr keypoints2,
								const std::vector<int> &correspondences,
								const std::vector<float> &correspondence_scores)
{
	// We want to visualize two clouds side-by-side, so do to this, we'll make copies of the clouds and transform them
	// by shifting one to the left and the other to the right.  Then we'll draw lines between the corresponding points

	// Create some new point clouds to hold our transformed data
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr points_left(new pcl::PointCloud<pcl::PointXYZRGB>);
	pcl::PointCloud<pcl::PointWithScale>::Ptr keypoints_left(new pcl::PointCloud<pcl::PointWithScale>);
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr points_right(new pcl::PointCloud<pcl::PointXYZRGB>);
	pcl::PointCloud<pcl::PointWithScale>::Ptr keypoints_right(new pcl::PointCloud<pcl::PointWithScale>);

	// Shift the first clouds' points to the left
	//const Eigen::Vector3f translate (0.0, 0.0, 0.3);
	const Eigen::Vector3f translate(0.4, 0.0, 0.0);
	const Eigen::Quaternionf no_rotation(0, 0, 0, 0);
	pcl::transformPointCloud(*points1, *points_left, -translate, no_rotation);
	pcl::transformPointCloud(*keypoints1, *keypoints_left, -translate, no_rotation);

	// Shift the second clouds' points to the right
	pcl::transformPointCloud(*points2, *points_right, translate, no_rotation);
	pcl::transformPointCloud(*keypoints2, *keypoints_right, translate, no_rotation);

	// Add the clouds to the vizualizer
	pcl::visualization::PCLVisualizer viz;
	viz.addPointCloud(points_left, "points_left");
	viz.addPointCloud(points_right, "points_right");

	// Compute the median correspondence score
	std::vector<float> temp(correspondence_scores);
	std::sort(temp.begin(), temp.end());
	float median_score = temp[temp.size() / 2];

	// Draw lines between the best corresponding points
	for (size_t i = 0; i < keypoints_left->size(); ++i)
	{
		if (correspondence_scores[i] > median_score)
		{
			continue; // Don't draw weak correspondences
		}

		// Get the pair of points
		const pcl::PointWithScale & p_left = keypoints_left->points[i];
		const pcl::PointWithScale & p_right = keypoints_right->points[correspondences[i]];

		// Generate a random (bright) color
		double r = (rand() % 100);
		double g = (rand() % 100);
		double b = (rand() % 100);
		double max_channel = std::max(r, std::max(g, b));
		r /= max_channel;
		g /= max_channel;
		b /= max_channel;

		// Generate a unique string for each line
		std::stringstream ss("line");
		ss << i;

		// Draw the line
		viz.addLine(p_left, p_right, r, g, b, ss.str());
	}

	// Give control over to the visualizer
	viz.spin();
}

#endif