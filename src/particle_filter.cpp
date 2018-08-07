/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	num_particles = 500;

	double std_x, std_y, std_theta; 

	std_x = std[0]; 
	std_y = std[1];
	std_theta = std[2];
	
	// Creates a normal (Gaussian) distribution
	normal_distribution<double> dist_x_init(x, std_x);
	normal_distribution<double> dist_y_init(y, std_y);
	normal_distribution<double> dist_theta_init(theta, std_theta);

	for (int i=0; i < num_particles; i++) 
	{
		Particle p;
		p.id = i;
		p.x = dist_x_init(gen);
		p.y = dist_y_init(gen);
		p.theta = dist_theta_init(gen);
		p.weight = 1.0;
		particles.push_back(p);
	}
	
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	normal_distribution<double> dist_x_pred(0, std_pos[0]);
	normal_distribution<double> dist_y_pred(0, std_pos[1]);
	normal_distribution<double> dist_theta_pred(0, std_pos[2]);
	
	for (int i=0; i < num_particles; i++) 
	{
		if (yaw_rate != 0)
		{
			particles[i].x += velocity/yaw_rate * (sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta));
			particles[i].y += velocity/yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate * delta_t));
			particles[i].theta += yaw_rate * delta_t;
		}
		else
		{
			particles[i].x += velocity * delta_t * cos(particles[i].theta);
			particles[i].y += velocity * delta_t * sin(particles[i].theta);
		}
		
		particles[i].x += dist_x_pred(gen);
		particles[i].y += dist_y_pred(gen);
		particles[i].theta += dist_theta_pred(gen);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

	// To associate each observation in map from one single particle to the closest landmark, and assign this landmark's ID to this observation in map.
	for (unsigned int i=0; i<observations.size();i++)
	{
		double closest_err = numeric_limits<double>::max();

		int closest_id = -1;

		for (unsigned int j=0; j<predicted.size();j++)
		{
			double err = dist(predicted[j].x, predicted[j].y, observations[i].x, observations[i].y);
			if (err < closest_err)
			{
				closest_err = err;
				closest_id = predicted[j].id;
			}
		}
		observations[i].id = closest_id;
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
	
	for (int i=0; i<num_particles; i++)
	{
		// vector predictd stores all landmark position and id which are within sensor range of one single particle.
		vector<LandmarkObs> predicted;
		
		for (unsigned int j=0; j<map_landmarks.landmark_list.size(); j++) 
		{
			double dist_p_lm = dist(particles[i].x, particles[i].y, map_landmarks.landmark_list[j].x_f, map_landmarks.landmark_list[j].y_f);
			if (dist_p_lm <= sensor_range)
			{
				LandmarkObs temp;

				temp.id = map_landmarks.landmark_list[j].id_i;
				temp.x = map_landmarks.landmark_list[j].x_f;
				temp.y = map_landmarks.landmark_list[j].y_f;

				predicted.push_back(temp);
			}
		}
		
		// vector tran_observations stores all observation positions in map from one single particle.
		vector<LandmarkObs> tran_observations;
		
		for (unsigned int k=0; k<observations.size(); k++)
		{
			LandmarkObs temp_1;
			temp_1.x = particles[i].x + cos(particles[i].theta) * observations[k].x - sin(particles[i].theta) * observations[k].y;
			temp_1.y = particles[i].y + sin(particles[i].theta) * observations[k].x + cos(particles[i].theta) * observations[k].y;
			temp_1.id = observations[k].id;
			tran_observations.push_back(temp_1);
		}

		dataAssociation(predicted, tran_observations);

		particles[i].weight = 1.0;

		for (unsigned int l=0; l<tran_observations.size(); l++)
		{	
			double tran_obs_x = tran_observations[l].x;
			double tran_obs_y = tran_observations[l].y;
			
			double asso_lm_x, asso_lm_y;
			
			// Attention: landmark ID and landmark index in landmark_list are totally different things!! At first I didn't notice this and spent a lot of time debugging other places.
			for (unsigned int m=0; m<map_landmarks.landmark_list.size(); m++)
			{
				if (map_landmarks.landmark_list[m].id_i == tran_observations[l].id)
				{
					asso_lm_x = map_landmarks.landmark_list[m].x_f;
					asso_lm_y = map_landmarks.landmark_list[m].y_f;
					break;
				}
			}

			particles[i].weight *= 1/(2 * M_PI * std_landmark[0] * std_landmark[1]) * exp(-0.5 * (pow((tran_obs_x - asso_lm_x)/std_landmark[0], 2) + pow((tran_obs_y - asso_lm_y)/std_landmark[1], 2)));
 		}
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	
	vector<Particle> new_particles;

	vector<double> weight_list;
	double weight_max = numeric_limits<double>::min();

	for (int i=0; i<num_particles; i++)
	{
		weight_list.push_back(particles[i].weight);
		if (particles[i].weight > weight_max)
		{
			weight_max = particles[i].weight;
		}
	}
	
	uniform_int_distribution<int> distribution_index(0, num_particles-1);
	int index = distribution_index(gen);

	double beta = 0.;
	
	uniform_real_distribution<double> distribution_beta(0., 2.*weight_max);
	
	for (int j=0; j<num_particles; j++)
	{
		beta += distribution_beta(gen);
		while (beta > weight_list[index])
		{
			beta -= weight_list[index];
			index = (index + 1) % num_particles;
		}
		new_particles.push_back(particles[index]);
	}
	particles = new_particles;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
	

	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
