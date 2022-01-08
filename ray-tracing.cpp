#include <iostream>
#include <fstream>		// to write results to ppm file
#include <atomic>		// to count number of completed threads
#include <chrono>		// to record start and end time and calculate elapsed time
#include <thread>		// split up image into rows and trace in threads
#include <vector>		// used for storing worker threads
#include <mutex>		// lock for some cout stuff that occurs at the end of a thread just before join()

#include "rtweekend.h"
#include "color.h"
#include "hittable_list.h"
#include "sphere.h"
#include "camera.h"
#include "material.h"

using namespace std;
using namespace chrono;

static mutex coutMutex;

// Image
const auto aspect_ratio = 3.0 / 2.0;
const int image_width = 600;
const int image_height = static_cast<int>(image_width / aspect_ratio);
const int samples_per_pixel = 250;
const int max_depth = 50;

// Camera
point3 lookfrom(14,1.5,-5);
point3 lookat(0,0.5,0);
vec3 vup(0,1,0);
auto dist_to_focus = (lookfrom-lookat).length();
auto aperture = 0.1;

camera cam(lookfrom, lookat, vup, 20, aspect_ratio, aperture, dist_to_focus);

// Multithreading
atomic<int> jobsDone{ 0 };
atomic<int> nextJob{ 0 };
const int numThreadsScalar = 3;

// Initialise world and pixelBuffer to null
hittable_list world;
int* pixelBuffer = nullptr;

color ray_color(const ray& r, hittable_list world, int depth)
{
	hit_record record;

	if (world.hit(r, 0.001, infinity, record))
	{
		ray scattered;
		color attenuation;

		if (depth < max_depth && record.mat_ptr->scatter(r, record, attenuation, scattered))
		{
			return attenuation * ray_color(scattered, world, depth + 1);
		}
		else
		{
			return color(0, 0, 0);
		}
	}
	else // hit background, gradient from white to light blue.
	{
		vec3 unit_direction = unit_vector(r.direction());
		double t = 0.5 * unit_direction.y() + 1.0;
		return (1.0 - t) * color(1.0, 1.0, 1.0) + t * color(0.5, 0.7, 1.0);
	}
}

hittable_list random_scene() {
    hittable_list world;

    auto ground_material = make_shared<lambertian>(color(0.11, 0.11, 0.11));
    world.add(make_shared<sphere>(point3(0,-1000,0), 1000, ground_material));

    for (int a = -11; a < 11; a++) {
        for (int b = -11; b < 11; b++) {
            auto choose_mat = random_double();
            point3 center(a + 0.9*random_double(), 0.2, b + 0.9*random_double());

            if ((center - point3(4, 0.2, 0)).length() > 0.9) {
                shared_ptr<material> sphere_material;

                if (choose_mat < 0.8) {
                    // diffuse
                    auto albedo = color::random() * color::random();
                    sphere_material = make_shared<lambertian>(albedo);
                    world.add(make_shared<sphere>(center, 0.2, sphere_material));
                } else if (choose_mat < 0.95) {
                    // metal
                    auto albedo = color::random(0.5, 1);
                    auto fuzz = random_double(0, 0.5);
                    sphere_material = make_shared<metal>(albedo, fuzz);
                    world.add(make_shared<sphere>(center, 0.2, sphere_material));
                } else {
                    // glass
                    sphere_material = make_shared<dielectric>(1.5);
                    world.add(make_shared<sphere>(center, 0.2, sphere_material));
                }
            }
        }
    }

    auto material1 = make_shared<dielectric>(1.5);
    world.add(make_shared<sphere>(point3(4, 1, 0), 1.0, material1));

    auto material2 = make_shared<lambertian>(color(0.15, 0.6, 0.5));
    world.add(make_shared<sphere>(point3(-4, 1, -0.4), 1.0, material2));

    auto material3 = make_shared<metal>(color(0.90, 0.32, 0.64), 0.0);
    world.add(make_shared<sphere>(point3(0, 1, 0), 1.0, material3));

    return world;
}

void WritePpmFile(int* buffer, int width, int height, const string fileName)
{
	ofstream outFile;
	outFile.open(fileName);

	outFile << "P3\n" << width << " " << height << "\n255\n";

	int length = width * height * 3;

	for (int i = 0; i < length; i += 3)
	{
		outFile << buffer[i] << " " << buffer[i + 1] << " " << buffer[i + 2] << "\n";
	}

	outFile.close();
}

void trace()
{
	do
	{
		int y = nextJob++;

		int index = ((image_height - 1) - y) * image_width * 3;
		double oneOverWidth = 1.0 / image_width;
		double oneOverHeight = 1.0 / image_height;

		for (int i = 0; i < image_width; ++i)
		{
            color col(0, 0, 0);
			for (int s = samples_per_pixel - 1; s >= 0; --s)
			{
				double u = double(i + random_double()) * oneOverWidth;
				double v = double(y + random_double()) * oneOverHeight;
				ray r = cam.get_ray(u, v);
				col += ray_color(r, world, 0);
			}

			col /= double(samples_per_pixel);
			col = color(sqrt(col[0]), sqrt(col[1]), sqrt(col[2]));

			pixelBuffer[index++] = int(255.99f * col[0]);
			pixelBuffer[index++] = int(255.99f * col[1]);
			pixelBuffer[index++] = int(255.99f * col[2]);
		}

		++jobsDone;

		if (jobsDone % 10 == 0)
		{
			lock_guard<mutex> lock(coutMutex);
			cout << jobsDone << "\\" << image_height << " jobs finished." << endl;
		}
	} while (nextJob < image_height);
}

int main() {
    
    // Generate world
    world = random_scene();

    time_point<steady_clock> start = steady_clock::now();

	pixelBuffer = new int[image_width * image_height * 3];

	unsigned int numThreads = thread::hardware_concurrency() * numThreadsScalar;
	vector<thread> workers;

	unsigned int nextThreadIndex = 0;
	while (nextThreadIndex < numThreads && nextThreadIndex < image_height)
	{
		workers.emplace_back(thread(trace));
		++nextThreadIndex;
	}

	numThreads = (int)workers.size();
	{
		lock_guard<mutex> lock(coutMutex);
		cout << numThreads << " workers launched to complete " << image_height << " jobs." << endl;
	}
	
	for (auto& worker : workers)
	{
		worker.join();
	}

	duration<double> elapsedSeconds = steady_clock::now() - start;
	cout << "Elapsed time: " << elapsedSeconds.count() << " seconds" << endl;

	string fileName = "";
	cout << "Finished. The output will be saved in the given file." << endl;
	cout << "Please enter the filename (without extension): ";
	getline(cin, fileName);
	string s = string(fileName) + string(".ppm");
	WritePpmFile(pixelBuffer, image_width, image_height, s);
	delete[] pixelBuffer;
}