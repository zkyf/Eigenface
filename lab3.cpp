#include <opencv2\opencv.hpp>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

using namespace std;
using namespace cv;

Mat average;
Mat EigenValue;
Mat EigenVec;
Mat A;
Mat L;
Size size;
vector<Mat> eigenfaces;

void train(string path);
void recognize();
Mat averageface(vector<Mat> input);
Mat gendiff(Mat average, vector<Mat> input, vector<Mat> output);

int main(int argc, char* argv[])
{
	if (argc < 2)
	{
		return 0;
	}
	train(argv[1]);
}

void train(string path)
{
	/// Generate file list
	string cmd;
	cmd = "dir " + path + " /B > template.txt";
	system(cmd.c_str());
	fstream __template("template.txt", ios::in);
	if (__template.fail())
	{
		cerr << "Error: fail to get the template face list\n";
		return;
	}
	vector<Mat> templates;

	/// Load templates
	while (true)
	{
		string nextfile;
		__template >> nextfile;
		nextfile = path + "\\" + nextfile;
		if (__template.eof()) break;
		Mat __nextfile;
		__nextfile = imread(nextfile);
		
		if (__nextfile.empty())
		{
			cerr << "Error: failed to load file " << nextfile << endl;
			return;
		}
		Mat __gray;
		cvtColor(__nextfile, __gray, CV_BGR2GRAY);
		__gray.convertTo(__nextfile, CV_64FC1, 1 / 255.0);
		templates.push_back(__nextfile);
	}

	/// Generate average face
	average = averageface(templates);
	//cout << average << endl;
	system("pause");

	/// Get Eigenvectors
	vector<Mat> diff;
	A = gendiff(average, templates, diff);
	//cout << A << endl;
	Mat At;
	At = A.t();
	L = At * A;
	eigen(L, EigenValue, EigenVec);

	// Get Eigenfaces
	eigenfaces.clear();
	for (int i = 0; i < EigenVec.rows; i++)
	{
		Mat __eigen(EigenVec.cols, 1, CV_64FC1, Scalar(0, 0, 0));
		for (int j = 0; j < EigenVec.cols; j++)
		{
			__eigen.at<double>(j, 0) = EigenVec.at<double>(i,j);
			//cout << EigenVec.at<double>(i, j) << " ";
		}
		//cout << endl;
		Mat eigen = A * __eigen;
		Mat toshow(size, CV_64FC1, Scalar(0, 0, 0));
		for (int y = 0; y < size.height; y++)
		{
			for (int x = 0; x < size.width; x++)
			{
				int index = x + y*size.width;
				toshow.at<double>(y, x) = eigen.at<double>(index, 0);
			}
		}
		imshow("eigenface", toshow);
		waitKey();
	}
}

Mat averageface(vector<Mat> input)
{
	Mat output;
	double count = input.size();
	for (vector<Mat>::iterator i = input.begin();
			 i != input.end(); i++)
	{
		if (output.empty())
		{
			output = Mat(i->size().height, i->size().width, CV_64FC1, Scalar(0, 0, 0));
			size = i->size();
		}
		else
		{
			if (i->size() != size)
			{
				resize(*i, *i, size);
			}
		}
		output += *i / count;
	}
	return output;
}

Mat gendiff(Mat average, vector<Mat> input, vector<Mat> output)
{
	int width = size.width;
	int height = size.height;
	int count = input.size();
	int now = 0;
	Mat A(width*height, count, CV_64FC1);
	for (vector<Mat>::iterator i = input.begin();
			 i != input.end(); i++)
	{
		Mat __diff = *i - average;
		output.push_back(__diff);
		for (int y = 0; y < height; y++)
		{
			double *row = __diff.ptr<double>(y);
			for (int x = 0; x < width; x++)
			{
				double *pos = A.ptr<double>(x + y*width);
				pos[now] = row[x];
			}
		}
	}
	return A;
}