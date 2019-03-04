#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <iostream>

using namespace std;
using namespace cv;

void paste(cv::Mat dst, cv::Mat src, int x, int y, int width, int height){
  cv::Mat resized_img;
	cv::resize(src, resized_img, cv::Size(width, height));

	if (x >= dst.cols || y >= dst.rows) return;
	int w = (x >= 0) ? std::min(dst.cols - x, resized_img.cols) : std::min(std::max(resized_img.cols + x, 0), dst.cols);
	int h = (y >= 0) ? std::min(dst.rows - y, resized_img.rows) : std::min(std::max(resized_img.rows + y, 0), dst.rows);
	int u = (x >= 0) ? 0 : std::min(-x, resized_img.cols - 1);
	int v = (y >= 0) ? 0 : std::min(-y, resized_img.rows - 1);
	int px = std::max(x, 0);
	int py = std::max(y, 0);

	cv::Mat roi_dst = dst(cv::Rect(px, py, w, h));
	cv::Mat roi_resized = resized_img(cv::Rect(u, v, w, h));
	roi_resized.copyTo(roi_dst);
}

void PlaySound(string file){
	static char cmd[256];

	if (strlen(file.c_str()) > 200) return;	// buffer overflow prevention
	sprintf(cmd, "paplay %s.wav &> /dev/null &", file.c_str());
	system(cmd);
}


std::pair<Point2f, float> findMinEnclosingCircle(const vector<Point>& goodPolyCurve){
  std::pair<Point2f, float> c;
    
  if(goodPolyCurve.size() > 0){
    minEnclosingCircle(goodPolyCurve, c.first, c.second);    
  }
  return c;
}

float getDistance(Point a, Point b){
	float d= sqrt(fabs( pow(a.x-b.x,2) + pow(a.y-b.y,2) )) ;  
	return d;
}

float getRadian(Point s, Point f, Point e){
	float l1 = getDistance(f,s);
	float l2 = getDistance(f,e);
	float dot=(s.x-f.x)*(e.x-f.x) + (s.y-f.y)*(e.y-f.y);
	float angle = acos(dot/(l1*l2));
	angle=angle;
	return angle;
}

vector<int> k_curvature(vector<Point> contour, vector<Vec4i> defects, int k, float threshold){
	vector<float> curvature = vector<float>(contour.size());
  vector<int> one_fin_defects;

	for (int i = 0; i < defects.size(); i++) {
		for(int j=-5; j<=5; j++){
        Point p0 = contour[defects[i][2] - k + j];
        Point p1 = contour[defects[i][2] + j];
        Point p2 = contour[defects[i][2] + k + j];
        curvature[i] = getRadian(p0, p1, p2);
        if (curvature[i] < threshold){
          one_fin_defects.push_back(i);
        }
		}
	}
	return one_fin_defects;
}

std::pair<Point, double> findMaxInscribedCircle(
        const vector<Point> & polyCurves, 
        const Mat& frame)
{
  std::pair<Point, double> c;
  double dist    = -1;
  double maxdist = -1;

  if (polyCurves.size() > 0) {
    for (int i = 0; i < frame.cols; i+=10) {
      for (int j = 0; j < frame.rows; j+=10) {
        dist = pointPolygonTest(polyCurves, Point(i,j), true);
        if (dist > maxdist) {
          maxdist = dist;
          c.first = Point(i,j);
        }
      }
    }
    c.second = maxdist;
  }

  return c;
}


int main(int argc, char *argv[])
{
  std::string cascadeName = "/usr/share/opencv/haarcascades/haarcascade_frontalface_alt.xml";
  cv::CascadeClassifier cascade;
  if(!cascade.load(cascadeName)){
    printf("ERROR: cascadeFile not found\n");
    return -1;
  }

  string state;
  int set_number = 20;
  std::vector<string> state_set(set_number+1);

  string most_common;
  string pre_most_common;
  string jpg = ".jpg";
  string png = ".png";

  cv::VideoCapture cap;
  cv::Mat frame;
  cv::Mat dst_img, msk_img;
  cv::Mat hsv_image;
  cv::Mat hsv_back, msk_back;
  cv::Mat dst, diff, gray;
  Mat picture;
  Point pre_bye;

  int count_open = 0;
  int count_bye = 0;
  int shut_down = 0;

  int take_picture = 0;
  int sound_ok;
  int sound_good;

  // 1. VideoCapture
  std::string input_index;
  if (argc >= 2){
    input_index = argv[1];
    cap.open(input_index);
  }else{
    cap.open(1);
  }
  if (!cap.isOpened()){
    cap.open(0);
  }else{
    printf("USB camera opened.\n");
  }
  if (!cap.isOpened()){
    printf("Cannot open the video.\n");
    exit(0);
  }

  // 2.prepare window for showing images
  cv::namedWindow("Input", 1);
  cv::namedWindow("finger", 1);
  cv::namedWindow("back", 1);

  cap >> frame;

  cv::Size s = frame.size();
  double scale = 6.8;
  int first = 0;

  bool loop_flag = true;

  while (loop_flag){

    cap >> frame;

    cv::Mat pure_frame = frame.clone();
    cv::Mat real_pure_frame = frame.clone();
    cv::Mat face_frame;
    cv::Mat back_frame;
    frame.copyTo(face_frame);
    frame.copyTo(back_frame);
    // 3. prepare for recognition 
    cv::Mat gray, smallImg(cv::saturate_cast<int>(frame.rows/scale),
    cv::saturate_cast<int>(frame.cols/scale), CV_8UC1);
    // convert to gray scale
    cv::cvtColor(face_frame, gray, CV_BGR2GRAY);
    
    // scale-down the image
    cv::resize(gray, smallImg, smallImg.size(), 0, 0, cv::INTER_LINEAR);
    cv::equalizeHist(smallImg, smallImg);
    
    // detect face using Haar-classifier
    std::vector<cv::Rect> faces;
    // multi-scale face searching
    cascade.detectMultiScale(smallImg, faces, 1.01, 2, CV_HAAR_SCALE_IMAGE, cv::Size(30, 30));
    // face region
    for(int i = 0; i < faces.size(); i++){
      cv::Point center;
      int radius;
      center.x = cv::saturate_cast<int>((faces[i].x + faces[i].width * 0.5) * scale);
      center.y = cv::saturate_cast<int>((faces[i].y + faces[i].height * 0.5) * scale);
      radius = cv::saturate_cast<int>((faces[i].width + faces[i].height) * 0.25 * scale);
    //cover face
      cv::Rect roi_rect(center.x - radius, center.y - radius, radius * 2, radius * 2);
      cv::Mat mosaic = face_frame(roi_rect);
      if(first == 0){
        cv::Mat mosaic_back = back_frame(roi_rect);
        mosaic_back = cv::Scalar(0,0,0);
        cv::cvtColor(back_frame, hsv_back, cv::COLOR_BGR2HSV);
        cv::inRange(hsv_back, cv::Scalar(0, 30, 60), cv::Scalar(20, 150, 255), msk_back);
        first++;
      }
      cv::Mat tmp;
      mosaic = cv::Scalar(0,0,0);
    }

    //converting BGR to HSV and skin color extraction
    if(faces.size()==0){
      cv::cvtColor(face_frame, hsv_image, cv::COLOR_BGR2HSV);
    }else{
      cv::cvtColor(face_frame, hsv_image, cv::COLOR_BGR2HSV);
    }
    if(first == 0){
        cv::cvtColor(frame, hsv_back, cv::COLOR_BGR2HSV);
        cv::inRange(hsv_back, cv::Scalar(0, 30, 60), cv::Scalar(20, 150, 255), msk_back);
        first++;
    }
    imshow("back",msk_back);
    cv::inRange(hsv_image, cv::Scalar(0, 30, 60), cv::Scalar(20, 150, 255), msk_img);
    absdiff(msk_img, msk_back, diff); //Background subtraction
    cv::medianBlur(diff, diff, 21);

    // 4. recognize hand shape
    vector< vector<cv::Point> > contours;
     vector<cv::Vec4i> hierarchy;
    cv::Mat temp = diff.clone();
    cv::findContours(temp, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);

    int maxId=-1;
    int maximum=0;
    int numContours = contours.size();
    for(int i=0;i<numContours;i++){
      if(maximum<contours[i].size()){
        maxId = i;
        maximum = contours[i].size();
      }
    }
    
    vector<cv::Point> hull;
    vector<int> inthull;
    vector<cv::Vec4i> defects;
    vector<cv::Vec4i> fin_defects;
    vector<int> one_fin_defects;
    int fin=0;
    bool one_fin = false;

    if(maxId>=0){

      cv::Moments mu1 = cv::moments( contours[maxId]);
      cv::Point2f mc1 = cv::Point2f( mu1.m10/mu1.m00 , mu1.m01/mu1.m00 );

      pair<Point, double> inscribed = findMaxInscribedCircle(contours[maxId], diff);
      if(inscribed.second>0){
        circle(diff, inscribed.first, inscribed.second, cv::Scalar(100), 2, 4);//内接円
      }else continue;

      pair<Point2f, float> enclose = findMinEnclosingCircle(contours[maxId]);

      cv::convexHull(contours[maxId], hull, false);
      cv::convexHull(contours[maxId], inthull, false);
      if(inthull.size()>3){
        cv::convexityDefects(contours[maxId],inthull,defects);
        int num_defects = defects.size();
        for(int i=0; i<num_defects; i++){
          if(defects[i][3]>inscribed.second && 
          getRadian(contours[maxId][defects[i][0]], contours[maxId][defects[i][2]], contours[maxId][defects[i][1]])<atan(1)*4*95/180
          ){
          fin_defects.push_back(defects[i]);
          fin++;
          circle(diff,contours[maxId][defects[i][0]], 8, cv::Scalar(180,180,180),2,4);
          circle(diff,contours[maxId][defects[i][1]], 8, cv::Scalar(180,180,180),2,4);
          circle(diff,contours[maxId][defects[i][2]], 4, cv::Scalar(100,0,0),2,4);
          }
    
        }
        if(fin==0){
          one_fin_defects = k_curvature(contours[maxId], defects, 8, atan(1)*4*70/180);
          if(one_fin_defects.size() > 0){
            for(int i=0; i<one_fin_defects.size(); i++){
              circle(diff,contours[maxId][defects[one_fin_defects[i]][0]], 4, cv::Scalar(0,0,100),2,4);
              circle(diff,contours[maxId][defects[one_fin_defects[i]][0]-8], 4, cv::Scalar(100,0,0),2,4);
              circle(diff,contours[maxId][defects[one_fin_defects[i]][0]+8], 4, cv::Scalar(100,0,0),2,4);
              one_fin = true;
            }
          }
        }

        // 5. gesture classfication
        if(fin>3){
          if(getDistance(inscribed.first, contours[maxId][fin_defects[2][0]])/inscribed.second>2.6){
            state = "open";
          }else{
            state = "grab";
          }
        }else if(fin == 3){
          if(hierarchy[maxId][2]>=0 && contours[hierarchy[maxId][2]].size() > contours[maxId].size()/15) {
            state = "ok"; 
          }else{
            state = "4";
          }
        }else if(fin == 2){
          if(hierarchy[maxId][2]>=0 && contours[hierarchy[maxId][2]].size() > contours[maxId].size()/15) {
            state = "ok"; 
          }else{
            state = "3";
          }
        }else if(fin == 1){
          if(getRadian(contours[maxId][fin_defects[0][0]],contours[maxId][fin_defects[0][2]],contours[maxId][fin_defects[0][1]]) > atan(1)*4*55/180){
            state = "fox";
          }else{
            state = "peace";
          }
        }else if(fin == 0){
          if(one_fin){
            if(getDistance(contours[maxId][defects[one_fin_defects[0]][2]], inscribed.first) < inscribed.second * 2.9){
              if(getDistance(contours[maxId][defects[one_fin_defects[0]][2]], inscribed.first) > inscribed.second *1.8 ){
                state = "good";
              }else{
                state = "0";
              }
            }else{
              state = "1";
            }
          }else{
            state = "0";
          }
        }
        state_set.insert(state_set.begin(), state);
        if(state_set.size() > set_number){
          state_set.pop_back();
        }
        int max = 0;
        map<string,int> m;

        for ( vector<string>::iterator vi = state_set.begin(); vi != state_set.end(); vi++) {
          m[*vi]++;
          if (m[*vi] > max) {
            max = m[*vi]; 
            most_common = *vi;
          }
        }

        if(most_common == "0"){
          if(m["1"]>set_number/5 || m["good"]>set_number/5){
            if(m["1"]<=m["good"]){
              most_common == "good";
            }else{
              most_common == "1";
            }
          }
        }
        // 6. function for each gesture
        if(most_common == "open"){
          string pic_state = most_common + jpg;
          picture = imread(pic_state,1);
          paste(frame, picture, 10, 250, 220, 220);
        }
        if(most_common == "grab"){
          string pic_state = most_common + jpg;
          picture = imread(pic_state,1);
          paste(frame, picture, 10, 300, 220, 160);
        }
        if(most_common == "4"){
          string pic_state = most_common + jpg;
          picture = imread(pic_state,1);
          paste(frame, picture, 10, 300, 220, 160);
        }
        if(most_common == "ok"){
          string pic_state = most_common + jpg;
          picture = imread(pic_state,1);
          paste(frame, picture, 10, 300, 220, 160);
          sound_ok++;
        }else{
          sound_ok = 0;
        }
        if(most_common == "3"){
          string pic_state = most_common + jpg;
          picture = imread(pic_state,1);
          paste(frame, picture, 10, 300, 220, 160);
        }
        if(most_common == "fox"){
          string pic_state = most_common + jpg;
          picture = imread(pic_state,1);
          paste(frame, picture, 10, 300, 200, 160);
        }
        if(most_common == "peace"){
          string pic_state = most_common + jpg;
          picture = imread(pic_state,1);
          paste(frame, picture, 10, 300, 220, 160);
          take_picture++;
        }else{
          take_picture = 0;
        }
        if(most_common == "good"){
          string pic_state = most_common + jpg;
          picture = imread(pic_state,1);
          paste(frame, picture, 10, 300, 220, 160);
          sound_good++;
        }else{
          sound_good = 0;
        }
        if(most_common == "1"){
          string pic_state = most_common + jpg;
          picture = imread(pic_state,1);
          paste(frame, picture, 10, 300, 220, 160);
        }
        if(most_common == "0"){
          string pic_state = most_common + jpg;
          picture = imread(pic_state,1);
          paste(frame, picture, 10, 300, 220, 160);
        }
        

        if(most_common == "open" && pre_most_common == "open"){
          if(count_bye >= 5){
            shut_down++;
          }
          count_open++;
          if(count_open % 10 == 0 && count_open > 0){
            if(fin_defects.size() > 3){
              if(count_bye == 0){
                pre_bye = contours[maxId][fin_defects[2][0]];
                count_bye++;
              }else{
                if(getDistance(pre_bye, contours[maxId][fin_defects[2][0]]) > 30 *inscribed.second/35){
                  count_bye++;
                }else{
                  count_bye = 0;
                }
                pre_bye = contours[maxId][fin_defects[2][0]];
              }
            }
          }
        }else{
          count_open = 0;
          count_bye = 0;
        }

        putText(frame, most_common, Point(20,100), FONT_HERSHEY_SIMPLEX, 2, Scalar(0,0,255), 1, CV_AA); 
        pre_most_common = most_common;
      }
    }
    if(sound_ok == 10){
      PlaySound(most_common);
    }
    if(sound_good == 10){
      PlaySound(most_common);
    }
    if(take_picture == 20){   
        PlaySound(most_common);
        picture = imread("peace.jpg",1);
        paste(pure_frame, picture, 10, 300, 220, 160);
        cv::imwrite("peace_photo.jpg", pure_frame);
    }

    if(shut_down > 0){
      shut_down++;
      putText(frame, "Thank you", Point(25,190), FONT_HERSHEY_SIMPLEX, 2.5, Scalar(0,0,255), 3, CV_AA);
      putText(frame, "for listening!!", Point(75,280), FONT_HERSHEY_SIMPLEX, 2.5, Scalar(0,0,255), 3, CV_AA);
    }
    if(shut_down > 75){
      loop_flag = false;
    }
    cv::imshow("finger", diff);
    cv::imshow("Input", frame);

    char key = cv::waitKey(10);
    if (key == 27){
      loop_flag = false;
    }else if(key == 32){
      first = 0;
    }
  }
  return 0;
}
