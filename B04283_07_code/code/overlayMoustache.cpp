// MOUSTACHE USING MOUTH DETECTOR

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/objdetect/objdetect.hpp"
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char* argv[])
{
    string faceCascadeName = argv[1];
    string mouthCascadeName = argv[2];
    
    CascadeClassifier faceCascade, mouthCascade;
    
    if( !faceCascade.load(faceCascadeName) )
    {
        cerr << "Error loading face cascade file. Exiting!" << endl;
        return -1;
    }

    if( !mouthCascade.load(mouthCascadeName) )
    {
        cerr << "Error loading mouth cascade file. Exiting!" << endl;
        return -1;
    }

    Mat mouthMask = imread(argv[3]);
    
    if ( !mouthMask.data )
    {
        cerr << "Error loading moustache image. Exiting!" << endl;
        return -1;
    }
    
    // Current frame
    Mat frame, frameGray;
    Mat frameROI, mouthMaskSmall;
    Mat grayMaskSmall, grayMaskSmallThresh, grayMaskSmallThreshInv;
    Mat maskedMouth, maskedFrame;
    
    char ch;
    
    // Create the capture object
    // 0 -> input arg that specifies it should take the input from the webcam
    VideoCapture cap(0);
    
    // If you cannot open the webcam, stop the execution!
    if( !cap.isOpened() )
        return -1;
    
    //create GUI windows
    namedWindow("Frame");
    
    // Scaling factor to resize the input frames from the webcam
    float scalingFactor = 0.75;
    
    vector<Rect> faces;
    
    // Iterate until the user presses the Esc key
    while(true)
    {
        // Capture the current frame
        cap >> frame;
        
        // Resize the frame
        resize(frame, frame, Size(), scalingFactor, scalingFactor, INTER_AREA);
        
        // Convert to grayscale
        cvtColor(frame, frameGray, CV_BGR2GRAY);
        
        // Equalize the histogram
        equalizeHist(frameGray, frameGray);
        
        // Detect faces
        faceCascade.detectMultiScale(frameGray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );
        
        vector<Point> centers;
        
        // Look for mouth in the face ROI
        for(int i = 0; i < faces.size(); i++)
        {
            Mat faceROI = frameGray(faces[i]);
            vector<Rect> mouths;
            
            // In each face, detect mouths
            mouthCascade.detectMultiScale(faceROI, mouths, 1.1, 5, 0 |CV_HAAR_SCALE_IMAGE, Size(30, 30));
            
            for(int j = 0; j < mouths.size(); j++)
            {
                Point center( faces[i].x + mouths[j].x + int(mouths[j].width*0.5), faces[i].y + mouths[j].y + int(mouths[j].height*0.5) );
                int radius = int( (mouths[j].width + mouths[j].height)*0.25 );
                //circle( frame, center, radius, Scalar( 0, 255, 0 ), 4, 8, 0 );
                
                // Overlay moustache
                int w = 1.8 * mouths[j].width;
                int h = mouths[j].height;
                int x = faces[i].x + mouths[j].x - 0.2*w;
                int y = faces[i].y + mouths[j].y - 0.65*h;
                
                frameROI = frame(Rect(x,y,w,h));
                resize(mouthMask, mouthMaskSmall, Size(w,h));
                cvtColor(mouthMaskSmall, grayMaskSmall, CV_BGR2GRAY);
                threshold(grayMaskSmall, grayMaskSmallThresh, 245, 255, CV_THRESH_BINARY_INV);
                bitwise_not(grayMaskSmallThresh, grayMaskSmallThreshInv);
                bitwise_and(mouthMaskSmall, mouthMaskSmall, maskedMouth, grayMaskSmallThresh);
                bitwise_and(frameROI, frameROI, maskedFrame, grayMaskSmallThreshInv);
                add(maskedMouth, maskedFrame, frame(Rect(x,y,w,h)));
            }
        }
        
        // Show the current frame
        imshow("Frame", frame);
        
        // Get the keyboard input and check if it's 'Esc'
        // 27 -> ASCII value of 'Esc' key
        ch = waitKey( 30 );
        if (ch == 27) {
            break;
        }
         
    }
    
    // Release the video capture object
    cap.release();
    
    // Close all windows
    destroyAllWindows();
    
    return 1;
}