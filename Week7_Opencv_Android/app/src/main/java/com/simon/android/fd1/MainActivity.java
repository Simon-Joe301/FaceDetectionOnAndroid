package com.simon.android.fd1;


import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.*;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

import android.Manifest;
import android.annotation.SuppressLint;
import android.content.Context;
import android.os.Bundle;
import android.util.Log;
import android.view.SurfaceView;
import android.view.WindowManager;
import android.widget.Toast;
import androidx.appcompat.app.AppCompatActivity;
//import com.tbruyelle.rxpermissions2.RxPermissions;
import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;


public class MainActivity extends AppCompatActivity implements CvCameraViewListener2 {
    private static final String TAG = "OCVSample::Activity";
    private CameraBridgeViewBase mOpenCvCameraView;
    private Mat mIntermediateMat;
    private CascadeClassifier classifier;
    private Mat mGray;
    private Mat mRgba;
    private int mAbsoluteFaceSize = 0;

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS: {
                    Log.i(TAG, "OpenCV loaded successfully");
                    initClassifier();
                    mOpenCvCameraView.setCameraPermissionGranted();
                    mOpenCvCameraView.enableView();

                }
                break;
                default: {
                    super.onManagerConnected(status);
                }
                break;
            }
        }
    };



    /**
     * Called when the activity is first created.
     */
    @Override
    public void onCreate(Bundle savedInstanceState) {
        Log.i(TAG, "called onCreate");
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        setContentView(R.layout.image_manipulations_surface_view);
        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.cjv);
        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
        mOpenCvCameraView.setCameraIndex(CameraBridgeViewBase.CAMERA_ID_FRONT);
        //这里的cjv也是我的项目中JavaCameraView的id，自己改一下
        mOpenCvCameraView.setCvCameraViewListener(this);
    }

    @Override
    public void onPause() {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onResume() {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION, this, mLoaderCallback);
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    public void onDestroy() {
        super.onDestroy();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    public void onCameraViewStarted(int width, int height) {
        mGray = new Mat();
        mRgba = new Mat();
    }

    public void onCameraViewStopped() {
        // Explicitly deallocate Mats
        if (mIntermediateMat != null)
            mIntermediateMat.release();
        mIntermediateMat = null;
        mGray.release();
        mRgba.release();
    }

    public Mat onCameraFrame(CvCameraViewFrame inputFrame) {
//        Mat rgba = inputFrame.rgba();
//        return rgba;
        mRgba = inputFrame.rgba();
        mGray = inputFrame.gray();
        float mRelativeFaceSize = 0.2f;
        if (mAbsoluteFaceSize == 0) {
            int height = mGray.rows();
            if (Math.round(height * mRelativeFaceSize) > 0) {
                mAbsoluteFaceSize = Math.round(height * mRelativeFaceSize);
            }
        }
        MatOfRect faces = new MatOfRect();
        if (classifier != null)
            classifier.detectMultiScale(mGray, faces, 1.1, 2, 2,
                    new Size(mAbsoluteFaceSize, mAbsoluteFaceSize), new Size());
        Rect[] facesArray = faces.toArray();
        Scalar faceRectColor = new Scalar(0, 255, 0, 255);
        for (Rect faceRect : facesArray)
            Imgproc.rectangle(mRgba, faceRect.tl(), faceRect.br(), faceRectColor, 1);
        return mRgba;
    }


    // 初始化人脸级联分类器，必须先初始化
    private void initClassifier() {
        try {
            InputStream is = getResources()
                    .openRawResource(R.raw.lbpcascade_frontalface);
            File cascadeDir = getDir("cascade", Context.MODE_PRIVATE);
            File cascadeFile = new File(cascadeDir, "lbpcascade_frontalface.xml");
            FileOutputStream os = new FileOutputStream(cascadeFile);
            byte[] buffer = new byte[4096];
            int bytesRead;
            while ((bytesRead = is.read(buffer)) != -1) {
                os.write(buffer, 0, bytesRead);
            }
            is.close();
            os.close();
            classifier = new CascadeClassifier(cascadeFile.getAbsolutePath());
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

}
/*
    @SuppressLint("CheckResult")
    private void getPermission(){
        RxPermissions rxPermissions = new RxPermissions(OpenCVActivity.this);
        rxPermissions.request(Manifest.permission.CAMERA).subscribe(aBoolean -> {
            if (aBoolean) {
                initClassifier();
                mOpenCvCameraView.setCameraPermissionGranted();
            } else {
                Toast.makeText(OpenCVActivity.this, "您还没有开启相机权限,请前往设置->应用管理->开启", Toast.LENGTH_SHORT).show();
            }
        });
    }
}*/