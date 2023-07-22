package com.singh.tuhina.facialanalysis;

import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Path;
import android.graphics.drawable.BitmapDrawable;
import android.provider.MediaStore;
import android.support.annotation.Nullable;
import android.support.design.widget.TabLayout;
import android.support.design.widget.FloatingActionButton;
import android.support.v4.app.FragmentTransaction;
import android.support.v4.view.PagerAdapter;
import android.support.v7.app.AppCompatActivity;
import android.support.v7.widget.Toolbar;

import android.support.v4.app.Fragment;
import android.support.v4.app.FragmentManager;
import android.support.v4.app.FragmentPagerAdapter;
import android.support.v4.view.ViewPager;
import android.os.Bundle;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.Menu;
import android.view.MenuItem;
import android.view.View;
import android.view.ViewGroup;

import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import com.google.android.gms.tasks.OnSuccessListener;
import com.google.firebase.FirebaseApp;
import com.google.firebase.ml.vision.FirebaseVision;
import com.google.firebase.ml.vision.common.FirebaseVisionImage;
import com.google.firebase.ml.vision.common.FirebaseVisionPoint;
import com.google.firebase.ml.vision.face.FirebaseVisionFace;
import com.google.firebase.ml.vision.face.FirebaseVisionFaceContour;
import com.google.firebase.ml.vision.face.FirebaseVisionFaceDetector;
import com.google.firebase.ml.vision.face.FirebaseVisionFaceDetectorOptions;
import com.google.firebase.ml.vision.face.FirebaseVisionFaceLandmark;

import java.util.List;
import java.util.Random;

public class StaticAnalysis extends AppCompatActivity {
    private SectionsPagerAdapter mSectionsPagerAdapter;
    private ViewPager mViewPager;
    private static final int REQ_CODE = 100;
    private static final String TAG = "MY MESSAGE";
    static ImageView imageView, ivhealthy, ivhappy;
    static Bitmap b, happyb, healthyb;
    static boolean bitmapset = false, happybset = false, healthybset = false;
    static double prob_smile = 0, prob_lefteyeopen = 0, prob_righteyeopen = 0;
    static double eyedist = 0, eardist = 0, cheekdist = 0, noselength = 0, mouthwidth = 0, lipheight = 0;
    Fragment f1 = null, f2 = null, f3 = null;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_static_analysis);

        Toolbar toolbar = findViewById(R.id.toolbar);
        setSupportActionBar(toolbar);
        mSectionsPagerAdapter = new SectionsPagerAdapter(getSupportFragmentManager());
        mViewPager = findViewById(R.id.container);
        mViewPager.setAdapter(mSectionsPagerAdapter);

        TabLayout tabLayout = findViewById(R.id.tabs);

        mViewPager.addOnPageChangeListener(new TabLayout.TabLayoutOnPageChangeListener(tabLayout));
        tabLayout.addOnTabSelectedListener(new TabLayout.ViewPagerOnTabSelectedListener(mViewPager));
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        getMenuInflater().inflate(R.menu.menu_static_analysis, menu);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        int id = item.getItemId();
        if (id == R.id.action_settings) {
            return true;
        }
        return super.onOptionsItemSelected(item);
    }

    public static class PlaceholderFragment extends Fragment {
        private static final String ARG_SECTION_NUMBER = "section_number";
        public PlaceholderFragment() {
        }

        public static PlaceholderFragment newInstance(int sectionNumber) {
            PlaceholderFragment fragment = new PlaceholderFragment();
            Bundle args = new Bundle();
            args.putInt(ARG_SECTION_NUMBER, sectionNumber);
            fragment.setArguments(args);
            return fragment;
        }

        @Override
        public View onCreateView(LayoutInflater inflater, ViewGroup container,
                                 Bundle savedInstanceState) {
            View rootView = null;
            int i = getArguments().getInt(ARG_SECTION_NUMBER);

            if (i==1) {
                rootView = inflater.inflate(R.layout.fragment_static_image, container, false);

                imageView = rootView.findViewById(R.id.img);
                FloatingActionButton btnaddimg = rootView.findViewById(R.id.btnaddimg);

                if (bitmapset)
                    imageView.setImageBitmap(b);

                btnaddimg.setOnClickListener(new View.OnClickListener() {
                    @Override
                    public void onClick(View v) {
                        Intent intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                        getActivity().startActivityForResult(intent, REQ_CODE);
                    }
                });
            }

            else if (i==2) {
                rootView = inflater.inflate(R.layout.fragment_static_healthy, container, false);

                ivhealthy = rootView.findViewById(R.id.ivhealthy);
                if (healthybset)
                    ivhealthy.setImageBitmap(healthyb);

                TextView tveyedist = rootView.findViewById(R.id.tveyedist);
                TextView tveardist = rootView.findViewById(R.id.tveardist);
                TextView tvcheekdist = rootView.findViewById(R.id.tvcheekdist);
                TextView tvnoselength = rootView.findViewById(R.id.tvnoselength);
                TextView tvmouthwidth = rootView.findViewById(R.id.tvmouthwidth);
                TextView tvlipheight = rootView.findViewById(R.id.tvlipheight);
                TextView tvhealth = rootView.findViewById(R.id.tvhealth);

                tveyedist.setText(String.format("%.2f mm", eyedist));
                tveardist.setText(String.format("%.2f mm", eardist));
                tvcheekdist.setText(String.format("%.2f mm", cheekdist));
                tvnoselength.setText(String.format("%.2f mm", noselength));
                tvmouthwidth.setText(String.format("%.2f mm", mouthwidth));
                tvlipheight.setText(String.format("%.2f mm", lipheight));

                int healthindex = 80;
                if (prob_smile!=0 && prob_smile*100<80) {
                    healthindex -= 10;
                }
                if (eardist!=0 && eyedist!=0 && eardist/eyedist<2)
                    healthindex -=5;
                if (eardist!=0 && eyedist!=0 && eardist/eyedist>3)
                    healthindex +=5;
                healthindex = healthindex + new Random().nextInt(10);
                if (healthindex >= 100)
                    healthindex = 99;

                if (!healthybset)
                    tvhealth.setText(Integer.toString(0) + " / 100");
                else
                    tvhealth.setText(Integer.toString(healthindex) + " / 100");
            }

            else if (i==3) {
                rootView = inflater.inflate(R.layout.fragment_static_happy, container, false);

                ivhappy = rootView.findViewById(R.id.ivhappy);
                if (happybset)
                    ivhappy.setImageBitmap(happyb);

                TextView tvsmile = rootView.findViewById(R.id.tvsmile);
                TextView tvwink = rootView.findViewById(R.id.tvwink);
                TextView tvhappy = rootView.findViewById(R.id.tvhappy);

                boolean smile = false;
                if (prob_smile >= 0.5)
                    smile = true;
                tvsmile.setText(Boolean.toString(smile));

                boolean wink = false;
                if (prob_lefteyeopen > 0.5 && prob_righteyeopen < 0.2 || prob_righteyeopen > 0.5 && prob_lefteyeopen < 0.2)
                    wink = true;
                tvwink.setText(Boolean.toString(wink));

                int happyindex = (int) (prob_smile*100);
                if (happybset && happyindex < 20)
                    happyindex += new Random().nextInt(10);
                tvhappy.setText(Integer.toString(happyindex) + " / 100");
            }

            return rootView;
        }
    }

    public class SectionsPagerAdapter extends FragmentPagerAdapter {

        public SectionsPagerAdapter(FragmentManager fm) {
            super(fm);
        }

        @Override
        public Fragment getItem(int position) {
            if (f1 == null && position == 0)
                f1 = PlaceholderFragment.newInstance(1);
            if (f2 == null && position == 1)
                f2 = PlaceholderFragment.newInstance(2);
            if (f3 == null && position == 2)
                f3 = PlaceholderFragment.newInstance(3);
            if (position == 0)
                return f1;
            else if (position == 1)
                return f2;
            else
                return f3;
        }

        @Override
        public int getCount() {
            return 3;
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if (requestCode == REQ_CODE) {
            if (resultCode == RESULT_OK) {
                b = (Bitmap) data.getExtras().get("data");
                imageView.setImageBitmap(b);
                bitmapset = true;
                Log.d(TAG, "image taken");
                faceAnalysis(b);
            }
            else if (resultCode == RESULT_CANCELED) {
                Toast.makeText(StaticAnalysis.this, "Operation cancelled by user.",
                        Toast.LENGTH_SHORT).show();
            }
            else {
                Toast.makeText(StaticAnalysis.this, "Failed to capture image. Retry.",
                        Toast.LENGTH_SHORT).show();
            }
        }
    }

    private void faceAnalysis(Bitmap b) {
        FirebaseApp.initializeApp(this);

        FirebaseVisionFaceDetectorOptions options = new FirebaseVisionFaceDetectorOptions.Builder()
                .setClassificationMode(FirebaseVisionFaceDetectorOptions.ALL_CLASSIFICATIONS)
                .setLandmarkMode(FirebaseVisionFaceDetectorOptions.ALL_LANDMARKS)
                .setContourMode(FirebaseVisionFaceDetectorOptions.ALL_CONTOURS)
                .enableTracking()
                .build();

        FirebaseVisionFaceDetector detector = FirebaseVision.getInstance().getVisionFaceDetector(options);

        FirebaseVisionImage firebaseImage = FirebaseVisionImage.fromBitmap(b);

        detector.detectInImage(firebaseImage).addOnSuccessListener(new OnSuccessListener<List<FirebaseVisionFace>>() {
            @Override
            public void onSuccess(List<FirebaseVisionFace> firebaseVisionFaces) {
                for (FirebaseVisionFace face : firebaseVisionFaces) {
                    Log.d(TAG, "Smiling Prob [" + face.getSmilingProbability() + "]");
                    Log.d(TAG, "Left eye open [" + face.getLeftEyeOpenProbability() + "]");
                    Log.d(TAG, "Right eye open [" + face.getRightEyeOpenProbability() + "]");
                    Log.d(TAG, "left ear: " + face.getLandmark(FirebaseVisionFaceLandmark.LEFT_EAR));
                    if (face.getLandmark(FirebaseVisionFaceLandmark.LEFT_EAR) != null)
                        Log.d(TAG, "leftearpos: " + face.getLandmark(FirebaseVisionFaceLandmark.LEFT_EAR).getPosition());

                    calchappiness(face);
                    calchealthiness(face);

                    contourhappy(face);
                    contourhealthy(face);
                }

                FragmentTransaction ft = getSupportFragmentManager().beginTransaction();
                if (f2 != null)
                    ft.detach(f2).attach(f2);
                if (f3 != null)
                    ft.detach(f3).attach(f3);
                ft.commit();
            }
        });
    }

    private void contourhealthy(FirebaseVisionFace face)
    {
        FirebaseVisionFaceContour contour = face.getContour(FirebaseVisionFaceContour.FACE);
        List<FirebaseVisionPoint> points = contour.getPoints();

        Bitmap mutableBitmap = b.copy(b.getConfig(), true);
        Canvas canvasHealthy = new Canvas(mutableBitmap);
        Paint paint = new Paint(Paint.ANTI_ALIAS_FLAG);
        paint.setColor(Color.parseColor("#904298F5"));
        Path path = new Path();

        path.moveTo(points.get(0).getX(), points.get(0).getY());
        for (FirebaseVisionPoint p : points) {
            path.lineTo(p.getX(), p.getY());
        }
        path.close();

        canvasHealthy.drawPath(path, paint);

        healthyb = mutableBitmap;
        healthybset = true;
        if (ivhealthy != null)
            ivhealthy.setImageBitmap(healthyb);
    }

    private void contourhappy(FirebaseVisionFace face)
    {
        FirebaseVisionFaceContour c1 = face.getContour(FirebaseVisionFaceContour.LEFT_EYE);
        FirebaseVisionFaceContour c2 = face.getContour(FirebaseVisionFaceContour.RIGHT_EYE);
        FirebaseVisionFaceContour c3 = face.getContour(FirebaseVisionFaceContour.NOSE_BRIDGE);
        FirebaseVisionFaceContour c4 = face.getContour(FirebaseVisionFaceContour.NOSE_BOTTOM);
        FirebaseVisionFaceContour c5 = face.getContour(FirebaseVisionFaceContour.LOWER_LIP_BOTTOM);
        FirebaseVisionFaceContour c6 = face.getContour(FirebaseVisionFaceContour.LOWER_LIP_TOP);
        FirebaseVisionFaceContour c7 = face.getContour(FirebaseVisionFaceContour.UPPER_LIP_BOTTOM);
        FirebaseVisionFaceContour c8 = face.getContour(FirebaseVisionFaceContour.UPPER_LIP_TOP);
        FirebaseVisionFaceContour c9 = face.getContour(FirebaseVisionFaceContour.LEFT_EYEBROW_BOTTOM);
        FirebaseVisionFaceContour c10 = face.getContour(FirebaseVisionFaceContour.LEFT_EYEBROW_TOP);
        FirebaseVisionFaceContour c11 = face.getContour(FirebaseVisionFaceContour.RIGHT_EYEBROW_BOTTOM);
        FirebaseVisionFaceContour c12 = face.getContour(FirebaseVisionFaceContour.RIGHT_EYEBROW_TOP);

        List<FirebaseVisionPoint> p1 = c1.getPoints();
        List<FirebaseVisionPoint> p2 = c2.getPoints();
        List<FirebaseVisionPoint> p3 = c3.getPoints();
        List<FirebaseVisionPoint> p4 = c4.getPoints();
        List<FirebaseVisionPoint> p5 = c5.getPoints();
        List<FirebaseVisionPoint> p6 = c6.getPoints();
        List<FirebaseVisionPoint> p7 = c7.getPoints();
        List<FirebaseVisionPoint> p8 = c8.getPoints();
        List<FirebaseVisionPoint> p9 = c9.getPoints();
        List<FirebaseVisionPoint> p10 = c10.getPoints();
        List<FirebaseVisionPoint> p11 = c11.getPoints();
        List<FirebaseVisionPoint> p12 = c12.getPoints();

        Bitmap mutableBitmap = b.copy(b.getConfig(), true);
        Canvas canvasHappy = new Canvas(mutableBitmap);
        Paint paint = new Paint(Paint.ANTI_ALIAS_FLAG);
        paint.setColor(Color.parseColor("#904298F5"));
        Path path = new Path();

        path.moveTo(p1.get(0).getX(), p1.get(0).getY());
        for (FirebaseVisionPoint p : p1) {
            path.lineTo(p.getX(), p.getY());
        }

        path.moveTo(p2.get(0).getX(), p2.get(0).getY());
        for (FirebaseVisionPoint p : p2) {
            path.lineTo(p.getX(), p.getY());
        }

        path.moveTo(p3.get(0).getX(), p3.get(0).getY());
        for (FirebaseVisionPoint p : p3) {
            path.lineTo(p.getX(), p.getY());
        }

        path.moveTo(p4.get(0).getX(), p4.get(0).getY());
        for (FirebaseVisionPoint p : p4) {
            path.lineTo(p.getX(), p.getY());
        }

        path.moveTo(p5.get(0).getX(), p5.get(0).getY());
        for (FirebaseVisionPoint p : p5) {
            path.lineTo(p.getX(), p.getY());
        }

        path.moveTo(p6.get(0).getX(), p6.get(0).getY());
        for (FirebaseVisionPoint p : p6) {
            path.lineTo(p.getX(), p.getY());
        }

        path.moveTo(p7.get(0).getX(), p7.get(0).getY());
        for (FirebaseVisionPoint p : p7) {
            path.lineTo(p.getX(), p.getY());
        }

        path.moveTo(p8.get(0).getX(), p8.get(0).getY());
        for (FirebaseVisionPoint p : p8) {
            path.lineTo(p.getX(), p.getY());
        }

        path.moveTo(p9.get(0).getX(), p9.get(0).getY());
        for (FirebaseVisionPoint p : p9) {
            path.lineTo(p.getX(), p.getY());
        }

        path.moveTo(p10.get(0).getX(), p10.get(0).getY());
        for (FirebaseVisionPoint p : p10) {
            path.lineTo(p.getX(), p.getY());
        }

        path.moveTo(p11.get(0).getX(), p11.get(0).getY());
        for (FirebaseVisionPoint p : p11) {
            path.lineTo(p.getX(), p.getY());
        }

        path.moveTo(p12.get(0).getX(), p12.get(0).getY());
        for (FirebaseVisionPoint p : p12) {
            path.lineTo(p.getX(), p.getY());
        }

        path.close();

        canvasHappy.drawPath(path, paint);

        happyb = mutableBitmap;
        happybset = true;
        if (ivhappy != null)
            ivhappy.setImageBitmap(happyb);
    }

    private void calchappiness(FirebaseVisionFace face)
    {
        prob_smile = face.getSmilingProbability();
        prob_lefteyeopen = face.getLeftEyeOpenProbability();
        prob_righteyeopen = face.getRightEyeOpenProbability();
    }

    private void calchealthiness(FirebaseVisionFace face)
    {
        FirebaseVisionFaceLandmark leftear = face.getLandmark(FirebaseVisionFaceLandmark.LEFT_EAR);
        FirebaseVisionFaceLandmark rightear = face.getLandmark(FirebaseVisionFaceLandmark.RIGHT_EAR);
        FirebaseVisionFaceLandmark leftcheek = face.getLandmark(FirebaseVisionFaceLandmark.LEFT_CHEEK);
        FirebaseVisionFaceLandmark rightcheek = face.getLandmark(FirebaseVisionFaceLandmark.RIGHT_CHEEK);
        FirebaseVisionFaceLandmark lefteye = face.getLandmark(FirebaseVisionFaceLandmark.LEFT_EYE);
        FirebaseVisionFaceLandmark righteye = face.getLandmark(FirebaseVisionFaceLandmark.RIGHT_EYE);
        FirebaseVisionFaceLandmark mouthleft = face.getLandmark(FirebaseVisionFaceLandmark.MOUTH_LEFT);
        FirebaseVisionFaceLandmark mouthright = face.getLandmark(FirebaseVisionFaceLandmark.MOUTH_RIGHT);
        FirebaseVisionFaceLandmark mouthbottom = face.getLandmark(FirebaseVisionFaceLandmark.MOUTH_BOTTOM);
        FirebaseVisionFaceLandmark nosebase = face.getLandmark(FirebaseVisionFaceLandmark.NOSE_BASE);

        eyedist = 6.0;
        if(lefteye!=null && righteye!=null)
        {
            FirebaseVisionPoint pleft = lefteye.getPosition();
            FirebaseVisionPoint pright = righteye.getPosition();
            eyedist = pright.getX() - pleft.getX();
        }
        Log.d(TAG, "calchealthiness:eye " + eyedist);

        cheekdist = 13.0;
        if (leftcheek!=null && rightcheek!=null)
        {
            FirebaseVisionPoint pleft = leftcheek.getPosition();
            FirebaseVisionPoint pright = rightcheek.getPosition();
            cheekdist = pright.getX() - pleft.getX();
        }
        Log.d(TAG, "calchealthiness:cheek " + cheekdist);

        eardist = 14.0;
        if (leftear!=null && rightear!=null)
        {
            FirebaseVisionPoint pleft = leftear.getPosition();
            FirebaseVisionPoint pright = rightear.getPosition();
            eardist = pright.getX() - pleft.getX();
        }

        noselength = 5.0;
        if (nosebase!=null)
        {
            if (lefteye!=null)
            {
                FirebaseVisionPoint ptop = lefteye.getPosition();
                FirebaseVisionPoint pbase = nosebase.getPosition();
                noselength = pbase.getY() - ptop.getY();
            }
            else if (righteye!=null)
            {
                FirebaseVisionPoint ptop = righteye.getPosition();
                FirebaseVisionPoint pbase = nosebase.getPosition();
                noselength = pbase.getY() - ptop.getY();
            }
        }

        mouthwidth = 4.0;
        lipheight = 1.0;
        if (mouthleft!=null && mouthright!=null && mouthbottom!=null)
        {
            FirebaseVisionPoint pleft = mouthleft.getPosition();
            FirebaseVisionPoint pright = mouthright.getPosition();
            FirebaseVisionPoint pbase = mouthbottom.getPosition();
            mouthwidth = pright.getX() - pleft.getX();
            lipheight =  pbase.getY() - pleft.getY();
        }
    }
}
