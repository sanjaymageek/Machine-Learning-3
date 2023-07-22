from tkinter import *
from tkinter import ttk
import tkinter.messagebox
import cv2
import dlib
import numpy as np
from imutils import face_utils
from tkinter import filedialog
import math
import pickle
np.set_printoptions(suppress=True)
# Load the pickled model


def isEyebrow(d1,d2):
    if(int(d1)<int(d2)):
        return True
    else:
        return False
def isLips(d1,d2):
    if(int(d1)<int(d2)):
        return False
    else:
        return True


def mentalState(smile,puck,nod,yaw,roll,eyebrow,lips):
    filename="model_11.sav"
    model_from_pickle = pickle.load(open(filename,"rb"))
# Use the loaded pickled model to make predictions
    x=[smile,puck,nod,yaw,roll,eyebrow,lips]
    for i in range(len(x)):
        x[i]= 1 if(x[i]==True) else 0
    X=[x]

    return [model_from_pickle.predict_proba(X),model_from_pickle.predict(X)]

def calDis(x1,x2):
     dist = math.sqrt((x2[0] - x1[0])**2 + (x2[1] - x1[1])**2)
     return dist

def isnod(sumX,gesture_threshold):
    if(sumX>gesture_threshold):
        return True
    else:
        return False
def isyaw(sumY,gesture_thresholdY):
    if(sumY> gesture_thresholdY ):
        return True
    else:
        return False

max_head_movement = 20
movement_threshold = 50
gesture_threshold = 75
gesture_thresholdY = 50
def fileDialog():

    filename = filedialog.askopenfilename( title = "Select A File", filetype =(("mp4 Files","*.mp4"),("all files","*.*")) )
    realTime(name=filename)

face_landmark_path = './shape_predictor_68_face_landmarks.dat'

K = [6.5308391993466671e+002, 0.0, 3.1950000000000000e+002,
     0.0, 6.5308391993466671e+002, 2.3950000000000000e+002,
     0.0, 0.0, 1.0]
D = [7.0834633684407095e-002, 6.9140193737175351e-002, 0.0, 0.0, -1.3073460323689292e+000]

cam_matrix = np.array(K).reshape(3, 3).astype(np.float32)
dist_coeffs = np.array(D).reshape(5, 1).astype(np.float32)

object_pts = np.float32([[6.825897, 6.760612, 4.402142],
                         [1.330353, 7.122144, 6.903745],
                         [-1.330353, 7.122144, 6.903745],
                         [-6.825897, 6.760612, 4.402142],
                         [5.311432, 5.485328, 3.987654],
                         [1.789930, 5.393625, 4.413414],
                         [-1.789930, 5.393625, 4.413414],
                         [-5.311432, 5.485328, 3.987654],
                         [2.005628, 1.409845, 6.165652],
                         [-2.005628, 1.409845, 6.165652],
                         [2.774015, -2.080775, 5.048531],
                         [-2.774015, -2.080775, 5.048531],
                         [0.000000, -3.116408, 6.097667],
                         [0.000000, -7.415691, 4.070434]])

reprojectsrc = np.float32([[10.0, 10.0, 10.0],
                           [10.0, 10.0, -10.0],
                           [10.0, -10.0, -10.0],
                           [10.0, -10.0, 10.0],
                           [-10.0, 10.0, 10.0],
                           [-10.0, 10.0, -10.0],
                           [-10.0, -10.0, -10.0],
                           [-10.0, -10.0, 10.0]])

line_pairs = [[0, 1], [1, 2], [2, 3], [3, 0],
              [4, 5], [5, 6], [6, 7], [7, 4],
              [0, 4], [1, 5], [2, 6], [3, 7]]



def get_head_pose(shape):
    image_pts = np.float32([shape[17], shape[21], shape[22], shape[26], shape[36],
                            shape[39], shape[42], shape[45], shape[31], shape[35],
                            shape[48], shape[54], shape[57], shape[8]])

    _, rotation_vec, translation_vec = cv2.solvePnP(object_pts, image_pts, cam_matrix, dist_coeffs)

    reprojectdst, _ = cv2.projectPoints(reprojectsrc, rotation_vec, translation_vec, cam_matrix,
                                        dist_coeffs)

    reprojectdst = tuple(map(tuple, reprojectdst.reshape(8, 2)))

    # calc euler angle
    rotation_mat, _ = cv2.Rodrigues(rotation_vec)
    pose_mat = cv2.hconcat((rotation_mat, translation_vec))
    _, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(pose_mat)

    return reprojectdst, euler_angle


def realTime(rec=0,name=None):
    nod=False
    yaw=False
    roll=False
    smile=False
    puck=False
    # return
    count=0
    font=cv2.FONT_HERSHEY_SIMPLEX
    sumX=0
    sumY=0
    gestDur=50
    gesture=False
    if(name==None):
        cap = cv2.VideoCapture(0)
    else:

        cap=cv2.VideoCapture(name)
    if(rec!=0):
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640,480))

    if not cap.isOpened():
        print("Unable to connect to camera.")
        return
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(face_landmark_path)

    while cap.read()[0]:
        ret, frame = cap.read()
        frame=cv2.flip(frame,1,0)
        if ret:
            face_rects = detector(frame, 0)

            if len(face_rects) > 0:
                shape = predictor(frame, face_rects[0])
                shape = face_utils.shape_to_np(shape)

                reprojectdst, euler_angle = get_head_pose(shape)
                cX = int((shape[51][0] + shape[57][0])/2)
                cY = int((shape[51][1] + shape[57][1])/2)
                # put text and highlight the center
                bX = int((shape[23][0] + shape[25][0])/2)
                bY = int((shape[23][1] + shape[25][1])/2)

                eX = int((shape[42][0] + shape[ 45][0])/2)
                eY = int((shape[42][1] + shape[45][1])/2)

                d1=calDis(tuple(shape[48]),(cX,cY))
                d2=calDis(tuple(shape[54]),(cX,cY))
                # isEyebrow
                # d3=calDis(tuple(shape[48]),(cX,cY))
                d4=calDis((eX,eY),(bX,bY))
                # isEyebrow()
                d5=calDis(tuple(shape[33]),tuple(shape[57]))
                if(count==0):
                    df1=d1
                    df2=d2
                    dB=d4
                    dL=d5
                    count=1
                    # display the image
                # print(dB,d4)
                eyebrow=isEyebrow(dB,d4)
                lips=isLips(dL,d5)
                # print()
                # eyebrow=False
                # lips=False
                if(df1<d1 and df2<d2):
                    smile=True
                    puck=False
                if(df1>d1 and df2>d2):
                    puck=True
                    smile=False
                # if(dB>d4):
                #     eyebrow=True
                # if(dL>d5):
                #     lips=True

                if(abs(euler_angle[0,0])>8 ):
                    sumX+=abs(euler_angle[0,0])
                if(abs(euler_angle[1,0])>15):
                   sumY+=abs(euler_angle[1,0])
                if(gestDur):
                    nod=isnod(sumX,gesture_threshold)
                    yaw=isyaw(sumY,gesture_thresholdY)
                    gestDur-=1
                if(gestDur==0):
                    nod,yaw=False,False
                    sumX=0
                    sumY=0
                    gestDur=50

                roll= True if(abs(euler_angle[2,0])>10) else False
                state=mentalState(smile,puck,nod,yaw,roll,eyebrow,lips)
                ment= state[1][0]
                l=list(state[0][0])
                for i in range(len(l)):
                    x="%.2f" % (l[i]/1*100)
                    l[i]=x
                state=l
                #print(state)
                #print(smile,"Smile\n",puck,"Puck\n",nod,"Nod\n",yaw,"yaw\n",roll,"roll\n")
                # print(l,state)

                cv2.putText(frame, "Press 'p' to  Pause The feed" , (370, 20), font,
                            0.5, (255,0,0), thickness=1)
                cv2.putText(frame, "Press 'c' to  Continue The feed" , (370, 40), font,
                            0.5, (255,0,0), thickness=1)
                cv2.putText(frame, "Press 'q' to Quit" , (370, 60),font,
                            0.5, (255,0,0), thickness=1)
                cv2.putText(frame, "Don't Close The window" , (370,80), font,
                            0.5, (0,255,0), thickness=1)

                # cv2.putText(frame, str(smile)+"Smile\n"+str(puck)+"Puck\n"+str(nod)+"Nod\n"+str(yaw)+"yaw\n"+str(roll)+"roll\n"+str(eyebrow)+"Eyebrow\n"+str(lips)+"lips", (50,300), cv2.FONT_HERSHEY_SIMPLEX,
                #             0.5, (0,255,0), thickness=1)
                if(ment=="A"):
                    cv2.putText(frame, "Agreement: " + str(state[0])+"%", (20, 20), font,
                            0.75, (255,255,0), thickness=2)
                    cv2.putText(frame, "Confused: " + str(state[1])+"%", (20, 60), font,
                            0.75, (255,255,255), thickness=2)
                    cv2.putText(frame, "DisAgreement: " + str(state[2])+"%", (20, 40), font,
                                0.75, (255,255,255), thickness=2)
                    cv2.putText(frame, "Uninterested: " + str(state[3])+"%", (20, 80), font,
                                0.75, (255,255,255), thickness=2)
                if(ment=="C"):
                    cv2.putText(frame, "Confused: " + str(state[1])+"%", (20, 60), font,
                            0.75, (255,255,0), thickness=2)
                    cv2.putText(frame, "Agreement: " + str(state[0])+"%", (20, 20), font,
                            0.75, (255,255,255), thickness=2)
                    cv2.putText(frame, "DisAgreement: " + str(state[2])+"%", (20, 40), font,
                                0.75, (255,255,255), thickness=2)
                    cv2.putText(frame, "Uninterested: " + str(state[3])+"%", (20, 80), font,
                                0.75, (255,255,255), thickness=2)

                if(ment=="D"):
                    cv2.putText(frame, "DisAgreement: " + str(state[2])+"%", (20, 40), font,
                                0.75, (255,255,0), thickness=2)
                    cv2.putText(frame, "Agreement: " + str(state[0])+"%", (20, 20), font,
                                0.75, (255,255,255), thickness=2)
                    cv2.putText(frame, "Confused: " + str(state[1])+"%", (20, 60), font,
                            0.75, (255,255,255), thickness=2)
                    cv2.putText(frame, "Uninterested: " + str(state[3])+"%", (20, 80), font,
                                0.75, (255,255,255), thickness=2)
                if(ment=="N"):
                    cv2.putText(frame, "DisAgreement: " + str(state[2])+"%", (20, 40), font,
                                0.75, (255,255,255), thickness=2)
                    cv2.putText(frame, "Agreement: " + str(state[0])+"%", (20, 20), font,
                                0.75, (255,255,255), thickness=2)
                    cv2.putText(frame, "Confused: " + str(state[1])+"%", (20, 60), font,
                            0.75, (255,255,255), thickness=2)
                    cv2.putText(frame, "Uninterested: " + str(state[3])+"%", (20, 80), font,
                                0.75, (255,255,0), thickness=2)



                #cv2.putText(frame, "x: " + "{:7.2f}".format(euler_angle[0, 0]), (20, 20), font,
                #            0.75, (255,255,255), thickness=2)
                #cv2.putText(frame, "y: " + "{:7.2f}".format(euler_angle[1, 0]), (20, 50), font,
                #            0.75, (255,255,255), thickness=2)
                #cv2.putText(frame, "z: " + "{:7.2f}".format(euler_angle[2, 0]), (20, 80), font,
                #            0.75, (255,255,255), thickness=2)
                key = cv2.waitKey(1) & 0xff
                if key == ord('p'):

                    while True:
                        key2 = cv2.waitKey(1) or 0xff


                        if key2 == ord('c'):
                            key="c"
                            break
                        if key2== ord("q"):
                            cap.release()
                            cv2.destroyAllWindows()
                            break
                if key== ord("q"):
                    break
            if(rec!=0):
                out.write(frame)
            resize = cv2.resize(frame, (1280,720))
            cv2.imshow("Mental State Analyser", resize)

            # if key == ord('q'):
            #     break

    cap.release()
    if(rec!=0):
        out.release()
    cv2.destroyAllWindows()
root=Tk()

root.geometry("500x420")
frame = Frame(root, relief=RIDGE, borderwidth=2)
frame.pack(fill=BOTH, expand=1)
root.resizable(0,0)
root.title("Mental State Analyser")
frame.config(background="light blue")
label = Label(frame, text="Mental State Analyser",bg='light blue',font=('Times 35 bold'))
label.pack(side=TOP)
filename = PhotoImage(file="brain.png")
background_label = Label(frame,image=filename)
background_label.pack(side=TOP)

but1=Button(frame,padx=5,pady=5,width=39,bg='white',fg='black',relief=GROOVE,text='RealTime Analysis',command=realTime, font=('helvetica 15 bold'))
but1.place(x=5,y=104)

but2=Button(frame,padx=5,pady=5,width=39,bg='white',fg='black',relief=GROOVE,text='RealTime Analysis and Record',command=lambda a=1,eff=None:realTime(a),font=('helvetica 15 bold'))
but2.place(x=5,y=176)

but3=Button(frame,padx=5,pady=5,width=39,bg='white',fg='black',relief=GROOVE,text='Browse a Video',command=fileDialog,font=('helvetica 15 bold'))
but3.place(x=5,y=250)

but4=Button(frame,padx=5,pady=5,width=5,bg='white',fg='black',relief=GROOVE,text='EXIT',command=root.quit,font=('helvetica 15 bold'))
but4.place(x=215,y=320)

root.mainloop()
