import face_recognition as fr
import cv2
import numpy as np
import os
from datetime import datetime as dt

face_dataset="face_dataset"         # path of image dataset
images=[]
class_names=[]
mylist=os.listdir(face_dataset)
print('Processing...')

# ----------------read all images from dataset into list----------------------------------------------------------

for fx in mylist:
    current_img=cv2.imread(f'{face_dataset}/{fx}')
    images.append(current_img)
    class_names.append(os.path.splitext(fx)[0])

# ----------------------------------sort attendace according to date----------------------------------------------

def sorted_date():
    with open('attendance_list.csv','r') as al:
        next(al)
        data_list=al.readlines()

        for data in data_list:
            x=data.split(',')
            entry_date=str(x[2])[:10]
            entry_name=str(x[0])

            with open(f'attendance/{entry_date}.csv','a+') as sd:
                y=f'{x[0]},{x[1]},{entry_date}\n'

            with open(f'attendance/{entry_date}.csv','r+') as sd:
                data_list1=sd.readlines()
                name_list1=[]

                for data1 in data_list1:
                    entry1=data1.split(',')
                    name_list1.append(str(entry1[0]))

                if entry_name not in name_list1:
                    sd.writelines(y)
                    sd.flush()

# ----------------------------------calculate percentage of attendance of students--------------------------------

def attendance_perecentage():
    total_days=os.listdir('attendance')
    total_days=len(total_days)
    name_attendance=dict()
    total_attendance=[]

    total_faces=os.listdir('face_dataset')
    faces=[]
    for fx in total_faces:
        faces.append(os.path.splitext(fx)[0])

    with open('students_attendance_perecentage.csv','w') as sap:
        with open('attendance_list.csv','r') as al:
            sap.writelines('Name,AttendancePercentage')
            attendace_list=al.readlines()
            attendace_list.pop(0)
            for fx in attendace_list:
                x=fx.split(',')
                total_attendance.append(x[0])
            for fx in total_attendance:
                name_attendance[fx]=str(round(total_attendance.count(fx)*100/total_days, 2))+ '%'
            for fx,yz in name_attendance.items():
                sap.writelines(f'\n{fx},{yz}')
                al.flush()

# ----------------mark attendance in attendance_list.csv----------------------------------------------------------

def markattendance(name):
    with open ('attendance_list.csv','r+') as f:
        data_list= f.readlines()
        name_list=[]
        name_date=dict()
        # print(data_list)
        for data in data_list:
            entry=data.split(',')
            name_list.append(entry[0])
            name_date[str(entry[0])]=str(entry[2])[:10]

        current=dt.now()
        dt_string=current.strftime('%H:%M:%S')
        x=str(dt.today())
        if name not in name_list:
            f.writelines(f'\n{name},{dt_string},{x[:10]}')
            f.flush()

        if name in name_list:            
            if name_date[name]!= x[:10]:
                f.writelines(f'\n{name},{dt_string},{x[:10]}')
                f.flush()

        sorted_date()
        attendance_perecentage()

# ---------------------------------------use camera for face recognition------------------------------------------

def findencoding(images):
    encode_list=[]
    for img in images:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = fr.face_encodings(img)[0]
        encode_list.append(encode)
    return encode_list

encode_list_known = findencoding(images)
print("encoding completed")

cap=cv2.VideoCapture(0)

while True:
    ret,img=cap.read()
    imgs=cv2.resize(img,(0,0),None,0.25,0.25)
    imgs=cv2.cvtColor(imgs,cv2.COLOR_BGR2RGB)
    faces_cur_frame=fr.face_locations(imgs)
    encode_cur_frame = fr.face_encodings(imgs,faces_cur_frame)

    for encode_face,face_loc in zip(encode_cur_frame,faces_cur_frame):
        matches=fr.compare_faces(encode_list_known,encode_face)
        face_dis= fr.face_distance(encode_list_known,encode_face)
        match_index=np.argmin(face_dis)
        if matches[match_index]:
            name=class_names[match_index]
            markattendance(name)
            y1,x2,y2,x1=face_loc
            y1,x2,y2,x1=y1*4,x2*4,y2*4,x1*4
            cv2.putText(img,'Press SPACEBAR TO Exit',(200,20),cv2.FONT_HERSHEY_SIMPLEX,0.75,(255,255,0),2)
            cv2.putText(img,'-'*len('Press SPACEBAR TO Exit'),(200,33),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,0),2)
            cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0),2)
            cv2.putText(img,name,(x1+6,y2+23),cv2.FONT_HERSHEY_COMPLEX,0.6,(0,0,255),2)
    cv2.imshow('camera',img)
    if cv2.waitKey(1) & 0xFF == ord(' '):
        print('Exiting...')
        break