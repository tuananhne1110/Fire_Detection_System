import pymongo
from gridfs import GridFS
import cv2
import threading
import os
import numpy as np 


def store_and_display_media(file_path, database_name, filename):
    client = pymongo.MongoClient("mongodb://localhost:27017")
    database_name = database_name
    custom_prefix = "gridfs_media"  # Replace with the desired folder name

    # Create a GridFS instance with the custom prefix
    db = client[database_name]
    fs = GridFS(db, collection=custom_prefix)

    with open(file_path, "rb") as file:
        file_id = fs.put(file, filename=filename)

    print(f"File stored in GridFS with ID: {file_id}")
    print('cai nay la file name',filename)
    # print('cai nay la media file name',media_filename)
    print('store xong r nha')

def dispay_media(database_name,file_name,media_file_path):
    client = pymongo.MongoClient("mongodb://localhost:27017")
    database_name = database_name
    custom_prefix = "gridfs_media.files"
    db = client[database_name]
    collection = custom_prefix
    fs = GridFS(db, collection=custom_prefix)
    print('file name ne sao k kiem dc z',file_name)
    # document = fs.find_one({"filename": "flame.mp4"})
    # if document:
    #     print(f"Found object with filename: {document['filename']}")
    #     # Access other fields as needed
    # else:
    #     print("Object not found.")
    with open(media_file_path, "rb") as file:
        file_id = fs.put(file, filename=file_name)
    media_file = fs.get(file_id)
    temp_media_filename = file_name
    with open(temp_media_filename, "wb") as temp_media_file:
        temp_media_file.write(media_file.read())
        
        print('file_name.endswith',temp_media_filename)
        if file_name.endswith(".mp4"):
            print('day la video r ne')
            cap = cv2.VideoCapture(file_name)  
            print('cap',cap)     
            # cv2.destroyAllWindows() 
        else:
            print('cai nay image nha')
            image = cv2.imread(temp_media_filename)
            cv2.imshow("Image", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()       
        print(' cai duma done r ne')


# media_file_path = "C:\\Users\\CONG DUY\\OneDrive\\Desktop\\Fire_Detection_System_2410\\Fire_Detection_System-web_signin\\uploads\\flame.mp4"
# database_name = "GRIDFS_DBB"
# file_name = os.path.basename(media_file_path)
# media_filename = file_name  

# store_and_display_media(media_file_path, database_name, file_name)
# dispay_media(database_name,file_name,media_file_path)





    # def display_media(file_id):
    #     media_file = fs.get(file_id)

    #     temp_media_filename = "temp_media"

    #     with open(temp_media_filename, "wb") as temp_media_file:
    #         temp_media_file.write(media_file.read())

    #     if filename.endswith(".mp4"):
    #         # Display video
    #         cap = cv2.VideoCapture(temp_media_filename)

    #         while cap.isOpened():
    #             ret, frame = cap.read()
    #             if not ret:
    #                 break

    #             cv2.imshow("Video", frame)

    #             if cv2.waitKey(25) & 0xFF == ord("q"):
    #                 break

    #         cap.release()
    #         cv2.destroyAllWindows()
    #     else:
    #         # Display image
    #         image = cv2.imread(temp_media_filename)
    #         cv2.imshow("Image", image)
    #         cv2.waitKey(0)
    #         cv2.destroyAllWindows()

    # display_thread = threading.Thread(target=display_media, args=(file_id,))
    # display_thread.start()