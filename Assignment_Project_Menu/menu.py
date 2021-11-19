from face_detection_vid import read_video
from face_recognition_cam import cam_face_recog
from face_recognition_image import read_image
from handwritten_predictor import main
 

def main_menu():

    loop = True 

    while loop:
        print("\t MENU ")
        print("\n\t Option \n1) Face Recognition Video (Tensorflow)\n2) Face Recognition (HaarCascade)\n3) Face Recognition Saved Video\n4) Handwritten Predictor\n5) Exit")

        choice = input("Enter your choice: ")

        if choice == '1':
            read_image("/Users/azureennaja/Desktop/Perantis/Project 1/Project_1_Q3/class18.jpeg")
            
        elif choice == '2':
            cam_face_recog(0)

        elif choice == '3':
            read_video("/Users/azureennaja/Desktop/Perantis/Project 1/Project_1_Q3/1D_1_1.mp4")

        elif choice == '4':
            main()
        
        elif choice == '5':
            break
        
        else:
            print("Invalid entry")
    
    return choice


if __name__ == "__main__":
   choice = main_menu()





    


