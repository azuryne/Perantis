import labelme2coco
import os 
import shutil

# access the directory seperately 
def create_folder():
    for i in range (1, 4):
        if not os.path.exists("json_file_b"):
            os.mkdir('json_file_b')
        folder = f'json_file_b/json_file{i}'
        os.mkdir(folder)
        src1 = f'json_file/img_{i}.json'
        src2 = f'json_file/img_{i}.png'
        dest = folder 

        destination = shutil.copy(src1, dest)
        destination = shutil.copy(src2, dest)

def json_to_coco(): 
    for i in range (1,4):
        if os.path.exists('json_file_b'):
            labelme_folder = f'json_file_b/json_file{i}'
            coco_json = f'test2/test_coco{i}'
            labelme2coco.convert(labelme_folder, coco_json)

if __name__ == '__main__':
    create_folder()
    json_to_coco()



