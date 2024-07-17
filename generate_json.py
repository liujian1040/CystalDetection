import json
import os 
import numpy as np
from PIL import Image
import cv2
category2id = {"A":1,"B":2,"C":3,"D":4}

def p_annotation(data,image_id,anno_id,):  
    # annotations
    annotations=[]
    for i in range(len(data['shapes'])):
        annotation={}
        anno_id = anno_id + 1
        if data['shapes'][i]['shape_type'] == 'circle':
            #print(1)
            poly_points = []
            points = data['shapes'][i]['points']
            point1, point2 = np.array(points[0]) , np.array(points[1])
            x0 = point1[0]
            y0 = point1[1]
            theta = np.arange(0, 2*np.pi, 2*np.pi/30)
            r = np.sqrt(np.sum(np.square(point1-point2))) 
            x = x0 + r * np.cos(theta)
            y = y0 + r * np.sin(theta)
            poly_points = []
            for index in range(len(x)):
                poly_points.append([x[index],y[index]])
            # print(poly_points)
            annotation['segmentation']= [list(np.asarray(poly_points).flatten())]
            if len(list(np.asarray(poly_points).flatten()))<8:
                print(len(list(np.asarray(poly_points).flatten())))
                print('error')
                exit(0)
            #print(poly_points)
        elif data['shapes'][i]['shape_type'] == 'polygon':
            annotation['segmentation']=[list(np.asarray(data['shapes'][i]['points']).flatten())]
            if len(list(np.asarray(data['shapes'][i]['points']).flatten()))<8:
                print(len(list(np.asarray(data['shapes'][i]['points']).flatten())))
                print('error')
                exit(0)
        else:
            print('wtf')
            exit(0)
        annotation['iscrowd']=0
        annotation['image_id']=image_id
        annotation['id'] = anno_id
        #找出标注点中的外接矩形的四个点
        x = annotation['segmentation'][0][::2]#奇数个是x的坐标
        y = annotation['segmentation'][0][1::2]#偶数个是y的坐标
        
        x_left = min(x)#往外扩展1个像素，也可以不扩展
        y_left = min(y)
        w = max(x) - min(x)
        h = max(y) - min(y)
        annotation['bbox'] = [x_left,y_left,w,h] # [左上角x,y以及宽和高]
        annotation['area'] = w * h
        annotation['category_id']= category2id[data['shapes'][i]['label']]
        annotations.append(annotation)
    return annotations , anno_id

if __name__ =='__main__':
    coco = {}
    coco["info"] =  {
            "description": "Solution Crystallization",
            "url": "http://cocodataset.org",
            "version": "1.0",
            "year": 2021,
            "contributor": "TianJin University",
            "date_created": "2021/05/01"
        },
    coco['licenses'] = []
    coco['categories'] = [{'id':1,'name':'A'},{'id':2,'name':'B'},{'id':3,'name':'C'},{'id':4,'name':'D'}  ]
    #jsonDir = "C:/Users/ZQY/Desktop/project/CystalDetection-master/CystalDetection-master/first_dataset/"
    jsonDir = "C:/Users/Administrator/Desktop/CystalDetection-master/val"
    files = os.listdir(jsonDir)
    images = []
    annos = []
    json_list = []
    for file in files:
        if file.endswith('.json'):
            json_list.append(file)
    image_id = 1
    anno_id  = 1 
    for json_file in json_list: 
        fp = open(os.path.join(jsonDir,json_file),'r')
        imagesDict = {}
        data = json.load(fp)
        if os.path.exists(os.path.join(jsonDir,data["imagePath"])):
            imagesDict["id"] = image_id
            imagesDict["file_name"] = data["imagePath"]
            
            im = Image.open(os.path.join(jsonDir,data["imagePath"]))#返回一个Image对象
            cv2.imread(os.path.join(jsonDir,data["imagePath"]))
                
           
            print(os.path.join(jsonDir,data["imagePath"]))
            imagesDict['width'] = im.size[0]
            imagesDict['height'] = im.size[1]
            images.append(imagesDict)
            anno, anno_id = p_annotation(data,image_id,anno_id,)
            for single in anno:
                annos.append(single)
            image_id = image_id + 1
        else:
            print('???')
    coco['images'] = images
    coco['annotations'] = annos
    with open("val.json","w") as dump_f:
        json.dump(coco,dump_f, indent=2)
    print(image_id)
