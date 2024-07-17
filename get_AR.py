import re

pattern_dict = dict()
#(?<=exp2)exp1
pattern_dict['AR_maxDets=1']=re.compile(r'AR @ IoU=0.50:0.95  area=   all  maxDets=  1  = (.+)')
pattern_dict['AR_maxDets=10']=re.compile(r'AR @ IoU=0.50:0.95  area=   all  maxDets= 10  = (.+)')
pattern_dict['AR_maxDets=100']=re.compile(r'AR @ IoU=0.50:0.95  area=   all  maxDets=100  = (.+)')
pattern_dict['AR_small']=re.compile(r'AR @ IoU=0.50:0.95  area= small  maxDets=100  = (.+)')
pattern_dict['AR_medium']=re.compile(r'AR @ IoU=0.50:0.95  area=medium  maxDets=100  = (.+)')
pattern_dict['AR_large']=re.compile(r'AR @ IoU=0.50:0.95  area= large  maxDets=100  = (.+)')
pattern_dict['AP_IoU=0.500.95']=re.compile(r'AP @ IoU=0.50:0.95  area=   all  maxDets=100  = (.+)')
pattern_dict['AP_IoU=0.50']=re.compile(r'AP @ IoU=0.50       area=   all  maxDets=100  = (.+)')
pattern_dict['AP_IoU=0.75']=re.compile(r'AP @ IoU=0.75       area=   all  maxDets=100  = (.+)')
pattern_dict['AP_small']=re.compile(r'AP @ IoU=0.50:0.95  area= small  maxDets=100  = (.+)')
pattern_dict['AP_medium']=re.compile(r'AP @ IoU=0.50:0.95  area=medium  maxDets=100  = (.+)')
pattern_dict['AP_large']=re.compile(r'AP @ IoU=0.50:0.95  area= large  maxDets=100  = (.+)')

output_list_dict = dict()
for key in pattern_dict.keys():
    output_list_dict[key] = []
    

with open("segm.txt","r",encoding="utf-8") as f:
    for userline in f:
        userline=str(userline).replace('\n','')
        userline=str(userline).replace('(','')
        userline=str(userline).replace(')','')
        userline=str(userline).replace('[','')
        userline=str(userline).replace(']','')
        userline=str(userline).replace('|','')
        for key_word, pattern in pattern_dict.items():
            result=pattern.findall(userline)
            if len(result)>0:
                output_list_dict[key_word].append(result[0])
            else:
                print("no such key_word:"+str(key_word))

for key_word in output_list_dict.keys():
    with open('./metrics/'+key_word+'.txt',"w+") as f:
        for value in output_list_dict[key_word]:
            f.write(str(value)+'\n')
            
