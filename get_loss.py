import re

pattern_dict = dict()
pattern_dict['loss']=re.compile('loss:(.+)')
pattern_dict['segm_mAP']=re.compile('segm_mAP:(.+?),')
pattern_dict['segm_mAP_50']=re.compile('segm_mAP_50:(.+?),')
pattern_dict['segm_mAP_75']=re.compile('segm_mAP_75:(.+?),')
pattern_dict['segm_mAP_s']=re.compile('segm_mAP_s:(.+?),')
pattern_dict['segm_mAP_m']=re.compile('segm_mAP_m:(.+?),')
pattern_dict['segm_mAP_l']=re.compile('segm_mAP_l:(.+?),')
pattern_dict['bbox_mAP']=re.compile('bbox_mAP:(.+?),')
pattern_dict['bbox_mAP_50']=re.compile('bbox_mAP_50:(.+?),')
pattern_dict['bbox_mAP_75']=re.compile('bbox_mAP_75:(.+?),')
pattern_dict['bbox_mAP_s']=re.compile('bbox_mAP_s:(.+?),')
pattern_dict['bbox_mAP_m']=re.compile('bbox_mAP_m:(.+?),')
pattern_dict['bbox_mAP_l']=re.compile('bbox_mAP_l:(.+?),')

output_list_dict = dict()
for key in pattern_dict.keys():
    output_list_dict[key] = []
    

with open("20240607_225713.log","r",encoding="utf-8") as f:
    for userline in f:
        userline=str(userline).replace('\n','')
        for key_word, pattern in pattern_dict.items():
            result=pattern.findall(userline)
            if len(result)>0:
                output_list_dict[key_word].append(result[0])

for key_word in output_list_dict.keys():        
    with open('./metrics/'+key_word+'.txt',"w+") as f:
        for value in output_list_dict[key_word]:
            f.write(str(value)+'\n')
            

            