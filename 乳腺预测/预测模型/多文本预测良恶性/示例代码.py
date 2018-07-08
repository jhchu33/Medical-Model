# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 20:05:04 2017

@author: Administrator
"""
import numpy as np  
import pandas as pd
import jieba

from keras.preprocessing import sequence
from keras.models import Sequential  
from keras.layers import Dense  
from keras.models import load_model  
from keras.models import *





def get_data(str_data):
    comment = pd.DataFrame([[str_data]],index=[0],columns=['titlecontent'])
    comlen=len(comment)
    cw = lambda x: list(jieba.cut(x))
    comment = comment[comment['titlecontent'].notnull()]
    comment['words'] = comment['titlecontent'].apply(cw)
    w = []
    for i in comment['words']:
        w.extend(i)
    dict = pd.DataFrame(pd.Series(w).value_counts())
    del w
    dict['id']=list(range(1,len(dict)+1))
    get_sent = lambda x: list(dict['id'][x])
    comment['sent'] = comment['words'].apply(get_sent)  
    maxlen = 20000
    comment['sent'] = list(sequence.pad_sequences(comment['sent'], maxlen=maxlen))
    x = np.array(list(comment['sent']))[:]
    return x

x1 = get_data("乳腺：腺体回声增多，增强，分布欠均匀，腺体表面尚光整，腺管未见明显扩张。CDFI:未见明显异常血流信号。右侧乳腺外上象限可见多支腺管扩张，扩张腺管内部分透声欠佳，于约10点钟方向近乳晕处可见一大小约19.4×6.6mm团块,与扩张腺管相连，水平位生长，呈不规则形，边界清晰，边缘微小分叶，内部呈混合回声，分布尚均，未见明显钙化灶，后方回声无明显改变，CDFI示团块内边缘及中央区域见丰富血流信号，血流纤细，走行欠规则。三维超声显示团块边缘呈毛刺状或向周边不规则突起，未显示腺体和库氏韧带有异常聚集征象。右侧乳腺可见多个异常团块,之一位于约12点钟方向大小约3.8×2.9mm，水平位生长，呈不规则形，边界欠清晰，边缘模糊，内部呈低回声，分布尚均，未见明显钙化灶，后方回声无明显改变，CDFI示团块内边缘及中央区域见少许血流信号，血流较纤细，走行较规则。三维超声未显示团块边缘呈毛刺状或向周边不规则突起，未显示腺体和库氏韧带有异常聚集征象。双侧腋窝未见明显异常肿大淋巴结。")
x2 = get_data("右乳外上象限见一小斑片、结节状影，边界模糊，见长毛刺，右乳并见一点状钙化灶。两肺纹理增多增粗，两肺上叶后段胸膜下小结节影。左肺上叶小气囊影。左肺上叶、右肺下叶散在小片较淡磨玻璃影。两侧胸膜增厚粘连。两肺门区未见异常。所示气管支气管影正常。纵隔内未见异常增大的淋巴结。附见肝脏多发大小不一类圆形囊性低密度灶，边界清晰。")
x3 = get_data("双侧乳腺皮肤及乳晕未见明显增厚，乳头无凹陷。皮下脂肪组织结构层次清晰。双侧腺体部分退化，呈片状条索状改变。右乳晕后可见多发结节状、片状高密度影。双乳内可见小圆形钙化灶。双腋下环形淋巴结显示。")
x4 = get_data("平扫：双乳表面皮肤光整，皮肤及乳晕未见明显增厚，双侧乳头无凹陷，皮下结构层次清晰。双乳腺体丰富，大致对称。平扫所右侧乳晕后方多发条状、分枝状、小结节状高信号影，沿导管分布。双侧乳后间隙未见异常信号影。双侧腋下未见明显肿大淋巴结影。动态增强扫描：双乳腺体内见多发斑片状、微小结节样强化影，腺体边缘为著，时间信号曲线呈缓慢上升型。右乳上部及右乳偏内侧多发团片状、条状及结节状异常强化灶，沿导管分布，时间信号曲线呈缓慢上升型及流出型；与平扫所右侧乳晕后方多发条状、分枝状、小结节状高信号影为相同病变区域。")
    


model = model_from_json(open('my_model_architecture.json').read())
model.load_weights('breast-cancer-multext.h5',by_name=True)  

p =model.predict([x1,x2,x3,x4])
print(p)