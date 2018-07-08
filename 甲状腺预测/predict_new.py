import re
import jieba
import pickle
import numpy as np
from keras.preprocessing import sequence as keras_seq
from keras.models import load_model


def remove_num(content):
    if content == np.nan:
        return ''
    content = str(content)
    # return content
    new_content = re.sub('[0-9]*\.*[0-9]+', '', content)
    new_content = re.sub('（.*）', '', new_content)
    new_content = re.sub('\(.*\)', '', new_content)
    new_content = new_content.replace('×', '')
    new_content = new_content.replace('*', '')
    new_content = new_content.replace('mm', '')
    new_content = new_content.replace('cm', '')
    new_content = new_content.replace('%', '')
    new_content = new_content.replace('-', '')
    new_content = new_content.replace('-', '')
    new_content = new_content.replace('\r', '')
    new_content = new_content.replace('\n', '')

    #    if new_content.find('穿刺记录') != -1:
    #        new_content = new_content[:new_content.find('穿刺记录')]
    return new_content


def load_data(tokenizer, sen):
    sen = remove_num(sen)
    sen = ' '.join(list(jieba.cut(sen)))
    #print(sen)

    text_seq = tokenizer.texts_to_sequences([sen])

    text_array = keras_seq.pad_sequences(text_seq, maxlen=300, dtype='int32', padding='pre')
    return text_array

def predict(sen, model_type):
    [tokenizer] = pickle.load(open("tokenizer.p", "rb"))
    text_array = load_data(tokenizer, sen)
    if model_type == 'cnn':
        model = load_model("cnn_model")
    elif model_type == 'lstm':
        model = load_model("lstm_model")
    res = model.predict(text_array)
    return res[0][0]


if __name__ == '__main__':
    sen = "右侧甲状腺中上极（近腹侧，偏外）可见一个低回声结节，大小约14.3×9.4×10.2mm。内部结构呈实性，回声均匀，形状不规则，边界清晰，边缘光整，无声晕，内部见点状强回声，后方回声伴细钙化引起的衰减，无侧方声影。与包膜接触面积0-25%，未向甲状腺包膜外突出，CDFI示其内血供程度中等，血流模式为混合型。"
    sen2 = "甲状腺:甲状腺左、右叶大小及形态正常，峡部厚度正常，边界清楚，表面光滑、包膜完整，内部呈密集中等回声，回声分布欠均匀。CDFI：未见明显异常血流信号。双侧甲状腺内可见几个不均质回声、低回声及混合性回声,左侧之一约25×20mm，右侧之一约6×3.5mm，形状呈椭圆形，内部回声欠均匀，边界尚清，内部未见明显点状强回声，CDFI：未见明显异常血流信号。双侧甲状旁腺区未见明显占位性病变。双侧颈部未见明显异常肿大淋巴结。"
    #lstm_res = predict(sen2, 'lstm')
    cnn_res = predict(sen, 'cnn')

    print(cnn_res)
