import json
import os

name2id =  {'hero':0,'sodier':1,'tower':2}#标签名称


def convert(img_size, box):
    dw = 1. / (img_size[0])
    dh = 1. / (img_size[1])
    x = (box[0] + box[2]) / 2.0 - 1
    y = (box[1] + box[3]) / 2.0 - 1
    w = box[2] - box[0]
    h = box[3] - box[1]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def decode_json(floder_path_json, json_name, floder_path_txt):
    print(f'json_name:{json_name}')
    if not json_name.endswith('.json'):
        return
    txt_name = floder_path_txt + json_name[0:-5] + '.txt'
    #存放txt的绝对路径
    txt_file = open(txt_name, 'w')

    json_path = os.path.join(floder_path_json, json_name)
    # 检查JSON文件是否存在
    if not os.path.exists(json_path):
        print(f"警告: 文件 {json_path} 不存在.")
        return
    print(f'json_path:{json_path}')
    # data = json.load(open(json_path, 'r', encoding='gb2312',errors='ignore'))
    data = json.load(open(json_path, 'r', errors='ignore'))

    img_w = data['imageWidth']
    img_h = data['imageHeight']

    for i in data['shapes']:

        label_name = i['label']
        if (i['shape_type'] == 'rectangle'):
            x1 = int(i['points'][0][0])
            y1 = int(i['points'][0][1])
            x2 = int(i['points'][1][0])
            y2 = int(i['points'][1][1])

            bb = (x1, y1, x2, y2)
            bbox = convert((img_w, img_h), bb)
            txt_file.write(str(name2id[label_name]) + " " + " ".join([str(a) for a in bbox]) + '\n')


if __name__ == "__main__":
    # txsp1.json
    floder_path_json = '../../demo_guanfang/labels/train/'
    #存放json的文件夹的绝对路径
    json_names = os.listdir(floder_path_json)
    for json_name in json_names:
        decode_json(floder_path_json, json_name, '../../demo_guanfang/labels/train/')