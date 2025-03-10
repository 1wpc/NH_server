import copy
from http.server import BaseHTTPRequestHandler, HTTPServer
import cgi
import json
from urllib.parse import parse_qs
import ast
import torch
from net import Net


dp = 0.0
valence = 0.0
arousal = 0.0
dominance = 0.0

dp_model = Net(num_classes=1, in_channels=2048, grid_size=(4, 6))
dp_model.load_state_dict(torch.load('dp_model.pth'))
dp_model.eval()

valence_model = Net(num_classes=1, in_channels=2048, grid_size=(4, 6))
valence_model.load_state_dict(torch.load('deap_model_valence.pth'))
valence_model.eval()

arousal_model = Net(num_classes=1, in_channels=2048, grid_size=(4, 6))
arousal_model.load_state_dict(torch.load('deap_model_arousal.pth'))
arousal_model.eval()

dominance_model = Net(num_classes=1, in_channels=2048, grid_size=(4, 6))
dominance_model.load_state_dict(torch.load('deap_model_dominance.pth'))
dominance_model.eval()

def data_convert(data):#8*2048
    data2d = [] 
    for i in range(2048):
        p8 = []
        for j in range(8):
            p8.append(data[j][i])
        image = []
        hang = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        hang[2] = p8[0]
        hang[3] = p8[1]
        image.append(copy.deepcopy(hang))
        hang = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        hang[1] = p8[2]
        hang[4] = p8[3]
        image.append(copy.deepcopy(hang))
        hang = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        hang[0] = p8[4]
        hang[5] = p8[5]
        image.append(copy.deepcopy(hang))
        hang = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        hang[2] = p8[6]
        hang[3] = p8[7]
        image.append(copy.deepcopy(hang))
        data2d.append(image)
    return data2d


class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        # 处理GET请求
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        response = {
            'dp': dp,
            'valence': valence,
            'arousal': arousal,
            'dominance': dominance

        }
        self.wfile.write(json.dumps(response).encode('utf-8'))

    def do_POST(self):
        # 处理POST请求
        print('receive post data')
        # content_length = int(self.headers['Content-Length',0])
        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length)
        print(post_data)

        dict_data = ast.literal_eval(post_data.decode())
        print(dict_data)

        list_data = dict_data['data']
        tensor_data = torch.tensor(data_convert(list_data))*0.02235

        t_dp = dp_model(tensor_data)
        t_valence = valence_model(tensor_data)
        t_arousal = arousal_model(tensor_data)
        t_dominance = dominance_model(tensor_data)

        global dp, valence, arousal, dominance
        dp = t_dp.item()
        valence = t_valence.item()
        arousal = t_arousal.item()
        dominance = t_dominance.item()

        
        # 发送HTTP状态码
        self.send_response(200)
        # 发送响应头部
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        
        # 发送响应内容
        response = 'Received your message'
        self.wfile.write(response.encode('utf-8'))

def run(server_class=HTTPServer, handler_class=SimpleHTTPRequestHandler):
    server_address = ('0.0.0.0', 8000)  # 服务器地址和端口
    httpd = server_class(server_address, handler_class)
    print('Starting httpd...')
    httpd.serve_forever()

if __name__ == '__main__':
    run()