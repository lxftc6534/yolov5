
import socket
import threading
import traceback
import win32api
import win32con
from pynput import keyboard

# 键盘按键映射表
KEY_MAP = {
    'enter': 13,
    'space': 32,
    'esc': 27,
    'tab': 9,
    'backspace': 8,
    'left': 37,
    'right': 39,
    'up': 38,
    'down': 40,
}


# 模拟按键
def simulate_key(key: str) -> None:
    if isinstance(key, str) and len(key) == 1:
        vk_code = ord(key)
        win32api.keybd_event(vk_code, 0, 0, 0)  # 按下键
        win32api.keybd_event(vk_code, 0, win32con.KEYEVENTF_KEYUP, 0)  # 释放键
    elif key.lower() in KEY_MAP:
        vk_code = KEY_MAP[key.lower()]
        win32api.keybd_event(vk_code, 0, 0, 0)  # 按下键
        win32api.keybd_event(vk_code, 0, win32con.KEYEVENTF_KEYUP, 0)  # 释放键
    else:
        print(f"Unknown key: {key}")


def on_press(key):
    try:
        print(f'on_press({key})')
        # 如果是普通字符键
        if len(key.char) == 1:
            return key.char
        # 如果是特殊键
        special_key = key.name
        if special_key in KEY_MAP:
            return special_key
    except AttributeError:
        pass
    return None


def listen_keyboard(client_socket, server_address):
    def on_press(key):
        try:
            print(f'listen_keyboard on_press({key})')
            if hasattr(key, 'char'):
                print(f'key.char: {key.char}')
            if hasattr(key, 'name'):
                print(f'key.name: {key.name}')
            if hasattr(key, 'code'):
                print(f'key.code: {key.code}')
            # 如果是普通字符键
            if hasattr(key, 'char') and key.char:
                client_socket.sendto(key.char.encode('utf-8'), server_address)
            # 如果是特殊键
            if hasattr(key, 'name'):
                special_key = key.name
                if special_key in KEY_MAP:
                    client_socket.sendto(special_key.encode('utf-8'), server_address)
        except AttributeError as e:
            print(f"AttributeError: {e}")
            traceback.print_exc()
            pass

    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()


def lan_server():
    print("lan_server")
    # 创建UDP socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # 绑定地址和端口
    server_address = ('', 12000)  # 注意：''代表本地任意可用IP
    server_socket.bind(server_address)

    # 循环接收消息
    while True:
        message, client_address = server_socket.recvfrom(1024)
        message = message.decode('utf-8')
        print(f"服务端收到信息: {message} from {client_address}")
        server_socket.sendto(message.upper().encode('utf-8'), client_address)

        # 监听按键并发送给客户端
        if message == "listen":
            # 启动键盘监听线程
            threading.Thread(target=listen_keyboard, args=(server_socket, server_address)).start()


def lan_client():
    print("lan_client")
    # 创建UDP socket
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # 服务器地址
    server_address = ('192.168.111.155', 12000)  # 服务器IP和端口

    # 发送消息
    message = b'client:Hello, Server!'
    client_socket.sendto(message, server_address)

    # 接收响应
    response, server_address = client_socket.recvfrom(1024)
    print(f"客户端收到信息: {response.decode('utf-8')} from {server_address}")

    # 开始监听按键
    client_socket.sendto(b"listen", server_address)

    # 接收按键信息并模拟按键
    while True:
        key, server_address = client_socket.recvfrom(1024)
        key = key.decode('utf-8')
        print(f"客户端收到信息: {key} from {server_address}")
        simulate_key(key)

    # 关闭socket
    client_socket.close()


if __name__ == "__main__":
    # 创建并启动服务器线程
    server_thread = threading.Thread(target=lan_server)
    server_thread.start()

    # 创建并启动客户端线程
    client_thread = threading.Thread(target=lan_client)
    client_thread.start()