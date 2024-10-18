import requests

# 发起GET请求
def send_get_request(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # 如果响应状态码不是200，将抛出HTTPError异常
        return response.text
    except requests.exceptions.HTTPError as errh:
        return f"HTTP Error: {errh}"
    except requests.exceptions.ConnectionError as errc:
        return f"Error Connecting: {errc}"
    except requests.exceptions.Timeout as errt:
        return f"Timeout Error: {errt}"
    except requests.exceptions.RequestException as err:
        return f"OOps: Something Else: {err}"

# 发起POST请求
def send_post_request(url, data):
    try:
        response = requests.post(url, data=data)
        response.raise_for_status()  # 如果响应状态码不是200，将抛出HTTPError异常
        return response.text
    except requests.exceptions.HTTPError as errh:
        return f"HTTP Error: {errh}"
    except requests.exceptions.ConnectionError as errc:
        return f"Error Connecting: {errc}"
    except requests.exceptions.Timeout as errt:
        return f"Timeout Error: {errt}"
    except requests.exceptions.RequestException as err:
        return f"OOps: Something Else: {err}"

# 示例用法
url = 'http://localhost:8000'
post_url = 'http://localhost:8000/post'

# 发送GET请求
get_response = send_get_request(url)
print(f"GET Response: {get_response}")

# 发送POST请求
post_data = {"key": [1,2,3]}
post_response = send_post_request(post_url, str(post_data).encode())
print(f"POST Response: {post_response}")