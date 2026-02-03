
# Encrypted configuration: load private values from the process environment.
import os as _secure_os

def _secure_required_env(name):
    value = _secure_os.environ.get(name, "").strip()
    if not value or value.startswith(("__ENV_REQUIRED_", "REPLACE_", "<required")):
        raise RuntimeError("Required private configuration is missing: " + name + "; decrypt the private archive locally and set this environment variable.")
    return value

def _secure_expand_config(value):
    if isinstance(value, dict):
        return {key: _secure_expand_config(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_secure_expand_config(item) for item in value]
    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
        return _secure_required_env(value[2:-1])
    return value

from flask import Flask, request, Response
import requests
import urllib.parse

app = Flask(__name__)

BACKEND_SERVICES = {
    "/model": _secure_required_env("PRIVATE_API_BASE_URL"),
}

def get_target_url(path):
    """
    根据请求路径匹配对应的后端服务地址
    """
    for prefix, backend_url in BACKEND_SERVICES.items():
        if path.startswith(prefix):
            # 拼接完整的目标 URL（去掉前缀，保留剩余路径）
            target_path = path[len(prefix):] if len(prefix) < len(path) else ""
            if target_path and not target_path.startswith("/"):
                target_path = "/" + target_path
            return urllib.parse.urljoin(backend_url, target_path)
    # 默认返回最后一个配置（根路径）
    return urllib.parse.urljoin(BACKEND_SERVICES["/"], path)

@app.route('/', defaults={'path': ''}, methods=['GET', 'POST', 'PUT', 'DELETE', 'PATCH'])
@app.route('/<path:path>', methods=['GET', 'POST', 'PUT', 'DELETE', 'PATCH'])
def proxy(path):
    """
    核心代理函数，处理所有请求
    """
    try:
        # 1. 获取目标 URL
        target_url = get_target_url(request.path)

        # 2. 处理请求头（排除 Flask 内部的特殊头信息）
        headers = dict(request.headers)
        # 移除可能导致问题的头信息
        headers.pop('Host', None)
        headers.pop('Content-Length', None)

        # 3. 转发请求到后端服务
        response = requests.request(
            method=request.method,
            url=target_url,
            headers=headers,
            data=request.get_data(),
            params=request.args,
            cookies=request.cookies,
            allow_redirects=False,
            timeout=600  # 设置超时时间
        )

        # 4. 处理响应头（排除可能导致问题的头信息）
        response_headers = dict(response.headers)
        response_headers.pop('Transfer-Encoding', None)

        # 5. 返回响应给客户端
        return Response(
            response.content,
            status=response.status_code,
            headers=response_headers
        )

    except requests.exceptions.RequestException as e:
        # 请求异常处理
        return Response(f"Proxy error: {str(e)}", status=502)
    except Exception as e:
        # 其他异常处理
        return Response(f"Server error: {str(e)}", status=500)

if __name__ == '__main__':
    # 启动 Flask 服务（生产环境建议使用 Gunicorn/Uvicorn）
    app.run(host='0.0.0.0', port=5000, debug=False)