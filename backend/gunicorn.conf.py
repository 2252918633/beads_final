# Gunicorn配置文件
bind = "0.0.0.0:5001"
workers = 4  # 根据CPU核心数调整，通常为 2 * CPU核心数 + 1
worker_class = "sync"
worker_connections = 1000
timeout = 120  # 请求超时时间（图片处理可能较慢）
keepalive = 5
max_requests = 1000
max_requests_jitter = 100
preload_app = True
reload = False  # 生产环境设为False

# 日志配置
accesslog = "/root/project/logs/access.log"
errorlog = "/root/project/logs/error.log"
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# 进程管理
daemon = False  # 如果要后台运行设为True
pidfile = "/root/project/backend/gunicorn.pid"
user = None  # 可设置运行用户
group = None # 可设置运行组

# 安全
limit_request_line = 4094
limit_request_fields = 100
limit_request_field_size = 8190