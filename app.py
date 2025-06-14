import os
from flask import (
    Flask,
    request,
    render_template,
    redirect,
    url_for,
    session,
    jsonify,
    flash,
)
from flask_socketio import SocketIO, emit, join_room, leave_room, disconnect
import shutil
import threading
import time
from datetime import datetime
import queue
import io
import sys
import builtins
import base64
import bcrypt
import json
import logging

# Supabase集成
try:
    from supabase import create_client, Client
    from dotenv import load_dotenv

    SUPABASE_AVAILABLE = True

    # 加载环境变量
    load_dotenv()

    # Supabase配置
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")

    # 初始化Supabase客户端
    supabase: Client = None
    if SUPABASE_URL and SUPABASE_KEY:
        try:
            supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
            print("✅ Supabase连接成功")
        except Exception as e:
            print(f"⚠️ Supabase连接失败: {e}")
            print("🔄 将使用本地存储作为备用")
            SUPABASE_AVAILABLE = False
    else:
        print("⚠️ 未找到Supabase配置，使用本地存储")
        SUPABASE_AVAILABLE = False

except ImportError as e:
    print(f"⚠️ Supabase库未安装: {e}")
    print("🔄 将使用本地存储")
    SUPABASE_AVAILABLE = False
    supabase = None

# 从您现有的脚本导入训练函数
# 确保 src 目录在 Python 路径中，或者调整导入方式
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from federated_training import train_federated_model
from federated_inference import run_inference

app = Flask(__name__)
app.secret_key = "123456"
socketio = SocketIO(app, cors_allowed_origins="*")

# 在线用户跟踪
online_users = (
    {}
)  # {session_id: {'username': str, 'role': str, 'connected_at': datetime}}
user_sessions = {}  # {username: set of session_ids}

UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# 推理文件上传目录
INFERENCE_UPLOAD_FOLDER = os.path.join(UPLOAD_FOLDER, "server_inference")
if not os.path.exists(INFERENCE_UPLOAD_FOLDER):
    os.makedirs(INFERENCE_UPLOAD_FOLDER)

# 客户端推理文件上传目录
CLIENT_INFERENCE_UPLOAD_FOLDER = os.path.join(UPLOAD_FOLDER, "client_inference")
if not os.path.exists(CLIENT_INFERENCE_UPLOAD_FOLDER):
    os.makedirs(CLIENT_INFERENCE_UPLOAD_FOLDER)

# 本地用户备用存储
LOCAL_USERS_FILE = os.path.join(os.path.dirname(__file__), "local_users.json")

DEFAULT_USERS = {}


def hash_password(password: str) -> str:
    """使用bcrypt哈希密码"""
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, hashed: str) -> bool:
    """验证密码"""
    return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))


def load_local_users():
    """从本地JSON文件加载用户数据"""
    try:
        if os.path.exists(LOCAL_USERS_FILE):
            with open(LOCAL_USERS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        else:
            # 如果文件不存在，创建默认用户并保存
            hashed_users = {}
            for username, data in DEFAULT_USERS.items():
                hashed_users[username] = {
                    "password": hash_password(data["password"]),
                    "role": data["role"],
                    "email": data["email"],
                    "created_at": datetime.now().isoformat(),
                    "is_active": True,
                }
            save_local_users(hashed_users)
            return hashed_users
    except Exception as e:
        print(f"❌ 加载本地用户数据失败: {e}")
        return {}


def save_local_users(users_data):
    """保存用户数据到本地JSON文件"""
    try:
        with open(LOCAL_USERS_FILE, "w", encoding="utf-8") as f:
            json.dump(users_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"❌ 保存本地用户数据失败: {e}")


def get_user_from_supabase(username: str):
    """从Supabase获取用户信息"""
    if not SUPABASE_AVAILABLE or not supabase:
        return None

    try:
        response = (
            supabase.table("users").select("*").eq("username", username).execute()
        )
        if response.data and len(response.data) > 0:
            return response.data[0]
        return None
    except Exception as e:
        print(f"❌ 从Supabase获取用户失败: {e}")
        return None


def authenticate_user(username: str, password: str):
    """认证用户 - 优先使用Supabase，失败时使用本地存储"""

    # 首先尝试Supabase认证
    if SUPABASE_AVAILABLE and supabase:
        try:
            user_data = get_user_from_supabase(username)
            if user_data:
                # 验证密码
                if verify_password(password, user_data["password_hash"]):
                    # 更新最后登录时间
                    try:
                        supabase.table("users").update(
                            {"last_login": datetime.now().isoformat()}
                        ).eq("username", username).execute()
                    except:
                        pass  # 忽略更新错误

                    return {
                        "username": user_data["username"],
                        "role": user_data.get("role", "client"),  # 从数据库获取角色
                        "email": user_data.get("email", ""),
                        "source": "supabase",
                    }
                else:
                    print(f"🔐 Supabase密码验证失败: {username}")

        except Exception as e:
            print(f"⚠️ Supabase认证失败，尝试本地存储: {e}")

    # 备用：使用本地存储认证
    try:
        local_users = load_local_users()
        if username in local_users:
            user_data = local_users[username]
            if verify_password(password, user_data["password"]):
                return {
                    "username": username,
                    "role": user_data["role"],
                    "email": user_data.get("email", ""),
                    "source": "local",
                }

    except Exception as e:
        print(f"❌ 本地认证失败: {e}")

    return None


def create_user_in_supabase(
    username: str, email: str, password: str, role: str = "client"
):
    """在Supabase中创建用户"""
    if not SUPABASE_AVAILABLE or not supabase:
        return False

    try:
        hashed_password = hash_password(password)
        data = {
            "username": username,
            "email": email,
            "password_hash": hashed_password,
            "role": role,
            "created_at": datetime.now().isoformat(),
            "is_active": True,
        }

        response = supabase.table("users").insert(data).execute()
        return response.data is not None

    except Exception as e:
        print(f"❌ 在Supabase创建用户失败: {e}")
        return False


def initialize_default_users():
    """初始化默认用户到Supabase"""
    if not SUPABASE_AVAILABLE or not supabase:
        print("📝 Supabase不可用，跳过初始化")
        return

    try:
        for username, data in DEFAULT_USERS.items():
            # 检查用户是否已存在
            existing = get_user_from_supabase(username)
            if not existing:
                success = create_user_in_supabase(
                    username, data["email"], data["password"], data["role"]
                )
                if success:
                    print(f"✅ 已创建默认用户: {username}")
                else:
                    print(f"❌ 创建默认用户失败: {username}")

    except Exception as e:
        print(f"❌ 初始化默认用户失败: {e}")


# 模拟用户数据库 - 现在主要用作角色映射的备用
users = DEFAULT_USERS

# 存储客户端文件上传状态和数据路径
# 这个变量将在initialize_client_data_status()函数中初始化
client_data_status = {}

# 全局训练状态和日志
training_status = {
    "is_training": False,
    "current_round": 0,
    "total_rounds": 0,
    "active_clients": 0,
    "start_time": None,
    "end_time": None,
    "progress": 0,
}

# 设置全局变量，让训练函数能够访问
builtins.app_training_status = training_status
builtins.socketio = socketio

# 全局推理状态
inference_status = {
    "is_running": False,
    "progress": 0,
    "current_step": "",
    "result_image": None,
    "error": None,
    "start_time": None,
    "end_time": None,
    "uploaded_files": [],
}

# 客户端推理状态
client_inference_status = {
    "is_running": False,
    "progress": 0,
    "current_step": "",
    "result_image": None,
    "error": None,
    "start_time": None,
    "end_time": None,
    "uploaded_files": [],
}

# 日志队列
server_logs = queue.Queue(maxsize=1000)
training_logs = queue.Queue(maxsize=1000)

# 数据变化跟踪，用于触发页面刷新
data_change_timestamp = None


def broadcast_user_status():
    """广播用户状态更新"""
    try:
        # 统计在线用户
        online_clients = []
        online_servers = []

        for session_id, user_info in online_users.items():
            if user_info["role"] == "client":
                client_status = client_data_status.get(user_info["username"], {})
                online_clients.append(
                    {
                        "username": user_info["username"],
                        "connected_at": user_info["connected_at"].strftime("%H:%M:%S"),
                        "uploaded": client_status.get("uploaded", False),
                        "file_count": client_status.get("file_count", 0),
                    }
                )
            elif user_info["role"] == "server":
                online_servers.append(
                    {
                        "username": user_info["username"],
                        "connected_at": user_info["connected_at"].strftime("%H:%M:%S"),
                    }
                )

        status_data = {
            "online_clients": online_clients,
            "online_servers": online_servers,
            "total_online": len(online_users),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        # 广播给所有连接的用户
        socketio.emit("user_status_update", status_data)

    except Exception as e:
        print(f"广播用户状态失败: {e}", file=sys.stderr)


def broadcast_training_status():
    """广播训练状态更新"""
    try:
        socketio.emit("training_status_update", training_status)
    except Exception as e:
        print(f"广播训练状态失败: {e}", file=sys.stderr)


def broadcast_client_data_update():
    """广播客户端数据更新"""
    try:
        # 获取所有客户端数据状态
        client_info = []
        for uname, uinfo in users.items():
            if uinfo["role"] == "client":
                status = client_data_status.get(uname, {"uploaded": False})
                is_online = uname in user_sessions and len(user_sessions[uname]) > 0

                client_info.append(
                    {
                        "username": uname,
                        "uploaded": status.get("uploaded", False),
                        "file_count": status.get("file_count", 0),
                        "upload_time": status.get("upload_time", "未上传"),
                        "is_online": is_online,
                        "last_login": status.get("last_login", "从未登录"),
                    }
                )

        data = {
            "clients": client_info,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        # 只广播给服务器端用户
        for session_id, user_info in online_users.items():
            if user_info["role"] == "server":
                socketio.emit("client_data_update", data, room=session_id)

    except Exception as e:
        print(f"广播客户端数据更新失败: {e}", file=sys.stderr)


def add_server_log(message):
    """添加服务器日志"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    try:
        server_logs.put_nowait(log_entry)
        # 实时广播日志更新
        socketio.emit(
            "logs_update",
            {
                "type": "server",
                "logs": get_logs_list(server_logs),
                "timestamp": timestamp,
            },
        )
    except queue.Full:
        # 如果队列满了，删除最旧的日志
        try:
            server_logs.get_nowait()
            server_logs.put_nowait(log_entry)
        except queue.Empty:
            pass
    except Exception as e:
        print(f"添加服务器日志失败: {e}", file=sys.stderr)


def add_training_log(message):
    """添加训练日志"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    try:
        training_logs.put_nowait(log_entry)
        # 实时广播日志更新
        socketio.emit(
            "logs_update",
            {
                "type": "training",
                "logs": get_logs_list(training_logs),
                "timestamp": timestamp,
            },
        )
    except queue.Full:
        # 如果队列满了，删除最旧的日志
        try:
            training_logs.get_nowait()
            training_logs.put_nowait(log_entry)
        except queue.Empty:
            pass


# 设置日志函数为全局可访问
builtins.add_server_log = add_server_log
builtins.add_training_log = add_training_log


def get_logs_list(log_queue):
    """获取日志列表"""
    logs = []
    temp_logs = []

    # 取出所有日志
    while not log_queue.empty():
        try:
            log = log_queue.get_nowait()
            temp_logs.append(log)
        except queue.Empty:
            break

    # 保留最新的100条日志
    recent_logs = temp_logs[-100:] if len(temp_logs) > 100 else temp_logs

    # 重新放回队列
    for log in recent_logs:
        try:
            log_queue.put_nowait(log)
        except queue.Full:
            break

    return recent_logs


# WebSocket 事件处理
@socketio.on("connect")
def handle_connect():
    """用户连接事件"""
    if "username" not in session:
        disconnect()
        return False

    username = session["username"]
    role = session["role"]
    session_id = request.sid

    # 记录用户在线状态
    online_users[session_id] = {
        "username": username,
        "role": role,
        "connected_at": datetime.now(),
    }

    # 维护用户会话映射
    if username not in user_sessions:
        user_sessions[username] = set()
    user_sessions[username].add(session_id)

    # 加入对应的房间
    join_room(f"{role}_room")
    join_room(f"user_{username}")

    add_server_log(f"用户 {username} ({role}) 建立 WebSocket 连接")

    # 广播用户状态更新
    broadcast_user_status()

    # 如果是服务器用户，发送客户端数据
    if role == "server":
        broadcast_client_data_update()

    return True


@socketio.on("disconnect")
def handle_disconnect():
    """用户断开连接事件"""
    session_id = request.sid

    if session_id in online_users:
        user_info = online_users[session_id]
        username = user_info["username"]
        role = user_info["role"]

        # 清理在线状态
        del online_users[session_id]

        if username in user_sessions:
            user_sessions[username].discard(session_id)
            if not user_sessions[username]:  # 如果用户没有其他连接
                del user_sessions[username]

        add_server_log(f"用户 {username} ({role}) 断开 WebSocket 连接")

        # 广播用户状态更新
        broadcast_user_status()


@socketio.on("request_status_update")
def handle_status_request():
    """处理状态更新请求"""
    if "username" not in session:
        return

    role = session["role"]

    # 发送用户状态
    broadcast_user_status()

    # 如果是服务器用户，还要发送客户端数据和训练状态
    if role == "server":
        broadcast_client_data_update()
        broadcast_training_status()


@socketio.on("heartbeat")
def handle_heartbeat():
    """心跳检测"""
    emit("heartbeat_response", {"timestamp": datetime.now().isoformat()})


@socketio.on("request_logs")
def handle_logs_request(data):
    """处理日志请求"""
    if "username" not in session or session["role"] != "server":
        return

    log_type = data.get("type", "server")

    if log_type == "server":
        logs = get_logs_list(server_logs)
    elif log_type == "training":
        logs = get_logs_list(training_logs)
    else:
        logs = []

    emit(
        "logs_update",
        {
            "type": log_type,
            "logs": logs,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
    )


# WebSocket handlers for training data
@socketio.on("join_training_room")
def handle_join_training_room(data):
    """客户端加入训练监控房间"""
    if "username" not in session:
        return

    username = session["username"]
    role = session["role"]
    client_id = data.get("client_id", username)

    print(f"用户 {username} (角色: {role}) 加入训练监控房间，客户端ID: {client_id}")

    # 加入对应的训练监控房间
    if role == "client":
        # 客户端加入特定的训练监控房间
        join_room(f"client_{client_id}_training")
        join_room("server_training")  # 也加入服务器房间接收总体数据
    elif role == "server":
        join_room("server_training")

    emit(
        "training_room_joined",
        {"status": "success", "room": f"{role}_training", "client_id": client_id},
    )


@socketio.on("request_training_data")
def handle_training_data_request(data):
    """处理训练数据请求"""
    if "username" not in session:
        return

    username = session["username"]
    role = session["role"]
    client_id = data.get("client_id", username)

    # 返回请求的训练数据（这里可以从存储中获取历史数据）
    emit(
        "training_data_response",
        {
            "client_id": client_id,
            "data": [],  # 这里可以添加历史数据获取逻辑
            "timestamp": datetime.now().isoformat(),
        },
    )


# REST API endpoints for training data
@app.route("/api/training/client/<client_id>/data", methods=["GET"])
def get_client_training_data(client_id):
    """获取特定客户端的训练数据"""
    if "username" not in session or session["role"] not in ["server", "client"]:
        return jsonify({"error": "未授权"}), 401

    # 如果是客户端，只能访问自己的数据
    if session["role"] == "client" and session["username"] != client_id:
        return jsonify({"error": "权限不足"}), 403

    # 这里可以从数据库或文件系统获取训练数据
    # 目前返回空数据，实际实现时可以添加数据持久化
    return jsonify(
        {
            "client_id": client_id,
            "training_data": [],
            "timestamp": datetime.now().isoformat(),
        }
    )


@app.route("/api/training/server/data", methods=["GET"])
def get_server_training_data():
    """获取服务器端训练数据"""
    if "username" not in session or session["role"] != "server":
        return jsonify({"error": "未授权"}), 401

    # 返回服务器端聚合训练数据
    return jsonify(
        {
            "server_data": [],
            "all_clients_data": {},
            "timestamp": datetime.now().isoformat(),
        }
    )


@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        # 使用supabase认证
        auth_result = authenticate_user(username, password)

        if auth_result:
            session["username"] = username
            session["role"] = auth_result["role"]
            session["email"] = auth_result.get("email", "")
            session["auth_source"] = auth_result["source"]

            add_server_log(
                f"用户 {username} ({auth_result['role']}) 登录成功 [来源: {auth_result['source']}]"
            )

            if auth_result["role"] == "server":
                return redirect(url_for("server_dashboard"))
            else:
                # 初始化客户端数据状态（如果尚不存在）
                if username not in client_data_status:
                    client_data_status[username] = {
                        "uploaded": False,
                        "data_path": None,
                        "last_login": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    }
                else:
                    client_data_status[username][
                        "last_login"
                    ] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                return redirect(url_for("client_dashboard"))

        add_server_log(f"用户 {username} 登录失败 - 无效凭据")
        # 对于AJAX请求，返回错误信息
        if (
            request.is_json
            or request.headers.get("Content-Type")
            == "application/x-www-form-urlencoded"
        ):
            return "无效的凭据", 400
        return redirect(url_for("login", error="1"))
    return render_template("login.html")


@app.route("/logout")
def logout():
    username = session.get("username", "未知用户")
    add_server_log(f"用户 {username} 登出")
    session.pop("username", None)
    session.pop("role", None)
    return redirect(url_for("login"))


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        # 基本验证
        if not username or not email or not password:
            flash("所有字段都是必填的", "error")
            return render_template("register.html")

        if len(username) < 3:
            flash("用户名至少需要3个字符", "error")
            return render_template("register.html")

        if len(password) < 6:
            flash("密码至少需要6个字符", "error")
            return render_template("register.html")

        if password != confirm_password:
            flash("密码不匹配", "error")
            return render_template("register.html")

        # 检查用户是否已存在
        if SUPABASE_AVAILABLE and supabase:
            existing_user = get_user_from_supabase(username)
            if existing_user:
                flash("用户名已存在", "error")
                return render_template("register.html")
        else:
            # 检查本地用户
            local_users = load_local_users()
            if username in local_users:
                flash("用户名已存在", "error")
                return render_template("register.html")

        # 创建用户
        success = False

        # 优先尝试Supabase
        if SUPABASE_AVAILABLE and supabase:
            success = create_user_in_supabase(
                username, email, password, "client"
            )  # 新注册用户默认为客户端
            if success:
                add_server_log(f"新用户 {username} 注册成功 [Supabase]")
                flash("注册成功！请登录", "success")
            else:
                flash("注册失败，请稍后重试", "error")

        # 备用：本地存储
        if not success:
            try:
                local_users = load_local_users()
                local_users[username] = {
                    "password": hash_password(password),
                    "role": "client",  # 新注册用户默认为客户端
                    "email": email,
                    "created_at": datetime.now().isoformat(),
                    "is_active": True,
                }
                save_local_users(local_users)
                add_server_log(f"新用户 {username} 注册成功 [本地存储]")
                flash("注册成功！请登录", "success")
                success = True
            except Exception as e:
                print(f"❌ 本地注册失败: {e}")
                flash("注册失败，请稍后重试", "error")

        if success:
            return redirect(url_for("login"))

    return render_template("register.html")


@app.route("/client/dashboard", methods=["GET"])
def client_dashboard():
    if "username" not in session or session["role"] != "client":
        return redirect(url_for("login"))

    status = client_data_status.get(
        session["username"], {"uploaded": False, "data_path": None}
    )
    return render_template(
        "client_dashboard.html", username=session["username"], status=status
    )  # 您需要创建此HTML文件


@app.route("/client/upload", methods=["POST"])
def upload_file():
    if "username" not in session or session["role"] != "client":
        return jsonify({"error": "未授权"}), 403

    username = session["username"]

    # 支持多文件上传 - 检查是否有文件
    uploaded_files = request.files.getlist("files")
    if not uploaded_files or len(uploaded_files) == 0:
        return jsonify({"error": "未选择文件"}), 400

    client_folder_name = f"{username}_data"
    client_upload_path = os.path.join(UPLOAD_FOLDER, client_folder_name)

    if not os.path.exists(client_upload_path):
        os.makedirs(client_upload_path)

    uploaded_count = 0
    mhd_count = 0
    raw_count = 0
    uploaded_filenames = []

    # 处理每个上传的文件
    for file in uploaded_files:
        if file.filename == "":
            continue

        filename = file.filename
        file_path = os.path.join(client_upload_path, filename)

        # 如果文件已存在，生成新的文件名
        counter = 1
        base_name, ext = os.path.splitext(filename)
        while os.path.exists(file_path):
            new_filename = f"{base_name}_{counter}{ext}"
            file_path = os.path.join(client_upload_path, new_filename)
            filename = new_filename
            counter += 1

        file.save(file_path)
        uploaded_filenames.append(filename)
        uploaded_count += 1

        # 统计文件类型
        if filename.lower().endswith(".mhd"):
            mhd_count += 1
        elif filename.lower().endswith(".raw"):
            raw_count += 1

    if uploaded_count == 0:
        return jsonify({"error": "没有成功上传文件"}), 400

    # 计算总文件数（包括之前上传的）
    all_files = os.listdir(client_upload_path)
    total_mhd = len([f for f in all_files if f.lower().endswith(".mhd")])
    total_raw = len([f for f in all_files if f.lower().endswith(".raw")])
    # 只计算有效的医学影像文件（.mhd和.raw）
    total_files = total_mhd + total_raw

    # 更新客户端状态
    client_data_status[username] = {
        "uploaded": True,
        "data_path": client_upload_path,
        "upload_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "file_count": total_files,
        "mhd_count": total_mhd,
        "raw_count": total_raw,
        "file_pairs": min(total_mhd, total_raw),  # 有效的文件对数
        "last_login": client_data_status.get(username, {}).get("last_login", "未知"),
    }

    add_server_log(
        f"客户端 {username} 上传 {uploaded_count} 个文件: {', '.join(uploaded_filenames[:3])}{('...' if len(uploaded_filenames) > 3 else '')}"
    )

    # 更新数据变化时间戳，用于触发页面刷新
    global data_change_timestamp
    data_change_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 实时广播客户端数据更新
    broadcast_client_data_update()
    broadcast_user_status()

    return jsonify(
        {
            "message": f"成功上传 {uploaded_count} 个文件",
            "data_path": client_upload_path,
            "uploaded_files": uploaded_filenames,
            "total_files": total_files,
            "mhd_count": total_mhd,
            "raw_count": total_raw,
            "file_pairs": min(total_mhd, total_raw),
        }
    )


@app.route("/server/dashboard", methods=["GET"])
def server_dashboard():
    if "username" not in session or session["role"] != "server":
        return redirect(url_for("login"))

    # 获取所有客户端用户信息
    active_clients_info = []

    # 优先从Supabase获取用户列表
    if SUPABASE_AVAILABLE and supabase:
        try:
            response = supabase.table("users").select("*").execute()
            if response.data:
                for user_data in response.data:
                    if user_data["username"] != "server":  # 排除服务器用户
                        status = client_data_status.get(
                            user_data["username"],
                            {
                                "uploaded": False,
                                "data_path": None,
                                "last_login": "从未登录",
                                "upload_time": "从未上传",
                                "file_count": 0,
                            },
                        )
                        active_clients_info.append(
                            {
                                "username": user_data["username"],
                                "logged_in": True,  # 简化：假设如果存在于数据库就可能登录
                                "file_uploaded": status["uploaded"],
                                "last_login": status.get("last_login", "从未登录"),
                                "upload_time": status.get("upload_time", "从未上传"),
                                "file_count": status.get("file_count", 0),
                                "data_path": status.get("data_path", "无"),
                            }
                        )
        except Exception as e:
            print(f"⚠️ 从Supabase获取用户列表失败: {e}")

    # 备用：从本地用户数据获取
    if not active_clients_info:  # 如果Supabase获取失败，使用本地数据
        try:
            local_users = load_local_users()
            for uname, uinfo in local_users.items():
                if uinfo.get("role") == "client":
                    status = client_data_status.get(
                        uname,
                        {
                            "uploaded": False,
                            "data_path": None,
                            "last_login": "从未登录",
                            "upload_time": "从未上传",
                            "file_count": 0,
                        },
                    )
                    active_clients_info.append(
                        {
                            "username": uname,
                            "logged_in": True,  # 简化：假设如果存在于本地用户就可能登录
                            "file_uploaded": status["uploaded"],
                            "last_login": status.get("last_login", "从未登录"),
                            "upload_time": status.get("upload_time", "从未上传"),
                            "file_count": status.get("file_count", 0),
                            "data_path": status.get("data_path", "无"),
                        }
                    )
        except Exception as e:
            print(f"❌ 从本地获取用户列表失败: {e}")

    # 最后的备用：使用默认用户（只有在前面都失败时）
    if not active_clients_info:
        for uname, uinfo in DEFAULT_USERS.items():
            if uinfo["role"] == "client":
                status = client_data_status.get(
                    uname,
                    {
                        "uploaded": False,
                        "data_path": None,
                        "last_login": "从未登录",
                        "upload_time": "从未上传",
                        "file_count": 0,
                    },
                )
                active_clients_info.append(
                    {
                        "username": uname,
                        "logged_in": True,  # 简化：假设如果存在于默认用户就可能登录
                        "file_uploaded": status["uploaded"],
                        "last_login": status.get("last_login", "从未登录"),
                        "upload_time": status.get("upload_time", "从未上传"),
                        "file_count": status.get("file_count", 0),
                        "data_path": status.get("data_path", "无"),
                    }
                )

    return render_template(
        "server_dashboard.html",
        clients=active_clients_info,
        training_status=training_status,
    )  # 您需要创建此HTML文件


@app.route("/server/start_training", methods=["POST"])
def start_training():
    if "username" not in session or session["role"] != "server":
        return jsonify({"error": "未授权"}), 403

    if training_status["is_training"]:
        return jsonify({"error": "训练正在进行中"}), 400

    # 获取训练参数（从请求中）
    data = request.get_json() if request.is_json else {}
    global_rounds = data.get("global_rounds", 5)  # 默认5轮
    local_epochs = data.get("local_epochs", 2)  # 默认2个本地epochs

    # 参数验证
    global_rounds = max(1, min(20, int(global_rounds)))  # 限制在1-20之间
    local_epochs = max(1, min(10, int(local_epochs)))  # 限制在1-10之间

    client_paths_for_training = []
    for client_name, status in client_data_status.items():
        if status.get("uploaded") and status.get("data_path"):
            client_paths_for_training.append(status["data_path"])

    if not client_paths_for_training:
        add_server_log("训练启动失败 - 没有客户端上传数据")
        return jsonify({"error": "没有客户端上传数据用于训练"}), 400

    num_active_clients = len(client_paths_for_training)

    # 更新训练状态
    training_status.update(
        {
            "is_training": True,
            "current_round": 0,
            "total_rounds": global_rounds,
            "active_clients": num_active_clients,
            "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": None,
            "progress": 0,
        }
    )

    add_server_log(
        f"开始联邦学习训练 - {num_active_clients} 个客户端参与 (全局轮数: {global_rounds}, 本地轮数: {local_epochs})"
    )
    add_training_log(f"训练初始化 - 客户端数据目录: {client_paths_for_training}")

    def run_training():
        try:
            add_training_log("正在启动联邦学习训练...")
            add_training_log(f"参与训练的客户端数量: {num_active_clients}")
            add_training_log(f"全局训练轮数: {global_rounds}")
            add_training_log(f"本地训练轮数: {local_epochs}")
            add_training_log(f"客户端数据路径: {client_paths_for_training}")

            add_training_log("训练日志系统已初始化")

            coordinator = train_federated_model(
                num_clients=num_active_clients,
                global_rounds=global_rounds,
                local_epochs=local_epochs,
                client_data_dirs=client_paths_for_training,
            )

            # 更新训练状态
            training_status.update(
                {
                    "is_training": False,
                    "current_round": global_rounds,
                    "end_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "progress": 100,
                }
            )

            if coordinator:
                add_server_log("联邦学习训练成功完成")
                add_training_log("训练完成，模型已保存")
            else:
                add_server_log("联邦学习训练失败")
                add_training_log("训练失败 - 协调器未成功初始化")

        except Exception as e:
            training_status.update(
                {
                    "is_training": False,
                    "end_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
            )
            add_server_log(f"训练过程中发生错误: {e}")
            add_training_log(f"训练异常终止: {str(e)}")
        finally:
            # 无论成功还是失败，都广播训练状态更新
            broadcast_training_status()

    # 在后台线程中运行训练
    training_thread = threading.Thread(target=run_training)
    training_thread.daemon = True
    training_thread.start()

    # 广播训练开始状态
    broadcast_training_status()

    return jsonify({"message": "训练已启动，请查看训练日志获取进度"})


@app.route("/api/server/status", methods=["GET"])
def get_server_status():
    """获取服务器状态API"""
    if "username" not in session or session["role"] != "server":
        return jsonify({"error": "未授权"}), 403

    client_count = len([c for c in client_data_status.values() if c.get("uploaded")])
    total_clients = len([u for u, info in users.items() if info["role"] == "client"])

    response_data = {
        "training_status": training_status,
        "client_count": client_count,
        "total_clients": total_clients,
        "server_logs": get_logs_list(server_logs),
        "training_logs": get_logs_list(training_logs),
        "data_change_timestamp": data_change_timestamp,
    }

    return jsonify(response_data)


@app.route("/api/server/logs", methods=["GET"])
def get_logs():
    """获取日志API"""
    if "username" not in session or session["role"] != "server":
        return jsonify({"error": "未授权"}), 403

    log_type = request.args.get("type", "server")

    if log_type == "training":
        logs = get_logs_list(training_logs)
    else:
        logs = get_logs_list(server_logs)

    return jsonify({"logs": logs})


# 自动检测和初始化客户端数据状态
def initialize_client_data_status():
    """自动检测uploads目录中的客户端数据并初始化状态"""
    global client_data_status

    print("正在检测客户端数据...")

    for username in users:
        if users[username]["role"] == "client":
            client_data_dir = os.path.join(UPLOAD_FOLDER, f"{username}_data")

            # 检查客户端数据目录是否存在
            if os.path.exists(client_data_dir):
                # 检查是否有有效的数据文件
                files = os.listdir(client_data_dir)
                mhd_files = [f for f in files if f.endswith(".mhd")]
                raw_files = [f for f in files if f.endswith(".raw")]

                if mhd_files and raw_files:
                    client_data_status[username] = {
                        "uploaded": True,
                        "data_path": client_data_dir,
                        "last_login": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "file_count": len(mhd_files),
                    }
                    print(f"✅ 检测到 {username} 的数据: {len(mhd_files)} 个文件")
                else:
                    print(f"⚠️ {username} 的数据目录存在但无有效数据文件")
            else:
                print(f"❌ {username} 的数据目录不存在: {client_data_dir}")


# 推理相关API端点
@app.route("/api/server/upload_inference_file", methods=["POST"])
def upload_inference_file():
    """上传推理文件（支持多文件上传）"""
    if "username" not in session or session["role"] != "server":
        return jsonify({"error": "未授权"}), 403

    files = request.files.getlist("files")
    if not files or len(files) == 0:
        return jsonify({"error": "未选择文件"}), 400

    uploaded_files = []
    errors = []

    # 验证文件类型和配对
    mhd_files = []
    raw_files = []

    for file in files:
        if file.filename == "":
            continue

        if file.filename.lower().endswith(".mhd"):
            mhd_files.append(file)
        elif file.filename.lower().endswith(".raw"):
            raw_files.append(file)
        else:
            errors.append(f"不支持的文件类型: {file.filename}")

    if not mhd_files:
        return jsonify({"error": "必须包含至少一个.mhd文件"}), 400

    # 验证每个.mhd文件都有对应的.raw文件
    for mhd_file in mhd_files:
        base_name = os.path.splitext(mhd_file.filename)[0]
        raw_filename = base_name + ".raw"

        # 查找对应的.raw文件
        corresponding_raw = None
        for raw_file in raw_files:
            if raw_file.filename == raw_filename:
                corresponding_raw = raw_file
                break

        if not corresponding_raw:
            errors.append(f"缺少对应的.raw文件: {raw_filename}")
            continue

        # 上传文件对
        try:
            # 处理文件名冲突
            mhd_filename = mhd_file.filename
            raw_filename = corresponding_raw.filename

            counter = 1
            base_name = os.path.splitext(mhd_filename)[0]

            while os.path.exists(
                os.path.join(INFERENCE_UPLOAD_FOLDER, mhd_filename)
            ) or os.path.exists(os.path.join(INFERENCE_UPLOAD_FOLDER, raw_filename)):
                mhd_filename = f"{base_name}_{counter}.mhd"
                raw_filename = f"{base_name}_{counter}.raw"
                counter += 1

            # 保存.mhd文件
            mhd_path = os.path.join(INFERENCE_UPLOAD_FOLDER, mhd_filename)
            mhd_file.save(mhd_path)

            # 保存.raw文件
            raw_path = os.path.join(INFERENCE_UPLOAD_FOLDER, raw_filename)
            corresponding_raw.save(raw_path)

            # 只将.mhd文件添加到状态中（推理时使用）
            inference_status["uploaded_files"].append(
                {
                    "name": mhd_filename,
                    "path": mhd_path,
                    "raw_file": raw_filename,
                    "upload_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
            )

            uploaded_files.extend([mhd_filename, raw_filename])
            add_server_log(f"推理文件对上传成功: {mhd_filename}, {raw_filename}")

        except Exception as e:
            errors.append(f"上传文件对失败 {mhd_file.filename}: {str(e)}")

    if not uploaded_files and errors:
        return jsonify({"error": f"上传失败: {'; '.join(errors)}"}), 500

    response_message = f"成功上传 {len(uploaded_files)} 个文件"
    if errors:
        response_message += f" (部分错误: {'; '.join(errors)})"

    return jsonify(
        {
            "message": response_message,
            "uploaded_files": uploaded_files,
            "errors": errors,
        }
    )


@app.route("/api/server/list_inference_files", methods=["GET"])
def list_inference_files():
    """获取已上传的推理文件列表"""
    if "username" not in session or session["role"] != "server":
        return jsonify({"error": "未授权"}), 403

    files = []
    if os.path.exists(INFERENCE_UPLOAD_FOLDER):
        for filename in os.listdir(INFERENCE_UPLOAD_FOLDER):
            if filename.lower().endswith(".mhd"):
                file_path = os.path.join(INFERENCE_UPLOAD_FOLDER, filename)
                stat = os.stat(file_path)

                # 检查对应的.raw文件是否存在
                base_name = os.path.splitext(filename)[0]
                raw_filename = base_name + ".raw"
                raw_path = os.path.join(INFERENCE_UPLOAD_FOLDER, raw_filename)
                has_raw_file = os.path.exists(raw_path)

                raw_size = 0
                if has_raw_file:
                    raw_stat = os.stat(raw_path)
                    raw_size = raw_stat.st_size

                files.append(
                    {
                        "name": filename,
                        "path": file_path,
                        "size": stat.st_size,
                        "raw_file": raw_filename if has_raw_file else None,
                        "raw_size": raw_size,
                        "total_size": stat.st_size + raw_size,
                        "has_pair": has_raw_file,
                        "upload_time": datetime.fromtimestamp(stat.st_mtime).strftime(
                            "%Y-%m-%d %H:%M:%S"
                        ),
                    }
                )

    return jsonify({"files": files})


@app.route("/api/server/run_inference", methods=["POST"])
def run_inference_api():
    """运行推理"""
    if "username" not in session or session["role"] != "server":
        return jsonify({"error": "未授权"}), 403

    if inference_status["is_running"]:
        return jsonify({"error": "推理正在运行中"}), 400

    data = request.get_json()
    filename = data.get("filename")
    use_federated = data.get("use_federated", True)
    fast_mode = data.get("fast_mode", False)

    if not filename:
        return jsonify({"error": "未指定文件名"}), 400

    file_path = os.path.join(INFERENCE_UPLOAD_FOLDER, filename)
    if not os.path.exists(file_path):
        return jsonify({"error": "MHD文件不存在"}), 404

    # 检查对应的.raw文件是否存在
    base_name = os.path.splitext(filename)[0]
    raw_filename = base_name + ".raw"
    raw_path = os.path.join(INFERENCE_UPLOAD_FOLDER, raw_filename)
    if not os.path.exists(raw_path):
        return jsonify({"error": f"对应的RAW文件不存在: {raw_filename}"}), 404

    # 重置推理状态
    inference_status.update(
        {
            "is_running": True,
            "progress": 0,
            "current_step": "初始化推理...",
            "result_image": None,
            "error": None,
            "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": None,
        }
    )

    def run_inference_thread():
        try:
            add_server_log(f"开始推理: {filename}")
            inference_status["current_step"] = "加载模型..."
            inference_status["progress"] = 10

            # 导入推理函数
            if use_federated:
                from federated_inference_utils import (
                    predict_with_federated_model,
                    visualize_federated_results,
                )

                inference_status["current_step"] = "使用联邦模型进行预测..."
                inference_status["progress"] = 30

                # 运行推理，支持快速模式
                nodules, prob_map, image, spacing, origin = (
                    predict_with_federated_model(file_path, fast_mode=fast_mode)
                )

                inference_status["current_step"] = "生成可视化结果..."
                inference_status["progress"] = 70

                # 生成结果图像并保存为base64
                result_path = visualize_federated_results(
                    image, prob_map, nodules, spacing, origin, save_path=True
                )

            else:
                from show_nodules import show_predicted_nodules

                inference_status["current_step"] = "使用快速模式进行预测..."
                inference_status["progress"] = 50

                result_path = show_predicted_nodules(
                    file_path, confidence_threshold=0.3, save_result=True
                )

            # 读取结果图像并转换为base64
            if result_path and os.path.exists(result_path):
                with open(result_path, "rb") as img_file:
                    img_data = img_file.read()
                    base64_image = base64.b64encode(img_data).decode("utf-8")
                    inference_status["result_image"] = (
                        f"data:image/png;base64,{base64_image}"
                    )

            inference_status["current_step"] = "推理完成"
            inference_status["progress"] = 100
            inference_status["is_running"] = False
            inference_status["end_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            add_server_log(f"推理完成: {filename}")

        except Exception as e:
            inference_status.update(
                {
                    "is_running": False,
                    "error": str(e),
                    "end_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
            )
            add_server_log(f"推理失败: {str(e)}")

    # 在后台线程中运行推理
    inference_thread = threading.Thread(target=run_inference_thread)
    inference_thread.daemon = True
    inference_thread.start()

    return jsonify({"message": "推理已启动"})


@app.route("/api/server/get_inference_status", methods=["GET"])
def get_inference_status():
    """获取推理状态"""
    if "username" not in session or session["role"] != "server":
        return jsonify({"error": "未授权"}), 403

    return jsonify(inference_status)


@app.route("/api/server/get_inference_result", methods=["GET"])
def get_inference_result():
    """获取推理结果图像"""
    if "username" not in session or session["role"] != "server":
        return jsonify({"error": "未授权"}), 403

    if inference_status["result_image"]:
        return jsonify({"result_image": inference_status["result_image"]})
    else:
        return jsonify({"error": "暂无结果图像"}), 404


@app.route("/api/server/delete_inference_file", methods=["DELETE"])
def delete_inference_file():
    """删除推理文件"""
    if "username" not in session or session["role"] != "server":
        return jsonify({"error": "未授权"}), 403

    data = request.get_json()
    filename = data.get("filename")

    if not filename:
        return jsonify({"error": "未指定文件名"}), 400

    # 删除.mhd文件
    mhd_path = os.path.join(INFERENCE_UPLOAD_FOLDER, filename)

    # 获取对应的.raw文件名
    base_name = os.path.splitext(filename)[0]
    raw_filename = base_name + ".raw"
    raw_path = os.path.join(INFERENCE_UPLOAD_FOLDER, raw_filename)

    deleted_files = []
    errors = []

    # 删除.mhd文件
    if os.path.exists(mhd_path):
        try:
            os.remove(mhd_path)
            deleted_files.append(filename)
            add_server_log(f"已删除推理文件: {filename}")
        except Exception as e:
            errors.append(f"删除{filename}失败: {str(e)}")

    # 删除对应的.raw文件
    if os.path.exists(raw_path):
        try:
            os.remove(raw_path)
            deleted_files.append(raw_filename)
            add_server_log(f"已删除推理文件: {raw_filename}")
        except Exception as e:
            errors.append(f"删除{raw_filename}失败: {str(e)}")

    if not deleted_files:
        return jsonify({"error": "文件不存在"}), 404

    # 从上传文件列表中移除
    inference_status["uploaded_files"] = [
        f
        for f in inference_status["uploaded_files"]
        if f["name"] not in [filename, raw_filename]
    ]

    response_message = f"已删除文件: {', '.join(deleted_files)}"
    if errors:
        response_message += f" (部分错误: {', '.join(errors)})"

    return jsonify({"message": response_message, "deleted_files": deleted_files})


# ============================
# 客户端推理相关API端点
# ============================


@app.route("/api/client/upload_inference_file", methods=["POST"])
def client_upload_inference_file():
    """客户端上传推理文件（支持多文件上传）"""
    if "username" not in session or session["role"] != "client":
        return jsonify({"error": "未授权"}), 403

    username = session["username"]
    files = request.files.getlist("files")
    if not files or len(files) == 0:
        return jsonify({"error": "未选择文件"}), 400

    # 为每个客户端创建独立的推理文件夹
    client_inference_folder = os.path.join(
        CLIENT_INFERENCE_UPLOAD_FOLDER, f"{username}_inference"
    )
    if not os.path.exists(client_inference_folder):
        os.makedirs(client_inference_folder)

    uploaded_files = []
    errors = []

    # 验证文件类型和配对
    mhd_files = []
    raw_files = []

    for file in files:
        if file.filename == "":
            continue

        if file.filename.lower().endswith(".mhd"):
            mhd_files.append(file)
        elif file.filename.lower().endswith(".raw"):
            raw_files.append(file)
        else:
            errors.append(f"不支持的文件类型: {file.filename}")

    if not mhd_files:
        return jsonify({"error": "必须包含至少一个.mhd文件"}), 400

    # 验证每个.mhd文件都有对应的.raw文件
    for mhd_file in mhd_files:
        base_name = os.path.splitext(mhd_file.filename)[0]
        raw_filename = base_name + ".raw"

        # 查找对应的.raw文件
        corresponding_raw = None
        for raw_file in raw_files:
            if raw_file.filename == raw_filename:
                corresponding_raw = raw_file
                break

        if not corresponding_raw:
            errors.append(f"缺少对应的.raw文件: {raw_filename}")
            continue

        # 上传文件对
        try:
            # 处理文件名冲突
            mhd_filename = mhd_file.filename
            raw_filename = corresponding_raw.filename

            counter = 1
            base_name = os.path.splitext(mhd_filename)[0]

            while os.path.exists(
                os.path.join(client_inference_folder, mhd_filename)
            ) or os.path.exists(os.path.join(client_inference_folder, raw_filename)):
                mhd_filename = f"{base_name}_{counter}.mhd"
                raw_filename = f"{base_name}_{counter}.raw"
                counter += 1

            # 保存.mhd文件
            mhd_path = os.path.join(client_inference_folder, mhd_filename)
            mhd_file.save(mhd_path)

            # 保存.raw文件
            raw_path = os.path.join(client_inference_folder, raw_filename)
            corresponding_raw.save(raw_path)

            # 只将.mhd文件添加到状态中（推理时使用）
            client_inference_status["uploaded_files"].append(
                {
                    "name": mhd_filename,
                    "path": mhd_path,
                    "raw_file": raw_filename,
                    "upload_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
            )

            uploaded_files.extend([mhd_filename, raw_filename])
            add_server_log(
                f"客户端 {username} 推理文件对上传成功: {mhd_filename}, {raw_filename}"
            )

        except Exception as e:
            errors.append(f"上传文件对失败 {mhd_file.filename}: {str(e)}")

    if not uploaded_files and errors:
        return jsonify({"error": f"上传失败: {'; '.join(errors)}"}), 500

    response_message = f"成功上传 {len(uploaded_files)} 个文件"
    if errors:
        response_message += f" (部分错误: {'; '.join(errors)})"

    return jsonify(
        {
            "message": response_message,
            "uploaded_files": uploaded_files,
            "errors": errors,
        }
    )


@app.route("/api/client/list_inference_files", methods=["GET"])
def client_list_inference_files():
    """获取客户端已上传的推理文件列表"""
    if "username" not in session or session["role"] != "client":
        return jsonify({"error": "未授权"}), 403

    username = session["username"]
    client_inference_folder = os.path.join(
        CLIENT_INFERENCE_UPLOAD_FOLDER, f"{username}_inference"
    )

    files = []
    if os.path.exists(client_inference_folder):
        for filename in os.listdir(client_inference_folder):
            if filename.lower().endswith(".mhd"):
                file_path = os.path.join(client_inference_folder, filename)
                stat = os.stat(file_path)

                # 检查对应的.raw文件是否存在
                base_name = os.path.splitext(filename)[0]
                raw_filename = base_name + ".raw"
                raw_path = os.path.join(client_inference_folder, raw_filename)
                has_raw_file = os.path.exists(raw_path)

                raw_size = 0
                if has_raw_file:
                    raw_stat = os.stat(raw_path)
                    raw_size = raw_stat.st_size

                files.append(
                    {
                        "name": filename,
                        "path": file_path,
                        "size": stat.st_size,
                        "raw_file": raw_filename if has_raw_file else None,
                        "raw_size": raw_size,
                        "total_size": stat.st_size + raw_size,
                        "has_pair": has_raw_file,
                        "upload_time": datetime.fromtimestamp(stat.st_mtime).strftime(
                            "%Y-%m-%d %H:%M:%S"
                        ),
                    }
                )

    return jsonify({"files": files})


@app.route("/api/client/run_inference", methods=["POST"])
def client_run_inference_api():
    """客户端运行推理"""
    if "username" not in session or session["role"] != "client":
        return jsonify({"error": "未授权"}), 403

    if client_inference_status["is_running"]:
        return jsonify({"error": "推理正在运行中"}), 400

    username = session["username"]
    data = request.get_json()
    filename = data.get("filename")
    use_federated = data.get("use_federated", True)
    fast_mode = data.get("fast_mode", False)

    if not filename:
        return jsonify({"error": "未指定文件名"}), 400

    client_inference_folder = os.path.join(
        CLIENT_INFERENCE_UPLOAD_FOLDER, f"{username}_inference"
    )
    file_path = os.path.join(client_inference_folder, filename)
    if not os.path.exists(file_path):
        return jsonify({"error": "MHD文件不存在"}), 404

    # 检查对应的.raw文件是否存在
    base_name = os.path.splitext(filename)[0]
    raw_filename = base_name + ".raw"
    raw_path = os.path.join(client_inference_folder, raw_filename)
    if not os.path.exists(raw_path):
        return jsonify({"error": f"对应的RAW文件不存在: {raw_filename}"}), 404

    # 重置客户端推理状态
    client_inference_status.update(
        {
            "is_running": True,
            "progress": 0,
            "current_step": "初始化推理...",
            "result_image": None,
            "error": None,
            "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": None,
        }
    )

    def run_client_inference_thread():
        try:
            add_server_log(f"客户端 {username} 开始推理: {filename}")
            client_inference_status["current_step"] = "加载模型..."
            client_inference_status["progress"] = 10

            # 导入推理函数
            if use_federated:
                from federated_inference_utils import (
                    predict_with_federated_model,
                    visualize_federated_results,
                )

                client_inference_status["current_step"] = "使用联邦模型进行预测..."
                client_inference_status["progress"] = 30

                # 运行推理，支持快速模式
                nodules, prob_map, image, spacing, origin = (
                    predict_with_federated_model(file_path, fast_mode=fast_mode)
                )

                client_inference_status["current_step"] = "生成可视化结果..."
                client_inference_status["progress"] = 70

                # 生成结果图像并保存为base64
                result_path = visualize_federated_results(
                    image, prob_map, nodules, spacing, origin, save_path=True
                )

            else:
                from show_nodules import show_predicted_nodules

                client_inference_status["current_step"] = "使用快速模式进行预测..."
                client_inference_status["progress"] = 50

                result_path = show_predicted_nodules(
                    file_path, confidence_threshold=0.3, save_result=True
                )

            # 读取结果图像并转换为base64
            if result_path and os.path.exists(result_path):
                with open(result_path, "rb") as img_file:
                    img_data = img_file.read()
                    base64_image = base64.b64encode(img_data).decode("utf-8")
                    client_inference_status["result_image"] = (
                        f"data:image/png;base64,{base64_image}"
                    )

            client_inference_status["current_step"] = "推理完成"
            client_inference_status["progress"] = 100
            client_inference_status["is_running"] = False
            client_inference_status["end_time"] = datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )

            add_server_log(f"客户端 {username} 推理完成: {filename}")

        except Exception as e:
            client_inference_status.update(
                {
                    "is_running": False,
                    "error": str(e),
                    "end_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
            )
            add_server_log(f"客户端 {username} 推理失败: {str(e)}")

    # 在后台线程中运行推理
    inference_thread = threading.Thread(target=run_client_inference_thread)
    inference_thread.daemon = True
    inference_thread.start()

    return jsonify({"message": "推理已启动"})


@app.route("/api/client/get_inference_status", methods=["GET"])
def client_get_inference_status():
    """获取客户端推理状态"""
    if "username" not in session or session["role"] != "client":
        return jsonify({"error": "未授权"}), 403

    return jsonify(client_inference_status)


@app.route("/api/client/delete_inference_file", methods=["DELETE"])
def client_delete_inference_file():
    """删除客户端推理文件"""
    if "username" not in session or session["role"] != "client":
        return jsonify({"error": "未授权"}), 403

    username = session["username"]
    data = request.get_json()
    filename = data.get("filename")

    if not filename:
        return jsonify({"error": "未指定文件名"}), 400

    client_inference_folder = os.path.join(
        CLIENT_INFERENCE_UPLOAD_FOLDER, f"{username}_inference"
    )

    # 删除.mhd文件
    mhd_path = os.path.join(client_inference_folder, filename)

    # 获取对应的.raw文件名
    base_name = os.path.splitext(filename)[0]
    raw_filename = base_name + ".raw"
    raw_path = os.path.join(client_inference_folder, raw_filename)

    deleted_files = []
    errors = []

    # 删除.mhd文件
    if os.path.exists(mhd_path):
        try:
            os.remove(mhd_path)
            deleted_files.append(filename)
            add_server_log(f"客户端 {username} 已删除推理文件: {filename}")
        except Exception as e:
            errors.append(f"删除{filename}失败: {str(e)}")

    # 删除对应的.raw文件
    if os.path.exists(raw_path):
        try:
            os.remove(raw_path)
            deleted_files.append(raw_filename)
            add_server_log(f"客户端 {username} 已删除推理文件: {raw_filename}")
        except Exception as e:
            errors.append(f"删除{raw_filename}失败: {str(e)}")

    if not deleted_files:
        return jsonify({"error": "文件不存在"}), 404

    # 从上传文件列表中移除
    client_inference_status["uploaded_files"] = [
        f
        for f in client_inference_status["uploaded_files"]
        if f["name"] not in [filename, raw_filename]
    ]

    response_message = f"已删除文件: {', '.join(deleted_files)}"
    if errors:
        response_message += f" (部分错误: {', '.join(errors)})"

    return jsonify({"message": response_message, "deleted_files": deleted_files})


# 在应用启动时初始化客户端数据状态
initialize_client_data_status()

if __name__ == "__main__":
    # 添加一些初始日志
    add_server_log("服务器启动")
    add_server_log("等待客户端连接和上传数据")

    # 使用 SocketIO 运行应用
    socketio.run(app, port=5000)
