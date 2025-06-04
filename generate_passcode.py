import bcrypt


def hash_password(password: str) -> str:
    """使用bcrypt哈希密码"""
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


password = "1"
hashed_password = hash_password(password)
print(f"原始密码: {password}")
print(f"哈希密码: {hashed_password}")
