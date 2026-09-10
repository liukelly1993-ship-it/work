import uvicorn # uvicorn是ASGI服务器，用于运行FastAPI应用 #http1.1  /http2  nginx->uvicore

if __name__ == "__main__":
    # api目录下有个main.py文件，app是main.py中定义的FastAPI实例
    uvicorn.run("api.main:app", host="0.0.0.0", port=8080, reload=True)