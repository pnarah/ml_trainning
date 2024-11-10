import uvicorn

if __name__ == "__main__":
    uvicorn.run("apicalls:app", host="0.0.0.0", port=80)

    # However tortoise ORM register_tortoise function relies on the lifespan event startup to register models. So I think
    # the lifespan mode on uvicorn should be on instead of the default auto. This way your server process will terminate
    # rather than serving a misconfigured app.
    # uvicorn.run("start_server:app", host="0.0.0.0", port=8080, log_level="info", lifespan='on')
