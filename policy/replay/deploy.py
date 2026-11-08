from policy import loop


def eval_one_episode(TASK_ENV, model_client):
    loop.run(TASK_ENV, model_client, TASK_ENV.config["horizon"])
