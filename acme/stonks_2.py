import ray
from time import sleep

import jax
jax.config.update('jax_platform_name', "cpu")

@ray.remote
class ActorOne():
  def __init__(self):
    pass

  def run(self):
    while True:
      print("actor one")
      sleep(2)

@ray.remote
class ActorTwo():
  def __init__(self):
    pass
  def run(self):
    while True:
      print("actor two")
      sleep(3)


if __name__ == "__main__":
  ray.init()

  one = ActorOne.remote()
  two = ActorTwo.remote()

  one.run.remote()
  two.run.remote()

  while True:
    sleep(5)