from launchpad.nodes.python.local_multi_processing import PythonProcess
from absl import flags
FLAGS = flags.FLAGS

# Let's just write out what we need. We'll start with just actors.
# num_actors, num_actors_per_node, then we calculate num_actor_nodes.
# Then we need the CPU range for actors.
# Then we have a helper function that divides it up correctly
# Then we change the interpreter. Nice.
# We also want learner_gpus and 

def _get_num_actor_nodes(num_actors, num_actors_per_node):
  num_actor_nodes, remainder = divmod(num_actors, num_actors_per_node)
  num_actor_nodes += int(remainder > 0)
  return num_actor_nodes

def get_cpu_range(*, cpu_start, cpu_end, num_actors, num_actors_per_node, actor_num):
  # We'll add one to cpu_end to make the range contain all.
  # How about we make sure the range is evenly divisble.
  # Oh, this isn't right if we're doing actors_per_node. Darn.

  # If we had 4 actors and 2 CPUs you want the first 2 actors to share and the second 2.
  # 0-0.5, 0.5-1, 1-1.5, 1.5-2. Hmm. seems like it wants a round down, which is dumb.
  # Maybe we do a lt somehow. Yeah. You assign it to the ones it overlaps with.
  # so, [cpu for cpu in cpu_range if ]
  # assert num_actors_per_node == 1, f"for now num_actors_per_node is 1, got {num_actors_per_node}"
  # assert isinstance(cpu_start, int) and isinstance(cpu_end, int) and cpu_end >= cpu_start
  # assert isinstance(actor_num, int) and actor_num >= 0 and actor_num < num_actors 
  # assert isinstance(num_actors, int) and num_actors > 0
  # cpu_range = list(range(cpu_start, cpu_end + 1))
  # cpu_length = len(cpu_range)
  # assert cpu_length % num_actors == 0, f"just for now we'll assume evenly divisible, but got {num_actors} actors and {cpu_length} cpus"
  # cpus_per = cpu_length // num_actors
  # return cpu_range[actor_num*cpus_per:(actor_num+1)*cpus_per]

  assert num_actors_per_node == 1, f"for now num_actors_per_node is 1, got {num_actors_per_node}"
  assert isinstance(cpu_start, int) and isinstance(cpu_end, int) and cpu_end >= cpu_start
  assert isinstance(actor_num, int) and actor_num >= 0 and actor_num < num_actors 
  assert isinstance(num_actors, int) and num_actors > 0
  cpu_range = list(range(cpu_start, cpu_end + 1))
  cpu_length = len(cpu_range)
  # assert cpu_length % num_actors == 0, f"just for now we'll assume evenly divisible, but got {num_actors} actors and {cpu_length} cpus"
  cpus_per = cpu_length / num_actors
  first_cpu = cpu_start + (actor_num*cpus_per)
  last_cpu = cpu_start+ ((actor_num+1)*cpus_per)
  
  return [cpu for cpu in cpu_range if cpu >= int(first_cpu) and cpu < last_cpu]
  # cpu_start = actor_num*cpus_per
  # return cpu_range[actor_num*cpus_per:(actor_num+1)*cpus_per]




def get_range_str_from_list(range_list):
  return ",".join([str(cpu) for cpu in range_list])

def test_cpu_range():
  assert get_cpu_range(cpu_start=0, cpu_end=0, num_actors=1, num_actors_per_node=1, actor_num=0) == [0]
  assert get_cpu_range(cpu_start=3, cpu_end=3, num_actors=1, num_actors_per_node=1, actor_num=0) == [3]
  assert get_cpu_range(cpu_start=3, cpu_end=3, num_actors=2, num_actors_per_node=1, actor_num=0) == [3]
  assert get_cpu_range(cpu_start=3, cpu_end=3, num_actors=2, num_actors_per_node=1, actor_num=1) == [3]
  assert get_cpu_range(cpu_start=3, cpu_end=4, num_actors=1, num_actors_per_node=1, actor_num=0) == [3,4]
  assert get_cpu_range(cpu_start=3, cpu_end=4, num_actors=2, num_actors_per_node=1, actor_num=0) == [3]
  assert get_cpu_range(cpu_start=3, cpu_end=4, num_actors=2, num_actors_per_node=1, actor_num=1) == [4]
  print('great success on test_cpu_range!')

def modify_python_process(python_process, cpu_range=None):
  if not cpu_range:
    raise Exception("don't call")
    return python_process
  cpu_range_string = get_range_str_from_list(cpu_range)
  interpreter = python_process.absolute_interpreter_path
  new_interpreter_path = f"taskset -c {cpu_range_string} {interpreter}"
  print('setting interpreter path to', new_interpreter_path)

  python_process._absolute_interpreter_path = new_interpreter_path
  return python_process # not necessary

def make_actor_resources(num_actors, cpu_start=0, cpu_end=1, gpu_str="-1"):
  # If you don't specify CPU range, then just do them all like before I guess.
  if cpu_start < 0 or cpu_end < 0:
    print('doing the regular way')
    return {
      "actor" : PythonProcess(env={
        "CUDA_VISIBLE_DEVICES": gpu_str,
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "TF_FORCE_GPU_ALLOW_GROWTH": "true",
      })
    }
  process_dict = {}
  for actor_num in range(num_actors):
    print('doing the other way!')
    actor_key = f"actor_{actor_num}"
    process = PythonProcess(env={
      "CUDA_VISIBLE_DEVICES": gpu_str,
      "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
      "TF_FORCE_GPU_ALLOW_GROWTH": "true",
    })
    # Maybe this should be where we do the whole ulimit thing actually, it would be much less burid this way.
    # cpu_range = get_cpu_range(cpu_start=cpu_start, cpu_end=cpu_end, num_actors=num_actors, num_actors_per_node=1, actor_num=actor_num)
    # process = modify_python_process(process, cpu_range)
    process_dict[actor_key] = process

  return process_dict
    

def make_with_gpu_dict(gpu_str="0"):
  return PythonProcess(env={
    # "CUDA_VISIBLE_DEVICES": str(0),
    "CUDA_VISIBLE_DEVICES": gpu_str,
    "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    "TF_FORCE_GPU_ALLOW_GROWTH": "true",
    # # "XLA_FLAGS": r"--xla_cpu_multi_thread_eigen=false\ intra_op_parallelism_threads=1\ inter_op_parallelism_threads=1",
    # # "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false",
    # # "XLA_FLAGS": "intra_op_parallelism_threads=1",
    "XLA_FLAGS": r"--xla_force_host_platform_device_count=1\ --xla_cpu_multi_thread_eigen=false\ intra_op_parallelism_threads=1\ inter_op_parallelism_threads=1",
  },
  # pre_command="ulimit -u 100000"
  )

def make_without_gpu_dict():
  return PythonProcess(env={
    "CUDA_VISIBLE_DEVICES": str(-1),
    # # "XLA_FLAGS": "intra_op_parallelism_threads=1",
    # # "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false",
    # "XLA_FLAGS": r"--xla_cpu_multi_thread_eigen=false\ intra_op_parallelism_threads=1\ inter_op_parallelism_threads=1",
    # # "XLA_FLAGS": "--xla_force_host_platform_device_count=1"
    "XLA_FLAGS": r"--xla_force_host_platform_device_count=1\ --xla_cpu_multi_thread_eigen=false\ intra_op_parallelism_threads=1\ inter_op_parallelism_threads=1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OMP_NUM_THREAD": "1",
    },
    # pre_command="ulimit -u 100000"
  )

def _get_local_resources(launch_type):
  num_actors = FLAGS.num_actors
  num_actors_per_node = FLAGS.num_actors_per_node
  actor_cpu_start = FLAGS.actor_cpu_start
  actor_cpu_end = FLAGS.actor_cpu_end
  actor_use_gpu = FLAGS.actors_on_gpu

  assert num_actors_per_node == 1, num_actors_per_node

  assert launch_type in ('local_mp', 'local_mt'), launch_type
  from launchpad.nodes.python.local_multi_processing import PythonProcess
  if launch_type == 'local_mp':
    local_resources = {
      # "learner": make_without_gpu_dict() if FLAGS.learner_on_cpu else make_with_gpu_dict("0,1,2,3"), # reversed from other
      # "learner": make_without_gpu_dict() if FLAGS.learner_on_cpu else make_with_gpu_dict("0,1,2,3"), # reversed from other      # "inference_server": make_with_gpu_dict(),
      # "learner": make_without_gpu_dict() if FLAGS.learner_on_cpu else make_with_gpu_dict("0,1,2,3"), # reversed from other      # "inference_server": make_with_gpu_dict(),
      # "inference_server": make_with_gpu_dict("4,5,6,7"),
      "learner": make_without_gpu_dict() if FLAGS.learner_on_cpu else make_with_gpu_dict("0"), # reversed from other      # "inference_server": make_with_gpu_dict(),
      "inference_server": make_with_gpu_dict("1"),
      "counter": make_without_gpu_dict(),
      "replay": make_without_gpu_dict(),
      "evaluator": make_without_gpu_dict(),
      # "actor": make_with_gpu_dict() if FLAGS.actors_on_gpu else make_without_gpu_dict(),
    }
    actor_resources = make_actor_resources(
      num_actors=num_actors, cpu_start=actor_cpu_start, cpu_end=actor_cpu_end, gpu_str= "0" if actor_use_gpu else "-1")
    local_resources.update(actor_resources)
  else:
    local_resources = {}
  # import ipdb; ipdb.set_trace()
  return local_resources

if __name__ == '__main__':
  test_cpu_range()