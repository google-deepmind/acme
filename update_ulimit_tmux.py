import os

# vs_extensions = "~/miniconda3/envs/jax_rl/lib/python3.9/site-packages/launchpad/nodes/courier/"
venv_dir = "./venv" # e.g. /home/sam/Code/ML/acme_testing/acme
local_run_dir = os.path.join(venv_dir, "lib/python3.8/site-packages/launchpad/launch/run_locally/")
# vs_extensions = os.path.join(venv_dir, )"~/miniconda3/envs/jax_rl/lib/python3.9/site-packages/launchpad/nodes/courier/"
# print(os.getcwd())
# os.chdir(os.path.expanduser(vs_extensions))
# print(os.getcwd())
# print(os.listdir())

file_name = os.path.join(local_run_dir, "launch_local_tmux.py")
cached_filename = os.path.join(local_run_dir, "launch_local_tmux.py.old")

old_str = r"""

    window_name = command_to_launch.title
"""

new_str = r"""
    print('[launch_local_tmux] inner_command: ', inner_command)

    print('[launch_local_tmux] changing ulimit here always')
    inner_command = f'ulimit -u 1000000 ; {inner_command}'
    print('[launch_local_tmux] inner_command: ', inner_command)
    window_name = command_to_launch.title
"""

replace_tuples = [
  (old_str, new_str),
]

if not os.path.exists(file_name):
    print('file not found. Maybe not installed, maybe wrong version of python in dir string.')
if os.path.exists(cached_filename):
    raise Exception("Seems like you probably already did the replacement already? old file already there")


with open(file_name) as f:
  contents = f.read()

with open(cached_filename, "w") as f:
    f.write(contents)
    print('old version written to courier_utils.py.old')

for og_txt, new_txt in replace_tuples:
  contents = contents.replace(og_txt, new_txt)
  if og_txt not in contents:
    print(og_txt)


with open(file_name, "w") as f:
  f.write(contents)

print('new version written to courier_utils.py')