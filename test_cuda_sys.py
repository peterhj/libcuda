from guppy import latest as gp

def main():
  task_configs = [
      ("cuda_8_0",  "target-ubuntu_16_04.cached", (("require_cuda", "8.0"),  ("require_distro", "ubuntu 16.04"))),
  ]
  tasks = []
  #batch = gp.batch()
  for taskcfg in task_configs:
    task_kwargs = dict(taskcfg[-1])
    #batch.append(gp.task(
    tasks.append(gp.taskspec(
        name="hello",
        toolchain="rust_nightly",
        **task_kwargs,
        allow_errors=True,
        sh=[
            "CUDA_HOME=/usr/local/cuda cargo -v test --release --features cuda_sys,{} --target-dir {}".format(*taskcfg[:-1]),
        ]
    ))
  gp.print_tasks(tasks)
  #batch.print()

if __name__ == "__main__":
  main()
