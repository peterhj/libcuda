import guppy.v1 as gp

def main():
  task_configs = [
      (("cuda_10_0", "target-ubuntu_16_04.cached"), (("require_cuda", "10.0"), ("require_distro", "ubuntu 16.04"))),
      (("cuda_9_2",  "target-ubuntu_16_04.cached"), (("require_cuda", "9.2"),  ("require_distro", "ubuntu 16.04"))),
      (("cuda_9_1",  "target-ubuntu_16_04.cached"), (("require_cuda", "9.1"),  ("require_distro", "ubuntu 16.04"))),
      (("cuda_9_0",  "target-ubuntu_16_04.cached"), (("require_cuda", "9.0"),  ("require_distro", "ubuntu 16.04"))),
      (("cuda_8_0",  "target-ubuntu_16_04.cached"), (("require_cuda", "8.0"),  ("require_distro", "ubuntu 16.04"))),
      (("cuda_7_5",  "target-ubuntu_14_04.cached"), (("require_cuda", "7.5"),  ("require_distro", "ubuntu 14.04"))),
      (("cuda_7_0",  "target-ubuntu_14_04.cached"), (("require_cuda", "7.0"),  ("require_distro", "ubuntu 14.04"))),
      (("cuda_6_5",  "target-ubuntu_14_04.cached"), (("require_cuda", "6.5"),  ("require_distro", "ubuntu 14.04"))),
  ]
  run = gp.run()
  for taskcfg in task_configs:
    task_kwargs = dict(taskcfg[1])
    run.append(gp.task(
        name="{}".format(taskcfg[0][0]),
        toolchain="rust_nightly",
        **task_kwargs,
        sh=[
            "CUDA_HOME=/usr/local/cuda cargo -v test --release --features {} --target-dir {}".format(*taskcfg[0]),
        ]
    ))
    if taskcfg[0][0] == "cuda_8_0":
      # NB: test cuda-sys.
      run.append(gp.task(
          name="{}+cuda_sys".format(taskcfg[0][0]),
          toolchain="rust_nightly",
          **task_kwargs,
          sh=[
              "CUDA_HOME=/usr/local/cuda cargo -v test --release --features cuda_sys,{} --target-dir {}".format(*taskcfg[0]),
          ]
      ))
  gp.out(run)

if __name__ == "__main__":
  main()
