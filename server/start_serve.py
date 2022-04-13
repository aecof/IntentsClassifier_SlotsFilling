import ray
from ray import serve
import subprocess 


def main():
    # Connect to the local running Ray cluster.
    ray.init(address='auto', namespace="serve-example", ignore_reinit_error=True)
    # Start the Ray Serve processes within the Ray cluster.
    serve.start(detached=True, http_options={"host": "0.0.0.0"})

if __name__ == '__main__':
    main()
