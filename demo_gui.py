"""Quick PyBullet GUI demo: random end-effector motions in peg-in-hole scene."""
import time

import numpy as np

from rlenv import PegInHoleGymEnv


def main():
    env = PegInHoleGymEnv()
    obs, info = env.reset()

    try:
        for episode in range(3):
            print(f"--- Episode {episode + 1} ---")
            for _ in range(env.max_steps):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                time.sleep(1.0 / 60.0)
                if terminated or truncated:
                    print(f"Episode ended: success={info.get('insertion_success', False)}")
                    obs, info = env.reset()
                    break
    finally:
        env.close()


if __name__ == "__main__":
    main()
