"""Evaluate trained policies with optional GUI and save results to outputs/."""
import argparse
import csv
import json
import os
import time
from datetime import datetime

from stable_baselines3 import A2C, PPO, SAC

from rlenv import PegInHoleGymEnv

AGENTS = {
    "sac": (SAC, "./checkpoints/sac_model_160000_steps"),
    "ppo": (PPO, "./checkpoints/ppo_model_160000_steps"),
    "a2c": (A2C, "./checkpoints/a2c_model_160000_steps"),
}

# Reported in README (1000-episode test)
PAPER_SUCCESS_RATE = {
    "sac": 95.6,
    "ppo": 26.9,
    "a2c": 0.0,
}


def evaluate_agent(agent_name, episodes, gui, step_sleep, output_dir, device="cuda"):
    model_class, checkpoint = AGENTS[agent_name]
    env = PegInHoleGymEnv(gui=gui, verbose=gui)
    model = model_class.load(checkpoint, env=env, device=device)

    results = []
    success_count = 0
    t0 = time.time()

    for ep in range(1, episodes + 1):
        obs, _ = env.reset()
        steps = 0
        final_reward = 0.0
        ok = False

        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1
            final_reward = reward
            if step_sleep > 0:
                time.sleep(step_sleep)
            if terminated or truncated:
                ok = info.get("insertion_success", False)
                success_count += int(ok)
                results.append({
                    "episode": ep,
                    "success": ok,
                    "steps": steps,
                    "final_reward": round(float(final_reward), 4),
                })
                status = "Success" if ok else "Failure"
                print(f"[{agent_name.upper()}] Episode {ep}/{episodes}: {status} ({steps} steps)")
                break

    elapsed = time.time() - t0
    success_rate = success_count / episodes * 100
    paper_rate = PAPER_SUCCESS_RATE[agent_name]

    summary = {
        "agent": agent_name,
        "checkpoint": checkpoint,
        "episodes": episodes,
        "successes": success_count,
        "failures": episodes - success_count,
        "success_rate_pct": round(success_rate, 2),
        "paper_success_rate_pct": paper_rate,
        "delta_vs_paper_pct": round(success_rate - paper_rate, 2),
        "elapsed_sec": round(elapsed, 1),
        "gui": gui,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "episodes_detail": results,
    }

    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(output_dir, f"{agent_name}_eval_{ts}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    env.close()
    print(
        f"\n[{agent_name.upper()}] {success_count}/{episodes} = {success_rate:.1f}% "
        f"(paper: {paper_rate}%, delta: {success_rate - paper_rate:+.1f}%) "
        f"-> {out_path}\n"
    )
    return summary


def write_summary_csv(summaries, output_dir):
    path = os.path.join(output_dir, "summary.csv")
    fieldnames = [
        "agent", "episodes", "successes", "failures",
        "success_rate_pct", "paper_success_rate_pct", "delta_vs_paper_pct",
        "elapsed_sec", "result_file",
    ]
    write_header = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for s in summaries:
            writer.writerow({
                "agent": s["agent"],
                "episodes": s["episodes"],
                "successes": s["successes"],
                "failures": s["failures"],
                "success_rate_pct": s["success_rate_pct"],
                "paper_success_rate_pct": s["paper_success_rate_pct"],
                "delta_vs_paper_pct": s["delta_vs_paper_pct"],
                "elapsed_sec": s["elapsed_sec"],
                "result_file": s.get("_result_file", ""),
            })
    return path


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate peg-in-hole RL checkpoints")
    p.add_argument(
        "--agent", choices=["sac", "ppo", "a2c", "all"], default="sac",
        help="Algorithm to evaluate (default: sac)",
    )
    p.add_argument(
        "--episodes", type=int, default=50,
        help="Evaluation episodes (README official test uses 1000; 50 is enough for quick check)",
    )
    p.add_argument(
        "--gui", action="store_true",
        help="Show PyBullet GUI (slower; default headless)",
    )
    p.add_argument(
        "--step-sleep", type=float, default=0.0,
        help="Sleep seconds per step (use ~0.033 with --gui)",
    )
    p.add_argument(
        "--output-dir", default="./outputs",
        help="Directory to save JSON/CSV results",
    )
    p.add_argument("--device", default="cuda", help="cuda or cpu")
    return p.parse_args()


def main():
    args = parse_args()
    agents = list(AGENTS.keys()) if args.agent == "all" else [args.agent]
    step_sleep = args.step_sleep if args.step_sleep > 0 else (1.0 / 30.0 if args.gui else 0.0)

    summaries = []
    for name in agents:
        s = evaluate_agent(
            agent_name=name,
            episodes=args.episodes,
            gui=args.gui,
            step_sleep=step_sleep,
            output_dir=args.output_dir,
            device=args.device,
        )
        summaries.append(s)

    csv_path = write_summary_csv(summaries, args.output_dir)
    print(f"Summary appended -> {csv_path}")

    print("\n=== Comparison vs README ===")
    for s in summaries:
        print(
            f"  {s['agent'].upper():3s}: ours {s['success_rate_pct']:5.1f}%  "
            f"paper {s['paper_success_rate_pct']:5.1f}%  "
            f"delta {s['delta_vs_paper_pct']:+5.1f}%"
        )


if __name__ == "__main__":
    main()
