import os
import sys
import yaml
import time
import json
import torch
import random
import argparse
import numpy as np

import ray  

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from model import PolicyNet
from worker import Worker
from parameter import *  
from mapinpaint.networks import Generator
from mapinpaint.evaluator import Evaluator

def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_predictor(device: torch.device):

    config_path = f'{generator_path}/config.yaml'
    ckpts = sorted([f for f in os.listdir(generator_path) if f.startswith('gen') and f.endswith('.pt')])
    if not ckpts:
        raise FileNotFoundError(f"No generator checkpoint found in {generator_path}")
    checkpoint_path = os.path.join(generator_path, ckpts[0])

    with open(config_path, 'r') as stream:
        config = yaml.load(stream, Loader=yaml.SafeLoader)

    use_cuda = (device.type == "cuda")
    generator = Generator(config['netG'], use_cuda)
    generator.load_state_dict(torch.load(checkpoint_path, map_location=device))

    predictor = Evaluator(config, generator, use_cuda, N_GEN_SAMPLE)
    print(f"[Predictor] Map predictor loaded from {checkpoint_path} (device={device})")
    return predictor


def maybe_load_policy_weights(policy_net: PolicyNet, device: torch.device):
    ckpt_path = os.path.join(model_path, 'checkpoint.pth')
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        if 'policy_model' in ckpt:
            policy_net.load_state_dict(ckpt['policy_model'])
            print(f"[Policy] Loaded policy weights from {ckpt_path}")
        else:
            print(f"[Policy] checkpoint.pth found but no 'policy_model' key; skipping.")
    else:
        print(f"[Policy] No checkpoint found at {ckpt_path}; using randomly initialized policy.")


def infer_gif_path(worker: Worker):
    return os.path.join(worker.run_dir, f"episode_{worker.global_step}_w{worker.meta_agent_id}.gif")


def _fmt_nan(x):
    try:
        return f"{float(x):.6f}"
    except Exception:
        return ""


# ----------------- Ray Actor -----------------
@ray.remote
class Runner:

    def __init__(self, meta_agent_id: int, want_gpu: bool):
        gpu_ids = ray.get_gpu_ids()  
        if want_gpu and torch.cuda.is_available() and len(gpu_ids) > 0:
            cuda_idx = int(gpu_ids[0])
            torch.cuda.set_device(cuda_idx)
            self.device = torch.device(f"cuda:{cuda_idx}")
        else:
            self.device = torch.device("cpu")

        self.policy_net = PolicyNet(NODE_INPUT_DIM, EMBEDDING_DIM).to(self.device)
        self.recorder = None

        self.predictor = load_predictor(self.device)
        self.meta_agent_id = meta_agent_id

    def set_weights(self, weights_state_dict: dict):
        self.policy_net.load_state_dict(weights_state_dict)

    def run_one(self, episode_number: int, save_gif: bool, seed: int = 0):
        if seed is not None:
            set_global_seed(seed + 1000 * self.meta_agent_id + episode_number)

        worker = Worker(meta_agent_id=self.meta_agent_id,
                        policy_net=self.policy_net,
                        predictor=self.predictor,
                        global_step=episode_number,
                        device=self.device,
                        save_image=bool(save_gif))

        t0 = time.time()
        worker.run_episode()
        dt = time.time() - t0

        pm = worker.perf_metrics
        total_travel  = float(pm.get('travel_dist', np.nan))
        max_travel    = float(pm.get('max_travel', np.nan))
        explored_rate = float(pm.get('explored_rate', np.nan))
        success       = bool(pm.get('success_rate', False))

        disc_free_mean_m2 = pm.get('disc_free_mean_m2', np.nan)
        disc_free_std_m2  = pm.get('disc_free_std_m2', np.nan)
        disc_free_cv      = pm.get('disc_free_cv', np.nan)
        disc_both_mean_m2 = pm.get('disc_both_mean_m2', np.nan)
        disc_both_std_m2  = pm.get('disc_both_std_m2', np.nan)
        disc_both_cv      = pm.get('disc_both_cv', np.nan)
        disc_free_per_agent = pm.get('disc_free_m2_per_agent', None)
        disc_occ_per_agent  = pm.get('disc_occ_m2_per_agent', None)

        gif_path = infer_gif_path(worker) if save_gif else None
        out_row = {
            "episode": episode_number,
            "elapsed_sec": dt,
            "total_travel": total_travel,
            "max_travel": max_travel,
            "explored_rate": explored_rate,
            "success": success,
            "gif": gif_path if gif_path and os.path.exists(gif_path) else None,

            "disc_free_mean_m2": float(disc_free_mean_m2) if disc_free_mean_m2 is not None else None,
            "disc_free_std_m2":  float(disc_free_std_m2)  if disc_free_std_m2  is not None else None,
            "disc_free_cv":      float(disc_free_cv)      if disc_free_cv      is not None else None,
            "disc_both_mean_m2": float(disc_both_mean_m2) if disc_both_mean_m2 is not None else None,
            "disc_both_std_m2":  float(disc_both_std_m2)  if disc_both_std_m2  is not None else None,
            "disc_both_cv":      float(disc_both_cv)      if disc_both_cv      is not None else None,

            "disc_free_m2_per_agent": disc_free_per_agent,
            "disc_occ_m2_per_agent":  disc_occ_per_agent,
            "disc_free_ts": pm.get("disc_free_ts", None),
            "disc_occ_ts":  pm.get("disc_occ_ts",  None),

            "area_std_m2": float(pm.get('area_std_m2', np.nan))
        }

        info = {
            "id": self.meta_agent_id,
            "map_path": getattr(worker.env, "map_path", None)
        }
        return out_row, info


def main():
    parser = argparse.ArgumentParser(description="Evaluate DRL planner (Ray parallel) and export GIF/metrics.")
    parser.add_argument("--episodes", type=int, default=60, help="Number of episodes to run")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--save-gif", action="store_true", default=True, help="Save GIFs during evaluation")
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"],
                        help="Force device for MAIN process; actors still choose by Ray resources automatically")
    parser.add_argument("--metrics-out", type=str, default=None,
                        help="Optional: path to save a JSON metrics summary")
    parser.add_argument("--csv-out", type=str, default=None,
                        help="Optional: path to save a CSV (default: checkpoints/.../test_metrics.csv)")
    parser.add_argument("--num-workers", type=int, default=15, help="Number of Ray actors (parallel workers)")
    parser.add_argument("--gpus-per-worker", type=float, default=0.0,
                        help="GPUs reserved per worker (Ray). 0 = CPU-only actor")
    parser.add_argument("--ray-address", type=str, default=None,
                        help="Ray cluster address (None = start local Ray)")
    args = parser.parse_args()

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda") if (USE_GPU_GLOBAL and torch.cuda.is_available()) else torch.device("cpu")

    print(f"[Info] Main process using device: {device}")
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(gifs_path, exist_ok=True)
    set_global_seed(args.seed)

    policy_net = PolicyNet(NODE_INPUT_DIM, EMBEDDING_DIM).to(device)
    maybe_load_policy_weights(policy_net, device)
    weights_state = {k: v.cpu() for k, v in policy_net.state_dict().items()}

    if args.ray_address:
        ray.init(address=args.ray_address)
    else:
        ray.init()
    try:
        print("[Ray] cluster_resources:", ray.cluster_resources())
    except Exception:
        pass

    want_gpu = (args.gpus_per_worker > 0.0)  
    runners = []
    for wid in range(args.num_workers):
        actor = Runner.options(num_cpus=1, num_gpus=args.gpus_per_worker).remote(
            meta_agent_id=wid,
            want_gpu=want_gpu
        )
        ray.get(actor.set_weights.remote(weights_state))
        runners.append(actor)

    all_metrics = []
    total_eps = args.episodes
    next_ep = 1
    inflight = []

    for r in runners:
        if next_ep <= total_eps:
            inflight.append(
                r.run_one.remote(episode_number=next_ep, save_gif=args.save_gif, seed=args.seed)
            )
            next_ep += 1

    while inflight:
        done_ids, inflight = ray.wait(inflight, num_returns=1, timeout=None)
        for did in done_ids:
            row, info = ray.get(did)
            print(f"\n=== Episode {row['episode']}/{total_eps} ===")
            print(f"Elapsed: {row['elapsed_sec']:.2f}s")
            print(f"Total travel distance: {row['total_travel']:.3f}")
            print(f"Max travel distance (per-agent): {row['max_travel']:.3f}")
            print(f"Explored rate: {row['explored_rate']:.4f}")
            print(f"Success flag: {bool(row['success'])}")
            dbcv = row.get("disc_both_cv", None)
            if dbcv is not None and not (isinstance(dbcv, float) and np.isnan(dbcv)):
                print("[Balance] discovered FREE: mean={:.2f} m^2, std={:.2f} m^2, CV={:.3f}".format(
                    float(row.get("disc_free_mean_m2", np.nan)),
                    float(row.get("disc_free_std_m2",  np.nan)),
                    float(row.get("disc_free_cv",      np.nan))))
                print("[Balance] discovered FREE+OCC: mean={:.2f} m^2, std={:.2f} m^2, CV={:.3f}".format(
                    float(row.get("disc_both_mean_m2", np.nan)),
                    float(row.get("disc_both_std_m2",  np.nan)),
                    float(row.get("disc_both_cv",      np.nan))))
                if row.get("disc_free_m2_per_agent", None) is not None:
                    print("[Per-agent] discovered FREE m^2:", [f"{float(x):.1f}" for x in row["disc_free_m2_per_agent"]])
                if row.get("disc_occ_m2_per_agent", None) is not None:
                    print("[Per-agent] discovered OCC  m^2:", [f"{float(x):.1f}" for x in row["disc_occ_m2_per_agent"]])
            else:
                print("[Balance] discovery stats unavailable (did Env.pop_discovery_masks / get_discovered_area / get_map_balance_stats run?)")
            print(f"GIF: {row['gif'] if row['gif'] else '(disabled or not found)'}")
            all_metrics.append(row)

            if next_ep <= total_eps:
                target_actor = runners[(next_ep - 1) % len(runners)]
                inflight.append(
                    target_actor.run_one.remote(episode_number=next_ep, save_gif=args.save_gif, seed=args.seed)
                )
                next_ep += 1

    # -------- 保存 JSON 汇总 --------
    if args.metrics_out:
        out_dir = os.path.dirname(os.path.abspath(args.metrics_out))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.metrics_out, "w", encoding="utf-8") as f:
            json.dump(all_metrics, f, ensure_ascii=False, indent=2)
        print(f"\n[Metrics] Saved metrics to {args.metrics_out}")

    # -------- 保存 CSV--------
    csv_path = args.csv_out if args.csv_out else os.path.join(model_path, "test_metrics.csv")
    try:
        import csv
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "episode", "elapsed_sec", "total_travel", "max_travel",
                "explored_rate", "success", "gif",
                "disc_free_mean_m2", "disc_free_std_m2", "disc_free_cv",
                "disc_both_mean_m2", "disc_both_std_m2", "disc_both_cv",
                "area_std_m2"   # <--- 新增
            ])
            for r in all_metrics:
                def _fmt(x):
                    try:
                        return f"{float(x):.6f}"
                    except Exception:
                        return ""
                writer.writerow([
                    r["episode"], f"{r['elapsed_sec']:.3f}",
                    _fmt(r.get('total_travel', np.nan)),
                    _fmt(r.get('max_travel', np.nan)),
                    _fmt(r.get('explored_rate', np.nan)),
                    int(bool(r.get("success", False))),
                    r["gif"] if r.get("gif") else "",
                    _fmt(r.get("disc_free_mean_m2")),
                    _fmt(r.get("disc_free_std_m2")),
                    _fmt(r.get("disc_free_cv")),
                    _fmt(r.get("disc_both_mean_m2")),
                    _fmt(r.get("disc_both_std_m2")),
                    _fmt(r.get("disc_both_cv")),
                    _fmt(r.get("area_std_m2")),   # <--- 新增
                ])
        print(f"[Metrics] CSV saved to {csv_path}")
    except Exception as e:
        print(f"[Metrics] Failed to write CSV: {e}")


if __name__ == "__main__":
    main()