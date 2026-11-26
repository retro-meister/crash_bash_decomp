import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import json
import math
from pathlib import Path
import random
import time

import numpy as np

from crashball import Crashball, TINY, score, winner
from polar_ppo import OBS_SIZE, edge_potential, observation


CHOICES = 4
DEFAULT_MARGIN = 0.15


def features(event):
    angle = event["charge_angle"] * math.tau / 4096
    return [*observation(event)[0], math.sin(angle), math.cos(angle)]


def open_workers(args, count, stack):
    return [stack.enter_context(Crashball(args.exe, args.data, arena="polar-panic", hero=TINY, hybrid=True))
            for _ in range(count)]


def benchmark(args, workers, choose):
    def run(env, seeds):
        rows = []
        for seed in seeds:
            packet = env.reset(seed)
            charges = overrides = 0
            while packet["event"] == "charge":
                action = choose(packet)
                charges += 1
                overrides += action != 0
                packet = env.command(f"charge {action} 0")
            if packet["event"] != "end":
                raise RuntimeError(f"Unexpected benchmark packet {packet['event']}")
            victory = winner(packet) if score(packet)[0] else -2
            rows.append((seed, victory, packet["tick"], charges, overrides))
        return rows

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=len(workers)) as pool:
        jobs = [pool.submit(run, env, range(args.seed + i, args.seed + args.episodes, len(workers)))
                for i, env in enumerate(workers)]
        rows = sorted(row for job in jobs for row in job.result())
    _, outcomes, ticks, charges, overrides = zip(*rows)
    elapsed = time.perf_counter() - started
    wins, draws = outcomes.count(0), outcomes.count(-1)
    return dict(episodes=args.episodes, wins=wins, losses=args.episodes - wins - draws, draws=draws,
                charges=sum(charges), overrides=sum(overrides), ticks=sum(ticks),
                seconds=round(elapsed, 3)), outcomes


def branch_value(before, after):
    prior = score(before)
    later = score(after)
    if not later[0]:
        return -1.0 + 0.15 * (sum(prior[1:]) - sum(later[1:]))
    if winner(after) == 0:
        return 1.4
    return (0.65 * (sum(prior[1:]) - sum(later[1:]))
            + 0.06 * min(after["kicks"][0], 3) + 0.4 * edge_potential(after))


def branches(args, workers):
    rng = random.Random(args.seed)
    groups = [workers[i:i + CHOICES] for i in range(0, len(workers), CHOICES)]
    states, returns = [], []
    for start in range(0, args.episodes, len(groups)):
        pending = [(group, rng.getrandbits(32), rng.randrange(7))
                   for group in groups[:min(len(groups), args.episodes - start)]]
        for group, seed, _ in pending:
            for env in group:
                env.send_reset(seed)
        for opportunity in range(7):
            following, probing = [], []
            for group, seed, target in pending:
                packets = [env.receive() for env in group]
                before = packets[0]
                if any(packet != before for packet in packets[1:]):
                    raise RuntimeError(f"Same-seed branches diverged before intervention at seed {seed}")
                if before["event"] == "end":
                    continue
                if before["event"] != "charge":
                    raise RuntimeError(f"Unexpected branch packet {before['event']}")
                if opportunity == target:
                    for action, env in enumerate(group):
                        env.send(f"charge {action} {args.horizon}")
                    probing.append((group, before))
                else:
                    for env in group:
                        env.send("charge 0 0")
                    following.append((group, seed, target))
            for group, before in probing:
                probes = [env.receive() for env in group]
                states.append(features(before))
                values = [branch_value(before, after) for after in probes]
                returns.append([value - values[0] for value in values[1:]])
            pending = following
    return np.asarray(states, dtype=np.float32), np.asarray(returns, dtype=np.float32)


def fit(states, returns, seed):
    import torch
    from torch import nn

    torch.manual_seed(seed)
    torch.set_num_threads(1)
    model = nn.Sequential(nn.Linear(OBS_SIZE + 2, 64), nn.Tanh(), nn.Linear(64, 32), nn.Tanh(), nn.Linear(32, CHOICES - 1))
    nn.init.zeros_(model[-1].weight)
    nn.init.zeros_(model[-1].bias)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    x, y = torch.from_numpy(states), torch.from_numpy(returns)
    for _ in range(22):
        for indices in torch.randperm(len(states)).split(128):
            error = model(x[indices]) - y[indices]
            loss = (error.square() * (1 + 2 * y[indices].abs())).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return {name: value.detach().numpy() for name, value in model.state_dict().items()}


def policy(weights, margin):
    def choose(packet):
        y = np.asarray(features(packet), dtype=np.float32)
        for layer in (0, 2, 4):
            y = weights[f"{layer}.weight"] @ y + weights[f"{layer}.bias"]
            if layer < 4:
                y = np.tanh(y)
        choice = int(y.argmax())
        return choice + 1 if y[choice] > margin else 0

    return choose


def main():
    parser = argparse.ArgumentParser(description="Train or benchmark a charge override for built-in Tiny")
    parser.add_argument("mode", choices=("train", "evaluate"))
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--episodes", type=int, default=400)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=90)
    parser.add_argument("--margin", type=float)
    parser.add_argument("--outcomes", type=Path)
    args = parser.parse_args()
    if args.episodes < 1 or args.workers < 1 or not 1 <= args.horizon <= 900:
        parser.error("Episodes, workers and horizon must be positive and horizon cannot exceed 900")
    if not 0 <= args.seed < 2**32 or args.seed + args.episodes > 2**32:
        parser.error("Seed range must fit in 32 bits")
    if args.mode == "train" and (args.workers < CHOICES or args.weights is None or args.weights.suffix != ".npz"):
        parser.error("Training requires at least four workers and an .npz weights output")
    with ExitStack() as stack:
        count = min(args.workers, args.episodes) if args.mode == "evaluate" else args.workers // CHOICES * CHOICES
        workers = open_workers(args, count, stack)
        if args.mode == "train":
            started = time.perf_counter()
            states, returns = branches(args, workers)
            if len(states) < 32:
                raise RuntimeError("Fewer than 32 matched charge opportunities")
            weights = fit(states, returns, args.seed)
            args.weights.parent.mkdir(parents=True, exist_ok=True)
            margin = DEFAULT_MARGIN if args.margin is None else args.margin
            np.savez(args.weights, polar_hybrid=1, margin=margin, **weights)
            print(json.dumps(dict(branches=len(states), samples=len(states) * CHOICES,
                                  positive_alternatives=int((returns.max(axis=1) > 0.1).sum()),
                                  negative_alternatives=int((returns.min(axis=1) < -0.1).sum()),
                                  seconds=round(time.perf_counter() - started, 3))))
        else:
            if args.weights:
                with np.load(args.weights, allow_pickle=False) as checkpoint:
                    if int(checkpoint["polar_hybrid"]) != 1:
                        raise ValueError("Unsupported hybrid weights")
                    margin = float(checkpoint["margin"]) if "margin" in checkpoint else DEFAULT_MARGIN
                    choose = policy({name: checkpoint[name] for name in checkpoint.files}, margin if args.margin is None else args.margin)
            else:
                choose = lambda packet: 0
            result, outcomes = benchmark(args, workers, choose)
            if args.outcomes:
                args.outcomes.write_text(json.dumps(outcomes) + "\n")
            print(json.dumps(result))


if __name__ == "__main__":
    main()
