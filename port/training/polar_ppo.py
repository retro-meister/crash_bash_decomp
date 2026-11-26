import argparse
from collections import defaultdict
from contextlib import ExitStack
import json
import math
from pathlib import Path
import random
import time
from types import SimpleNamespace

import numpy as np

from crashball import POLAR_ACTIONS, TINY, checkpoint_policy, measure, score, simulator, winner
from polar import steer


GLOBAL_SIZE = 32
OPPONENT_SIZE = 14
OPPONENTS = 3
OBS_SIZE = GLOBAL_SIZE + OPPONENT_SIZE * OPPONENTS
DIRECTIONS = len(POLAR_ACTIONS) // 2
STAGES = ("easy", "normal", "hard", "drill")


def encode_player(player, origin):
    x, z, vx, vz, heading, charge, mode, _, _, _, phase, frozen, ready, mass = player
    angle = heading * math.tau / 4096
    return [(x - origin[0]) / 2048, (z - origin[1]) / 2048, (vx - origin[2]) / 128,
            (vz - origin[3]) / 128, math.sin(angle), math.cos(angle), charge / 256, mode == 0x51,
            phase / 24, min(frozen, 30) / 30, ready, mass / 512, mode in (0x4B, 0x4C, 0x4D)]


def observation(result):
    frame = result["frames"][-1]
    if "edge" not in frame:
        raise ValueError("Rebuild the native port to use the Polar Panic PPO observation")
    me, *opponents = frame["polar"]
    features = encode_player(me, (0, 0, 0, 0))
    features.extend(d / 2048 for d in frame["edge"])
    for x, z in frame["obstacles"]:
        features.extend((x / 2048, z / 2048, 1))
    features.extend((0, 0, 0) * (3 - len(frame["obstacles"])))
    minutes, seconds = (frame["clock"] >> 16) & 255, (frame["clock"] >> 8) & 255
    left = 60 * (10 * (minutes >> 4) + (minutes & 15)) + 10 * (seconds >> 4) + (seconds & 15)
    features.extend((left / 90, frame["square_held"]))
    for opponent in opponents:
        features.extend([*encode_player(opponent, me), 1] if opponent[7] else [0] * OPPONENT_SIZE)
    can_charge = me[12] and not (frame["square_held"] or me[11]) and me[6] in (0x49, 0x4A, 0x54, 0x55)
    return features, bool(can_charge)


def edge_potential(result):
    frame = result["frames"][-1]
    vx, vz = frame["polar"][0][2:4]
    direction = steer(vx, vz)
    speed = math.hypot(vx, vz)
    if not direction:
        return 0.0
    clearance = frame["edge"][direction - 1]
    return -0.25 * min(1.0, max(0.0, (speed * 12 + 160 - clearance) / 1000))


def load_policy(checkpoint):
    if int(checkpoint["polar_ppo"]) != 1:
        raise ValueError("Unsupported Polar Panic PPO checkpoint")
    weights = {name.removeprefix("ppo."): checkpoint[name] for name in checkpoint.files if name.startswith("ppo.")}

    def linear(x, name):
        return x @ weights[name + ".weight"].T + weights[name + ".bias"]

    def action(result, _rng):
        features, can_charge = observation(result)
        values = np.asarray(features, dtype=np.float32)
        opponents = values[GLOBAL_SIZE:].reshape(OPPONENTS, OPPONENT_SIZE)
        active = opponents[:, -1:]
        encoded = np.tanh(linear(opponents, "opponents")) * active
        pooled = [encoded.sum(axis=0) / max(active.sum(), 1), encoded.max(axis=0)]
        shared = np.tanh(linear(np.concatenate([values[:GLOBAL_SIZE], *pooled]), "body.0"))
        shared = np.tanh(linear(shared, "body.2"))
        logits = linear(shared, "actor")
        return int(logits.argmax() if can_charge else logits[:DIRECTIONS].argmax())

    return action


def save(model, path, repeat):
    weights = {"ppo." + name: value.detach().cpu().numpy() for name, value in model.state_dict().items()}
    np.savez(path, polar_ppo=1, hero=TINY, repeat=repeat, **weights)


def difficulty(rng, elapsed, curriculum):
    if elapsed < curriculum // 3:
        return rng.choices((0, 1, 2), (80, 15, 5))[0]
    if elapsed < curriculum:
        return rng.choices((0, 1, 2), (20, 50, 30))[0]
    return rng.choices((0, 1, 2), (5, 15, 80))[0]


def demonstrations(args, envs, rng, torch):
    coach = checkpoint_policy(SimpleNamespace(arena=args.arena, weights=args.coach))
    results = [env.reset(rng.getrandbits(32), difficulty=args.difficulty) for env in envs]
    states, masks, actions = [], [], []
    while len(actions) < args.coach_steps:
        active = envs[:min(len(envs), args.coach_steps - len(actions))]
        for env, result in zip(active, results):
            features, can_charge = observation(result)
            action = coach(result, rng)
            if not can_charge:
                action %= DIRECTIONS
            states.append(features)
            masks.append(can_charge)
            actions.append(action)
            env.send_step(action)
        for i, env in enumerate(active):
            result = env.receive()
            if result["done"] or score(result)[0] == 0:
                result = env.reset(rng.getrandbits(32), difficulty=args.difficulty)
            results[i] = result
    return torch.tensor(states), torch.tensor(masks), torch.tensor(actions)


def train(args):
    import torch
    from torch import nn
    from torch.distributions import Categorical

    class Policy(nn.Module):
        def __init__(self):
            super().__init__()
            self.opponents = nn.Linear(OPPONENT_SIZE, 32)
            self.body = nn.Sequential(nn.Linear(GLOBAL_SIZE + 64, 128), nn.Tanh(), nn.Linear(128, 128), nn.Tanh())
            self.actor = nn.Linear(128, len(POLAR_ACTIONS))
            self.critic = nn.Linear(128, 1)
            for layer in (self.opponents, self.body[0], self.body[2], self.actor, self.critic):
                scale = 0.01 if layer is self.actor else 1.0 if layer is self.critic else math.sqrt(2)
                nn.init.orthogonal_(layer.weight, scale)
                nn.init.zeros_(layer.bias)

        def forward(self, state, can_charge):
            opponents = state[:, GLOBAL_SIZE:].reshape(-1, OPPONENTS, OPPONENT_SIZE)
            alive = opponents[:, :, -1:]
            encoded = self.opponents(opponents).tanh() * alive
            pooled = torch.cat((encoded.sum(1) / alive.sum(1).clamp_min(1), encoded.amax(1)), dim=1)
            shared = self.body(torch.cat((state[:, :GLOBAL_SIZE], pooled), dim=1))
            logits = self.actor(shared)
            illegal = (~can_charge[:, None]) & (torch.arange(len(POLAR_ACTIONS)) >= DIRECTIONS)
            return Categorical(logits=logits.masked_fill(illegal, -1e9)), self.critic(shared).squeeze(1)

    started = time.perf_counter()
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    model = Policy()
    if args.weights:
        with np.load(args.weights, allow_pickle=False) as checkpoint:
            if "polar_ppo" not in checkpoint or int(checkpoint.get("repeat", 1)) != args.repeat:
                raise ValueError("Initial checkpoint must be PPO weights with the same action repeat")
            model.load_state_dict({name: torch.from_numpy(checkpoint["ppo." + name])
                                   for name in model.state_dict()})
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, eps=1e-5)
    gamma, gae_lambda = 0.998 ** args.repeat, 0.98

    done = 0
    updates = 0
    next_report = args.report_every
    results = []
    levels = []
    counters = defaultdict(lambda: [0, 0, 0])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        envs = [stack.enter_context(simulator(args)) for _ in range(args.workers)]
        if args.coach:
            examples, legal, targets = demonstrations(args, envs, rng, torch)
            charge_examples = targets >= DIRECTIONS
            weights = torch.where(charge_examples, args.coach_charge_weight, 1.0)
            for _ in range(8):
                for indices in torch.randperm(len(examples)).split(512):
                    distribution, _ = model(examples[indices], legal[indices])
                    loss = -(distribution.log_prob(targets[indices]) * weights[indices]).mean()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            optimizer.state.clear()
            save(model, args.output.with_stem(args.output.stem + "-coach"), args.repeat)
            print(json.dumps(dict(demonstrations=len(examples), charges=int(charge_examples.sum()),
                                  seconds=round(time.perf_counter() - started, 1))), flush=True)
        if args.validate_rounds:
            validation = [stack.enter_context(simulator(args)) for _ in range(min(args.workers, args.validate_rounds))]
            validation_args = SimpleNamespace(difficulty=2, max_steps=args.max_steps)
            seeds = range(args.validate_seed, args.validate_seed + args.validate_rounds)
            best_wins = -1

        def reset(env):
            level = difficulty(rng, done, args.curriculum) if args.curriculum else args.difficulty
            if done < args.drill_steps and rng.random() > done / args.drill_steps:
                return env.reset(rng.getrandbits(32), players=rng.choice((2, 3, 4)), difficulty=0), 3
            return env.reset(rng.getrandbits(32), difficulty=level), level

        for env in envs:
            result, level = reset(env)
            results.append(result)
            levels.append(level)
        potentials = [edge_potential(result) for result in results]
        while done < args.steps:
            horizon = min(args.horizon, (args.steps - done) // args.workers)
            shape = (horizon, args.workers)
            states = torch.empty((*shape, OBS_SIZE), dtype=torch.float32)
            masks = torch.empty(shape, dtype=torch.bool)
            actions = torch.empty(shape, dtype=torch.int64)
            log_probs = torch.empty(shape)
            values = torch.empty(shape)
            rewards = torch.empty(shape)
            continuing = torch.empty(shape)
            for t in range(horizon):
                obs, legal = zip(*(observation(result) for result in results))
                states[t] = torch.tensor(obs)
                masks[t] = torch.tensor(legal)
                with torch.no_grad():
                    distribution, values[t] = model(states[t], masks[t])
                    actions[t] = distribution.sample()
                    log_probs[t] = distribution.log_prob(actions[t])
                for env, action in zip(envs, actions[t].tolist(), strict=True):
                    env.send_step(action)
                done += args.workers
                for i, env in enumerate(envs):
                    result = env.receive()
                    alive = score(result)[0] > 0
                    terminal = result["done"] or not alive
                    won = terminal and alive and winner(result) == 0
                    eliminated = sum(score(results[i])[1:]) - sum(score(result)[1:])
                    contacts = result["kicks"][0]
                    potential = 0.0 if terminal else edge_potential(result)
                    rewards[t, i] = ((1.0 if won else -1.0) if terminal else 0.0) + 0.12 * eliminated
                    rewards[t, i] += gamma * potential - potentials[i] + 0.02 * min(contacts, 1)
                    continuing[t, i] = not terminal
                    counters[levels[i]][2] += contacts
                    if terminal:
                        counters[levels[i]][0] += 1
                        counters[levels[i]][1] += won
                        result, levels[i] = reset(env)
                        potential = edge_potential(result)
                    results[i] = result
                    potentials[i] = potential

            with torch.no_grad():
                obs, legal = zip(*(observation(result) for result in results))
                _, next_value = model(torch.tensor(obs), torch.tensor(legal))
                advantage = torch.empty_like(rewards)
                estimate = torch.zeros(args.workers)
                for t in reversed(range(horizon)):
                    alive = continuing[t]
                    delta = rewards[t] + gamma * alive * next_value - values[t]
                    estimate = delta + gamma * gae_lambda * alive * estimate
                    advantage[t] = estimate
                    next_value = values[t]
                returns = advantage + values
            states, masks = states.flatten(0, 1), masks.flatten()
            actions, old_log_probs = actions.flatten(), log_probs.flatten()
            advantage, returns = advantage.flatten(), returns.flatten()
            advantage = (advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1e-8)
            for _ in range(4):
                for indices in torch.randperm(len(states)).split(512):
                    distribution, value = model(states[indices], masks[indices])
                    ratio = (distribution.log_prob(actions[indices]) - old_log_probs[indices]).exp()
                    actor = torch.minimum(ratio * advantage[indices], ratio.clamp(0.8, 1.2) * advantage[indices])
                    loss = -actor.mean() + 0.5 * (value - returns[indices]).square().mean() - 0.01 * distribution.entropy().mean()
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    optimizer.step()
            updates += 1
            if done >= next_report or done == args.steps:
                save(model, args.output, args.repeat)
                report = dict(steps=done, updates=updates, seconds=round(time.perf_counter() - started, 1),
                              training={STAGES[k]: dict(rounds=v[0], wins=v[1], contacts=v[2])
                                        for k, v in sorted(counters.items())})
                counters.clear()
                if args.validate_rounds:
                    with np.load(args.output, allow_pickle=False) as checkpoint:
                        action = load_policy(checkpoint)
                    wins, losses, draws, _, _, _ = measure(validation_args, validation, seeds, action)
                    report["validation"] = dict(wins=wins, losses=losses, draws_or_truncated=draws)
                    if wins > best_wins:
                        best_wins = wins
                        save(model, args.output.with_stem(args.output.stem + "-best"), args.repeat)
                print(json.dumps(report), flush=True)
                next_report = done + args.report_every


def main():
    parser = argparse.ArgumentParser(description="Train a masked PPO policy for Tiny in Polar Panic")
    parser.add_argument("--exe", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--coach", type=Path)
    parser.add_argument("--coach-steps", type=int, default=120_000)
    parser.add_argument("--coach-charge-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=52193)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--steps", type=int, default=1_200_000)
    parser.add_argument("--horizon", type=int, default=256)
    parser.add_argument("--repeat", type=int, choices=(1, 2, 3), default=1,
                        help="Game ticks to hold each action")
    parser.add_argument("--curriculum", type=int, default=300_000)
    parser.add_argument("--drill-steps", type=int, default=0,
                        help="Fade out practice rounds with stationary opponents over this many steps")
    parser.add_argument("--difficulty", type=int, choices=range(3), default=2)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--report-every", type=int, default=200_000)
    parser.add_argument("--validate-rounds", type=int, default=64)
    parser.add_argument("--validate-seed", type=int, default=51001)
    parser.set_defaults(arena="polar-panic", hero=TINY, max_steps=3600)
    args = parser.parse_args()
    if args.workers < 1 or args.steps < args.workers or args.horizon < 1 or args.report_every < 1:
        parser.error("Workers, horizon and report interval must be positive, and steps must cover at least one tick per worker")
    if args.curriculum < 0 or args.drill_steps < 0 or args.validate_rounds < 0 or args.learning_rate <= 0:
        parser.error("Curriculum and validation counts must be nonnegative and learning rate must be positive")
    if args.coach_steps < 1 or args.coach_charge_weight <= 0 or (args.coach and args.weights):
        parser.error("Coach steps and charge weight must be positive and a coach cannot be combined with initial weights")
    if not 0 <= args.seed < 2**32 or not 0 <= args.validate_seed < 2**32 or args.validate_seed + args.validate_rounds > 2**32:
        parser.error("Seeds must fit in an unsigned 32-bit integer")
    if args.output.suffix != ".npz":
        parser.error("Output must have an .npz suffix")
    args.steps -= args.steps % args.workers
    train(args)


if __name__ == "__main__":
    main()
